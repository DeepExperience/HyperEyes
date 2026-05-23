# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Migrated from RedAccel sly_agent:aipet_rl/agent/video_agent.py
# Changes:
#   1. ToolBase → BaseInteractionEnv
#   2. reset(raw_prompt, multi_modal_data, ...) → reset() returning (obs, info)
#   3. execute(action_string) → step(response_text) returning (obs_dict, done, info)
#   4. 初始图片和 question 通过 build_env(sample, args) 传入
#   5. asyncio.run() 内的搜索改为 await（Relax rollout 已在 async context）

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple 

from examples.deepeyes.base_env import BaseInteractionEnv
from relax.utils.types import Sample

from .sly_tools.search_tools.formatter import text_search_observation, image_search_observation
from .sly_tools.search_tools.config import TextSearchConfig, ImageSearchConfig, WikiConfig

WIKI_SERVER_URL = "EMPTY"
MAX_TOOL_CALLS_NUM = 8
MAX_ITERATIONS = MAX_TOOL_CALLS_NUM + 1

TEXT_SEARCH_CONFIG = TextSearchConfig(
    version = "sly",
    keys=["google"],
    threshold=0.6,
    wiki=WikiConfig(mode="off", WIKI_SERVER_URL=WIKI_SERVER_URL),
)

IMAGES_SEARCH_CONFIG = ImageSearchConfig(
    version = "sly",
    keys=["google"],
    threshold=0.6,
    wiki=WikiConfig(mode="off", WIKI_SERVER_URL=WIKI_SERVER_URL), 
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  并行搜索 helpers（保留原逻辑）
# ---------------------------------------------------------------------------

def crop_area_from_image(source_pil: Image.Image, area: List[float]) -> Optional[Image.Image]:
    """
    根据归一化坐标 [x1, y1, x2, y2] 裁剪图像。
    裁剪结果 resize 到 max_length=1000 保持比例。
    失败返回 None。
    """
    img_w, img_h = source_pil.size
    x1_r, y1_r, x2_r, y2_r = area

    x1 = max(0,     int(x1_r * img_w))
    y1 = max(0,     int(y1_r * img_h))
    x2 = min(img_w, int(x2_r * img_w))
    y2 = min(img_h, int(y2_r * img_h))

    if x2 <= x1 or y2 <= y1:
        print(f"    -> 区域无效: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return None

    cropped = source_pil.crop((x1, y1, x2, y2))
    cur_w, cur_h = cropped.size
    if max(cur_w, cur_h) > 1000:
        ratio   = 1000.0 / max(cur_w, cur_h)
        cropped = cropped.resize(
            (int(cur_w * ratio), int(cur_h * ratio)),
            Image.Resampling.LANCZOS
        )
    return cropped


async def parallel_image_search_observation(img_ids: List[str], registry: Dict) -> str:
    """
    并行图像搜索。
    img_ids: 图片 ID 列表，如 ["img_1", "img_2"]
    registry 中每个条目的 "image" 字段存放 PIL Image 对象。
    """
    print(f"image search 并行搜索: {img_ids}")
    if not img_ids:
        return "未检测到有效的图片ID。"

    tasks, valid_ids = [], []
    st = time.time()
    for tid in img_ids:
        if tid in registry:
            img_obj = registry[tid]["image"]   # PIL Image
            tasks.append(image_search_observation(img_obj, IMAGES_SEARCH_CONFIG))
            valid_ids.append(tid)
        else:
            print(f"    -> ⚠️ 警告: ID {tid} 不存在于 registry")

    if not tasks:
        return "提供的图片ID均无效。"

    results = await asyncio.gather(*tasks)
    combined_obs = ""
    if not results:
        print("parallel_image_search_observation 结果为空")
    for tid, (obs_text, _) in zip(valid_ids, results):
        combined_obs += f"\n[ {tid} 图像搜索结果]: \n{obs_text}\n"
    print(f"image search cost time: {time.time() - st:.2f}s")
    return combined_obs


async def parallel_text_search_observation(raw_input) -> str:
    """
    并行文本搜索。
    raw_input 支持 list[str]（新格式）或 '|' 分隔的字符串（旧格式兼容）。
    """
    if isinstance(raw_input, list):
        queries = [q.strip() for q in raw_input if isinstance(q, str) and q.strip()]
    else:
        queries = [q.strip() for q in str(raw_input).split('|') if q.strip()]

    print(f"text search 并行搜索：{queries}")
    if not queries:
        return "未检测到有效的搜索关键词"

    tasks = []
    st = time.time()
    for q in queries:
        tasks.append(text_search_observation(q, TEXT_SEARCH_CONFIG))

    results = await asyncio.gather(*tasks)

    combined_obs = ""
    if not results:
        print("parallel_text_search_observation 结果为空")
    for i, (q, res) in enumerate(zip(queries, results)):
        combined_obs += f"[查询词 {i+1}]: {q} 搜索结果:\n{res}\n"
    print(f"text search cost time: {time.time() - st:.2f}s")
    return combined_obs



# ---------------------------------------------------------------------------
#  Action 解析（保留原逻辑）
# ---------------------------------------------------------------------------


def _extract_action(text: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
    """解析模型输出，提取 reason, tool_call, answer。"""
    reason, tool_call, answer = None, None, None

    # 提取 <reason>
    m_reason = re.search(r"<reason>(.*?)</reason>", text, flags=re.S)
    if not m_reason:
        m_reason_open = re.search(r"<reason>(.*)", text, flags=re.S)
        if m_reason_open:
            reason = m_reason_open.group(1).strip()
    else:
        reason = m_reason.group(1).strip()

    # 提取 <tool_call>
    m_tool = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.S)
    if not m_tool:
        m_tool_open = re.search(r"<tool_call>(.*)", text, flags=re.S)
        if m_tool_open:
            raw_json = m_tool_open.group(1).strip()
            try:
                tool_call = json.loads(raw_json)
            except Exception:
                tool_call = None
    else:
        try:
            tool_call = json.loads(m_tool.group(1).strip())
        except json.JSONDecodeError:
            tool_call = None

    # 提取 <answer>
    m_ans = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if not m_ans:
        m_ans_open = re.search(r"<answer>(.*)", text, flags=re.S)
        if m_ans_open:
            answer = m_ans_open.group(1).strip()
    else:
        answer = m_ans.group(1).strip()

    if answer:
        answer = re.sub(r"[<>\n\r\t]+$", "", answer).strip()

    return reason, tool_call, answer


# ---------------------------------------------------------------------------
#  VideoSearchEnv — Relax 交互环境
# ---------------------------------------------------------------------------


class HyperEyesEnv(BaseInteractionEnv):
    """搜索 Agent 交互环境（文字搜索 + 图像搜索）。

    对应 RedAccel 的 ``HyperEyesTools(ToolBase)``。
    """

    def __init__(
        self,
        *,
        max_turns: int | None = None,
        images: list | None = None,
        question: str = "",
    ):
        self.max_turns = max_turns or MAX_ITERATIONS
        self.max_tool_calls = MAX_TOOL_CALLS_NUM
        self.question = question

        # 图像注册表
        self.image_registry: Dict[str, Dict] = {}
        self.image_counter = 0
        if images:
            for idx, img in enumerate(images):
                img_id = f"img_{self.image_counter}"
                self.image_registry[img_id] = {"image": img, "desc": f"Initial Environment Image {idx}"}
                self.image_counter += 1

        self.iterations = 0
        self.tool_calls_num = 0


    # -- BaseInteractionEnv 接口 --

    def reset(self):
        self.iterations = 0
        self.tool_calls_num = 0
        observation: dict[str, Any] = {}
        info = {"has_images": len(self.image_registry) > 0}
        return observation, info

    def close(self):
        self.image_registry.clear()

    async def step(self, response_text: str):
        """处理模型回复，执行工具调用或检测 answer。

        Returns:
            (obs_dict, done, info) — 遵循 Relax BaseInteractionEnv 协议。

        NOTE: 此方法必须为 async，因为工具调用涉及网络 I/O。
        使用 time.sleep / run_until_complete 会阻塞事件循环，
        导致同一循环上的其他协程（包括其他 sample 的 generate）全部卡住。
        """
        self.iterations += 1
        reason, tool_call, answer = _extract_action(response_text)

        # 如果有 answer 或无 tool_call，episode 结束
        if answer:
            return {"obs_str": "Answer received.", "role": "user"}, True, {"final_answer": True}

        if not tool_call:
            return {"obs_str": "No tool call or answer detected.", "role": "user"}, True, {}

        # 达到最大轮次
        if self.iterations >= self.max_turns:
            return {"obs_str": "Max iterations reached.", "role": "user"}, True, {}

        try:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})
            current_image: list = []

            # 随机延迟，防止瞬间请求过高（用 asyncio.sleep 避免阻塞事件循环）
            await asyncio.sleep(random.uniform(0.1, 0.5))

            if tool_name == "image_search":
                src_id = tool_args.get("image_id", "img_0").strip()
                areas  = tool_args.get("area", None)

                area_results   = []

                if src_id not in self.image_registry:
                    obs_content = f"图片ID {src_id} 无效"
                elif areas:
                    source_pil   = self.image_registry[src_id]["image"]
                    area_results = []   # [(img_id, success:bool), ...]
                    for area in areas:
                        new_id      = f"img_{self.image_counter}"
                        cropped_pil = crop_area_from_image(source_pil, area)

                        if cropped_pil is None:
                            self.image_counter += 1
                            area_results.append((new_id, False))
                            continue
                        self.image_counter += 1
                        self.image_registry[new_id] = {
                            "image": cropped_pil,
                            "desc":  f"Crop from {src_id}: area={area}",
                        }
                        area_results.append((new_id, True))
                        current_image.append(cropped_pil)
                    success_ids = [nid for nid, ok in area_results if ok]
                    failed_ids  = [nid for nid, ok in area_results if not ok]
                    if not success_ids:
                        obs_content = "所有区域裁剪均失败"
                    else:
                        search_obs = await parallel_image_search_observation(success_ids, self.image_registry)
                    
                        header_parts = []
                        if success_ids: header_parts.append(f"裁剪成功: {', '.join(success_ids)}")
                        if failed_ids:  header_parts.append(f"裁剪失败: {', '.join(failed_ids)}")
                        header_line = "  ".join(header_parts)

                        failed_obs = "".join(
                            f"\n[ {fid} 图像搜索结果]: 裁剪失败\n"
                            for fid in failed_ids
                        )
                        obs_content = f"{header_line}\n{search_obs}{failed_obs}"

                else:
                    # ── Case B：不带 area，整图搜索 ───────────────
                    obs_content = await parallel_image_search_observation([src_id], self.image_registry)

            elif tool_name == "text_search":
                text_query = tool_args.get("input", "")
                obs_content = await parallel_text_search_observation(text_query)
                

            else:
                raise ValueError(f"Unknown tool name: {tool_name}")

            self.tool_calls_num += 1
            ori_question = f"原始问题（Original Question）： {self.question}"

            remaining = self.max_tool_calls - self.tool_calls_num
            if remaining == 0:
                tool_info = (
                    "\n(系统提示: 您不能再调用工具了；请直接输出 <reason><answer>)"
                    "（System hint: you have no tool calls remaining, please directly output <reason><answer>）"
                )
            elif remaining == 1:
                tool_info = (
                    f"\n(系统提示: 您还能再调用 {remaining} 次工具，请继续输出新的<reason>/<tool_call>或<answer>)"
                    f"（System hint: you still have {remaining} tool call(s) remaining, please continue to output <reason>/<tool_call> or <answer>）"
                )
            else:
                tool_info = ""

            obs_text = (
                f"{'<image>' * len(current_image)}"
                f"<tool_response>{obs_content}</tool_response>"
                f"{ori_question}{tool_info}"
            )

            obs: dict[str, Any] = {"obs_str": obs_text, "role": "user"}
            if current_image:
                obs["multi_modal_data"] = {"image": current_image}

            info = {"status": "success", "tool_used": tool_name}
            await asyncio.sleep(1)
            return obs, False, info

        except Exception as e:
            logger.exception("Tool execution failed")  # 添加详细堆栈
            obs = {"obs_str": f"Error: {str(e)}", "role": "user"}
            self.tool_calls_num += 1
            return obs, False, {"error": str(e), "status": "failed"}


# ---------------------------------------------------------------------------
#  build_env 工厂函数 — Relax rollout 框架调用入口
# ---------------------------------------------------------------------------


def _extract_images(sample: Sample | None) -> list:
    """从 sample 的 multimodal_inputs 中提取图片列表。"""
    if sample is None:
        return []
    multimodal = sample.multimodal_inputs or {}
    if isinstance(multimodal, dict):
        for key in ("images", "image"):
            imgs = multimodal.get(key)
            if imgs:
                return list(imgs)
    return []


def _extract_question(sample: Sample | None) -> str:
    """从 sample 的 prompt / metadata 中提取用户问题文本。"""
    if sample is None:
        return ""
    meta = sample.metadata or {}
    question = meta.get("question", "")
    if question:
        return str(question).replace("<image>", "").strip()

    # fallback: 从 prompt messages 中提取
    prompt = sample.prompt
    if isinstance(prompt, list):
        for msg in prompt:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item["text"].replace("<image>", "").strip()
                elif isinstance(content, str):
                    return content.replace("<image>", "").strip()
    return ""


def build_env(sample: Sample | None = None, args: Any = None, **_: Any) -> HyperEyesEnv:
    """构建 HyperEyesEnv 实例。"""
    max_turns = getattr(args, "max_turns", None) or MAX_ITERATIONS
    images = _extract_images(sample)
    question = _extract_question(sample)

    if not images:
        logger.warning("No images found in sample.multimodal_inputs.")

    return HyperEyesEnv(max_turns=max_turns, images=images, question=question)
