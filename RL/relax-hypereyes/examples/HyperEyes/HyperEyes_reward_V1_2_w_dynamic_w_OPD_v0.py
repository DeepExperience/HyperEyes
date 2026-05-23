
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from typing import List

from examples.HyperEyes.llm_judge import get_acc
from relax.utils.types import Sample


# ---------------------------------------------------------------------------
#  常量 & 工具
# ---------------------------------------------------------------------------

_CACHE_FILE_NAME = "baseline_cache.json"
_EMPTY_TOOL = {"count": 0, "total": 0, "max_parallel": 0, "per_turn": []}


def _make_stat(
    acc_reward: float,
    answer_text: str,
    tool: dict,
    reward_type: str,
    question_text: str,
) -> dict:
    """统一构造单条 rollout 的 stat dict。"""
    return dict(
        acc_reward=acc_reward,
        answer_text=answer_text,
        tool=tool,
        reward_type=reward_type,
        question_text=question_text,
    )


# ---------------------------------------------------------------------------
#  Completion 解析
# ---------------------------------------------------------------------------


def extract_completion(completion_str: str) -> dict:
    """
    将 completion 解析为结构化信息：
    - segments: [{"role": "...", "content": "..."}]
    - actions: [{"tool_call", "tool_name", "tool_response", "next_assistant", ...}]
    - final_answer
    - tool_count / search_count / max_parallel / per_turn
    """
    result = {
        "segments": [],
        "actions": [],
        "final_answer": "",
        "tool_count": 0,
        "search_count": 0,
        "max_parallel": 0,
        "per_turn": [],
    }
    raw_tool = ""
    try:
        if not completion_str:
            return result

        if not completion_str.lstrip().startswith("<|im_start|>"):
            completion_str = f"<|im_start|>assistant\n{completion_str}"

        turn_pattern = re.compile(
            r"<\|im_start\|>(assistant|user|system)\n?(.*?)(?:<\|im_end\|>|$)",
            re.DOTALL,
        )

        matches = list(turn_pattern.finditer(completion_str))

        if matches:
            for m in matches:
                role = m.group(1).strip()
                content = m.group(2).strip()
                result["segments"].append({"role": role, "content": content})
        else:
            result["segments"].append({"role": "assistant", "content": completion_str})

        actions = []

        for seg_idx, seg in enumerate(result["segments"]):
            if seg["role"] != "assistant":
                continue

            tool_matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", seg["content"], re.DOTALL))
            for tm in tool_matches:
                raw_tool = tm.group(1).strip()

                tool_name = ""
                tool_args = {}
                try:
                    tool_data = json.loads(raw_tool)
                    tool_name = tool_data.get("name", "")
                    tool_args = tool_data.get("arguments", {})

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {}

                    if not isinstance(tool_args, dict):
                        tool_args = {}

                except Exception:
                    pass

                search_count = 0
                max_parallel = 0

                if "text_search" in tool_name:
                    q = str(tool_args.get("input", "")).strip()
                    parts = [x.strip() for x in q.split("|") if x.strip()]
                    search_count = len(parts)
                    max_parallel = len(parts)

                elif "image_search" in tool_name:
                    image_id = str(tool_args.get("image_id", tool_args.get("img_id", ""))).strip()
                    if image_id:
                        parts = [x.strip() for x in image_id.split("|") if x.strip()]
                        search_count = len(parts)
                        max_parallel = len(parts)

                actions.append({
                    "tool_call": raw_tool,
                    "tool_name": tool_name,
                    "tool_response": "",
                    "next_assistant": "",
                    "search_count": search_count,
                    "max_parallel": max_parallel,
                    "assistant_seg_idx": seg_idx,
                })

        # 统计每轮工具调用
        per_turn_stats = defaultdict(lambda: {"tool_calls": 0, "searches": 0})
        for action in actions:
            idx = action["assistant_seg_idx"]
            per_turn_stats[idx]["tool_calls"] += 1
            per_turn_stats[idx]["searches"] += action["search_count"]

        per_turn_list = [per_turn_stats[k] for k in sorted(per_turn_stats.keys())]

        for action in actions:
            seg_idx = action["assistant_seg_idx"]

            for j in range(seg_idx + 1, len(result["segments"])):
                nxt = result["segments"][j]
                if nxt["role"] == "user" and "<tool_response>" in nxt["content"]:
                    m = re.search(r"<tool_response>(.*?)</tool_response>", nxt["content"], re.DOTALL)
                    action["tool_response"] = m.group(1).strip() if m else nxt["content"].strip()
                    break
                if nxt["role"] == "assistant":
                    break

            for j in range(seg_idx + 1, len(result["segments"])):
                nxt = result["segments"][j]
                if nxt["role"] == "assistant":
                    action["next_assistant"] = nxt["content"].strip()
                    break

            action.pop("assistant_seg_idx", None)

        result["actions"] = actions
        result["tool_count"] = len(actions)
        result["search_count"] = sum(x["search_count"] for x in actions)
        result["per_turn"] = per_turn_list
        result["max_parallel"] = max(
            (v["searches"] for v in per_turn_stats.values()), default=0
        )

        assistant_segments = [x for x in result["segments"] if x["role"] == "assistant"]
        if assistant_segments:
            last_text = assistant_segments[-1]["content"]
            m = re.search(r"<answer>(.*?)</answer>", last_text, re.DOTALL)
            if m:
                result["final_answer"] = m.group(1).strip()

    except Exception as e:
        print(f"json extract warning: {e} | tool_call: {str(raw_tool)}")

    return result


# ---------------------------------------------------------------------------
#  单条 rollout 统计量
# ---------------------------------------------------------------------------


async def compute_single_stats_async(
    predict_str: str,
    ground_truth: str,
    question_text: str,
    reward_type: str,
) -> dict:
    """计算单条 rollout 的 acc + quality_reward。"""
    base = dict(reward_type=reward_type, question_text=question_text)

    # 过滤: 搜索全失败
    # if predict_str.count("无相关信息返回") >= 3:
    #     return _make_stat(acc_reward=-100, answer_text="", tool=_EMPTY_TOOL, **base)

    parsed = extract_completion(predict_str)
    answer_text = parsed["final_answer"]
    tool = {
        "count": parsed["tool_count"],
        "total": parsed["search_count"],
        "max_parallel": parsed["max_parallel"],
        "per_turn": parsed["per_turn"],
    }

    if not answer_text:
        return _make_stat(acc_reward=-0.5, answer_text="", tool=_EMPTY_TOOL, **base)

    ground_truth = ground_truth.replace("<answer>", "").replace("</answer>", "")
    answer_text = answer_text.replace("<answer>", "").replace("</answer>", "")

    # acc 计算
    if reward_type == "only_acc":         # ✓ 严格相等（避免旧 substring 匹配 bug）
        acc_reward = await get_acc(answer_text, ground_truth, question_text)
    else:
        # 与 acc=-0.5 同量纲，避免 magic number 污染 GRPO advantage
        return _make_stat(acc_reward=-0.5, answer_text=answer_text, tool=tool, **base)

    return _make_stat(acc_reward=acc_reward, answer_text=answer_text, tool=tool, **base)


# ---------------------------------------------------------------------------
#  Group Bonus
# ---------------------------------------------------------------------------


def compute_group_bonus(
    stats            : list[dict],
    sft_tc           : int = -1,   # -1 表示无参考标准
    sft_total_search : int = -1,   # -1 表示无参考标准
) -> list[float]:
    
    bonuses: list[float | None] = [None] * len(stats)

    # ── 通用排名赋值：rank=0（最优键）→ hi，rank=M-1（最差键）→ lo ──────────
    def _rank_assign(idx_list, lo, hi, key_fn):
        if not idx_list:
            return
        keys     = [key_fn(i) for i in idx_list]
        unique   = sorted(set(keys))
        M        = len(unique)
        rank_map = {k: r for r, k in enumerate(unique)}
        for i, k in zip(idx_list, keys):
            r          = rank_map[k]
            bonuses[i] = round(hi if M == 1 else hi - (hi - lo) * r / (M - 1), 4)

    def _sft_key(i: int):
        t = stats[i]["tool"]
        return (t["count"], t["total"])

    for i, s in enumerate(stats):
        acc      = s["acc_reward"]
        tc       = s["tool"]["count"]
        tc_delta = tc - sft_tc

        if acc <= -0.5:                      # 解析异常
            bonuses[i] = 0.0

        elif acc <= 0:                       # 答错
            if   tc_delta < 0 or tc > int(sft_tc*1.5):  bonuses[i] =  -0.1 # 答错且工具调用偏少 鼓励探索
            else:               bonuses[i] = 0.0

        elif tc == 0:                        # tc=0 + 答对
            bonuses[i] = +0.0               # 固定奖励，防止模型为高分猜测

        # tc>0 + 答对 → None，留待下方排名

    rank_pool = [i for i in range(len(stats)) if bonuses[i] is None]
    if rank_pool:
        good_set = {
            i for i in rank_pool
            if stats[i]["tool"]["count"] <= sft_tc
            and stats[i]["tool"]["total"] <= sft_total_search
        }
        good = [i for i in rank_pool if i in good_set]
        bad  = [i for i in rank_pool if i not in good_set]
        _rank_assign(good, lo=+0.05, hi=+0.20, key_fn=_sft_key)
        _rank_assign(bad,  lo=-0.10, hi=-0.02, key_fn=_sft_key)

    return [round(b if b is not None else 0.0, 4) for b in bonuses]


# ---------------------------------------------------------------------------
#  调试打印
# ---------------------------------------------------------------------------


def _print_group(rows: list[dict], step: int) -> None:
    if not rows:
        print(f"\n{'=' * 60} Step #{step} | no valid rows\n")
        return

    first = rows[0]
    lines = [f"\n{'=' * 50} Step:{step} {'=' * 50}"]
    lines.append(
        f"Type:{first['stat']['reward_type']} | Q:{first['stat']['question_text']}\n"
        f"GT:{first['ground_truth']}"
    )

    for idx, row in enumerate(rows):
        s = row["score_dict"]
        stat = row["stat"]
        tool = stat["tool"]
        per_turn_searches = [r["searches"] for r in tool["per_turn"]]
        sft_ref = row.get("sft_ref", {})
        opd_branch = s.get("opd_branch", "?")
        lines.append(
            f"  [{idx}] Acc: {s['acc_reward']:.2f} | Bonus: {s['t_bonus']:.2f} | "
            f"tc={tool['count']} total={tool['total']} per_turn={per_turn_searches} | "
            f"SFT_ref: tc={sft_ref.get('tc', -1)} total={sft_ref.get('total', -1)} | "
            f"OPD: {opd_branch} | Final: {s['score']:.2f}"
        )
        lines.append(f"      Pred: {stat['answer_text']}")

    lines.append("=" * 100)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
#  BaselineCache：线程安全的 JSON 持久化 cache
# ---------------------------------------------------------------------------


class BaselineCache:
    """
    线程安全的本地 JSON 持久化 Cache。

    设计原则：
      - 单一内存 cache（_mem），初始化时从磁盘加载
      - get()    加锁读 _mem，返回 (tc, total, parallel)
      - update() 加锁写 _mem，并原子落盘
      - 同一次 reward_func 内"先 get 后 update"，天然无同批污染
      - 跨 step 更新立即生效：Step N 写入的最优 baseline，Step N+1 即可读到

    存储格式：
        { "<index>": {"tool_call": int, "parallel": [int, ...]} }

    选优规则：
      - primary key  : tool_call 越小越优（tc=0 排除，可能是 lucky guess）
      - secondary key: sum(parallel) 越小越优（tc 相同时）
      - tc 和 parallel 来自同一条 rollout，不跨条混合
    """

    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self._path = os.path.join(cache_dir, _CACHE_FILE_NAME)
        self._lock = threading.Lock()

        # 单一内存 cache：启动时从磁盘加载，之后在内存中维护并实时落盘
        self._mem: dict = self._load_from_disk()

        print(
            f"[BaselineCache] 初始化完成 | cache 路径: {self._path} | "
            f"已有 {len(self._mem)} 条历史 baseline"
        )

    # ── 磁盘 I/O ───────────────────────────────────────────────────────────────

    def _load_from_disk(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[BaselineCache] 加载 cache 失败，使用空 cache: {e}")
            return {}

    def _flush(self):
        """将 _mem 原子写回磁盘（调用方须持有 self._lock）。"""
        dir_name = os.path.dirname(self._path)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name, suffix=".tmp",
                delete=False, encoding="utf-8"
            ) as f:
                json.dump(self._mem, f, ensure_ascii=False)
                tmp_path = f.name
            os.replace(tmp_path, self._path)
        except Exception as e:
            print(f"[BaselineCache] 写入 cache 失败: {e}")

    # ── 比较逻辑 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_better(new_tc: int, new_total: int, old_tc: int, old_total: int) -> bool:
        """
        判断 (new_tc, new_total) 是否优于 (old_tc, old_total)。
        优先 tc 小，tc 相同时 total 小。tc=0 视为无效（可能是猜测）。
        """
        if new_tc <= 0:
            return False
        if new_tc < old_tc:
            return True
        if new_tc == old_tc and new_total < old_total:
            return True
        return False

    # ── 公共接口 ───────────────────────────────────────────────────────────────

    def get(self, index: str, orig_tc: int, orig_parallel: list) -> tuple:
        """
        读取 index 对应的当前最优 baseline。
        若 cache 中存在且优于原始 SFT 值，使用 cache 值；否则使用原始值。
        返回 (effective_tc, effective_total, effective_parallel)。
        """
        orig_total = sum(orig_parallel) if orig_parallel else 0

        with self._lock:
            entry = self._mem.get(str(index))

        if entry is None:
            return orig_tc, orig_total, orig_parallel

        cached_tc       = entry.get("tool_call", orig_tc)
        cached_parallel = entry.get("parallel", orig_parallel)
        cached_total    = sum(cached_parallel) if cached_parallel else 0

        if self._is_better(cached_tc, cached_total, orig_tc, orig_total):
            return cached_tc, cached_total, cached_parallel
        else:
            return orig_tc, orig_total, orig_parallel

    def update(
        self,
        index: str,
        new_tc: int,
        new_parallel: list,
        orig_tc: int,
        orig_parallel: list,
    ):
        """
        用本次 rollout 中找到的最优 (tc, parallel) 更新 cache（内存 + 磁盘）。
        只有新值优于 cache 中已有值时才写入（单调递减）。
        更新立即对后续 step 生效。
        """
        key = str(index)
        new_total = sum(new_parallel) if new_parallel else 0

        with self._lock:
            entry        = self._mem.get(key, {"tool_call": orig_tc, "parallel": orig_parallel})
            old_tc       = entry.get("tool_call", orig_tc)
            old_parallel = entry.get("parallel", orig_parallel)
            old_total    = sum(old_parallel) if old_parallel else 0

            if self._is_better(new_tc, new_total, old_tc, old_total):
                self._mem[key] = {"tool_call": new_tc, "parallel": new_parallel}
                self._flush()
                print(
                    f"[BaselineCache] index={index} 更新 baseline（立即生效）: "
                    f"tc {old_tc} -> {new_tc} | "
                    f"parallel {old_parallel} -> {new_parallel} "
                    f"(total {old_total} -> {new_total})"
                )


# ---------------------------------------------------------------------------
#  全局 Cache 单例（延迟初始化）
#
#  cache_dir 优先级：
#    1. 环境变量 RL_BASELINE_CACHE_DIR（推荐在启动脚本中设置）
#    2. /tmp/rl_baseline_cache（最后 fallback，打印警告）
# ---------------------------------------------------------------------------

# _cache_instance: BaselineCache | None = None
# _cache_init_lock = threading.Lock()



# ---------------------------------------------------------------------------
#  从 rollout group 提取最优 baseline
# ---------------------------------------------------------------------------


def _best_rollout_baseline(stats: list) -> tuple | None:
    """
    从一个 group 的所有 rollout 中，找到答对（acc > 0）且使用了工具（tc >= 1）的样本，
    选出最优的单条 rollout 的 (tc, parallel)：
      - primary key  : tc (tool_call 次数) 最小
      - secondary key: total=sum(per_turn searches) 最小（tc 相同时）
      - tc 和 parallel 来自同一条 rollout，不跨条混合

    排除 tc=0 的原因：
      tc=0 表示模型未调用任何工具就给出了答案，极有可能是猜测（lucky guess），
      不应将其作为 baseline 更新依据。

    若 group 内无任何满足条件的 rollout（全错或全 tc=0），返回 None 不更新。
    """
    candidates = []
    for s in stats:
        tc = s["tool"]["count"]
        if s["acc_reward"] > 0 and tc >= 1:   # tc=0 排除：未用工具，可能是猜测
            per_turn = s["tool"]["per_turn"]   # list[{"tool_calls": int, "searches": int}]
            parallel = [t["searches"] for t in per_turn]
            total = sum(parallel)
            candidates.append((tc, total, parallel))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    best_tc, _, best_parallel = candidates[0]
    return best_tc, best_parallel


# ---------------------------------------------------------------------------
#  Relax reward_func 入口（batched, per-group）
# ---------------------------------------------------------------------------

_local_step = 0


async def reward_func(args, samples: list[Sample], **kwargs) -> list[dict]:
    """Relax batched reward 入口（动态 baseline 版）。

    接受同一 prompt 的一组 samples（n-samples-per-prompt），
    在计算 reward 前读取当前最优 baseline，计算完成后将本 group 的最优 rollout
    写回 cache，供后续 step 使用（单调递减，逐步收紧参考标准）。

    返回 list[dict]，格式 {"score": float}，与 --reward-key score 配合使用。
    """
    global _local_step
    _local_step += 1

    completions: list[str] = []
    solutions: list[str] = []
    extra_info_list: list[dict] = []

    for s in samples:
        completions.append(s.response or "")
        label = s.label
        if isinstance(label, dict):
            label = label.get("ground_truth", "")
        solutions.append(label or "")
        extra_info_list.append(s.metadata or {})

    tasks = [
        compute_single_stats_async(
            predict_str=completion,
            ground_truth=solution,
            question_text=(extra_info_list[i] if i < len(extra_info_list) else {})
                          .get("question", "").replace("<image>", ""),
            reward_type=(extra_info_list[i] if i < len(extra_info_list) else {})
                        .get("type", "only_acc"),
        )
        for i, (completion, solution) in enumerate(zip(completions, solutions))
    ]
    stats: list[dict] = list(await asyncio.gather(*tasks))

    eval_mode = (len(stats) == 1)
    if eval_mode:
        results: list[dict] = []
        for stat in stats:
            acc = stat["acc_reward"]
            score = max(0.0, acc)
            results.append({"score": score, "opd_branch": "skip"})
        print(f"[Eval] step={_local_step} | n_samples=1 | "
              f"acc={stats[0]['acc_reward']} → score={results[0]['score']}")
        return results

    ref = extra_info_list[0] if extra_info_list else {}
    index = ref.get("index", None)    # 用于 cache key

    raw_tc = ref.get("tool_call", None)
    orig_sft_tc = int(raw_tc) if raw_tc is not None else -1

    orig_parallel: list = []
    orig_sft_total = -1
    if orig_sft_tc >= 0:
        try:
            raw_parallel = ref.get("parallel", "[]")
            orig_parallel = (
                json.loads(raw_parallel)
                if isinstance(raw_parallel, str)
                else list(raw_parallel)
            )
            orig_sft_total = int(sum(orig_parallel)) if orig_parallel else 0
        except (json.JSONDecodeError, TypeError, ValueError):
            orig_parallel = []
            orig_sft_total = -1


    # skip cache update
    sft_tc = orig_sft_tc
    sft_total = orig_sft_total
    eff_parallel = orig_parallel

    print(
        f"[Dynamic] step={_local_step} index={index} | "
        f"orig SFT: tc={orig_sft_tc} parallel={orig_parallel} total={orig_sft_total} | "
        f"effective SFT: tc={sft_tc} parallel={eff_parallel} total={sft_total}"
    )


    t_bonuses = compute_group_bonus(
        stats,
        sft_tc=sft_tc,
        sft_total_search=sft_total,
    )


    results: list[dict] = []
    rows: list[dict] = []

    for stat, solution, t_bonus in zip(stats, solutions, t_bonuses):
        ground_truth = solution.replace("<answer>", "").replace("</answer>", "")
        acc = stat["acc_reward"]
        tc = stat["tool"]["count"]

        if acc <= -0.5:
            final_score = acc
        else:
            final_score = acc + t_bonus

        # ── OPD 分支分配 ──────────────────────────────────────────────────────
        if acc <= 0: # v3
            opd_branch = "reverse_kl"   # 分支 A：已启用
        else:
            opd_branch = "skip"         # 分支 B/C 暂时统一 skip，先只跑分支 A
            
        score_dict = dict(
            score=final_score,
            acc_reward=acc,
            t_bonus=t_bonus,
            opd_branch=opd_branch,
        )
        rows.append(dict(
            ground_truth=ground_truth,
            stat=stat,
            score_dict=score_dict,
            sft_ref={"tc": sft_tc, "total": sft_total, "parallel": eff_parallel},
        ))
        results.append({"score": final_score, "opd_branch": opd_branch})

    _print_group(rows, _local_step)


    return results


if __name__ == "__main__":
    print("HyperEyes_reward_sft_dynamic loaded.")
