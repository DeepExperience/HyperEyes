

from __future__ import annotations

import os
import random
import re


TIMEOUT = int(os.environ.get("SLYCHAT_JUDGE_TIMEOUT", "120"))

# 保存原始 proxy 环境变量 key，供 _clear_proxy_env / _restore_proxy_env 使用
_PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
]


def _make_no_proxy_httpx_client():
    """创建一个不使用任何 HTTP proxy 的 httpx.Client。

    训练集群环境中通常配置了 Squid 代理（HTTP_PROXY / HTTPS_PROXY）， 这会导致发往 127.0.0.1（本地 judge
    服务）的请求被代理拦截并返回 ERR_CONNECT_FAIL。httpx.Client(proxy=None) 不能禁用从环境变量 读取的
    proxy，因此这里在创建 client 前临时清除 proxy 环境变量。
    """
    import httpx

    saved = {}
    for key in _PROXY_ENV_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        client = httpx.Client()
    finally:
        os.environ.update(saved)
    return client


def _make_no_proxy_async_httpx_client():
    """创建一个不使用任何 HTTP proxy 的 httpx.AsyncClient。"""
    import httpx

    saved = {}
    for key in _PROXY_ENV_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        client = httpx.AsyncClient()
    finally:
        os.environ.update(saved)
    return client


# ---------------------------------------------------------------------------
#  Judge 客户端初始化
# ---------------------------------------------------------------------------


def _get_judge_client():
    """创建 OpenAI SDK 客户端，读取环境变量进行配置。

    Returns:
        (client, model_list) 元组。
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required. Install via: pip install openai") from exc

    api_key = os.environ.get("SLYCHAT_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SLYCHAT_JUDGE_API_KEY or OPENAI_API_KEY env var.")

    base_url = os.environ.get("SLYCHAT_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    
    http_client = _make_no_proxy_httpx_client()
    client = (
        OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        if base_url
        else OpenAI(api_key=api_key, http_client=http_client)
    )

    models_str = os.environ.get("SLYCHAT_JUDGE_MODELS") or os.environ.get("SLYCHAT_JUDGE_MODEL") or "model"
    model_list = [m.strip() for m in models_str.split(",") if m.strip()]
    return client, model_list


# ---------------------------------------------------------------------------
#  通用调用
# ---------------------------------------------------------------------------


def call_judge(prompt: str, timeout: int = TIMEOUT) -> str:
    client, model_list = _get_judge_client()
    model_name = random.choice(model_list)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=timeout,
            )
            return resp.choices[0].message.content.strip()
        except BaseException as e:
            print(f"[llm_judge] ERROR model={model_name} attempt={attempt + 1}/3: {e}")
            if attempt == 2:
                return ""
    return ""


# ---------------------------------------------------------------------------
#  异步版 Judge 客户端与调用
# ---------------------------------------------------------------------------


def _get_async_judge_client():
    """创建 AsyncOpenAI SDK 客户端，读取环境变量进行配置。

    Returns:
        (async_client, model_list) 元组。
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required. Install via: pip install openai") from exc

    api_key = os.environ.get("SLYCHAT_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SLYCHAT_JUDGE_API_KEY or OPENAI_API_KEY env var.")

    base_url = os.environ.get("SLYCHAT_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    http_client = _make_no_proxy_async_httpx_client()
    client = (
        AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        if base_url
        else AsyncOpenAI(api_key=api_key, http_client=http_client)
    )

    models_str = os.environ.get("SLYCHAT_JUDGE_MODELS") or os.environ.get("SLYCHAT_JUDGE_MODEL") or "model"
    model_list = [m.strip() for m in models_str.split(",") if m.strip()]
    return client, model_list


async def async_call_judge(prompt: str, timeout: int = TIMEOUT) -> str:
    """异步调用 Judge 模型并返回文本响应。

    替代同步版 ``call_judge``，不阻塞事件循环。最多重试 3 次。
    """
    client, model_list = _get_async_judge_client()
    model_name = random.choice(model_list)
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=timeout,
            )
            return resp.choices[0].message.content.strip()
        except BaseException as e:
            print(f"[llm_judge] ERROR model={model_name} attempt={attempt + 1}/3: {e}")
            if attempt == 2:
                return ""
    return ""


# ---------------------------------------------------------------------------
#  异步 Gemini Flash 调用（替代原同步 requests.post 版本）
# ---------------------------------------------------------------------------

GEMINI_HEADERS = {
    "content-type": "application/json",
    "api-key": os.environ.get("GEMINI_API_KEY", ""),
}
GEMINI_FLASH_URL = os.environ.get(
    "GEMINI_FLASH_URL",
    "judge_model_url",
)


async def async_call_gemini_flash(prompt: str, timeout: int = TIMEOUT) -> str:
    """异步调用 Gemini Flash API，不阻塞事件循环。最多重试 3 次。"""
    import aiohttp

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 30,
            "temperature": 0.0,
        },
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GEMINI_FLASH_URL,
                    headers=GEMINI_HEADERS,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    resp.raise_for_status()
                    output = await resp.json()
                    return output["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"[llm_judge] ERROR Gemini Flash attempt={attempt + 1}/3: {e}")
            if attempt == 2:
                return ""
    return ""


# ---------------------------------------------------------------------------
#  Agent 版 Judge 函数（从 sly_agent 的 llm_judge.py 迁移）
#  全部改为 async，内部调用 async_call_gemini_flash（Gemini Flash API）
# ---------------------------------------------------------------------------


async def get_acc(pred: str, gt: str, question: str) -> float:
    """二分类准确度判断：Correct → 1.0, Incorrect → 0.0。"""
    content = f"""
Question: {question}
Ground truth: {gt}
Model prediction: {pred}

Please compare the model's prediction with the ground truth to determine if the model's prediction contains the correct answer. Answer only "Correct" or "Incorrect".
    """
    for i in range(3):
        try:
            resp = await async_call_gemini_flash(content, timeout=TIMEOUT)
            if not resp:
                continue
            resp = resp.strip()
            if "correct" in resp.lower() and "incorrect" not in resp.lower():
                return 1.0
            elif "incorrect" in resp.lower():
                return 0.0
            else:
                print(f"[llm_judge] judge format error: {resp}")
                continue
        except Exception as e:
            print(f"[llm_judge] async_call_gemini_flash error: {e} (retry {i + 1}/3)")
            continue
    return 0.0