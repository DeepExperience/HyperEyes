
import asyncio
import json
import logging
import random
import time
from typing import List

import httpx
from .config import get_config_by_version, API_VERSION

logging.getLogger("httpx").setLevel(logging.WARNING)

_TIMEOUT = httpx.Timeout(300.0, connect=16.0)

def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_TIMEOUT)

async def rag_request(
    client: httpx.AsyncClient,
    payload: dict,
    tag: str,
    max_retries: int = 3,
    version: str = None
) -> List[dict]:
    """POST 到 RAG API，根据版本解析响应"""
    # 动态获取当前请求应该使用的 URL 和 Headers
    ver = version or API_VERSION
    url, headers = get_config_by_version(ver)

    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, headers=headers, json=payload)
            logging.info(
                f"[{tag}] version={ver} status={resp.status_code} "
                f"cost={time.perf_counter()-t0:.2f}s (attempt {attempt})"
            )
            resp.raise_for_status()
            resp_json = resp.json()
            
            data = resp_json.get("data")
                
            return data if isinstance(data, list) else []

        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as e:
            logging.warning(f"[{tag}] 第 {attempt} 次失败: {e}")
            await asyncio.sleep(random.uniform(0.5 * attempt, attempt * 3.0))

        except Exception as e:
            logging.error(f"[{tag}] 未知错误: {e}")
            break

    return []
