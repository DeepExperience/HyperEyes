"""
文本检索 / 图片检索接口。
支持在 Config 中显示指定版本。
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Union, Optional

from PIL import Image

from .config import TextSearchConfig, ImageSearchConfig, API_VERSION
from .rag_client import rag_request, make_client
from .utils import *
from .payload_builder import (
    build_text_payload,
    build_image_base_payload,
    build_google_image_payload
)
from .fetch_detail import enrich, fetch_detail_summarize

async def text_search(
    query: str,
    cfg: Optional[TextSearchConfig] = None,
) -> List[dict]:
    """文本检索"""
    cfg = cfg or TextSearchConfig()
    payload = build_text_payload(query, cfg)

    async with make_client() as client:
        # 将 cfg.version 传给底层，实现动态切换
        data = await rag_request(client, payload, "TextSearch", version=cfg.version)
        data = [x for x in data if x.get("score", 0) >= cfg.threshold][: cfg.topk]

        if cfg.wiki.mode != "off" and data:
            data = await enrich(client, clear_detail(data), cfg.wiki, query=query)
        elif cfg.fetch_detail and data:
            # 当 Wiki 未开启但 fetch_detail 开启时，对普通内容进行摘要
            tasks = [
                fetch_detail_summarize(
                    item.get("detail"), 
                    query, 
                    cfg.wiki
                )
                for item in data
            ]
            summaries = await asyncio.gather(*tasks)
            for item, summary in zip(data, summaries):
                item["detail"] = summary
        else:
            # 如果两者都没开启，则清空原始数据中可能存在的 detail 字段，避免不完整的噪音返回
            for item in data:
                item.pop("detail", None)

    return data

async def image_search(
    image_input: Union[str, Image.Image],
    cfg: Optional[ImageSearchConfig] = None,
) -> List[dict]:
    """图片检索"""
    cfg     = cfg or ImageSearchConfig()
    img_b64 = compress_and_b64(image_input, cfg.max_img_size)
    base_params = build_image_base_payload(img_b64, cfg)

    async with make_client() as client:
        tasks = []

        if "google" in cfg.keys:
            async def _google():
                p = build_google_image_payload(base_params, version=cfg.version)
                res = await rag_request(client, p, "GoogleImg", version=cfg.version)
                return res[:3]
            tasks.append(_google())

        results = await asyncio.gather(*tasks)
        merged  = sorted(
            [item for sub in results for item in sub],
            key=lambda x: x.get("score", 0.8),
            reverse=True,
        )

        if cfg.wiki.mode != "off" and merged:
            wiki_query = getattr(cfg, 'query', "")
            merged = await enrich(client, clear_detail(merged), cfg.wiki, query=wiki_query)
        elif cfg.fetch_detail and merged:
            # 当 Wiki 未开启但 fetch_detail 开启时，对普通内容进行摘要
            tasks = [
                fetch_detail_summarize(
                    item.get("detail"), 
                    None, 
                    cfg.wiki
                )
                for item in merged
            ]
            summaries = await asyncio.gather(*tasks)
            for item, summary in zip(merged, summaries):
                item["detail"] = summary
        else:
            # 如果两者都没开启，则清空原始数据中可能存在的 detail 字段，避免不完整的噪音返回
            for item in merged:
                item.pop("detail", None)

    return merged
