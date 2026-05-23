
from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse, unquote

import httpx
import openai

from .config import WikiConfig

# ── 正则 ──────────────────────────────────────────────────────────────
_WIKI_URL_RE  = re.compile(r"https?://([a-z\-]+)\.wikipedia\.org(/[^\s\"'>]+)", re.I)
_ZH_PREFIX_RE = re.compile(r"^/(zh(?:-\w+)?)/(.+)", re.I)
_SECTION_RE   = re.compile(r'\n={2,}[^=]+=+\s*\n')
_SENT_END_RE  = re.compile(r'[。！？!?]|(?<=[a-zA-Z0-9])\. ')
_CN_RE        = re.compile(r'[\u4e00-\u9fa5]')
_WORD_RE      = re.compile(r'[a-zA-Z0-9]+')
_URL_CLEAN_RE = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

# ── LLM Prompt ────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
你是专业的信息提取助手。请从提供的内容中提取最相关的信息，生成简洁、准确、信息密度高的摘要。
要求：
  1. 重点关注核心事实、人物、时间或事件；
  2. 如果提供了搜索词，请优先提取与搜索词相关的信息；
  3. 保持客观，不添加原文没有的内容；
  4. 摘要建议在 300 字以内。\
"""

_USER_TPL_WITH_QUERY = "【用户搜索词】:\n{query}\n\n【原文内容】\n{text}\n\n请生成摘要："
_USER_TPL_NO_QUERY = "【原文内容】\n{text}\n\n请生成摘要："

_IMAGE_SEARCH_DEFAULT_QUERY = "请提取原文内容中的核心事实、人物、时间与事件信息"

# ══════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """清理文本，去除超链接等噪音"""
    if not text:
        return ""
    # 去除 URL
    text = _URL_CLEAN_RE.sub("", text)
    # 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ══════════════════════════════════════════════════════════════════════
#  URL 工具
# ══════════════════════════════════════════════════════════════════════

def _normalize_wiki_url(url: str) -> str:
    """/zh-cn/xxx → /wiki/xxx"""
    try:
        parsed = urlparse(url)
        m = _ZH_PREFIX_RE.match(parsed.path)
        if m:
            return urlunparse(parsed._replace(path=f"/wiki/{m.group(2)}"))
    except Exception:
        pass
    return url


def _extract_wiki_url(item: dict) -> Optional[str]:
    """从 result 的 url/content/title 字段提取并规范化 Wikipedia URL"""
    def _safe_get(item: dict, key: str) -> str:
        return item.get(key) or ""

    text = " ".join([
        _safe_get(item, "url"),
        _safe_get(item, "content"),
        _safe_get(item, "title"),
    ])
    m = _WIKI_URL_RE.search(text)
    return _normalize_wiki_url(m.group(0)) if m else None


# ══════════════════════════════════════════════════════════════════════
#  规则截断
# ══════════════════════════════════════════════════════════════════════

def _norm_len(text: str) -> int:
    """归一化长度：中文按字，英文/数字按词"""
    cn = len(_CN_RE.findall(text))
    en = len(_WORD_RE.findall(re.sub(_CN_RE, " ", text)))
    return cn + en


def smart_truncate(text: str, max_units: int) -> str:
    """
    规则截断，三步兜底：

    Step 1  取导言段（首个 == 章节标题 == 之前的内容）
    Step 2  在窗口内找最后一个句子边界截断
    Step 3  兜底按字符比例硬截 + 补「…」
    """
    if not text or _norm_len(text) <= max_units:
        return text

    # Step 1: 导言段
    m         = _SECTION_RE.search(text)
    lead      = text[: m.start()].strip() if m else text
    candidate = lead if _norm_len(lead) >= max_units * 0.2 else text

    if _norm_len(candidate) <= max_units:
        return candidate

    # Step 2: 句子边界
    best_pos = -1
    for sm in _SENT_END_RE.finditer(candidate):
        pos = sm.end()
        if _norm_len(candidate[:pos]) <= max_units:
            best_pos = pos
        else:
            break

    if best_pos > 0:
        return candidate[:best_pos].strip()

    # Step 3: 硬截
    ratio  = len(candidate) / max(_norm_len(candidate), 1)
    cutoff = int(max_units * ratio)
    return candidate[:cutoff].rstrip() + "…"


# ══════════════════════════════════════════════════════════════════════
#  LLM 摘要
# ══════════════════════════════════════════════════════════════════════

async def _llm_summarize(text: str, query: Optional[str], cfg: WikiConfig) -> Optional[str]:
    """
    LLM 内容摘要。
    """
    if not text:
        return None

    safe_text = text[: cfg.pre_trunc_chars]
    # safe_text = text

    kw: dict = {"api_key": cfg.openai_api_key or None}
    if cfg.openai_base_url:
        kw["base_url"] = cfg.openai_base_url

    if query:
        user_content = _USER_TPL_WITH_QUERY.format(query=query, text=safe_text)
    else:
        user_content = _USER_TPL_NO_QUERY.format(text=safe_text)

    try:
        async with openai.AsyncOpenAI(**kw) as client:
            resp   = await client.chat.completions.create(
                model       = cfg.openai_model,
                messages    = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens  = cfg.max_llm_chars * 2,
                temperature = 0.2,
            )
            summary = resp.choices[0].message.content.strip()
            logging.info(f"[Summary] ✅ 完成 {len(summary)} chars")
            return summary

    except Exception as e:
        logging.warning(f"[Summary] LLM 失败: {e}")

    return None


async def fetch_detail_summarize(text: str, query: Optional[str], cfg: WikiConfig) -> str:
    """
    通用内容摘要，用于 fetch_detail 参数。
    在总结前会先进行文本清洗。
    """
    if not text:
        return ""

    cleaned = clean_text(text)
    summary = await _llm_summarize(cleaned, query, cfg)
    
    if summary:
        return summary
    
    # 兜底
    return cleaned[:300] + "..." if len(cleaned) > 300 else cleaned


# ══════════════════════════════════════════════════════════════════════
#  Wiki Server 拉取
# ══════════════════════════════════════════════════════════════════════

async def _wiki_fetch(client: httpx.AsyncClient, wiki_url: str) -> Optional[str]:
    timeout = httpx.Timeout(30.0, connect=2.0)
    try:
        decoded_url = unquote(wiki_url)          # ✅ 避免双重编码
        params = urlencode({"url": decoded_url, "preview": 99_999_999})
        resp = await client.get(
            f"{WikiConfig.WIKI_SERVER_URL}/query?{params}",
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("hit"):
            logging.info(
                f"[Wiki] ✅ {decoded_url}"
                f" → {data['title']} ({data['length']:,} chars)"
            )
            return data.get("preview", "")

        logging.info(f"[Wiki] ❌ 未命中: {decoded_url}")
        return None

    except Exception as e:
        logging.warning(f"[Wiki] 请求失败 {unquote(wiki_url)}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  对外主入口
# ══════════════════════════════════════════════════════════════════════

async def enrich(
    client: httpx.AsyncClient,
    results: List[dict],
    cfg: WikiConfig,
    query: str = "",
) -> List[dict]:
    """
    对 RAG 检索结果做 Wiki 增强，最终文本写入 result["detail"]。

    处理流程
    ────────
    1. 提取每条 result 的 Wiki URL
    2. 并发拉取所有唯一 URL 的原文
    3. 规则截断（始终执行）
    4. mode="llm" 时并发调用 LLM 摘要，失败自动降级
    5. 相同 URL 只保留得分最高的一条（results 已降序）
    """
    if cfg.mode == "off" or not results:
        return results

    # 判断文搜或图搜
    first_search_from = results[0].get("search_from", "")
    if "image" in first_search_from.lower():
        query = _IMAGE_SEARCH_DEFAULT_QUERY


    # Step 1
    item_urls   = [_extract_wiki_url(item) for item in results]
    unique_urls = list({u for u in item_urls if u})
    logging.info(f"[Wiki] mode={cfg.mode} | {len(results)} 条 → {len(unique_urls)} 个唯一 URL")

    # Step 2: 并发拉取
    fetched  = await asyncio.gather(*[_wiki_fetch(client, u) for u in unique_urls])
    raw_cache: Dict[str, Optional[str]] = dict(zip(unique_urls, fetched))

    # Step 3: 规则截断
    rule_cache: Dict[str, str] = {}
    for url, raw in raw_cache.items():
        if raw:
            truncated = smart_truncate(raw, cfg.max_rule_units)
            logging.info(
                f"[Wiki] ✂️  {len(raw):,} → {len(truncated):,} chars ({unquote(url)})"
            )
            rule_cache[url] = truncated

    # Step 4: LLM 摘要（可选，并发）
    llm_cache: Dict[str, Optional[str]] = {}
    if cfg.mode == "llm" and rule_cache:
        url_list    = list(rule_cache.keys())
        # 对于 Wiki 增强，文搜使用 query，图搜使用默认 prompt 逻辑在 _llm_summarize 处理
        summary_query = query if query else _IMAGE_SEARCH_DEFAULT_QUERY
        llm_results = await asyncio.gather(
            *[_llm_summarize(rule_cache[u], summary_query, cfg) for u in url_list]
        )
        llm_cache = dict(zip(url_list, llm_results))

    # Step 5: 去重 + 写入 detail
    seen:   set        = set()
    output: List[dict] = []

    for item, wiki_url in zip(results, item_urls):
        if wiki_url is None:
            output.append(item)
            continue

        if wiki_url in seen:
            logging.info(f"[Wiki] 去重跳过: {unquote(wiki_url)}")
            output.append(item)
            continue
        seen.add(wiki_url)

        # LLM 摘要 → 规则截断（降级）
        final: Optional[str] = None
        if cfg.mode == "llm":
            final = llm_cache.get(wiki_url)
            if final:
                logging.info(f"[Wiki] 🤖 LLM 摘要  ({unquote(wiki_url)})")
        if not final:
            final = rule_cache.get(wiki_url)
            if final:
                logging.info(f"[Wiki] 📐 规则截断  ({unquote(wiki_url)})")

        output.append({**item, "detail": final} if final else item)

    return output
