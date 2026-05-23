
from typing import List, Tuple, Union

from PIL import Image

from .config import TextSearchConfig, ImageSearchConfig
from .search import text_search, image_search


# ══════════════════════════════════════════════════════════════════════
#  核心格式化
# ══════════════════════════════════════════════════════════════════════

def fmt_text_items(items: List[dict], wiki_enabled: bool = False) -> str:
    """
    文本检索结果 → 可读字符串。

    格式：网页{i}. 相关性: {score:.2f} 标题: {title} 内容: {content}

    wiki_enabled=True 且 item 含 detail 字段时，
    在内容后追加「Wiki: {detail}」段落。
    """
    if not items:
        return "无相关信息返回"

    txt_list = []
    for i, obs in enumerate(items):
        score   = obs.get("score", 0)
        title   = obs.get("title")
        content = obs.get("content", "")
        detail  = obs.get("detail", "")

        line = f"网页{i+1}. 相关性: {score:.2f} 标题: {title}\n 内容: {content}\n"

        if detail:
            line += f"\n 详细信息: {detail}\n"

        txt_list.append(line)

    return "\n\n".join(txt_list)


def fmt_image_items(items: List[dict], wiki_enabled: bool = False) -> Tuple[str, list]:
    """
    图片检索结果 → (可读字符串, [])。

    格式与原始 image_search_observation 完全一致。
    """
    if not items:
        return "无相关信息返回", []

    tool_str_list = []
    for i, item in enumerate(items):
        title   = item.get("title", "")
        content = item.get("content", "")
        score   = item.get("score", 0.8)
        detail  = item.get("detail", "")

        line = f"网页 {i+1}. 相关性: {score:.2f} 标题: {title}\n 内容: {content}\n"
        if detail:
            line += f" 详细信息: {detail}\n"

        tool_str_list.append(line)
        

    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(tool_str_list)]), []


# ══════════════════════════════════════════════════════════════════════
#  对外接口
# ══════════════════════════════════════════════════════════════════════

async def text_search_observation(
    query: str,
    cfg: TextSearchConfig = None,
) -> str:
    cfg   = cfg or TextSearchConfig()
    items = await text_search(query, cfg)
    return fmt_text_items(items, wiki_enabled=cfg.wiki.mode != "off")


async def image_search_observation(
    image_input: Union[str, Image.Image],
    cfg: ImageSearchConfig = None,
) -> Tuple[str, list]:
    cfg   = cfg or ImageSearchConfig()
    items = await image_search(image_input, cfg)
    return fmt_image_items(items, wiki_enabled=cfg.wiki.mode != "off")
