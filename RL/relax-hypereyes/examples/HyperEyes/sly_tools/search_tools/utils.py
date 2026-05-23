
import base64
import logging
import re
from io import BytesIO
from typing import Union, List

from PIL import Image


def compress_and_b64(
    image_input: Union[str, Image.Image],
    max_size: int = 1024,
) -> str:
    """
    图片压缩 → Base64 字符串（JPEG 格式）。

    Parameters
    ----------
    image_input : 文件路径 或 PIL Image 对象
    max_size    : 缩放后最大边长（像素）
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.copy().convert("RGB")
    else:
        raise ValueError(f"不支持的图片类型: {type(image_input)}")

    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def clear_detail(
    data: List[dict]
) -> List[dict]:
    """在 wiki enrichment 前清空每条记录的 detail 字段，避免脏数据干扰后续处理。"""
    for item in data:
        if "detail" in item:
            item["detail"] = ""
    return data



def setup_logging(level: int = logging.INFO) -> None:
    """统一日志配置（在入口处调用一次即可）"""
    logging.basicConfig(
        level  = level,
        format = "%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
