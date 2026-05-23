

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# ══════════════════════════════════════════════════════════════════════
#  底层服务地址与 API 版本配置
# ══════════════════════════════════════════════════════════════════════

# 这里的 API_VERSION 作为全局默认值
API_VERSION: Literal["sly", "old"] = "sly"

def get_config_by_version(version: Optional[Literal["sly", "old"]] = None):
    # ver = version or API_VERSION
    url = "http://xxx/ser/in" # search api
    headers = {"Content-Type": "application/json"}
    return url, headers

# ══════════════════════════════════════════════════════════════════════
#  Wiki 增强配置
# ══════════════════════════════════════════════════════════════════════

@dataclass
class WikiConfig:
    """
    Wiki 增强行为配置。
    """
    mode: Literal["off", "rule", "llm"] = "off"
    WIKI_SERVER_URL: str = "http://xxxx:7878"

    # rule
    max_rule_units: int = 1000

    # llm
    openai_api_key:  str = "xx"
    openai_base_url: str = "xx"
    openai_model:    str = "xxx"
    max_llm_chars:   int = 1024
    pre_trunc_chars: int = 8192


# ══════════════════════════════════════════════════════════════════════
#  文本检索配置
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TextSearchConfig:
    """
    文本检索配置。
    """
    version:      Optional[Literal["sly", "old"]] = None
    keys:         List[str]  = field(default_factory=lambda: ["google"])
    threshold:    float      = 0.6
    topk:         int        = 3
    fetch_detail: bool       = False
    extra_params: dict       = field(default_factory=dict)
    wiki:         WikiConfig = field(default_factory=WikiConfig)


# ══════════════════════════════════════════════════════════════════════
#  图片检索配置
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ImageSearchConfig:
    """
    图片检索配置。
    """
    version:      Optional[Literal["sly", "old"]] = None
    keys:         List[str]  = field(default_factory=lambda: ["google"])
    threshold:    float      = 0.6
    query:        str        = ""
    max_img_size: int        = 1024
    fetch_detail: bool       = False
    extra_params: dict       = field(default_factory=dict)
    wiki:         WikiConfig = field(default_factory=WikiConfig)
