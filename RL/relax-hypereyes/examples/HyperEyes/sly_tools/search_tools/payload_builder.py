

from typing import List, Union
from .config import API_VERSION, TextSearchConfig, ImageSearchConfig

def build_text_payload(query: str, cfg: TextSearchConfig) -> dict:
    ver = cfg.version or API_VERSION
    return {
        "messages":                   [{"content": query}],
        "num_search_google_text":     10,
        "query_text":                 query,
        "fetch_detail":               cfg.fetch_detail,
        **cfg.extra_params,
    }

def build_image_base_payload(img_b64: str, cfg: ImageSearchConfig) -> dict:
    ver = cfg.version or API_VERSION
    
    return {
        "messages":              [],
        "query_text":            "",
        "query_image_base64":    f"data:image/jpeg;base64,{img_b64}",
        "fetch_detail":          cfg.fetch_detail,
        **cfg.extra_params,
    }

def build_google_image_payload(base_payload: dict, version: str = None) -> dict:
    # ver = version or API_VERSION
    return {
        **base_payload,
        "activeGoogleImage":           True,
        "num_search_google_image":     3,
    }
