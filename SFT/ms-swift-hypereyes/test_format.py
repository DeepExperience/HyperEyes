


import re

STRUCTURE_PATTERNS = [
    (r"^\s*\d+[.、．]\s*", "numbered_list"),      # 1. 2. 3.
    (r"^\s*[-•*]\s+", "bullet_list"),             # - * •
    (r"[（(]\d+[)）]", "parenthesis_number"),     # （1）(1)
    (r"#{1,6}\s+", "markdown_header"),            # # ##
    (r"\*\*.*?\*\*", "bold_markdown"),            # **bold**
    (r"__.*?__", "underline_markdown"),           # __underline__
    (r"`{1,3}.*?`{1,3}", "code_inline"),          # `code`
    (r"!\[.*?\]\(.*?\)", "markdown_image"),       # ![alt](url)
    (r"\[.*?\]\(.*?\)", "markdown_link"),         # [text](url)
    (r"^\s*[-—]{3,}\s*$", "horizontal_rule"),     # --- 
]

ANTI_COLLOQUIAL_PATTERNS = [
    (r"(综上所述|由此可见|基于上述分析|因此我们可以得出结论|总而言之|一言以蔽之|归纳起来|总结如下)", "formal_conclusion"),
    (r"(首先.*其次.*最后|第一.*第二.*第三|第一点|第二点|第三点)", "enumerated_structure"),
    (r"(本文|本回答|笔者|本助手|该系统|本平台|作为一个人工智能|作为AI助手|作为智能助手|作为语音助手|哎呀)", "formal_reference"),
]

def structure_penalty_v2(text):
    hits = 0

    for pattern, _ in STRUCTURE_PATTERNS:
        if re.search(pattern, text, flags=re.MULTILINE):
            hits += 1
            print(pattern)

    for pattern, _ in ANTI_COLLOQUIAL_PATTERNS:
        if re.search(pattern, text):
            hits += 1
            print(pattern)

    if hits == 0:
        return 0.0

    # 每命中一个 -0.1，上限 -0.2
    return -min(0.1 * hits, 0.2)


text = "这所大学是巴西圣保罗大学，2025年QS拉丁美洲及加勒比地区排名是第1名。它常年稳居拉美第一，学术实力和国际化程度都很强。<|im_end|>"

print(structure_penalty_v2(text))
