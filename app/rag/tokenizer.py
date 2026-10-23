"""中文检索分词：字 bigram + 单字，无需额外词典依赖。

对政策类短文本检索，bigram 的精度已经足够，且完全离线、可解释。
"""
from __future__ import annotations

import re
import unicodedata

CJK_RE = re.compile(r"[一-鿿]+")
LATIN_RE = re.compile(r"[a-z0-9]+")

# 领域同义词归一化，替换时按长度倒序，避免短词先命中
SYNONYMS = {
    "7天无理由": "七天无理由",
    "7天": "七天",
    "七天无理由退货": "七天无理由",
    "无理由退货": "七天无理由",
    "退款不退货": "仅退款",
    "只退款": "仅退款",
    "价保": "价格保护",
    "保价": "价格保护",
    "运费保险": "运费险",
    "物流信息": "物流",
    "快递": "物流",
    "开发票": "发票",
    "开票": "发票",
    "客服": "人工客服",
    "赔付标准": "赔付金额",
    "赔偿标准": "赔付金额",
    "赔多少": "赔付金额",
    "能赔多少": "赔付金额",
    "退多少": "退款金额",
    "能退多少": "退款金额",
    "多少钱": "金额",
    "期限": "自然日",
    "时效": "自然日",
    "几天": "自然日",
}
_SYNONYM_ITEMS = sorted(SYNONYMS.items(), key=lambda kv: len(kv[0]), reverse=True)

STOPWORDS = {"的", "了", "吗", "呢", "啊", "是", "我", "你", "请问", "一下"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    for src, dst in _SYNONYM_ITEMS:
        text = text.replace(src, dst)
    return text


def tokenize(text: str) -> list[str]:
    norm = normalize(text)
    tokens: list[str] = []
    for run in CJK_RE.findall(norm):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(LATIN_RE.findall(norm))
    return [t for t in tokens if t not in STOPWORDS]
