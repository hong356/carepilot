"""政策知识库：Markdown 分节 → BM25 检索 → 带 policy_id 的引用片段。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bm25 import BM25
from .tokenizer import normalize, tokenize

SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

# 口语化疑问词会稀释检索信号，检索前剥离（等价于一次轻量 query rewrite）
QUERY_NOISE = (
    "为什么", "怎么样", "怎么", "多久", "多少", "能不能", "可以吗", "还能",
    "请问", "帮我", "一下", "是否", "不支持", "不适用", "可以", "处理", "吗", "呢", "呀", "啊",
)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class PolicyChunk:
    policy_id: str
    title: str
    category: str
    version: str
    section: str
    content: str
    tokens: list[str]


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, parts[2].strip()


class PolicyStore:
    def __init__(self, chunks: list[PolicyChunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25([chunk.tokens for chunk in chunks])
        self._heading_tokens = [set(tokenize(f"{c.title} {c.section}")) for c in chunks]

    @classmethod
    def load(cls, directory: Path) -> "PolicyStore":
        chunks: list[PolicyChunk] = []
        for path in sorted(Path(directory).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            policy_id = meta.get("id", path.stem)
            title = meta.get("title", path.stem)
            category = meta.get("category", "通用")
            version = meta.get("version", "")

            sections = list(SECTION_RE.finditer(body))
            if not sections:
                continue
            for i, match in enumerate(sections):
                start = match.end()
                end = sections[i + 1].start() if i + 1 < len(sections) else len(body)
                section = match.group(1).strip()
                content = body[start:end].strip()
                if not content:
                    continue
                tokens = tokenize(f"{title} {section} {section} {content}")
                chunks.append(
                    PolicyChunk(policy_id, title, category, version, section, content, tokens)
                )
        if not chunks:
            raise RuntimeError(f"政策库为空：{directory}")
        return cls(chunks)

    def search(self, query: str, top_k: int = 4) -> dict[str, Any]:
        # 先做同义词归一化（如「能赔多少」→「赔付金额」），再剔除口语噪声
        clean_query = normalize(query)
        for phrase in QUERY_NOISE:
            clean_query = clean_query.replace(phrase, " ")
        query_tokens = tokenize(clean_query)
        ranked = self.bm25.top_n(query_tokens, top_k * 2)
        query_set = set(query_tokens)

        query_numbers = set(NUMBER_RE.findall(clean_query))
        boosted: list[tuple[int, float]] = []
        for index, score in ranked:
            heading_overlap = len(query_set & self._heading_tokens[index])
            coverage = len(query_set & set(self.chunks[index].tokens))
            # 金额阈值、时效数字是强信号，命中数字额外加权
            number_bonus = len(query_numbers & set(NUMBER_RE.findall(self.chunks[index].content)))
            boosted.append(
                (index, score + 0.25 * heading_overlap + 0.06 * coverage + 1.0 * number_bonus)
            )
        boosted.sort(key=lambda item: item[1], reverse=True)

        results = []
        seen: set[tuple[str, str]] = set()
        for index, score in boosted:
            chunk = self.chunks[index]
            key = (chunk.policy_id, chunk.section)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "policy_id": chunk.policy_id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "version": chunk.version,
                    "section": chunk.section,
                    "content": chunk.content,
                    "score": round(score, 3),
                }
            )
            if len(results) >= top_k:
                break

        return {
            "query": query,
            "count": len(results),
            "results": results,
            "policy_ids": [r["policy_id"] for r in results],
        }

    def get(self, policy_id: str) -> list[PolicyChunk]:
        return [chunk for chunk in self.chunks if chunk.policy_id == policy_id]
