"""纯 Python BM25，零依赖、可解释、冷启动无 embedding 成本。"""
from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.idf: dict[str, float] = {}
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: list[str], index: int) -> float:
        freqs = self.doc_freqs[index]
        dl = self.doc_len[index] or 1
        score = 0.0
        for term in query_tokens:
            tf = freqs.get(term)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += idf * tf * (self.k1 + 1) / denom
        return score

    def top_n(self, query_tokens: list[str], n: int = 5) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(self.n_docs)]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:n]
