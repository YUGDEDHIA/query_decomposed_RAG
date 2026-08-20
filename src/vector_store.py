"""Hand-built cosine-similarity vector store over chunk embeddings.

Deliberately not a vector DB (see requirements.txt) -- at tens of thousands
of chunks, a dense numpy matrix and a dot product is simple and fast
enough. Embeddings are L2-normalized once at build time, so search is just
a matmul against a normalized query vector.

Metadata is kept as plain dicts (the same shape written to
data/processed/*.jsonl by src/ingest.py -- i.e. dataclasses.asdict(Chunk)),
not Chunk objects, so this module doesn't need to import src.chunker. Note
`page_range` is therefore a 2-element list, not a tuple, same JSON caveat as
ingest.py.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.section_topics import matches_topic


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid div-by-zero; shouldn't happen for real embeddings
    return matrix / norms


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a @ b, suppressing a known spurious RuntimeWarning.

    On macOS, numpy built against Apple's Accelerate BLAS backend can emit
    "divide by zero"/"overflow"/"invalid value encountered in matmul" on
    perfectly valid float32 matmuls -- verified the actual output has no
    NaN/Inf either way. Suppressed here rather than left to alarm callers
    on every search().
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return a @ b


@dataclass
class VectorStore:
    embeddings: np.ndarray  # (N, D), L2-normalized
    metadata: list[dict]  # index-aligned with embeddings

    @classmethod
    def build(cls, metadata: list[dict], embeddings: list[list[float]]) -> "VectorStore":
        if len(metadata) != len(embeddings):
            raise ValueError(f"metadata ({len(metadata)}) and embeddings ({len(embeddings)}) length mismatch")
        matrix = _normalize(np.array(embeddings, dtype=np.float32))
        return cls(embeddings=matrix, metadata=metadata)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        company: str | None = None,
        section: str | None = None,
        section_topic: str | None = None,
    ) -> list[tuple[dict, float]]:
        """Top-k chunks by cosine similarity, optionally pre-filtered by company/section.

        `section` is an exact match against the chunk's literal section
        string. `section_topic` is a canonical topic (see
        src.section_topics) matched via alias/punctuation-tolerant
        normalization -- use this when you don't know the exact wording a
        given company's RHP used for e.g. "risk factors" vs "OBJECTS OF THE
        ISSUE" vs "OBJECTS OF THE OFFER".
        """
        query = _normalize(np.array([query_embedding], dtype=np.float32))[0]

        if company is None and section is None and section_topic is None:
            scores = _matmul(self.embeddings, query)
            order = np.argsort(-scores)[:top_k]
            return [(self.metadata[i], float(scores[i])) for i in order]

        candidate_indices = [
            i for i, m in enumerate(self.metadata)
            if (company is None or m["company"] == company)
            and (section is None or m["section"] == section)
            and (section_topic is None or matches_topic(m["section"], section_topic))
        ]
        if not candidate_indices:
            return []
        scores = _matmul(self.embeddings[candidate_indices], query)
        order = np.argsort(-scores)[:top_k]
        return [(self.metadata[candidate_indices[i]], float(scores[i])) for i in order]

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        with open(out_dir / "metadata.jsonl", "w") as f:
            for m in self.metadata:
                f.write(json.dumps(m) + "\n")

    @classmethod
    def load(cls, in_dir: Path) -> "VectorStore":
        embeddings = np.load(in_dir / "embeddings.npy")
        metadata = []
        with open(in_dir / "metadata.jsonl") as f:
            for line in f:
                metadata.append(json.loads(line))
        return cls(embeddings=embeddings, metadata=metadata)
