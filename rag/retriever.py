"""
Retrieves the top-k most relevant regulatory chunks for a query, with citations, via
pgvector cosine search against the rag_chunks table (db/models.py) -- replaces the
original numpy-array-in-memory index. The embeddings, chunking, and source documents are
unchanged (still produced by rag/ingest.py and loaded into Postgres by db/seed.py); only
where the vectors live changed, from index.npz to a real DB so other services/queries
can join against the same data instead of each process loading its own copy of the index.

Used by the Orchestrator agent to ground incident reports in real regulatory text
instead of generating regulation numbers from the LLM's own (unverifiable) memory.

Run directly for a quick manual check: `python -m rag.retriever "confined space entry"`
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from db.models import RagChunk  # noqa: E402
from db.session import SessionLocal  # noqa: E402

_model = None


def _lazy_load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class RetrievedChunk:
    text: str
    citation: str
    score: float


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    _lazy_load_model()
    query_vec = _model.encode(query, normalize_embeddings=True).tolist()

    db = SessionLocal()
    try:
        # cosine_distance = 1 - cosine_similarity, so similarity score = 1 - distance
        rows = (
            db.query(RagChunk, RagChunk.embedding.cosine_distance(query_vec).label("distance"))
            .order_by("distance")
            .limit(k)
            .all()
        )
        return [RetrievedChunk(text=chunk.text, citation=chunk.citation, score=1 - distance)
                for chunk, distance in rows]
    finally:
        db.close()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "confined space entry gas testing permit"
    for r in retrieve(query, k=5):
        print(f"[{r.score:.3f}] {r.citation}")
        print("  " + r.text[:200].replace("\n", " "))
        print()
