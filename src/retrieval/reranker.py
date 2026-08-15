"""Optional cross-encoder reranking."""

from __future__ import annotations

from functools import lru_cache

from src import config
from src.models.schemas import RetrievalResult


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL)


def rerank(
    query: str,
    results: list[RetrievalResult],
    top_n: int | None = None,
) -> list[RetrievalResult]:
    """Rerank retrieval results using a cross-encoder."""
    if not results or not config.ENABLE_RERANKING:
        return results[: top_n or config.TOP_K_CONTEXT]

    model = _get_reranker()
    pairs = [(query, r.text) for r in results]
    scores = model.predict(pairs)

    scored = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    reranked: list[RetrievalResult] = []
    for rank, (result, score) in enumerate(scored[: top_n or config.TOP_K_CONTEXT], start=1):
        reranked.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                score=round(float(score), 4),
                rank=rank,
                document_id=result.document_id,
                document_name=result.document_name,
                page_number=result.page_number,
                section_title=result.section_title,
                text=result.text,
            )
        )
    return reranked
