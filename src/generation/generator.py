"""Grounded answer generation with abstention."""

from __future__ import annotations

from src import config
from src.citations.citation_engine import build_citations
from src.generation.prompts import GROUNDED_PROMPT, SYSTEM_PROMPT, build_context_block
from src.models.schemas import Answer, RetrievalResult
from src.retrieval.reranker import rerank


ABSTAIN_MESSAGE = (
    "I couldn't find enough information in your knowledge base to answer this reliably."
)


def _has_sufficient_evidence(results: list[RetrievalResult]) -> bool:
    if not results:
        return False
    return results[0].score >= config.MIN_RELEVANCE_SCORE


def _generate_with_openai(question: str, context: str) -> str | None:
    if not config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        prompt = GROUNDED_PROMPT.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
        )
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )
        return response.choices[0].message.content or None
    except Exception:
        return None


def _generate_extractive(question: str, results: list[RetrievalResult]) -> str:
    """Fallback grounded answer without external LLM."""
    parts = []
    for i, r in enumerate(results[:config.TOP_K_CONTEXT], start=1):
        excerpt = r.text[:400].strip()
        if excerpt and not excerpt.endswith("."):
            last_period = excerpt.rfind(".")
            if last_period > 100:
                excerpt = excerpt[: last_period + 1]
        parts.append(f"Based on [{i}], {excerpt}")
    intro = f"Regarding your question: {question}\n\n"
    return intro + "\n\n".join(parts)


def generate_answer(
    question: str,
    retrieval_results: list[RetrievalResult],
    use_reranking: bool | None = None,
) -> Answer:
    """Generate a grounded answer with citations and abstention."""
    rerank_enabled = use_reranking if use_reranking is not None else config.ENABLE_RERANKING

    vector_results = list(retrieval_results)
    reranked_results = (
        rerank(question, vector_results) if rerank_enabled else vector_results
    )
    context_results = reranked_results[: config.TOP_K_CONTEXT]

    debug = {
        "vector_retrieval": [
            {
                "rank": r.rank,
                "document": r.document_name,
                "page": r.page_number,
                "score": r.score,
                "chunk_id": r.chunk_id,
            }
            for r in vector_results
        ],
        "reranked": rerank_enabled,
        "selected_context": [
            {
                "rank": r.rank,
                "document": r.document_name,
                "page": r.page_number,
                "score": r.score,
            }
            for r in context_results
        ],
    }

    if not _has_sufficient_evidence(context_results):
        return Answer(
            question=question,
            response=ABSTAIN_MESSAGE,
            citations=[],
            retrieved_chunks=context_results,
            abstained=True,
            retrieval_debug=debug,
        )

    context = build_context_block(context_results)
    response = _generate_with_openai(question, context)
    if response is None:
        response = _generate_extractive(question, context_results)

    citations = build_citations(context_results)

    return Answer(
        question=question,
        response=response,
        citations=citations,
        retrieved_chunks=context_results,
        abstained=False,
        retrieval_debug=debug,
    )
