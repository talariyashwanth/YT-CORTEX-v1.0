"""Tests for RAG generation and abstention."""

from src.generation.generator import generate_answer
from src.models.schemas import RetrievalResult


def _make_result(score: float, text: str, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk_{rank}",
        score=score,
        rank=rank,
        document_id="doc1",
        document_name="ML Notes",
        page_number=42,
        section_title="Random Forest",
        text=text,
    )


def test_abstain_on_low_relevance():
    results = [_make_result(0.1, "Unrelated content about cooking recipes.")]
    answer = generate_answer("What is quantum computing?", results, use_reranking=False)
    assert answer.abstained
    assert "couldn't find enough information" in answer.response.lower()


def test_grounded_answer_on_good_relevance():
    results = [
        _make_result(
            0.85,
            "Random Forest combines multiple decision trees to reduce variance "
            "and generally improve generalization.",
            rank=1,
        )
    ]
    answer = generate_answer("What is Random Forest?", results, use_reranking=False)
    assert not answer.abstained
    assert len(answer.citations) == 1
    assert answer.citations[0].document_name == "ML Notes"
