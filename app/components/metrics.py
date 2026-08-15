"""Reusable UI components for YT CORTEX."""

from __future__ import annotations

import streamlit as st

from src.models.schemas import DiagnosticIssue, Recommendation, Severity


SEVERITY_COLORS = {
    Severity.CRITICAL: "#dc3545",
    Severity.HIGH: "#fd7e14",
    Severity.MEDIUM: "#ffc107",
    Severity.LOW: "#0d6efd",
}

SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
}


def severity_badge(severity: Severity) -> str:
    return f"{SEVERITY_EMOJI.get(severity, '')} {severity.value.upper()}"


def render_metric_row(metrics: dict[str, str | float | int]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def render_issue_card(issue: DiagnosticIssue) -> None:
    with st.expander(f"{severity_badge(issue.severity)} — {issue.title}", expanded=False):
        st.markdown(f"**Category:** {issue.category}")
        st.markdown(f"**Evidence:** {issue.evidence}")
        st.markdown(f"**Why it matters:** {issue.explanation}")
        st.markdown(f"**Recommendation:** {issue.recommendation}")


def render_recommendation_card(rec: Recommendation) -> None:
    st.markdown(
        f"**{rec.priority:02d}.** {severity_badge(rec.severity)} — **{rec.title}**  \n"
        f"*{rec.action}*  \n"
        f"<small>{rec.rationale}</small>",
        unsafe_allow_html=True,
    )
