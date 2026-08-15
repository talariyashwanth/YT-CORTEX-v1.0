"""YT CORTEX v1.0 — Main Streamlit Application."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.loader import infer_problem_type, load_dataframe_from_bytes
from src.models.schemas import AnalysisResult, ProblemType
from src.pipeline.orchestrator import run_analysis

st.set_page_config(
    page_title="YT CORTEX v1.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .tagline { color: #6c757d; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .health-good { color: #198754; font-weight: bold; }
    .health-mid { color: #ffc107; font-weight: bold; }
    .health-bad { color: #dc3545; font-weight: bold; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_session_state() -> None:
    defaults = {
        "df": None,
        "filename": "",
        "target_column": None,
        "problem_type": None,
        "analysis_result": None,
        "analysis_complete": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _health_class(score: float) -> str:
    if score >= 70:
        return "health-good"
    if score >= 40:
        return "health-mid"
    return "health-bad"


def render_landing() -> None:
    st.markdown('<p class="main-header">YT CORTEX v1.0</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tagline">Find out why your machine learning model is failing.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Load sample dataset", use_container_width=True):
            sample_path = ROOT / "examples" / "sample_churn.csv"
            st.session_state["df"] = pd.read_csv(sample_path)
            st.session_state["filename"] = "sample_churn.csv"
            st.rerun()

    if uploaded is not None:
        try:
            content = uploaded.getvalue()
            df = load_dataframe_from_bytes(content, uploaded.name)
            st.session_state["df"] = df
            st.session_state["filename"] = uploaded.name
            st.session_state["analysis_complete"] = False
            st.session_state["analysis_result"] = None
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
            return

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.success(f"Loaded **{st.session_state['filename']}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

        target = st.selectbox(
            "Select target column",
            options=list(df.columns),
            index=len(df.columns) - 1,
        )
        st.session_state["target_column"] = target

        pt = infer_problem_type(df[target])
        st.session_state["problem_type"] = pt
        st.info(f"Inferred problem type: **{pt.value.replace('_', ' ').title()}**")

        if st.button("Analyze Dataset", type="primary", use_container_width=True):
            with st.spinner("Running YT CORTEX analysis pipeline..."):
                try:
                    result = run_analysis(df, st.session_state["filename"], target, pt)
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_complete"] = True
                    st.rerun()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")


def render_sidebar() -> None:
    st.sidebar.markdown("## Navigation")
    pages = {
        "Overview": "overview",
        "Data Health": "data_health",
        "Leakage": "leakage",
        "Features": "features",
        "Models": "models",
        "Diagnostics": "diagnostics",
        "Recommendations": "recommendations",
    }
    if not st.session_state.get("analysis_complete"):
        st.sidebar.info("Upload a dataset and run analysis to unlock all pages.")
        return

    for label in pages:
        st.sidebar.page_link(
            f"app.py",
            label=label,
            icon=None,
        )


def render_overview(result: AnalysisResult) -> None:
    profile = result.dataset_profile
    score = profile.health_score
    health_cls = _health_class(score)

    st.markdown('<p class="main-header">Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{profile.n_rows:,}")
    c2.metric("Features", profile.n_features)
    c3.metric("Health Score", f"{score}/100")
    c4.metric("Best Model", result.best_model or "N/A")

    st.markdown(
        f'<p class="{health_cls}">{profile.health_summary}</p>',
        unsafe_allow_html=True,
    )

    crit = sum(1 for d in result.diagnostics if d.severity.value == "critical")
    high = sum(1 for d in result.diagnostics if d.severity.value == "high")
    med = sum(1 for d in result.diagnostics if d.severity.value == "medium")

    i1, i2, i3 = st.columns(3)
    i1.metric("Critical Issues", crit)
    i2.metric("High Issues", high)
    i3.metric("Medium Issues", med)

    if result.best_model:
        st.success(
            f"Best baseline: **{result.best_model}** — "
            f"{result.primary_metric.upper()} = {result.best_score:.3f} (validation)"
        )

    st.subheader("Dataset Summary")
    summary_data = {
        "Metric": [
            "Filename", "Missing %", "Duplicates %",
            "Numeric Features", "Categorical Features", "Problem Type",
        ],
        "Value": [
            profile.filename,
            f"{profile.missing_pct:.1%}",
            f"{profile.duplicate_pct:.1%}",
            profile.numeric_features,
            profile.categorical_features,
            result.problem_type.value.replace("_", " ").title(),
        ],
    }
    st.table(pd.DataFrame(summary_data))


def render_data_health(result: AnalysisResult) -> None:
    st.markdown('<p class="main-header">Data Health</p>', unsafe_allow_html=True)
    profile = result.dataset_profile

    rows = []
    for col in profile.columns:
        rows.append({
            "Feature": col.name,
            "Type": col.column_type.value,
            "Missing": f"{col.missing_pct:.1%}",
            "Unique": col.unique_count,
            "Risk": col.risk_level.value.upper(),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if result.quality_issues:
        st.subheader("Quality Issues")
        from app.components.metrics import render_issue_card
        from src.models.schemas import DiagnosticIssue
        for issue in result.quality_issues:
            render_issue_card(DiagnosticIssue(
                id=issue.id, title=issue.title, category=issue.category,
                severity=issue.severity, evidence=issue.evidence,
                explanation=issue.explanation, recommendation=issue.recommendation,
            ))
    else:
        st.success("No significant quality issues detected.")


def render_leakage(result: AnalysisResult) -> None:
    st.markdown('<p class="main-header">Leakage Analysis</p>', unsafe_allow_html=True)
    st.warning(
        "Flags indicate **possible** leakage based on statistical signals. "
        "They are not confirmed leakage — always investigate feature creation timing."
    )

    if not result.leakage_flags:
        st.success("No suspicious leakage signals detected.")
        return

    for flag in result.leakage_flags:
        with st.expander(
            f"🚨 {flag.severity.value.upper()} — {flag.feature} ({flag.signal_type})",
            expanded=flag.severity.value == "critical",
        ):
            st.markdown(f"**Signal value:** {flag.signal_value:.3f}")
            st.markdown(f"**Evidence:** {flag.evidence}")
            st.markdown(f"**Why suspicious:** {flag.explanation}")
            st.markdown(f"**Recommended action:** {flag.recommendation}")


def render_features(result: AnalysisResult) -> None:
    st.markdown('<p class="main-header">Feature Analysis</p>', unsafe_allow_html=True)
    rows = []
    for insight in result.feature_insights:
        rows.append({
            "Feature": insight.name,
            "Type": insight.column_type.value,
            "Summary": insight.summary,
            "Missing %": insight.stats.get("missing_pct", 0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_models(result: AnalysisResult) -> None:
    st.markdown('<p class="main-header">Model Comparison</p>', unsafe_allow_html=True)

    if not result.model_results:
        st.info("No model results available.")
        return

    primary = result.primary_metric
    rows = []
    for m in result.model_results:
        row = {
            "Model": m.name,
            f"Train {primary.upper()}": round(m.train_metrics.get(primary, 0), 3),
            f"Val {primary.upper()}": round(m.val_metrics.get(primary, 0), 3),
            f"Test {primary.upper()}": round(m.test_metrics.get(primary, 0), 3),
            "Overfitting": "Yes" if m.overfitting else "No",
            "Underfitting": "Yes" if m.underfitting else "No",
        }
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    best = result.model_results[0]
    if best.feature_importance:
        st.subheader(f"Top Features — {best.name}")
        imp_df = pd.DataFrame(
            list(best.feature_importance.items()),
            columns=["Feature", "Importance"],
        )
        st.bar_chart(imp_df.set_index("Feature"))

    if best.confusion_matrix:
        st.subheader("Confusion Matrix (Test)")
        st.dataframe(pd.DataFrame(best.confusion_matrix), hide_index=True)


def render_diagnostics(result: AnalysisResult) -> None:
    st.markdown('<p class="main-header">Diagnostics</p>', unsafe_allow_html=True)
    from app.components.metrics import render_issue_card

    if not result.diagnostics:
        st.success("No diagnostic issues found.")
        return

    for issue in result.diagnostics:
        render_issue_card(issue)


def render_recommendations(result: AnalysisResult) -> None:
    st.markdown('<p class="main-header">Recommendations</p>', unsafe_allow_html=True)
    from app.components.metrics import render_recommendation_card

    if not result.recommendations:
        st.info("No recommendations generated.")
        return

    for rec in result.recommendations:
        render_recommendation_card(rec)
        st.divider()


def main() -> None:
    _init_session_state()

    st.sidebar.markdown("# YT CORTEX v1.0")
    st.sidebar.markdown("*ML Diagnostics Platform*")
    st.sidebar.divider()

    if st.session_state.get("analysis_complete") and st.session_state.get("analysis_result"):
        result: AnalysisResult = st.session_state["analysis_result"]
        page = st.sidebar.radio(
            "Navigate",
            [
                "Overview", "Data Health", "Leakage",
                "Features", "Models", "Diagnostics", "Recommendations",
            ],
            label_visibility="collapsed",
        )
        st.sidebar.divider()
        if st.sidebar.button("New Analysis"):
            for key in ["df", "filename", "target_column", "analysis_result", "analysis_complete"]:
                st.session_state[key] = None if key != "analysis_complete" else False
            st.rerun()

        renderers = {
            "Overview": render_overview,
            "Data Health": render_data_health,
            "Leakage": render_leakage,
            "Features": render_features,
            "Models": render_models,
            "Diagnostics": render_diagnostics,
            "Recommendations": render_recommendations,
        }
        renderers[page](result)
    else:
        render_landing()


if __name__ == "__main__":
    main()
