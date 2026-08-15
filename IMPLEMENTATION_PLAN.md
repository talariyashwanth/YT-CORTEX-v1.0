# YT CORTEX v1.0 — Implementation Plan

> **Note:** The referenced Recall PRD describes a semantic-search/RAG system. The phased deliverables (dataset profiling, leakage detection, baseline ML, diagnostics, Streamlit UI) align with the **ML Debugger PRD**. YT CORTEX v1.0 is implemented as an automated ML diagnostics platform per that specification, branded as YT CORTEX.

## Product Summary

**YT CORTEX v1.0** is an automated ML diagnostics platform for tabular supervised learning. Users upload a dataset, select a target column, and the system profiles data quality, detects risks (including possible leakage), trains reproducible baselines, evaluates generalization, and produces evidence-backed recommendations.

**Core question:** *Why is my model performing poorly, and what should I investigate or change?*

---

## Phased Delivery

### Phase 1 — Dataset Ingestion & Profiling
| Component | Responsibility |
|-----------|----------------|
| `src/ingestion/loader.py` | CSV/XLSX upload, validation, schema detection, problem-type inference |
| `src/profiling/profiler.py` | Row/column stats, missing %, duplicates, dtype breakdown, health score |
| `src/models/schemas.py` | Shared dataclasses: `DatasetProfile`, `ColumnProfile`, `AnalysisResult` |
| `tests/test_ingestion.py`, `tests/test_profiling.py` | Unit tests with sample data |

**Exit criteria:** Load a CSV, infer schema, produce a dataset health profile with numeric/categorical breakdown and target distribution.

### Phase 2 — Data Quality & Leakage Diagnostics
| Component | Responsibility |
|-----------|----------------|
| `src/quality/detector.py` | Missing values, duplicates, constant/near-constant cols, ID detection, cardinality risks |
| `src/leakage/detector.py` | Target-like names, suspicious correlation/association, near-perfect predictors |
| `tests/test_quality.py`, `tests/test_leakage.py` | Issue detection tests |

**Exit criteria:** Flag quality issues with severity; surface *possible* leakage with evidence (never claim confirmed leakage).

### Phase 3 — Baseline ML Models & Evaluation
| Component | Responsibility |
|-----------|----------------|
| `src/preprocessing/pipeline.py` | sklearn `Pipeline` + `ColumnTransformer` (fit inside CV to prevent leakage) |
| `src/modeling/trainer.py` | Dummy, Logistic/Ridge, Random Forest, HistGradientBoosting |
| `src/evaluation/evaluator.py` | Classification/regression metrics, train/val gap, overfitting/underfitting |
| `src/features/analyzer.py` | Feature distributions, correlations, target relationships |
| `tests/test_modeling.py` | End-to-end training smoke tests |

**Exit criteria:** Train all baselines, compare metrics, detect overfitting/underfitting indicators.

### Phase 4 — Diagnostic Engine & Recommendations
| Component | Responsibility |
|-----------|----------------|
| `src/diagnostics/engine.py` | Aggregate issues: id, title, category, severity, evidence, explanation |
| `src/recommendations/engine.py` | Prioritized, evidence-backed next steps |
| `tests/test_diagnostics.py`, `tests/test_recommendations.py` | Engine output validation |

**Exit criteria:** Unified diagnostic report with CRITICAL → LOW severity; ranked recommendations.

### Phase 5 — Streamlit UI
| Component | Responsibility |
|-----------|----------------|
| `app/app.py` | Landing page, session state, orchestration |
| `app/pages/*.py` | Overview, Data Health, Leakage, Features, Models, Diagnostics, Recommendations |
| `app/components/*.py` | Reusable metric cards, issue cards, charts |

**Exit criteria:** Full interactive workflow: Upload → Select Target → Analyze → Explore all tabs.

---

## Repository Structure

```text
yt-cortex-v1/
├── app/
│   ├── app.py                      # Main Streamlit entry + landing
│   ├── pages/
│   │   ├── overview.py
│   │   ├── data_health.py
│   │   ├── leakage.py
│   │   ├── features.py
│   │   ├── models.py
│   │   ├── diagnostics.py
│   │   └── recommendations.py
│   └── components/
│       ├── metrics.py
│       ├── issue_card.py
│       └── charts.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   └── schemas.py              # Shared data models
│   ├── ingestion/
│   │   └── loader.py
│   ├── profiling/
│   │   └── profiler.py
│   ├── quality/
│   │   └── detector.py
│   ├── leakage/
│   │   └── detector.py
│   ├── features/
│   │   └── analyzer.py
│   ├── preprocessing/
│   │   └── pipeline.py
│   ├── modeling/
│   │   └── trainer.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── diagnostics/
│   │   └── engine.py
│   ├── recommendations/
│   │   └── engine.py
│   └── pipeline/
│       └── orchestrator.py         # End-to-end analysis runner
├── tests/
├── examples/
│   └── sample_churn.csv
├── data/                           # Runtime uploads (gitignored)
├── reports/                        # Exported reports
├── requirements.txt
├── README.md
├── IMPLEMENTATION_PLAN.md
└── .gitignore
```

---

## Technology Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Data | pandas, numpy, scipy |
| ML | scikit-learn, optional XGBoost later |
| Viz | Plotly |
| Tests | pytest |

---

## Data Flow

```text
Upload (CSV/XLSX)
    → Validate & Load
    → Select Target → Infer Problem Type
    → Profile Dataset (Phase 1)
    → Detect Quality Issues (Phase 2)
    → Detect Possible Leakage (Phase 2)
    → Analyze Features
    → Build Preprocessing Pipeline
    → Train/Evaluate Baselines (Phase 3)
    → Run Diagnostic Engine (Phase 4)
    → Generate Recommendations (Phase 4)
    → Render Streamlit Dashboard (Phase 5)
```

---

## Design Principles

1. **Evidence over confidence** — Every flag shows supporting numbers.
2. **Hypothesis language** — Use "possible leakage", not "confirmed leakage".
3. **Pipeline integrity** — Preprocessing fitted only on training folds.
4. **Modular phases** — Each `src/` package is independently testable.
5. **Preserve functionality** — Later phases extend, never replace, earlier ones.

---

## Future Roadmap (post v1.0)

- V2: SHAP explainability, hyperparameter search, time-series diagnostics
- V3: FastAPI backend, async training, experiment registry
- V4: LLM-assisted root-cause analysis (companion to Recall RAG system)
