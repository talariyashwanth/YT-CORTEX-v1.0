# YT CORTEX v1.0

**Find out why your machine learning model is failing.**

YT CORTEX is an automated ML diagnostics platform for tabular supervised-learning projects. Upload a dataset, select a target, and the system profiles data quality, detects common ML risks (including possible leakage), trains reproducible baselines, evaluates generalization, and produces evidence-backed recommendations.

## Features

- **Dataset Profiling** — Schema detection, missing values, duplicates, health score
- **Data Quality Detection** — Constant columns, ID detection, cardinality risks
- **Leakage Analysis** — Flags *possible* leakage with statistical evidence
- **Baseline Models** — Dummy, Logistic/Ridge, Random Forest, Gradient Boosting
- **Evaluation** — Classification & regression metrics, overfitting/underfitting detection
- **Diagnostics & Recommendations** — Prioritized, evidence-backed next steps
- **Streamlit UI** — Interactive dashboard for the full workflow

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Launch the app
streamlit run app/app.py
```

## Usage

1. Open the app in your browser
2. Upload a CSV/XLSX dataset (or load the sample dataset)
3. Select your target column
4. Click **Analyze Dataset**
5. Explore Overview, Data Health, Leakage, Features, Models, Diagnostics, and Recommendations

## Project Structure

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full architecture and phased delivery plan.

## Sample Dataset

`examples/sample_churn.csv` — A small customer churn dataset with intentional data quality issues (ID column, possible leakage via `churn_status`).

## License

MIT
