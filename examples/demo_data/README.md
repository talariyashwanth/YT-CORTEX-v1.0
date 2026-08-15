# YT CORTEX Demo Data

This folder contains sample documents for testing YT CORTEX v1.0.

## Files

| File | Topics | Good questions to try |
|------|--------|---------------------|
| `machine_learning_notes.md` | Random Forest, overfitting, ensembles | "What is Random Forest?" / "How does it reduce overfitting?" |
| `deep_learning_notes.md` | Neural networks, CNNs, transformers | "Compare traditional ML with deep learning" |
| `university_regulations.md` | Attendance, scholarships, exams | "What is the minimum attendance requirement?" |
| `rag_architecture_notes.txt` | RAG pipeline, evaluation, hybrid search | "What are the steps in a RAG pipeline?" |

## Quick load in the app

1. Create a knowledge base on the Home page
2. Click **Load Sample Documents** (loads ML + regulations from `examples/`)
3. Or upload all files from this folder manually

## Download

A pre-packaged zip is available at:

```
examples/demo_data.zip
```

Recreate the zip after editing demo files:

```bash
python scripts/package_demo_data.py
```
