# 🛡️ Prompt-Injection & Jailbreak Detector

A defensive machine-learning classifier that flags prompts attempting to bypass an AI model's safety guidelines (jailbreaks and prompt-injection attacks). Built for **Trust & Safety** and **red-teaming** workflows, where catching malicious inputs and being able to *explain why* both matter.

**Live demo:** *(add your Streamlit link here after deploying)*

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange) ![Tests](https://img.shields.io/badge/tests-7%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## What It Does

Given any prompt, the detector returns:
- a **label** (jailbreak attempt vs benign),
- a **confidence score**, and
- an **interpretable feature breakdown** explaining the decision.

This is a *defensive* tool: it detects malicious inputs. It does not generate attacks.

## How It Works

Each prompt is converted into two complementary families of features:

1. **TF-IDF** word (1–2 gram) and character (2–4 gram) features that capture suspicious wording and obfuscation.
2. **Interpretable structural features**: length, uppercase ratio, count of known injection phrases, roleplay markers, imperative-verb openings, and special-character ratio.

These are combined and fed to a **logistic-regression** classifier. Logistic regression is a deliberate choice: it is fast, robust on small and imbalanced datasets, and its coefficients are interpretable, so a Trust & Safety analyst can justify why a prompt was flagged instead of relying on an opaque score.

```
prompt ──► TF-IDF (word + char n-grams) ──┐
       └─► structural features ───────────┴──► logistic regression ──► label + probability + explanation
```

## Why These Metrics

In abuse detection, a single accuracy number is misleading on imbalanced data. This project reports **precision** (not flagging benign users), **recall** (catching real attacks), **F1**, and **ROC-AUC**, evaluated with **stratified 5-fold cross-validation** for an honest estimate of generalization. The precision/recall trade-off is treated as a policy choice, so both are always shown alongside the full confusion matrix.

> **Honest note on the numbers.** The bundled sample dataset is synthetic and cleanly separable, so cross-validated scores are near-perfect. That is *not* the interesting part. The value of this project is the end-to-end, tested, explainable pipeline; on real red-teaming data the task is genuinely hard and the same pipeline would give more realistic numbers.

## Project Structure

```
├── app.py                 # Streamlit demo (try prompts, view performance)
├── train.py               # CLI: train + print cross-validation metrics
├── src/
│   ├── features.py        # Interpretable feature extraction
│   ├── classifier.py      # Hybrid TF-IDF + structural classifier
│   ├── evaluate.py        # Precision/recall/F1/ROC-AUC + cross-validation
│   └── dataset.py         # Sample labeled dataset generator
└── tests/
    └── test_detector.py   # Unit tests (features + classifier)
```

## Quickstart

```bash
git clone https://github.com/Sidra25825/jailbreak-detector.git
cd jailbreak-detector
pip install -r requirements.txt

python train.py            # train and print cross-validation metrics
streamlit run app.py       # launch the interactive demo
```

## Running the Tests

```bash
python -m pytest tests/ -v
```

## Extending to Real Data

The pipeline reads any CSV with `prompt,label` columns (label 1 = attack, 0 = benign). Swap `src/dataset.py` for a curated red-teaming dataset and the rest of the pipeline works unchanged. The classifier in `src/classifier.py` is isolated behind a small interface, so it can be replaced with a stronger model without touching feature extraction or evaluation.

## Tech Stack

Python, scikit-learn, SciPy, NumPy, pandas, Streamlit, pytest

## License

MIT
