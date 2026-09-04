<div align="center">

# 📱 Smartphone Addiction Prediction

### Kaggle Machine Learning Project with Feature Engineering, Cross-Validation & Ensemble Modeling

**Python · XGBoost · LightGBM · CatBoost · Logistic Regression · Rank Blending**

</div>

---

## 🎯 Executive Summary

This project was developed for **Kaggle Playground Series S6E8**, a binary classification challenge focused on predicting smartphone addiction behavior.

The final solution was built through a sequence of reproducible experiments rather than a single model.

I compared linear and boosting approaches, engineered behavioral thresholds and relational features, tested sparse categorical representations of exact numeric values, and combined complementary model families through **out-of-fold rank blending**.

The final ensemble reached:

- **OOF ROC AUC: 0.967468**
- **Reported Public LB: 0.96876**

> **Main lesson:** model diversity mattered more than squeezing the last decimal out of a single learner.

---

## 📊 Project Highlights

| Metric | Result |
|---|---:|
| 🎯 Task | **Binary Classification** |
| 📏 Competition metric | **ROC AUC** |
| 🧪 Final OOF ROC AUC | **0.967468** |
| 🏁 Reported Public LB | **0.96876** |
| 📚 Train rows | **691,369** |
| 🧾 Test rows | **296,302** |
| 🔢 Predictive features | **12** |
| 🧠 Final ensemble | **XGBoost + Logistic + LightGBM** |

> The reported Public LB score is preserved from the competition workflow, but the repository does not include a downloaded Kaggle leaderboard artifact for that value.

---

## 🧩 What I Explored

The project evolved across several modeling directions:

- Logistic Regression baseline
- CatBoost
- XGBoost
- threshold-based feature engineering
- exact-value sparse categorical representations
- relational ratios
- LightGBM with high-resolution binning
- fold-safe frequency encoding
- seed ensembling
- weighted rank blending

Several ideas were also tested and rejected:

- ExtraTrees
- neural networks
- Factorization Machines
- target encoding
- ranking correctors
- dual representations
- broader frequency encodings

Those dead ends remain documented because they are part of the experimental reasoning, not noise to hide.

---

## 🧠 Modeling Strategy

The workflow followed a consistent pattern:

```text
Baseline
   ↓
Stratified CV
   ↓
OOF predictions
   ↓
Feature experiments
   ↓
Model-family comparison
   ↓
Diversity analysis
   ↓
Rank blending
   ↓
Final ensemble
```

All major comparisons were made using **stratified cross-validation** and **out-of-fold predictions**, allowing models to be compared on equivalent validation structure.

---

## 🔬 Feature Engineering

Feature engineering was one of the strongest sources of improvement.

### Threshold features

Explicit thresholds on variables such as screen time and social-media usage improved XGBoost from:

```text
0.964358 → 0.965233 OOF ROC AUC
```

This showed that nonlinear behavioral cut points contained useful signal that tree boosting could exploit more effectively when made explicit.

### Relational features

The project also tested relationships between variables rather than only raw values.

Useful features included:

- ratios between usage-hour variables,
- relationships between screen time and social usage,
- `screen_minus_social`.

One candidate ratio, `weekend_over_screen`, did not survive selection.

### Fold-safe frequency encoding

Selected features such as:

- `weekend_freq`
- `screen_freq`

were computed using only the training fold during cross-validation.

This avoided leakage while still capturing how common specific values were in the observed training distribution.

---

## 🧮 A Useful Surprise: Sparse Exact-Value Logistic Regression

One of the most interesting findings came from treating exact numeric values as categorical tokens and fitting a sparse Logistic Regression.

That representation performed much better than expected.

It did not replace boosting, but it captured a different type of structure and added useful ensemble diversity.

This is a good example of why model-family diversity can matter as much as individual model strength.

---

## 🌳 LightGBM High-Resolution Branch

A later branch focused on increasing LightGBM's histogram resolution.

Increasing:

```text
max_bin: 255 → 2047
```

produced a clear improvement.

Moving from:

```text
2047 → 4095
```

was nearly neutral during screening, but `4095` was retained in the final configuration.

The strongest LightGBM branch also included:

- relational numeric features,
- selected fold-safe frequencies,
- higher histogram resolution.

The individual LightGBM model from the final branch reached approximately:

```text
OOF ROC AUC ≈ 0.966684
```

and became a major component of the final blend.

---

## 🏁 Experiment Journey

| Experiment | Model | OOF ROC AUC | Public LB | Role in the journey |
|---|---|---:|---:|---|
| EXP-001 | Logistic baseline | 0.911452 | 0.91355 | Reproducible starting point |
| EXP-003 | CatBoost | 0.963593 | 0.96497 | First competitive boosting model |
| EXP-008 | XGBoost | 0.964358 | 0.96587 | Strong early individual model |
| EXP-012 | XGBoost + thresholds | 0.965233 | 0.96696 | Important feature-engineering gain |
| EXP-016 | Refined XGBoost | 0.965702 | 0.96730 | Strong XGBoost reference |
| EXP-027 | Seed ensemble XGBoost + CatBoost | 0.965919 | — | Variance reduction |
| EXP-035 | Exact-values Logistic + blend | 0.966810 | 0.96815* | Sparse representation adds diversity |
| EXP-036 | Logistic + ratios + blend | 0.967037 | 0.96838* | Consistent relational gain |
| EXP-037 | Relational Logistic + blend | 0.967068 | 0.96842* | Adds `screen_minus_social` |
| **EXP-039** | **Final ensemble** | **0.967468** | **0.96876*** | **Best result** |

\* Public LB values are preserved as reported competition results and are not backed by downloaded Kaggle leaderboard artifacts inside the repository.

---

## 🔀 Final Ensemble

The final solution is a weighted **rank blend** of three complementary branches:

| Component | Weight |
|---|---:|
| XGBoost ensemble | **37.5%** |
| Relational Logistic | **22.5%** |
| High-resolution LightGBM | **40.0%** |

Each component is converted to ranks with average tie handling, normalized to `[0, 1]`, and then combined through a weighted mean.

```text
XGBoost ensemble ─────── 37.5%
Relational Logistic ──── 22.5%
High-res LightGBM ────── 40.0%
             ↓
       Weighted Rank Blend
             ↓
     OOF ROC AUC 0.967468
```

The validated implementation is in:

[`src/ensembles/final_ensemble.py`](src/ensembles/final_ensemble.py)

---

## 🧪 What Did Not Work

Not every experiment improved the score.

That is intentional to preserve in the repository.

Documented unsuccessful or neutral directions include:

- ExtraTrees
- neural networks
- Factorization Machine
- target encoding
- ranking correction
- broader frequency encodings
- dual representations
- EXP-040 categorical-copy strategy

These experiments helped narrow the search space and clarified which signals were genuinely complementary.

See [`docs/decisiones_y_descartes.md`](docs/decisiones_y_descartes.md).

---

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-ML-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-Boosting-EC6B23">
  <img src="https://img.shields.io/badge/LightGBM-Boosting-3A7D44">
  <img src="https://img.shields.io/badge/CatBoost-Boosting-FFCC00">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white">
</p>

**Methods:** Stratified Cross-Validation · OOF Prediction · Feature Engineering · Frequency Encoding · Sparse Modeling · Gradient Boosting · Rank Blending

---

## 📂 Repository Structure

```text
src/
  models/         core migrated model implementations
  features/       shared feature transformations
  ensembles/      final ensemble logic
  diagnostics/    analyses outside the main training pipeline
  *.py            historical wrappers and experiment entrypoints

tests/             imports, contracts, smoke tests and parity checks

docs/              experiment log, methodology, decisions and results

assets/            documentation figures

outputs/
  metrics/         lightweight metrics
  reports/         reproducible tables and reports
  manifests/       inventories, hashes and traceability
```

Historical `EXP` scripts are preserved as wrappers or entrypoints to keep the experiment history traceable without duplicating core logic.

---

## 🧭 Where to Start

For a quick technical review:

- [`src/models/xgboost_thresholds.py`](src/models/xgboost_thresholds.py) — refined XGBoost with threshold features
- [`src/models/logistic_relational.py`](src/models/logistic_relational.py) — sparse Logistic model with relational features
- [`src/models/lightgbm_high_resolution.py`](src/models/lightgbm_high_resolution.py) — final LightGBM branch
- [`src/ensembles/final_ensemble.py`](src/ensembles/final_ensemble.py) — weighted rank blend
- [`docs/bitacora_competencia.md`](docs/bitacora_competencia.md) — full experiment history
- [`docs/metodologia.md`](docs/metodologia.md) — validation, leakage control, OOF and ensemble methodology

---

## ✅ Reproducibility

Install the main pipeline:

```bash
pip install -r requirements.txt
```

Download the official competition files and place them in:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Run validation checks:

```bash
python -m compileall -q src
pytest tests -q
```

The repository does **not** include:

- competition datasets,
- full OOF predictions,
- test predictions,
- submission files.

Neural and RealMLP experiments use separate dependencies in:

```text
requirements-experiments.txt
```

The repository preserves the historical experiment entrypoints, but it does not claim that all experiments can be reproduced with one command.

---

## 💡 Key Learnings

- Strong validation structure matters more than leaderboard chasing.
- Feature engineering can still materially improve gradient boosting on tabular data.
- Sparse linear models can add useful diversity even when their standalone score is lower.
- Fold-safe encodings are essential when frequencies are derived from the training distribution.
- A slightly weaker individual model can still improve the final blend if its errors are different.
- Failed experiments are useful when they reduce uncertainty about the search space.
- Rank blending can stabilize heterogeneous model outputs and improve ensemble performance.

---

## ⚠️ Limitations

- Public leaderboard values are preserved as reported results but are not backed by downloaded leaderboard artifacts in the repository.
- Competition data is not redistributed.
- The project is optimized for a Kaggle competition setting, not deployed production inference.
- Some historical experiments require separate dependencies.
- Not all 43 experiments are exposed through a single unified runner.
- OOF performance should not be interpreted as evidence of real-world causal prediction.

---

## 🚀 Potential Next Steps

- Reproduce the strongest branches under a fully unified experiment runner.
- Add experiment tracking with structured metadata.
- Add calibration and error-slice analysis.
- Compare rank blending against stacking or constrained linear blending.
- Add stronger permutation or SHAP-based diagnostics.
- Benchmark modern tabular deep-learning approaches under the same CV folds.
- Add a compact inference demo using synthetic or license-safe sample data.

---

## 👤 Author

**Guido Arturo Broccoli**

[LinkedIn](https://www.linkedin.com/in/guido-a-broccoli) ·
[GitHub](https://github.com/GabArg) ·
[Repository](https://github.com/GabArg/smartphone-addiction-prediction)

---

## 📄 License & Data

Competition data is not included in this repository.

Download it directly from the official Kaggle competition page.

Original code in this repository is released under the MIT License.
