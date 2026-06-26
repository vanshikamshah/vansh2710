# Insurance Claim Settlement Bias Analysis

A Streamlit dashboard for auditing potential bias in life insurance death-claim settlement decisions.

---

## Features

| Tab | Content |
|-----|---------|
| 📊 Descriptive Analysis | Cross-tabulations of all key variables vs Policy Status, summary statistics, distribution charts |
| 🔬 Diagnostic Bias Analysis | Chi-square significance tests, zone/income/age/occupation/state/gender deep-dive with statistical tests |
| 🤖 ML Classification Models | KNN · Decision Tree · Random Forest · Gradient Boosting — accuracy, precision/recall/F1, ROC curves, confusion matrices, FP/FN % |
| 📋 Findings & Recommendations | Key bias findings, severity ratings, actionable governance recommendations |

---

## Quick Start — Local

```bash
# 1. Clone / download this repo
git clone https://github.com/<your-username>/insurance-bias-analysis.git
cd insurance-bias-analysis

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser, then upload `Insurance.csv` via the sidebar.

---

## Deploy on Streamlit Cloud

1. Push this folder to a **public** (or private with Streamlit Cloud access) GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set:
   - **Repository**: `<your-username>/insurance-bias-analysis`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**. The app will be live in ~2 minutes.
5. Upload your CSV via the in-app sidebar uploader.

> The `.streamlit/config.toml` file sets the color theme automatically.

---

## Dataset Format

The app expects a CSV with these columns:

| Column | Type | Notes |
|--------|------|-------|
| `POLICY_NO` | int | Policy identifier |
| `PI_NAME` | str | Policyholder name |
| `PI_GENDER` | str | M / F |
| `SUM_ASSURED` | str | Indian number format, e.g. `1,68,300` |
| `ZONE` | str | Sales zone / team name |
| `PAYMENT_MODE` | str | Annual / Half-Yly / Monthly / Quarterly / Single |
| `EARLY_NON` | str | EARLY / NON EARLY |
| `PI_OCCUPATION` | str | Occupation (nullable) |
| `MEDICAL_NONMED` | str | MEDICAL / NON MEDICAL |
| `PI_STATE` | str | Indian state |
| `REASON_FOR_CLAIM` | str | Cause of death (nullable) |
| `PI_AGE` | int | Age in years |
| `PI_ANNUAL_INCOME` | str | Indian number format |
| `POLICY_STATUS` | str | **Target**: `Approved Death Claim` or `Repudiate Death` |

---

## ML Models — Tuned (v2 Feature Engineering)

Feature engineering pipeline:
- **Target encoding** (smoothed, s=5) for ZONE, PI_STATE, PI_OCCUPATION, REASON_FOR_CLAIM — computed on training split only (no leakage)
- **Frequency encoding** for same 4 high-cardinality columns
- **Ordinal encoding** for PAYMENT_MODE (domain-ordered: Single=6 → Quarterly=1)
- **Binary flags** + IS_EARLY × IS_MED interaction term
- **Log transforms** for SUM_ASSURED and PI_ANNUAL_INCOME
- **Ratio / polynomial features**: INC_SA_R, SA_INC_R, AGE_SQ, AGE_LI, SA_AGE, INC_AGE

| Model | Best Hyperparameters | Search Method | Test Acc | CV Mean | AUC |
|-------|---------------------|---------------|----------|---------|-----|
| KNN | k=19, manhattan, uniform | GridSearchCV | 74.6% | 75.1% | 0.751 |
| Decision Tree | entropy, max_depth=5, min_leaf=2 | GridSearchCV | 72.1% | 75.3% | 0.765 |
| Random Forest | 300 trees, max_depth=10, min_leaf=3 | RandomizedSearchCV | 73.5% | **78.1%** | **0.781** |
| Gradient Boosting | 150 trees, lr=0.05, max_depth=3, subsample=0.8 | RandomizedSearchCV | **74.9%** | 77.4% | 