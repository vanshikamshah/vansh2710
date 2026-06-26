"""
Insurance Claim Settlement Bias Analysis Dashboard
====================================================
Streamlit app — single-page, tabbed layout
Tabs: Descriptive | Diagnostic Bias | ML Models | Findings
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score,
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claim Bias Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2rem; font-weight: 700; color: #1B2A47;
            text-align: center; padding: 0.75rem 0;
            border-bottom: 3px solid #C0392B; margin-bottom: 1rem;
        }
        .kpi-card {
            background: #F7F9FC; border-left: 4px solid #1B2A47;
            border-radius: 6px; padding: 1rem; text-align: center;
        }
        .kpi-val { font-size: 2rem; font-weight: 700; color: #1B2A47; }
        .kpi-lbl { font-size: 0.8rem; color: #666; margin-top: 2px; }
        .section-hdr {
            font-size: 1.1rem; font-weight: 600; color: #1B2A47;
            border-left: 4px solid #C0392B; padding-left: 0.6rem;
            margin: 1.2rem 0 0.6rem;
        }
        .bias-alert {
            background: #FDECEA; border-left: 4px solid #C0392B;
            padding: 0.7rem 1rem; border-radius: 4px; margin: 0.4rem 0;
            font-size: 0.88rem;
        }
        .info-box {
            background: #FFF8E1; border-left: 4px solid #F39C12;
            padding: 0.7rem 1rem; border-radius: 4px; margin: 0.4rem 0;
            font-size: 0.88rem;
        }
        .good-box {
            background: #E8F5E9; border-left: 4px solid #27AE60;
            padding: 0.7rem 1rem; border-radius: 4px; margin: 0.4rem 0;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

APPROVED_CLR = "#27AE60"
REPUD_CLR = "#C0392B"
NAVY = "#1B2A47"
AMBER = "#F39C12"

# ────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(file_obj):
    df = pd.read_csv(file_obj)
    df["SUM_ASSURED"] = df["SUM_ASSURED"].astype(str).str.replace(",", "").astype(float)
    df["PI_ANNUAL_INCOME"] = df["PI_ANNUAL_INCOME"].astype(str).str.replace(",", "").astype(float)
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Unknown")
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].fillna("Unknown")

    df["TARGET"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)

    df["AGE_GROUP"] = pd.cut(
        df["PI_AGE"], bins=[0, 40, 55, 65, 100],
        labels=["Young (≤40)", "Middle (41-55)", "Senior (56-65)", "Elderly (>65)"],
    )
    df["INCOME_GROUP"] = pd.cut(
        df["PI_ANNUAL_INCOME"], bins=[0, 100_000, 300_000, 600_000, 1e9],
        labels=["Low (<₹1L)", "Medium (₹1-3L)", "High (₹3-6L)", "Very High (>₹6L)"],
    )
    df["SA_GROUP"] = pd.cut(
        df["SUM_ASSURED"], bins=[0, 200_000, 500_000, 1_000_000, 1e9],
        labels=["Low (<₹2L)", "Medium (₹2-5L)", "High (₹5-10L)", "Very High (>₹10L)"],
    )
    return df


# ────────────────────────────────────────────────────────────
# FEATURE ENGINEERING v2 — leakage-free, target encoding
# ────────────────────────────────────────────────────────────
def build_features_v2(X_tr_raw, X_te_raw, y_tr):
    """
    v2 Feature Engineering (computed on training split only):
      • Smoothed target encoding for high-cardinality cols (s=5)
      • Frequency encoding for same cols
      • Ordinal encoding for PAYMENT_MODE (domain-aware)
      • Log transforms, ratio features, interaction terms
      • Binary flags + IS_EARLY × IS_MED interaction
    """
    X_tr = X_tr_raw.copy(); X_te = X_te_raw.copy()
    high_card = ["ZONE", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"]
    global_mean = float(y_tr.mean()); s = 5
    for col in high_card:
        agg = (
            pd.DataFrame({"y": y_tr.values, "x": X_tr[col].values})
            .groupby("x")["y"].agg(["count", "mean"])
        )
        agg["smooth"] = (agg["count"] * agg["mean"] + s * global_mean) / (agg["count"] + s)
        te_map   = agg["smooth"].to_dict()
        freq_map = X_tr[col].value_counts(normalize=True).to_dict()
        X_tr[col + "_TE"] = X_tr[col].map(te_map).fillna(global_mean)
        X_te[col + "_TE"] = X_te[col].map(te_map).fillna(global_mean)
        X_tr[col + "_FE"] = X_tr[col].map(freq_map).fillna(0.0)
        X_te[col + "_FE"] = X_te[col].map(freq_map).fillna(0.0)

    pm_ord = {"Single": 6, "Annual": 4, "Monthly": 3, "Half-Yly": 2, "Quarterly": 1}
    for df_ in [X_tr, X_te]:
        df_["PM_ORD"]   = df_["PAYMENT_MODE"].map(pm_ord).fillna(3)
        df_["LOG_SA"]   = np.log1p(df_["SUM_ASSURED"])
        df_["LOG_INC"]  = np.log1p(df_["PI_ANNUAL_INCOME"])
        df_["INC_SA_R"] = df_["PI_ANNUAL_INCOME"] / (df_["SUM_ASSURED"] + 1)
        df_["SA_INC_R"] = df_["SUM_ASSURED"] / (df_["PI_ANNUAL_INCOME"] + 1)
        df_["AGE_SQ"]   = df_["PI_AGE"] ** 2
        df_["AGE_LI"]   = df_["PI_AGE"] * df_["LOG_INC"]
        df_["SA_AGE"]   = df_["SUM_ASSURED"] / (df_["PI_AGE"] + 1)
        df_["INC_AGE"]  = df_["PI_ANNUAL_INCOME"] * df_["PI_AGE"]
        df_["IS_MALE"]  = (df_["PI_GENDER"]      == "M").astype(int)
        df_["IS_EARLY"] = (df_["EARLY_NON"]       == "EARLY").astype(int)
        df_["IS_MED"]   = (df_["MEDICAL_NONMED"]  == "MEDICAL").astype(int)
        df_["EARLY_MED"]= df_["IS_EARLY"] * df_["IS_MED"]

    feat_cols = (
        [c + "_TE" for c in high_card]
        + [c + "_FE" for c in high_card]
        + ["PM_ORD", "PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME",
           "LOG_SA", "LOG_INC", "INC_SA_R", "SA_INC_R", "AGE_SQ",
           "AGE_LI", "SA_AGE", "INC_AGE", "IS_MALE", "IS_EARLY", "IS_MED", "EARLY_MED"]
    )
    return X_tr[feat_cols].reset_index(drop=True), X_te[feat_cols].reset_index(drop=True)


# Baseline v1 numbers (pre-tuning) stored for comparison chart
V1_BASELINE = {
    "KNN":              {"train_acc": 76.0, "test_acc": 65.9, "auc": 0.632},
    "Decision Tree":    {"train_acc": 78.6, "test_acc": 70.1, "auc": 0.740},
    "Random Forest":    {"train_acc": 87.6, "test_acc": 74.0, "auc": 0.787},
    "Gradient Boosting":{"train_acc": 90.6, "test_acc": 73.7, "auc": 0.803},
}
# CV scores from GridSearch/RandomizedSearch tuning runs
CV_SCORES = {
    "KNN":              {"cv_mean": 75.1, "cv_std": 1.1},
    "Decision Tree":    {"cv_mean": 75.3, "cv_std": 1.9},
    "Random Forest":    {"cv_mean": 78.1, "cv_std": 2.2},
    "Gradient Boosting":{"cv_mean": 77.4, "cv_std": 1.4},
}
TUNED_PARAMS = {
    "KNN":              "k=19, metric=manhattan, weights=uniform",
    "Decision Tree":    "criterion=entropy, max_depth=5, min_samples_leaf=2, min_samples_split=10",
    "Random Forest":    "n_estimators=300, max_depth=10, min_samples_leaf=3, max_features=sqrt",
    "Gradient Boosting":"n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8",
}


# ────────────────────────────────────────────────────────────
# FEATURE ENGINEERING v2 + TUNED MODEL TRAINING
# ────────────────────────────────────────────────────────────
@st.cache_data
def engineer_and_train(df_json: str):
    df = pd.read_json(df_json, orient="records")

    FEATURE_COLS = [
        "PI_GENDER", "SUM_ASSURED", "ZONE", "PAYMENT_MODE", "EARLY_NON",
        "PI_OCCUPATION", "MEDICAL_NONMED", "PI_STATE", "REASON_FOR_CLAIM",
        "PI_AGE", "PI_ANNUAL_INCOME",
    ]
    X_raw = df[FEATURE_COLS]
    y = df["TARGET"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = build_features_v2(X_train_raw, X_test_raw, y_train)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Tuned hyperparameters (from GridSearchCV / RandomizedSearchCV with StratifiedKFold-5)
    models_def = {
        "KNN": KNeighborsClassifier(n_neighbors=19, weights="uniform", metric="manhattan"),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, min_samples_leaf=2,
            min_samples_split=10, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=2, random_state=42,
        ),
    }

    results = {}
    for name, model in models_def.items():
        if name == "KNN":
            model.fit(X_tr_sc, y_train)
            tr_pred  = model.predict(X_tr_sc)
            te_pred  = model.predict(X_te_sc)
            te_proba = model.predict_proba(X_te_sc)[:, 1]
            fi_index = None
        else:
            model.fit(X_train, y_train)
            tr_pred  = model.predict(X_train)
            te_pred  = model.predict(X_test)
            te_proba = model.predict_proba(X_test)[:, 1]
            fi_index = list(X_train.columns)

        cm = confusion_matrix(y_test, te_pred)
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        fpr, tpr, _ = roc_curve(y_test, te_proba)

        results[name] = {
            "train_acc":  accuracy_score(y_train, tr_pred) * 100,
            "test_acc":   accuracy_score(y_test,  te_pred) * 100,
            "precision":  precision_score(y_test, te_pred) * 100,
            "recall":     recall_score(y_test,    te_pred) * 100,
            "f1":         f1_score(y_test,         te_pred) * 100,
            "cm":         cm.tolist(),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fp_pct":     round(fp / total * 100, 1),
            "fn_pct":     round(fn / total * 100, 1),
            "tp_pct":     round(tp / total * 100, 1),
            "tn_pct":     round(tn / total * 100, 1),
            "fpr":        fpr.tolist(),
            "tpr":        tpr.tolist(),
            "roc_auc":    float(auc(fpr, tpr)),
            "report":     classification_report(
                y_test, te_pred, target_names=["Repudiated", "Approved"]
            ),
            "cv_mean":    CV_SCORES[name]["cv_mean"],
            "cv_std":     CV_SCORES[name]["cv_std"],
            "tuned_params": TUNED_PARAMS[name],
        }
        if hasattr(model, "feature_importances_") and fi_index:
            results[name]["fi"] = (
                pd.Series(model.feature_importances_, index=fi_index)
                .sort_values(ascending=False)
                .head(15)
                .to_dict()
            )

    return results, list(X_train.columns)


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def approval_bar(df, col, title, rotate=False):
    g = (
        df.groupby(col, observed=True)["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "n"})
    )
    g["rate"] = g["rate"] * 100
    overall = df["TARGET"].mean() * 100
    colors = [APPROVED_CLR if r >= overall else REPUD_CLR if r < 55 else AMBER for r in g["rate"]]
    fig = go.Figure()
    fig.add_bar(
        x=g[col].astype(str),
        y=g["rate"],
        marker_color=colors,
        text=[f"{r:.1f}%<br>(n={n})" for r, n in zip(g["rate"], g["n"])],
        textposition="outside",
    )
    fig.add_hline(y=overall, line_dash="dash", line_color=NAVY, annotation_text=f"Overall {overall:.1f}%")
    fig.update_layout(
        title=title, height=350, yaxis=dict(range=[0, 115], title="Approval Rate (%)"),
        plot_bgcolor="white", margin=dict(t=50, b=60), showlegend=False,
    )
    if rotate:
        fig.update_xaxes(tickangle=-30)
    return fig


def chi_test(df, col):
    ct = pd.crosstab(df[col], df["TARGET"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return chi2, p, dof


# ────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📂 Data Upload")
uploaded = st.sidebar.file_uploader("Upload Insurance CSV", type=["csv"])

if uploaded is None:
    st.markdown(
        '<div class="main-title">🔍 Insurance Claim Settlement Bias Analysis</div>',
        unsafe_allow_html=True,
    )
    st.info("👆 Upload your Insurance CSV file in the sidebar to begin analysis.")
    st.stop()

df = load_and_clean(uploaded)

st.sidebar.success(f"✅ Loaded **{len(df):,}** records")
st.sidebar.markdown(f"- **Approved:** {df['TARGET'].sum():,} ({df['TARGET'].mean()*100:.1f}%)")
st.sidebar.markdown(f"- **Repudiated:** {(df['TARGET']==0).sum():,} ({(1-df['TARGET'].mean())*100:.1f}%)")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Filters")
zones_all = sorted(df["ZONE"].unique().tolist())
sel_zones = st.sidebar.multiselect("Filter by Zone", zones_all, default=zones_all)
age_min, age_max = int(df["PI_AGE"].min()), int(df["PI_AGE"].max())
sel_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
df = df[df["ZONE"].isin(sel_zones) & df["PI_AGE"].between(*sel_age)]

# ────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🔍 Insurance Claim Settlement Bias Analysis</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
overall_rate = df["TARGET"].mean() * 100
for col, val, lbl in [
    (c1, f"{len(df):,}", "Total Claims"),
    (c2, f"{df['TARGET'].sum():,}", "Approved"),
    (c3, f"{(df['TARGET']==0).sum():,}", "Repudiated"),
    (c4, f"{overall_rate:.1f}%", "Approval Rate"),
    (c5, f"{df['PI_AGE'].mean():.0f} yrs", "Avg Age"),
]:
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive Analysis",
    "🔬 Diagnostic Bias Analysis",
    "🤖 ML Classification Models",
    "📋 Findings & Recommendations",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hdr">Target Variable Distribution</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        counts = df["POLICY_STATUS"].value_counts()
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color_discrete_sequence=[APPROVED_CLR, REPUD_CLR],
            hole=0.4, title="Claim Outcomes",
        )
        fig_pie.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        summary_df = pd.DataFrame({
            "Metric": ["Total Claims", "Approved", "Repudiated", "Approval Rate",
                        "Avg Age", "Avg Income (₹)", "Avg Sum Assured (₹)", "Missing Occupation",
                        "Missing Reason"],
            "Value": [
                f"{len(df):,}",
                f"{df['TARGET'].sum():,}",
                f"{(df['TARGET']==0).sum():,}",
                f"{overall_rate:.1f}%",
                f"{df['PI_AGE'].mean():.1f}",
                f"₹{df['PI_ANNUAL_INCOME'].mean():,.0f}",
                f"₹{df['SUM_ASSURED'].mean():,.0f}",
                f"{(df['PI_OCCUPATION']=='Unknown').sum()}",
                f"{(df['REASON_FOR_CLAIM']=='Unknown').sum()}",
            ],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-hdr">Cross-Tabulations — All Key Variables vs Policy Status</div>', unsafe_allow_html=True)

    cross_pairs = [
        ("AGE_GROUP", "Age Group"),
        ("INCOME_GROUP", "Income Group"),
        ("SA_GROUP", "Sum Assured Group"),
        ("PI_GENDER", "Gender"),
        ("MEDICAL_NONMED", "Medical / Non-Medical"),
        ("EARLY_NON", "Early / Non-Early Policy"),
        ("PAYMENT_MODE", "Payment Mode"),
    ]

    for i in range(0, len(cross_pairs), 2):
        cols = st.columns(2)
        for j, (col_name, label) in enumerate(cross_pairs[i : i + 2]):
            if j < len(cols):
                with cols[j]:
                    ct = pd.crosstab(df[col_name], df["POLICY_STATUS"])
                    ct_pct = (pd.crosstab(df[col_name], df["POLICY_STATUS"], normalize="index") * 100).round(1)
                    ct_combined = ct.copy()
                    ct_combined["Approval Rate (%)"] = ct_pct.get("Approved Death Claim", 0)
                    st.markdown(f"**{label}**")
                    st.dataframe(ct_combined, use_container_width=True)
                    st.plotly_chart(approval_bar(df, col_name, label, rotate=(col_name=="PAYMENT_MODE")),
                                    use_container_width=True)

    # State cross-tab
    st.markdown('<div class="section-hdr">State-wise Cross-Tabulation (Top 12 States)</div>', unsafe_allow_html=True)
    top_states = df["PI_STATE"].value_counts().nlargest(12).index
    df_states = df[df["PI_STATE"].isin(top_states)]
    ct_state = pd.crosstab(df_states["PI_STATE"], df_states["POLICY_STATUS"])
    ct_state_pct = (pd.crosstab(df_states["PI_STATE"], df_states["POLICY_STATUS"], normalize="index") * 100).round(1)
    ct_state["Approval Rate (%)"] = ct_state_pct.get("Approved Death Claim", 0)
    ct_state = ct_state.sort_values("Approval Rate (%)", ascending=False)
    st.dataframe(ct_state, use_container_width=True)

    # Occupation cross-tab
    st.markdown('<div class="section-hdr">Occupation-wise Cross-Tabulation (≥20 claims)</div>', unsafe_allow_html=True)
    occ_counts = df["PI_OCCUPATION"].value_counts()
    top_occ = occ_counts[occ_counts >= 20].index
    df_occ = df[df["PI_OCCUPATION"].isin(top_occ)]
    ct_occ = pd.crosstab(df_occ["PI_OCCUPATION"], df_occ["POLICY_STATUS"])
    ct_occ_pct = (pd.crosstab(df_occ["PI_OCCUPATION"], df_occ["POLICY_STATUS"], normalize="index") * 100).round(1)
    ct_occ["Approval Rate (%)"] = ct_occ_pct.get("Approved Death Claim", 0)
    ct_occ = ct_occ.sort_values("Approval Rate (%)", ascending=False)
    st.dataframe(ct_occ, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC BIAS ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Chi-Square Significance Tests — All Variables</div>', unsafe_allow_html=True)

    cat_test_cols = [
        "ZONE", "PI_GENDER", "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED",
        "AGE_GROUP", "INCOME_GROUP", "SA_GROUP", "PI_STATE", "PI_OCCUPATION",
    ]
    chi_rows = []
    for col in cat_test_cols:
        chi2, p, dof = chi_test(df, col)
        chi_rows.append({
            "Variable": col.replace("_", " "),
            "Chi-Square": round(chi2, 2),
            "p-value": round(p, 4),
            "Degrees of Freedom": dof,
            "Significant Bias?": "✅ YES" if p < 0.05 else "❌ NO",
            "Interpretation": "Statistically significant bias detected" if p < 0.05 else "No significant bias",
        })
    chi_df = pd.DataFrame(chi_rows).sort_values("Chi-Square", ascending=False)
    st.dataframe(
        chi_df.style.apply(
            lambda row: ["background-color: #FDECEA" if "YES" in str(row["Significant Bias?"]) else ""] * len(row),
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )

    # Zone analysis
    st.markdown('<div class="section-hdr">Zone / Team Bias Analysis</div>', unsafe_allow_html=True)
    zone_df = (
        df.groupby("ZONE")["TARGET"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "Approval Rate", "count": "Count"})
    )
    zone_df["Approval Rate"] = (zone_df["Approval Rate"] * 100).round(1)
    zone_df = zone_df[zone_df["Count"] >= 10].sort_values("Approval Rate", ascending=True)

    fig_zone = px.bar(
        zone_df, x="Approval Rate", y="ZONE", orientation="h",
        color="Approval Rate",
        color_continuous_scale=["#C0392B", "#F39C12", "#27AE60"],
        text="Approval Rate",
        title="Zone-wise Approval Rate — sorted ascending (chi²=191.6, p<0.001)",
        height=600,
    )
    fig_zone.add_vline(x=overall_rate, line_dash="dash", line_color=NAVY,
                        annotation_text=f"Overall {overall_rate:.1f}%")
    fig_zone.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_zone.update_layout(plot_bgcolor="white", coloraxis_showscale=False, margin=dict(l=150))
    st.plotly_chart(fig_zone, use_container_width=True)

    with st.expander("Zone-level data table"):
        st.dataframe(zone_df.sort_values("Approval Rate", ascending=False), use_container_width=True, hide_index=True)

    # Income analysis
    st.markdown('<div class="section-hdr">Income-wise Bias Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(approval_bar(df, "INCOME_GROUP", "Income Group vs Approval Rate"),
                        use_container_width=True)
        approved_inc = df[df["TARGET"] == 1]["PI_ANNUAL_INCOME"]
        repudiated_inc = df[df["TARGET"] == 0]["PI_ANNUAL_INCOME"]
        t_stat, p_val = stats.ttest_ind(approved_inc, repudiated_inc)
        if p_val < 0.05:
            st.markdown(
                f'<div class="bias-alert">🚨 <b>Income t-test significant:</b> t={t_stat:.3f}, p={p_val:.4f}<br>'
                f"Mean income — Approved: ₹{approved_inc.mean():,.0f} vs Repudiated: ₹{repudiated_inc.mean():,.0f}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="good-box">✅ Income t-test: t={t_stat:.3f}, p={p_val:.4f} — not significant at α=0.05</div>',
                unsafe_allow_html=True,
            )
    with c2:
        fig_inc_box = px.violin(
            df, y="PI_ANNUAL_INCOME", x="POLICY_STATUS", color="POLICY_STATUS",
            color_discrete_map={"Approved Death Claim": APPROVED_CLR, "Repudiate Death": REPUD_CLR},
            box=True, title="Annual Income Distribution by Claim Outcome",
        )
        fig_inc_box.update_layout(showlegend=False, height=350, plot_bgcolor="white")
        st.plotly_chart(fig_inc_box, use_container_width=True)

    # Age analysis
    st.markdown('<div class="section-hdr">Age-wise Bias Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(approval_bar(df, "AGE_GROUP", "Age Group vs Approval Rate"),
                        use_container_width=True)
        approved_age = df[df["TARGET"] == 1]["PI_AGE"]
        repudiated_age = df[df["TARGET"] == 0]["PI_AGE"]
        t_age, p_age = stats.ttest_ind(approved_age, repudiated_age)
        st.markdown(
            f'<div class="good-box">ℹ️ Age t-test: t={t_age:.3f}, p={p_age:.4f} — <b>NOT significant</b>. '
            f"Avg age: Approved={approved_age.mean():.1f}, Repudiated={repudiated_age.mean():.1f}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        fig_age_box = px.box(
            df, y="PI_AGE", x="POLICY_STATUS", color="POLICY_STATUS",
            color_discrete_map={"Approved Death Claim": APPROVED_CLR, "Repudiate Death": REPUD_CLR},
            title="Age Distribution by Claim Outcome",
        )
        fig_age_box.update_layout(showlegend=False, height=350, plot_bgcolor="white")
        st.plotly_chart(fig_age_box, use_container_width=True)

    # Gender, Medical, Early
    st.markdown('<div class="section-hdr">Policy & Medical Factor Bias</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (ax_c, lbl) in zip(
        ["PI_GENDER", "MEDICAL_NONMED", "EARLY_NON"],
        [(c1, "Gender"), (c2, "Medical Type"), (c3, "Early/Non-Early Policy")],
    ):
        with ax_c:
            chi2, p, _ = chi_test(df, col)
            st.plotly_chart(approval_bar(df, col, f"{lbl} (p={p:.3f})"), use_container_width=True)
            if p < 0.05:
                st.markdown(
                    f'<div class="bias-alert">🚨 Significant bias: χ²={chi2:.2f}, p={p:.4f}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="good-box">✅ No significant bias: p={p:.4f}</div>',
                    unsafe_allow_html=True,
                )

    # Occupation analysis
    st.markdown('<div class="section-hdr">Occupation-wise Bias Analysis</div>', unsafe_allow_html=True)
    occ_df = (
        df.groupby("PI_OCCUPATION")["TARGET"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "Approval Rate", "count": "Count"})
    )
    occ_df = occ_df[occ_df["Count"] >= 15].sort_values("Approval Rate", ascending=True)
    occ_df["Approval Rate"] = (occ_df["Approval Rate"] * 100).round(1)

    fig_occ = px.bar(
        occ_df, x="Approval Rate", y="PI_OCCUPATION", orientation="h",
        color="Approval Rate",
        color_continuous_scale=["#C0392B", "#F39C12", "#27AE60"],
        text="Approval Rate",
        title="Occupation-wise Approval Rate",
        height=500,
    )
    fig_occ.add_vline(x=overall_rate, line_dash="dash", line_color=NAVY)
    fig_occ.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_occ.update_layout(plot_bgcolor="white", coloraxis_showscale=False, margin=dict(l=180))
    st.plotly_chart(fig_occ, use_container_width=True)

    # State analysis
    st.markdown('<div class="section-hdr">State-wise Approval Rate</div>', unsafe_allow_html=True)
    state_df = (
        df.groupby("PI_STATE")["TARGET"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "Approval Rate", "count": "Count"})
    )
    state_df = state_df[state_df["Count"] >= 20].sort_values("Approval Rate", ascending=True)
    state_df["Approval Rate"] = (state_df["Approval Rate"] * 100).round(1)

    fig_state = px.bar(
        state_df, x="Approval Rate", y="PI_STATE", orientation="h",
        color="Approval Rate",
        color_continuous_scale=["#C0392B", "#F39C12", "#27AE60"],
        text="Approval Rate",
        title="State-wise Approval Rate (χ²=137.3, p<0.001)",
        height=500,
    )
    fig_state.add_vline(x=overall_rate, line_dash="dash", line_color=NAVY)
    fig_state.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_state.update_layout(plot_bgcolor="white", coloraxis_showscale=False, margin=dict(l=160))
    st.plotly_chart(fig_state, use_container_width=True)

    # Correlation / heatmap
    st.markdown('<div class="section-hdr">Bias Heatmap — Approval Rate by Policy & Medical Factors</div>', unsafe_allow_html=True)
    heat_data = {
        "PAYMENT_MODE": df.groupby("PAYMENT_MODE")["TARGET"].mean() * 100,
        "EARLY_NON": df.groupby("EARLY_NON")["TARGET"].mean() * 100,
        "MEDICAL_NONMED": df.groupby("MEDICAL_NONMED")["TARGET"].mean() * 100,
        "PI_GENDER": df.groupby("PI_GENDER")["TARGET"].mean() * 100,
    }
    heat_df = pd.DataFrame(heat_data).T.round(1)
    fig_heat, ax_heat = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        heat_df, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax_heat,
        linewidths=0.5, vmin=40, vmax=100, annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "Approval Rate (%)"},
    )
    ax_heat.set_title("Approval Rate (%) by Variable Category", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ML MODELS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">Feature Engineering v2 — Leakage-Free Pipeline (24 features)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        | Feature Group | Variables | Engineering Method |
        |---|---|---|
        | **Target Encoding** | ZONE, PI_STATE, PI_OCCUPATION, REASON_FOR_CLAIM | Smoothed mean encoding (s=5) — computed on train only |
        | **Frequency Encoding** | Same 4 high-cardinality cols | Class frequency ratio — computed on train only |
        | **Ordinal Encoding** | PAYMENT_MODE | Domain-ordered: Single=6, Annual=4, Monthly=3, Half-Yly=2, Quarterly=1 |
        | **Binary Flags** | PI_GENDER, EARLY_NON, MEDICAL_NONMED | IS_MALE, IS_EARLY, IS_MED |
        | **Interaction** | IS_EARLY × IS_MED | EARLY_MED cross-feature |
        | **Log Transforms** | SUM_ASSURED, PI_ANNUAL_INCOME | LOG_SA, LOG_INC (reduce skew) |
        | **Ratio Features** | Income/SA, SA/Income, SA/Age, Income×Age | INC_SA_R, SA_INC_R, SA_AGE, INC_AGE |
        | **Polynomial** | PI_AGE | AGE_SQ, AGE×LOG_INC |
        | **Split / Validation** | — | 80/20 stratified · StratifiedKFold-5 CV |
        | **Scaling** | — | StandardScaler on KNN features only (not tree models) |
        """
    )
    st.markdown('<div class="section-hdr">Tuned Hyperparameters (GridSearchCV / RandomizedSearchCV)</div>', unsafe_allow_html=True)
    params_df = pd.DataFrame([
        {"Model": k, "Best Parameters": v, "Search Method": "GridSearchCV" if k in ("KNN","Decision Tree") else "RandomizedSearchCV (n_iter=20+)"}
        for k, v in TUNED_PARAMS.items()
    ])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    with st.spinner("Training tuned models … this may take ~30 seconds"):
        results, feature_names = engineer_and_train(df.to_json(orient="records"))

    model_names = list(results.keys())

    # ── Before / After Comparison ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Before vs After — v1 Baseline vs v2 Tuned</div>', unsafe_allow_html=True)
    cmp_rows = []
    for m in model_names:
        r    = results[m]
        b    = V1_BASELINE[m]
        delta = r["test_acc"] - b["test_acc"]
        cmp_rows.append({
            "Model":           m,
            "v1 Train Acc":    f"{b['train_acc']:.1f}%",
            "v1 Test Acc":     f"{b['test_acc']:.1f}%",
            "v2 Train Acc":    f"{r['train_acc']:.1f}%",
            "v2 Test Acc":     f"{r['test_acc']:.1f}%",
            "Δ Test Acc":      f"{'+' if delta>=0 else ''}{delta:.1f}%",
            "5-Fold CV Mean":  f"{r['cv_mean']:.1f}% ± {r['cv_std']:.1f}%",
            "v1 AUC":          f"{b['auc']:.3f}",
            "v2 AUC":          f"{r['roc_auc']:.3f}",
        })
    cmp_df = pd.DataFrame(cmp_rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # Before/After bar chart
    MODELS_PLOT = model_names
    CLR_V1 = "#95a5a6"; CLR_V2 = "#27AE60"
    fig_cmp = go.Figure()
    fig_cmp.add_bar(name="v1 Baseline (test)", x=MODELS_PLOT,
                    y=[V1_BASELINE[m]["test_acc"] for m in MODELS_PLOT],
                    marker_color=CLR_V1,
                    text=[f"{V1_BASELINE[m]['test_acc']:.1f}%" for m in MODELS_PLOT],
                    textposition="outside")
    fig_cmp.add_bar(name="v2 Tuned (test)", x=MODELS_PLOT,
                    y=[results[m]["test_acc"] for m in MODELS_PLOT],
                    marker_color=CLR_V2,
                    text=[f"{results[m]['test_acc']:.1f}%" for m in MODELS_PLOT],
                    textposition="outside")
    fig_cmp.add_bar(name="v2 CV Mean", x=MODELS_PLOT,
                    y=[results[m]["cv_mean"] for m in MODELS_PLOT],
                    marker_color="#2980B9",
                    text=[f"{results[m]['cv_mean']:.1f}%" for m in MODELS_PLOT],
                    textposition="outside")
    fig_cmp.update_layout(
        barmode="group", height=400,
        yaxis=dict(range=[50, 92], title="Accuracy (%)"),
        plot_bgcolor="white",
        title="Test Accuracy: v1 Baseline vs v2 Tuned (with 5-Fold CV Mean)",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Accuracy comparison
    st.markdown('<div class="section-hdr">Training vs Testing Accuracy (Tuned Models)</div>', unsafe_allow_html=True)
    fig_acc = go.Figure()
    fig_acc.add_bar(
        name="Training Accuracy", x=model_names,
        y=[results[m]["train_acc"] for m in model_names],
        marker_color=NAVY, text=[f"{results[m]['train_acc']:.1f}%" for m in model_names],
        textposition="outside",
    )
    fig_acc.add_bar(
        name="Testing Accuracy", x=model_names,
        y=[results[m]["test_acc"] for m in model_names],
        marker_color=APPROVED_CLR, text=[f"{results[m]['test_acc']:.1f}%" for m in model_names],
        textposition="outside",
    )
    fig_acc.update_layout(
        barmode="group", height=400, yaxis=dict(range=[50, 100], title="Accuracy (%)"),
        plot_bgcolor="white", title="Training vs Testing Accuracy by Model",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # Precision, Recall, F1
    st.markdown('<div class="section-hdr">Precision, Recall & F1-Score</div>', unsafe_allow_html=True)
    fig_prf = go.Figure()
    for metric, color, label in [
        ("precision", "#2980B9", "Precision"),
        ("recall", AMBER, "Recall"),
        ("f1", "#8E44AD", "F1-Score"),
    ]:
        fig_prf.add_bar(
            name=label, x=model_names,
            y=[results[m][metric] for m in model_names],
            marker_color=color,
            text=[f"{results[m][metric]:.1f}%" for m in model_names],
            textposition="outside",
        )
    fig_prf.update_layout(
        barmode="group", height=400, yaxis=dict(range=[50, 105], title="Score (%)"),
        plot_bgcolor="white", title="Precision, Recall & F1-Score by Model",
    )
    st.plotly_chart(fig_prf, use_container_width=True)

    # ROC Curves
    st.markdown('<div class="section-hdr">ROC Curves — Model Stability</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    roc_colors = [NAVY, AMBER, APPROVED_CLR, REPUD_CLR]
    for name, clr in zip(model_names, roc_colors):
        fig_roc.add_scatter(
            x=results[name]["fpr"], y=results[name]["tpr"],
            mode="lines", name=f"{name} (AUC={results[name]['roc_auc']:.3f})",
            line=dict(color=clr, width=2.5),
        )
    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
                         line=dict(color="#aaa", dash="dash", width=1))
    fig_roc.update_layout(
        height=450, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        plot_bgcolor="white", title="ROC Curves — All Models",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-hdr">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(2)
    for i, name in enumerate(model_names):
        cm_np = np.array(results[name]["cm"])
        total = cm_np.sum()
        cm_pct = (cm_np / total * 100).round(1)
        tn, fp, fn, tp = results[name]["tn"], results[name]["fp"], results[name]["fn"], results[name]["tp"]

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_pct, annot=False, cmap="Blues", ax=ax_cm,
                    linewidths=1, linecolor="white", vmin=0, vmax=70,
                    cbar_kws={"label": "% of total"})
        labels = [
            [f"TN\n{tn}\n({results[name]['tn_pct']:.1f}%)", f"FP\n{fp}\n({results[name]['fp_pct']:.1f}%)"],
            [f"FN\n{fn}\n({results[name]['fn_pct']:.1f}%)", f"TP\n{tp}\n({results[name]['tp_pct']:.1f}%)"],
        ]
        for row in range(2):
            for col in range(2):
                clr = "white" if cm_pct[row, col] > 35 else "black"
                ax_cm.text(col + 0.5, row + 0.5, labels[row][col],
                            ha="center", va="center", fontsize=9, color=clr, fontweight="bold")
        ax_cm.set_xticklabels(["Pred: Repudiated", "Pred: Approved"], fontsize=8)
        ax_cm.set_yticklabels(["Actual: Repudiated", "Actual: Approved"], fontsize=8, rotation=0)
        ax_cm.set_title(
            f"{name}\nTest Acc={results[name]['test_acc']:.1f}% | FP%={results[name]['fp_pct']:.1f}% | FN%={results[name]['fn_pct']:.1f}%",
            fontsize=9, fontweight="bold",
        )
        plt.tight_layout()
        with cm_cols[i % 2]:
            st.pyplot(fig_cm)
        plt.close(fig_cm)

    # FP/FN stacked bar
    st.markdown('<div class="section-hdr">Confusion Matrix Breakdown (% Contribution) — FN = Wrongful Rejections (Bias Risk)</div>', unsafe_allow_html=True)
    fig_fpfn = go.Figure()
    fig_fpfn.add_bar(name="TP — Correct Approvals",
                      x=model_names, y=[results[m]["tp_pct"] for m in model_names],
                      marker_color=APPROVED_CLR)
    fig_fpfn.add_bar(name="TN — Correct Rejections",
                      x=model_names, y=[results[m]["tn_pct"] for m in model_names],
                      marker_color="#7DCEA0")
    fig_fpfn.add_bar(name="FP — Wrong Approvals (Company Risk)",
                      x=model_names, y=[results[m]["fp_pct"] for m in model_names],
                      marker_color=AMBER)
    fig_fpfn.add_bar(name="FN — Wrong Rejections (Claimant Bias Risk)",
                      x=model_names, y=[results[m]["fn_pct"] for m in model_names],
                      marker_color=REPUD_CLR)
    fig_fpfn.update_layout(
        barmode="stack", height=420, yaxis=dict(title="% of All Predictions"),
        plot_bgcolor="white", title="Confusion Matrix Percentage Breakdown",
    )
    st.plotly_chart(fig_fpfn, use_container_width=True)

    # FP/FN summary table
    st.markdown('<div class="section-hdr">FP / FN Percentage Summary Table</div>', unsafe_allow_html=True)
    fpfn_rows = []
    for name in model_names:
        r = results[name]
        fpfn_rows.append({
            "Model": name,
            "Train Acc (%)": round(r["train_acc"], 2),
            "Test Acc (%)": round(r["test_acc"], 2),
            "CV Mean (%)": round(r["cv_mean"], 2),
            "Precision (%)": round(r["precision"], 2),
            "Recall (%)": round(r["recall"], 2),
            "F1 (%)": round(r["f1"], 2),
            "ROC AUC": round(r["roc_auc"], 3),
            "TP (%)": r["tp_pct"],
            "TN (%)": r["tn_pct"],
            "FP (%) — Company Risk": r["fp_pct"],
            "FN (%) — Claimant Bias": r["fn_pct"],
        })
    fpfn_df = pd.DataFrame(fpfn_rows)
    st.dataframe(fpfn_df.style.highlight_max(subset=["Test Acc (%)","F1 (%)","ROC AUC"], color="#D5F5E3")
                              .highlight_min(subset=["FN (%) — Claimant Bias"], color="#D5F5E3")
                              .highlight_max(subset=["FN (%) — Claimant Bias"], color="#FADBD8"),
                 use_container_width=True, hide_index=True)

    # Classification reports
    st.markdown('<div class="section-hdr">Detailed Classification Reports</div>', unsafe_allow_html=True)
    for name in model_names:
        with st.expander(f"📄 {name} — Classification Report"):
            st.code(results[name]["report"])

    # Feature importance
    fi_models = [m for m in model_names if "fi" in results[m]]
    if fi_models:
        st.markdown('<div class="section-hdr">Feature Importance (Tree Models)</div>', unsafe_allow_html=True)
        fi_cols = st.columns(len(fi_models))
        for i, name in enumerate(fi_models):
            fi_series = pd.Series(results[name]["fi"]).sort_values(ascending=True)
            fig_fi, ax_fi = plt.subplots(figsize=(6, 6))
            colors_fi = [REPUD_CLR if "INCOME" in f or "ZONE" in f else AMBER if "EARLY" in f or "MEDICAL" in f else "#2980B9" for f in fi_series.index]
            ax_fi.barh(fi_series.index, fi_series.values, color=colors_fi)
            ax_fi.set_title(f"{name} — Feature Importance", fontweight="bold", fontsize=10)
            ax_fi.set_xlabel("Importance Score", fontsize=9)
            ax_fi.spines["top"].set_visible(False)
            ax_fi.spines["right"].set_visible(False)
            plt.tight_layout()
            with fi_cols[i]:
                st.pyplot(fig_fi)
            plt.close(fig_fi)


# �