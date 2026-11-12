"""Batch Quality, Equipment Reliability and Commercial Results — Streamlit dashboard.

Companion to Batch_Quality_Final_Notebook.ipynb (same eleven sections). Every figure is computed live
from the four LAL CSV files in this folder.
Run locally:   streamlit run batch_quality_app.py
"""
import warnings                                                             # silence fitting notes
import altair as alt                                                        # charts
import numpy as np                                                          # numbers
import pandas as pd                                                         # tables
import streamlit as st                                                      # dashboard
from scipy.stats import chi2, chi2_contingency, norm                        # intervals and tests
from sklearn.impute import SimpleImputer                                    # fill gaps
from sklearn.linear_model import LogisticRegression                         # classifier and calibration
from sklearn.metrics import auc, precision_recall_curve, roc_curve          # model curves
from sklearn.pipeline import make_pipeline                                  # chain steps
from sklearn.preprocessing import StandardScaler                            # scale inputs

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Batch Quality Dashboard", layout="wide")
alt.data_transformers.disable_max_rows()
PAL = ["#6366F1", "#14B8A6", "#F59E0B", "#8B5CF6", "#0EA5E9", "#F43F5E", "#94A3B8"]  # notebook palette
IND, TEA, AMB, VIO, SKY, ROS, SLA = PAL
TONES = {"indigo": ("#4338CA", "#6366F1"), "teal": ("#0F766E", "#14B8A6"), "amber": ("#B45309", "#F59E0B"),
         "violet": ("#6D28D9", "#8B5CF6"), "sky": ("#0369A1", "#0EA5E9"), "rose": ("#BE123C", "#F43F5E")}
SEED = 42
NUM = ["equipment_age_years", "maintenance_days_ago", "temp_avg", "temp_variance", "pressure_avg", "pressure_variance",
       "humidity_avg", "raw_material_quality", "operator_experience_years", "production_hours", "batch_size"]
PROC = ["temp_avg", "pressure_avg", "humidity_avg", "temp_variance", "pressure_variance"]

st.markdown("""
<style>
.banner {background: linear-gradient(120deg, #1E1B4B 0%, #4338CA 55%, #6366F1 100%); color: #fff; padding: 20px 26px;
         border-radius: 16px; margin-bottom: 16px;}
.banner h1 {color: #fff; font-size: 1.8rem; margin: 0; padding: 0;}
.banner p {color: #E0E7FF; margin: 6px 0 0 0; font-size: .92rem;}
.kpi {border-radius: 14px; padding: 14px 16px 12px 16px; color: #fff; min-height: 118px; margin-bottom: 10px;
      box-shadow: 0 6px 16px rgba(0,0,0,.12);}
.kpi .label {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; opacity: .92; font-weight: 600;}
.kpi .value {font-size: clamp(1.2rem, 1.9vw, 1.85rem); font-weight: 800; line-height: 1.15; margin-top: 6px;}
.kpi .sub {font-size: .78rem; opacity: .92; margin-top: 2px;}
.kpi .bar {height: 6px; background: rgba(255,255,255,.25); border-radius: 4px; margin-top: 9px; overflow: hidden;}
.kpi .bar > div {height: 100%; background: #fff; border-radius: 4px;}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone="indigo", bar=None):                # one coloured value card
    a, z = TONES[tone]
    meter = f'<div class="bar"><div style="width:{min(max(bar, 0), 100):.0f}%"></div></div>' if bar is not None else ""
    col.markdown(f'<div class="kpi" style="background: linear-gradient(135deg, {a}, {z});"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div>{meter}</div>', unsafe_allow_html=True)


def mask(v):                                                                # last four characters only
    t = str(v)
    return "..." + t[-4:] if len(t) > 4 else t


def wilson(f, n, z=1.96):                                                   # 95% Wilson interval for a rate
    f, n = np.asarray(f, float), np.asarray(n, float)
    p = f / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def logit_fit(X, y, iters=30):                                              # logistic regression with standard errors
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


def chart_card(title):                                                      # chart heading with the notebook's label
    st.markdown(f"**{title}**")


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading the four data files...")
def load():
    b = pd.read_csv("LAL_Batch_Manufacturing_Production.csv")
    m = pd.read_csv("LAL_Equipment_Maintenance_Records.csv")
    s = pd.read_csv("sales_records.csv", dtype={"Supplier_ID": str})
    inv = pd.read_csv("inventory_records.csv", dtype={"Supplier_ID": str})
    b = b.merge(m[["batch_id", "maintenance_status"]], on="batch_id", how="left")
    b["production_start"] = pd.to_datetime(b["production_start"], errors="coerce")
    b["year"] = b["production_start"].dt.year
    b["batch_failed"] = (b["batch_passed"] == 0).astype(int)
    b["equipment_age_years"] = b["equipment_age_days"] / 365.25
    b["age_band"] = pd.cut(b["equipment_age_years"], [-1, 2, 5, 10, 100], labels=["0-2 yrs", "2-5 yrs", "5-10 yrs", "10+ yrs"])
    b["operator_band"] = pd.cut(b["operator_experience_years"], [-1, 2, 5, 10, 100], labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "11+ yrs"])
    s["Sale_Date"] = pd.to_datetime(s["Sale_Date"])
    s["year"] = s["Sale_Date"].dt.year
    s["cancelled"] = s["Order_Status"].eq("Cancelled")
    inv["Movement_Date"] = pd.to_datetime(inv["Movement_Date"])
    inv["year"] = inv["Movement_Date"].dt.year
    out = inv[inv["Movement_Type"] == "Sale_Out"].drop_duplicates("Reference_ID").set_index("Reference_ID")["Unit_Cost_KES"]
    s["cogs"] = s["Quantity"] * s["Sale_ID"].map(out)                       # cost of goods sold per order
    s["gp"] = s["Net_Amount_Excl_VAT_KES"] - s["cogs"]                      # gross profit per order
    return b, s, inv


@st.cache_resource(show_spinner="Training the classifier...")
def classifier(_b):
    b = _b
    X = pd.get_dummies(b[NUM + ["maintenance_status", "dosage_form", "manufacturing_section"]],
                       columns=["maintenance_status", "dosage_form", "manufacturing_section"], drop_first=False, dtype=float)
    X = X.drop(columns=["maintenance_status_Current", f"dosage_form_{sorted(b.dosage_form.unique())[0]}",
                        f"manufacturing_section_{sorted(b.manufacturing_section.unique())[0]}"])
    tr, va, te = b.year.between(2007, 2021), b.year.between(2022, 2023), b.year.between(2024, 2026)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=3000)).fit(X[tr], b.batch_failed[tr])
    raw = model.decision_function(X)                                         # log-odds (class-weighted)
    pv = 1 / (1 + np.exp(-raw[va]))
    order = np.sort(np.unique(pv))[::-1]
    yv = b.batch_failed[va].to_numpy()
    thr, best = order[-1], -1                                               # highest precision with recall >= 0.80
    for t in order[::max(1, len(order) // 400)]:
        f = pv >= t
        if f.sum() and yv[f].sum() / yv.sum() >= 0.80:
            prec = yv[f].mean()
            if prec > best:
                thr, best = t, prec
    cal = LogisticRegression().fit(raw[va].reshape(-1, 1), yv)              # Platt calibration on validation
    prob = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return model, cal, X.columns.tolist(), raw, prob, thr, tr.to_numpy(), va.to_numpy(), te.to_numpy()


b, s, inv = load()
live = s[~s.cancelled]
model, cal, cols, raw, prob, thr, TR, VA, TE = classifier(b)

st.markdown('<div class="banner"><h1>Batch Quality, Equipment Reliability and Commercial Results</h1>'
            '<p>Companion to Batch_Quality_Final_Notebook · LAL data 2007–2026 · seed 42 · train 2007–21, validate 2022–23, test 2024–26</p></div>',
            unsafe_allow_html=True)
NAMES = ["Overview", "Batch failure", "Maintenance", "Classifier", "Process control", "Receivables",
         "Margin", "Inventory & forecast", "Cost & suppliers", "Risk score", "Executive summary"]
T = dict(zip(NAMES, st.tabs(NAMES)))

# ------------------------------------------------------------------ overview (section 0)
with T["Overview"]:
    k = st.columns(5)
    kpi(k[0], "Batches", f"{len(b):,}", f"{int(b.year.min())}–{int(b.year.max())}", "indigo")
    kpi(k[1], "Failure rate", f"{b.batch_failed.mean():.2%}", f"{b.batch_failed.sum():,} failed", "rose")
    kpi(k[2], "Sales lines", f"{len(s):,}", f"{s.cancelled.sum():,} cancelled", "teal")
    kpi(k[3], "Inventory movements", f"{len(inv):,}", f"{(inv.Movement_Type == 'Goods_Received').sum():,} receipts", "violet")
    kpi(k[4], "Net sales", f"KES {live.Net_Amount_Excl_VAT_KES.sum() / 1e9:.1f} bn", "2014–2025", "amber")
    c = st.columns(2)
    miss = (b.drop(columns=["age_band", "operator_band"]).isna().mean().sort_values(ascending=False).head(12) * 100).reset_index()
    miss.columns = ["field", "pct"]
    c[0].altair_chart(alt.Chart(miss).mark_bar(color=IND).encode(y=alt.Y("field:N", sort="-x", title=None), x=alt.X("pct:Q", title="Missing (%)"),
                      tooltip=["field", alt.Tooltip("pct:Q", format=".1f")]).properties(height=300, title="[Univariate] Missing values by batch field"), width="stretch")
    py = b.groupby("year").size().reset_index(name="batches")
    c[1].altair_chart(alt.Chart(py).mark_bar(color=IND).encode(x=alt.X("year:O", title=None), y="batches:Q", tooltip=["year", "batches"])
                      .properties(height=300, title="[Univariate] Batches per production year"), width="stretch")
    st.caption("Quality register in the notebook: 25 checks · 20 pass · 0 fail · 5 information.")

# ------------------------------------------------------------------ 1 batch failure
def failure_by(col, order=None):
    g = b.groupby(col, observed=True).batch_failed.agg(n="size", f="sum").reset_index()
    g["rate"] = g.f / g.n * 100
    lo, hi = wilson(g.f, g.n)
    g["low"], g["high"] = lo * 100, hi * 100
    if order:
        g = g.set_index(col).loc[order].reset_index()
    g[col] = g[col].astype(str)
    return g


def rate_chart(col, title, order=None):
    g = failure_by(col, order)
    base = alt.Chart(g).encode(x=alt.X(f"{col}:N", sort=None if order else "-y", title=None))
    avg = alt.Chart(pd.DataFrame({"y": [b.batch_failed.mean() * 100]})).mark_rule(color=ROS, strokeDash=[5, 3]).encode(y="y:Q")
    return alt.layer(base.mark_bar(color=IND, opacity=0.85).encode(y=alt.Y("rate:Q", title="Failure rate (%)"),
                                                                   tooltip=[col, "n", "f", alt.Tooltip("rate:Q", format=".2f")]),
                     base.mark_rule(color="#334155").encode(y="low:Q", y2="high:Q"), avg).properties(height=260, title=title)


with T["Batch failure"]:
    tests = []
    for col in ["operator_band", "dosage_form", "age_band", "maintenance_status", "manufacturing_section"]:
        tab = pd.crosstab(b[col], b.batch_failed)
        tests.append([col, chi2_contingency(tab)[1], failure_by(col).rate.min(), failure_by(col).rate.max()])
    tests = pd.DataFrame(tests, columns=["factor", "p_value", "min_rate", "max_rate"]).sort_values("p_value")
    k = st.columns(4)
    op, do = failure_by("operator_band"), failure_by("dosage_form")
    kpi(k[0], "Operator experience range", f"{op.rate.min():.1f}% – {op.rate.max():.1f}%", "failure rate", "indigo")
    kpi(k[1], f"Highest dosage form: {do.loc[do.rate.idxmax(), 'dosage_form']}", f"{do.rate.max():.1f}%", f"lowest {do.rate.min():.1f}%", "rose")
    kpi(k[2], "Strongest factor (chi-square)", tests.factor.iloc[0].replace("_", " "), f"p = {tests.p_value.iloc[0]:.1e}", "violet")
    kpi(k[3], "Overall failure rate", f"{b.batch_failed.mean():.2%}", "dashed line on charts", "amber")
    c = st.columns(2)
    c[0].altair_chart(rate_chart("dosage_form", "[Univariate] By dosage form (95% Wilson interval)"), width="stretch")
    c[1].altair_chart(rate_chart("manufacturing_section", "[Univariate] By manufacturing section"), width="stretch")
    c = st.columns(3)
    c[0].altair_chart(rate_chart("maintenance_status", "[Univariate] By maintenance status", ["Current", "Due Soon", "Critical"]), width="stretch")
    c[1].altair_chart(rate_chart("age_band", "[Univariate] By equipment age band", ["0-2 yrs", "2-5 yrs", "5-10 yrs", "10+ yrs"]), width="stretch")
    c[2].altair_chart(rate_chart("operator_band", "[Univariate] By operator experience", ["0-2 yrs", "3-5 yrs", "6-10 yrs", "11+ yrs"]), width="stretch")
    yr = failure_by("year")
    yr["year"] = yr.year.astype(float).astype(int)
    band = alt.Chart(yr).encode(x=alt.X("year:O", title=None))
    st.altair_chart(alt.layer(band.mark_area(color=IND, opacity=0.2).encode(y=alt.Y("low:Q", title="Failure rate (%)"), y2="high:Q"),
                              band.mark_line(color=IND, point=True).encode(y="rate:Q", tooltip=["year", alt.Tooltip("rate:Q", format=".2f")]))
                    .properties(height=260, title="[Bivariate] Failure rate by production year (95% band)"), width="stretch")
    c = st.columns(3)
    out = b.assign(outcome=np.where(b.batch_failed == 1, "Failed", "Passed"))
    c[0].altair_chart(alt.Chart(out).mark_boxplot(extent="min-max").encode(x=alt.X("outcome:N", title=None), y=alt.Y("temp_avg:Q", scale=alt.Scale(zero=False)),
                      color=alt.Color("outcome:N", scale=alt.Scale(domain=["Passed", "Failed"], range=[TEA, ROS]), legend=None))
                      .properties(height=260, title="[Bivariate] Temperature by outcome"), width="stretch")
    c[1].altair_chart(alt.Chart(out).mark_boxplot(extent="min-max").encode(x=alt.X("outcome:N", title=None), y=alt.Y("production_hours:Q"),
                      color=alt.Color("outcome:N", scale=alt.Scale(domain=["Passed", "Failed"], range=[TEA, ROS]), legend=None))
                      .properties(height=260, title="[Bivariate] Production hours by outcome"), width="stretch")
    c[2].altair_chart(alt.Chart(b[["operator_experience_years"]]).mark_bar(color=TEA).encode(
        x=alt.X("operator_experience_years:Q", bin=alt.Bin(maxbins=24), title="Operator experience (years)"), y=alt.Y("count():Q", title="Batches"))
        .properties(height=260, title="[Univariate] Operator experience"), width="stretch")
    st.dataframe(tests.assign(p_value=tests.p_value.map(lambda v: f"{v:.1e}")).round(2), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 2 maintenance and equipment
@st.cache_data(show_spinner="Fitting the adjusted models...")
def odds_ratios(_b, fe=False):
    d = _b.dropna(subset=["maintenance_status", "equipment_age_years", "operator_experience_years"])
    X = pd.get_dummies(d[["maintenance_status", "dosage_form", "manufacturing_section"] + (["equipment_id"] if fe else [])], dtype=float)
    X = X.drop(columns=[c for c in X.columns if c in ("maintenance_status_Current",)] +
               [f"dosage_form_{sorted(d.dosage_form.unique())[0]}", f"manufacturing_section_{sorted(d.manufacturing_section.unique())[0]}"] +
               ([f"equipment_id_{sorted(d.equipment_id.unique())[0]}"] if fe else []))
    if fe:                                                                  # sections are fixed by machine: drop to avoid collinearity
        X = X.drop(columns=[c for c in X.columns if c.startswith("manufacturing_section_")])
    X["Equipment age (per year)"] = d.equipment_age_years
    X["Operator experience (per year)"] = d.operator_experience_years
    keep = X.columns[X.std() > 0]
    bcoef, se = logit_fit(X[keep].to_numpy(), d.batch_failed.to_numpy())
    r = pd.DataFrame({"term": keep, "coef": bcoef[1:], "se": se[1:]})
    r["odds_ratio"], r["low"], r["high"] = np.exp(r.coef), np.exp(r.coef - 1.96 * r.se), np.exp(r.coef + 1.96 * r.se)
    r["p_value"] = 2 * (1 - norm.cdf(np.abs(r.coef / r.se)))
    r["term"] = r.term.str.replace("maintenance_status_", "Maintenance: ").str.replace("dosage_form_", "Dosage: ").str.replace("manufacturing_section_", "Section: ")
    return r[~r.term.str.startswith("equipment_id_")]


def equipment_table(b):
    g = b.groupby(["manufacturing_section", "dosage_form"]).batch_failed.transform("mean")  # plant rate for each product mix
    e = b.assign(expected=g).groupby(["equipment_id", "equipment_name"]).agg(
        batches=("batch_id", "size"), failures=("batch_failed", "sum"), expected=("expected", "sum"), hours=("production_hours", "sum")).reset_index()
    e["ratio"] = e.failures / e.expected
    e["low"] = chi2.ppf(0.025, 2 * e.failures) / 2 / e.expected
    e["high"] = chi2.ppf(0.975, 2 * (e.failures + 1)) / 2 / e.expected
    e["priority"] = np.where(e.low > 1, "Priority", np.where(e.ratio > 1, "Watch", "Standard"))
    e["mtbf_hours"] = e.hours / e.failures.replace(0, np.nan)
    for h in (2, 8, 24):
        e[f"availability_{h}h"] = e.mtbf_hours / (e.mtbf_hours + h)
    e["name"] = e.equipment_name.map(mask)
    return e


with T["Maintenance"]:
    orr, orfe = odds_ratios(b), odds_ratios(b, fe=True)
    eq = equipment_table(b)
    crit = orr.set_index("term").loc["Maintenance: Critical"]
    critfe = orfe.set_index("term").loc["Maintenance: Critical"]
    k = st.columns(4)
    kpi(k[0], "Critical: adjusted odds ratio", f"{crit.odds_ratio:.2f}", f"95% CI {crit.low:.2f}–{crit.high:.2f}", "rose")
    kpi(k[1], "Within same machine", f"{critfe.odds_ratio:.2f}", "equipment fixed effects", "violet")
    kpi(k[2], "Machines on Watch", f"{(eq.priority == 'Watch').sum()} of {len(eq)}", "above expected, CI includes 1", "amber")
    kpi(k[3], "MTBF proxy range", f"{eq.mtbf_hours.min():,.0f}–{eq.mtbf_hours.max():,.0f} h", "production hours per failure", "indigo")
    c = st.columns(2)
    fp = alt.Chart(orr).encode(y=alt.Y("term:N", sort=None, title=None))
    c[0].altair_chart((fp.mark_rule(color=SLA).encode(x=alt.X("low:Q", scale=alt.Scale(type="log"), title="Odds ratio (log)"), x2="high:Q")
                       + fp.mark_point(color=IND, filled=True, size=70).encode(x="odds_ratio:Q", tooltip=["term", alt.Tooltip("odds_ratio:Q", format=".2f"),
                                                                                                         alt.Tooltip("p_value:Q", format=".3f")])
                       + alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=ROS, strokeDash=[4, 3]).encode(x="x:Q"))
                      .properties(height=340, title="[Multivariate] Adjusted odds ratios"), width="stretch")
    lim = float(max(eq.expected.max(), eq.failures.max()) * 1.1)
    c[1].altair_chart((alt.Chart(eq).mark_circle(stroke="black", strokeWidth=0.5).encode(
        x=alt.X("expected:Q", title="Expected failures", scale=alt.Scale(domain=[0, lim])), y=alt.Y("failures:Q", title="Observed failures", scale=alt.Scale(domain=[0, lim])),
        size=alt.Size("batches:Q", legend=None), color=alt.Color("ratio:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), title="Obs / exp"),
        tooltip=["name", "batches", "failures", alt.Tooltip("expected:Q", format=".1f"), alt.Tooltip("ratio:Q", format=".2f")])
        + alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(color=SLA, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q"))
        .properties(height=340, title="[Bivariate] Observed vs expected failures by machine"), width="stretch")
    c = st.columns(2)
    e2 = eq.sort_values("ratio")
    ep = alt.Chart(e2).encode(y=alt.Y("name:N", sort=None, title=None))
    c[0].altair_chart((ep.mark_rule(color=SLA).encode(x=alt.X("low:Q", title="Observed / expected"), x2="high:Q")
                       + ep.mark_point(filled=True, size=60).encode(x="ratio:Q", color=alt.Color("priority:N", scale=alt.Scale(domain=["Priority", "Watch", "Standard"], range=[ROS, AMB, IND]), title=None),
                                                                    tooltip=["name", alt.Tooltip("ratio:Q", format=".2f"), "priority"])
                       + alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=ROS, strokeDash=[4, 3]).encode(x="x:Q"))
                      .properties(height=320, title="[Univariate] Standardised failure ratio (95% CI)"), width="stretch")
    av = eq.melt(id_vars="name", value_vars=["availability_2h", "availability_8h", "availability_24h"], var_name="repair", value_name="availability")
    av["repair"] = av.repair.str.replace("availability_", "MTTR ")
    c[1].altair_chart(alt.Chart(av).mark_bar().encode(y=alt.Y("name:N", title=None), x=alt.X("availability:Q", scale=alt.Scale(domain=[0.85, 1]), title="Availability"),
                      yOffset="repair:N", color=alt.Color("repair:N", scale=alt.Scale(range=[TEA, IND, ROS]), title=None),
                      tooltip=["name", "repair", alt.Tooltip("availability:Q", format=".3f")]).properties(height=320, title="[Univariate] Availability scenarios (assumed repair times)"), width="stretch")
    st.dataframe(eq[["name", "batches", "failures", "expected", "ratio", "low", "high", "priority", "mtbf_hours"]].round(2), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 3 classifier and predictor
with T["Classifier"]:
    yte, rte, pte = b.batch_failed.to_numpy()[TE], 1 / (1 + np.exp(-raw[TE])), prob[TE]
    flag = rte >= thr
    tp, fp_, fn, tn = int((flag & (yte == 1)).sum()), int((flag & (yte == 0)).sum()), int((~flag & (yte == 1)).sum()), int((~flag & (yte == 0)).sum())
    fpr, tpr, _ = roc_curve(yte, rte)
    pr, rc, _ = precision_recall_curve(yte, rte)
    k = st.columns(4)
    kpi(k[0], "Test recall", f"{tp / (tp + fn):.3f}", "target 0.80", "teal", bar=tp / (tp + fn) * 100)
    kpi(k[1], "Test precision", f"{tp / (tp + fp_):.3f}", "target 0.19", "rose")
    kpi(k[2], "Test ROC-AUC", f"{auc(fpr, tpr):.3f}", "target 0.80", "indigo")
    kpi(k[3], "Batches flagged", f"{flag.mean():.1%}", "at the adopted threshold", "amber", bar=flag.mean() * 100)
    c = st.columns(3)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr}).iloc[:: max(1, len(fpr) // 400)]
    c[0].altair_chart((alt.Chart(roc).mark_line(color=IND).encode(x=alt.X("fpr:Q", title="False positive rate"), y=alt.Y("tpr:Q", title="True positive rate"))
                       + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=SLA, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q"))
                      .properties(height=280, title=f"[Bivariate] ROC curve (AUC {auc(fpr, tpr):.3f})"), width="stretch")
    prc = pd.DataFrame({"recall": rc, "precision": pr}).iloc[:: max(1, len(rc) // 400)]
    c[1].altair_chart((alt.Chart(prc).mark_line(color=TEA).encode(x="recall:Q", y="precision:Q")
                       + alt.Chart(pd.DataFrame({"y": [yte.mean()]})).mark_rule(color=ROS, strokeDash=[4, 3]).encode(y="y:Q"))
                      .properties(height=280, title="[Bivariate] Precision-recall (dashed: base rate)"), width="stretch")
    cm = pd.DataFrame({"actual": ["Pass", "Pass", "Fail", "Fail"], "predicted": ["Pass", "Flag", "Pass", "Flag"], "n": [tn, fp_, fn, tp]})
    c[2].altair_chart((alt.Chart(cm).mark_rect().encode(x=alt.X("predicted:N", sort=["Pass", "Flag"]), y=alt.Y("actual:N", sort=["Pass", "Fail"]),
                                                        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
                       + alt.Chart(cm).mark_text(fontSize=16).encode(x=alt.X("predicted:N", sort=["Pass", "Flag"]), y=alt.Y("actual:N", sort=["Pass", "Fail"]), text=alt.Text("n:Q", format=",")))
                      .properties(height=280, title="[Univariate] Confusion matrix (test)"), width="stretch")
    dec = pd.DataFrame({"p": pte, "y": yte})
    dec["decile"] = pd.qcut(dec.p.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    dd = dec.groupby("decile").agg(predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    dd[["predicted", "observed"]] *= 100
    gains = dd.assign(lift=dd.observed / (yte.mean() * 100)).sort_values("decile", ascending=False)
    c = st.columns(2)
    lim = float(dd[["predicted", "observed"]].max().max() * 1.1)
    c[0].altair_chart((alt.Chart(dd).mark_line(point=True, color=IND).encode(x=alt.X("predicted:Q", title="Predicted (%)"), y=alt.Y("observed:Q", title="Observed (%)"),
                                                                          tooltip=["decile", alt.Tooltip("predicted:Q", format=".2f"), alt.Tooltip("observed:Q", format=".2f")])
                       + alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(color=SLA, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q"))
                      .properties(height=280, title="[Bivariate] Calibration by decile (test)"), width="stretch")
    c[1].altair_chart(alt.Chart(gains).mark_bar(color=IND).encode(x=alt.X("decile:O", sort="descending", title="Risk decile (10 = highest)"),
                      y=alt.Y("lift:Q", title="Lift"), tooltip=["decile", alt.Tooltip("lift:Q", format=".2f")])
                      .properties(height=280, title="[Univariate] Lift by risk decile"), width="stretch")

    st.markdown("**Predictor: change one input**")
    lr = model[-1]
    med = pd.Series(model[0].statistics_, index=cols)
    typ = pd.Series(0.0, index=cols)
    typ[NUM] = med[NUM]

    def chance(changes):
        x = typ.copy()
        for kk, vv in changes.items():
            x[kk] = vv
        r = lr.intercept_[0] + ((x.to_numpy() - model[1].mean_) / model[1].scale_) @ lr.coef_.ravel()
        return float(cal.predict_proba(np.array([[r]]))[0, 1])

    dcols = [c_ for c_ in cols if c_.startswith("dosage_form_")]
    scen = {"Typical batch": {}, "Critical maintenance": {"maintenance_status_Critical": 1.0},
            "Due Soon maintenance": {"maintenance_status_Due Soon": 1.0},
            "Operator 0 years": {"operator_experience_years": 0.0}, "Operator 20 years": {"operator_experience_years": 20.0}}
    for dcol in [c_ for c_ in dcols if c_.endswith(("Injection", "Bolus"))]:
        scen[dcol.replace("dosage_form_", "") + " dosage"] = {dcol: 1.0}
    sp = pd.DataFrame({"scenario": list(scen), "chance": [chance(v) * 100 for v in scen.values()]})
    k = st.columns([1, 2, 2])
    kpi(k[0], "Typical batch", f"{sp.chance.iloc[0]:.2f}%", "predicted failure chance", "indigo")
    kpi(k[0], "Critical maintenance", f"{sp.set_index('scenario').chance['Critical maintenance']:.2f}%", "same batch", "rose")
    k[1].altair_chart(alt.Chart(sp).mark_bar().encode(y=alt.Y("scenario:N", sort=None, title=None), x=alt.X("chance:Q", title="Predicted failure chance (%)"),
                      color=alt.condition(alt.datum.scenario == "Typical batch", alt.value(IND), alt.value(AMB)), tooltip=["scenario", alt.Tooltip("chance:Q", format=".2f")])
                      .properties(height=260, title="[Multivariate] One change from a typical batch"), width="stretch")
    var = k[2].selectbox("Vary", ["temp_avg", "temp_variance", "operator_experience_years", "equipment_age_years", "raw_material_quality"], key="bq_vary")
    xs = np.linspace(b[var].quantile(0.01), b[var].quantile(0.99), 40)
    k[2].altair_chart(alt.Chart(pd.DataFrame({var: xs, "chance": [chance({var: v}) * 100 for v in xs]})).mark_line(color=TEA).encode(
        x=alt.X(f"{var}:Q", scale=alt.Scale(zero=False)), y=alt.Y("chance:Q", title="Predicted failure chance (%)", scale=alt.Scale(zero=False)))
        .properties(height=220, title=f"[Bivariate] Chance as {var} varies"), width="stretch")

# ------------------------------------------------------------------ 4 process control
@st.cache_data
def process_tables(_b):
    base = _b[(_b.batch_failed == 0) & (_b.year <= 2021)]
    rows = []
    for sec, g in base.groupby("manufacturing_section"):
        for v in PROC:
            m_, sd = g[v].mean(), g[v].std()
            rows.append([sec, v, m_, sd, 0.0 if "variance" in v else m_ - 3 * sd, m_ + 3 * sd])
    cl = pd.DataFrame(rows, columns=["manufacturing_section", "variable", "mean", "sd", "lcl", "ucl"])
    within = []
    for v in PROC:
        lim = cl[cl.variable == v].set_index("manufacturing_section")
        lo, hi = _b.manufacturing_section.map(lim.lcl), _b.manufacturing_section.map(lim.ucl)
        ok = _b[v].between(lo, hi) | _b[v].isna()
        within.append([v, ok.mean(), int((~ok).sum())])
    wl = pd.DataFrame(within, columns=["variable", "share_within", "out_of_control"])
    cap = []
    for sec, g in _b.groupby("manufacturing_section"):
        a = g.assay_result_purity.dropna()
        m_, sd = a.mean(), a.std()
        cap.append([sec, len(a), (105 - 95) / (6 * sd), min(105 - m_, m_ - 95) / (3 * sd), ((a < 95) | (a > 105)).mean()])
    return cl, wl, pd.DataFrame(cap, columns=["section", "n", "cp", "cpk", "out_of_spec"])


with T["Process control"]:
    cl, wl, cap = process_tables(b)
    k = st.columns(3)
    kpi(k[0], "Within limits", f"{wl.share_within.min():.1%} – {wl.share_within.max():.1%}", "across variables", "teal", bar=wl.share_within.min() * 100)
    kpi(k[1], "Cpk range", f"{cap.cpk.min():.2f} – {cap.cpk.max():.2f}", "target 1.33", "rose")
    kpi(k[2], "Sections capable", f"{int((cap.cpk >= 1.33).sum())} of {len(cap)}", "Cpk ≥ 1.33", "amber")
    c = st.columns(2)
    c[0].altair_chart((alt.Chart(wl.assign(pct=wl.share_within * 100)).mark_bar(color=IND, clip=True).encode(
        x=alt.X("variable:N", title=None), y=alt.Y("pct:Q", title="Within limits (%)", scale=alt.Scale(domain=[96, 100])), tooltip=["variable", alt.Tooltip("pct:Q", format=".2f")])
        + alt.Chart(pd.DataFrame({"y": [99.7]})).mark_rule(color=ROS, strokeDash=[4, 3]).encode(y="y:Q")).properties(height=260, title="[Univariate] Share within limits (dashed: 99.7%)"), width="stretch")
    c[1].altair_chart(alt.Chart(wl).mark_bar(color=ROS).encode(x=alt.X("variable:N", title=None), y=alt.Y("out_of_control:Q", title="Readings outside limits"), tooltip=["variable", "out_of_control"])
                      .properties(height=260, title="[Univariate] Out-of-control readings"), width="stretch")
    c = st.columns([1, 2])
    sec = c[0].selectbox("Section", sorted(b.manufacturing_section.unique()), key="bq_sec")
    v = c[0].selectbox("Reading", ["assay_result_purity"] + PROC, key="bq_var")
    ser = b.loc[b.manufacturing_section == sec].sort_values("production_start")[v].head(300).reset_index(drop=True)
    mu, sd = ser.mean(), ser.std()
    ic = pd.DataFrame({"seq": range(len(ser)), "value": ser})
    c[1].altair_chart((alt.Chart(ic).mark_line(point=alt.OverlayMarkDef(size=12), color=IND, strokeWidth=0.8).encode(x=alt.X("seq:Q", title="Batch sequence"), y=alt.Y("value:Q", scale=alt.Scale(zero=False), title=v))
                       + alt.Chart(pd.DataFrame({"y": [mu - 3 * sd, mu, mu + 3 * sd]})).mark_rule(color=ROS, strokeDash=[4, 3]).encode(y="y:Q"))
                      .properties(height=260, title=f"[Univariate] Individuals chart: {v}, {sec} (first 300)"), width="stretch")
    c = st.columns(2)
    c[0].altair_chart((alt.Chart(cap).mark_bar(color=TEA).encode(x=alt.X("section:N", title=None), y=alt.Y("cpk:Q", title="Cpk"), tooltip=["section", alt.Tooltip("cpk:Q", format=".2f")])
                       + alt.Chart(pd.DataFrame({"y": [1.33]})).mark_rule(color=ROS, strokeDash=[4, 3]).encode(y="y:Q")).properties(height=260, title="[Univariate] Assay Cpk by section"), width="stretch")
    c[1].altair_chart((alt.Chart(cap.assign(pct=cap.out_of_spec * 100)).mark_circle(color=IND).encode(x=alt.X("cpk:Q", title="Cpk"), y=alt.Y("pct:Q", title="Out of spec (%)"), size=alt.Size("n:Q", legend=None),
                                                                                                    tooltip=["section", alt.Tooltip("cpk:Q", format=".2f"), alt.Tooltip("pct:Q", format=".2f")])
                       + alt.Chart(cap.assign(pct=cap.out_of_spec * 100)).mark_text(dx=8, align="left", fontSize=11).encode(x="cpk:Q", y="pct:Q", text="section:N"))
                      .properties(height=260, title="[Bivariate] Out-of-spec share vs Cpk"), width="stretch")

# ------------------------------------------------------------------ 5 receivables
with T["Receivables"]:
    cr = live[live.Payment_Method == "Credit Account"].copy()
    end = s.Sale_Date.max()
    cr["age"] = (end - cr.Sale_Date).dt.days
    cr["bucket"] = pd.cut(cr.age, [-1, 30, 90, 180, 365, 100000], labels=["0-30 days", "31-90 days", "91-180 days", "181-365 days", "over 365 days"])
    opn = cr[cr.Balance_Due_KES > 0]
    ag = opn.groupby("bucket", observed=False).agg(balance=("Balance_Due_KES", "sum"), invoices=("Sale_ID", "size")).reset_index()
    ag["bucket"] = ag.bucket.astype(str)
    share = cr.groupby("year").apply(lambda g: (g.Balance_Due_KES > 0).mean()).reset_index(name="open_share")
    over = ag.loc[ag.bucket == "over 365 days", "balance"].sum() / ag.balance.sum()
    k = st.columns(3)
    kpi(k[0], "Open credit", f"KES {ag.balance.sum() / 1e6:,.0f} m", f"{len(opn):,} invoices", "amber")
    kpi(k[1], "Older than a year", f"{over:.1%}", "of the open balance", "rose", bar=over * 100)
    kpi(k[2], "Open share, latest year", f"{share.open_share.iloc[-1]:.0%}", "flat across years", "violet", bar=share.open_share.iloc[-1] * 100)
    c = st.columns(3)
    c[0].altair_chart(alt.Chart(ag.assign(m=ag.balance / 1e6)).mark_bar(color=AMB).encode(x=alt.X("bucket:N", sort=None, title=None), y=alt.Y("m:Q", title="KES million"),
                      tooltip=["bucket", alt.Tooltip("m:Q", format=",.1f")]).properties(height=260, title="[Univariate] Open balance by age"), width="stretch")
    c[1].altair_chart(alt.Chart(ag).mark_bar(color=IND).encode(x=alt.X("bucket:N", sort=None, title=None), y="invoices:Q", tooltip=["bucket", "invoices"])
                      .properties(height=260, title="[Univariate] Open invoices by age"), width="stretch")
    c[2].altair_chart(alt.Chart(share.assign(pct=share.open_share * 100)).mark_line(point=True, color=ROS).encode(x=alt.X("year:O", title=None), y=alt.Y("pct:Q", title="Open share (%)", scale=alt.Scale(zero=False)))
                      .properties(height=260, title="[Bivariate] Credit invoices still open, by year"), width="stretch")
    by = cr.groupby("year").Balance_Due_KES.sum().div(1e6).reset_index(name="m")
    st.altair_chart(alt.Chart(by).mark_bar(color=AMB).encode(x=alt.X("year:O", title=None), y=alt.Y("m:Q", title="KES million"), tooltip=["year", alt.Tooltip("m:Q", format=",.1f")])
                    .properties(height=220, title="[Multivariate] Open balance by sale year (credit accounts)"), width="stretch")

# ------------------------------------------------------------------ 6 margin and discounts
@st.cache_data
def discount_model(_live, max_year=None):                                   # units response per discount point (within product)
    d = _live if max_year is None else _live[_live.year <= max_year]
    y = np.log(d.Quantity.clip(lower=1))
    x = d.Discount_Pct.astype(float)
    yd = y - y.groupby(d.Product_ID).transform("mean")                      # remove product level
    xd = x - x.groupby(d.Product_ID).transform("mean")
    beta = (xd * yd).sum() / (xd ** 2).sum()
    resid = yd - beta * xd
    se = np.sqrt((resid ** 2).sum() / (len(d) - d.Product_ID.nunique() - 1) / (xd ** 2).sum())
    f = lambda v: (np.exp(v) - 1) * 100
    return {"coef": beta, "pct": f(beta), "low": f(beta - 1.96 * se), "high": f(beta + 1.96 * se)}


def cap_scenarios(_live, coef, years=(2024, 2025), cap=5.0):
    d = _live[_live.year.isin(years) & (_live.Discount_Pct > cap)]
    price_factor = (1 - cap / 100) / (1 - d.Discount_Pct / 100)             # same list price, smaller discount
    new_rev = d.Net_Amount_Excl_VAT_KES * price_factor
    full_loss = np.exp(-coef * (d.Discount_Pct - cap))                      # volume kept if the whole response is lost
    out = []
    for name, keep in [("none", 1.0), ("half", None), ("all", None)]:
        k_ = 1.0 if name == "none" else (1 - (1 - full_loss) / 2 if name == "half" else full_loss)
        new_gp = ((new_rev - d.cogs) * k_).sum()
        vol = (d.Quantity * k_).sum() / d.Quantity.sum() * 100 if name != "none" else 100.0
        out.append([name, new_gp - d.gp.sum(), vol])
    return pd.DataFrame(out, columns=["volume case", "profit_change", "volume_retained_pct"])


with T["Margin"]:
    mg = live.groupby("Discount_Pct").agg(orders=("Sale_ID", "size"), gp=("gp", "sum"), rev=("Net_Amount_Excl_VAT_KES", "sum")).reset_index()
    mg["margin"] = mg.gp / mg.rev * 100
    full, oot = discount_model(live), discount_model(live, 2023)
    sc = cap_scenarios(live, oot["coef"])
    k = st.columns(4)
    kpi(k[0], "Margin: 0% vs deepest discount", f"{mg.margin.iloc[0]:.1f}% vs {mg.margin.iloc[-1]:.1f}%", f"at {mg.Discount_Pct.iloc[-1]:g}%", "rose")
    kpi(k[1], "Units per discount point", f"+{oot['pct']:.2f}%", f"2014–2023 · full period +{full['pct']:.2f}%", "sky")
    kpi(k[2], "5% cap, no volume loss", f"KES {sc.profit_change.iloc[0] / 1e6:+,.1f} m", "profit change 2024–25", "teal")
    kpi(k[3], "5% cap, all response lost", f"KES {sc.profit_change.iloc[2] / 1e6:+,.1f} m", "profit change 2024–25", "amber")
    c = st.columns(2)
    base = alt.Chart(mg).encode(x=alt.X("Discount_Pct:O", title="Discount (%)"))
    c[0].altair_chart(alt.layer(base.mark_bar(color=SKY, opacity=0.5).encode(y=alt.Y("orders:Q", title="Orders")),
                                base.mark_line(color=ROS, point=True).encode(y=alt.Y("margin:Q", title="Gross margin (%)", scale=alt.Scale(zero=False)),
                                                                             tooltip=["Discount_Pct", "orders", alt.Tooltip("margin:Q", format=".1f")]))
                      .resolve_scale(y="independent").properties(height=280, title="[Bivariate] Margin and orders by discount"), width="stretch")
    smp = live.sample(min(3000, len(live)), random_state=SEED)
    c[1].altair_chart(alt.Chart(smp).mark_circle(size=10, opacity=0.4, color=IND).encode(
        x=alt.X("Discount_Pct:Q", title="Discount (%)"), y=alt.Y("Quantity:Q", scale=alt.Scale(type="log"), title="Units (log)"))
        .properties(height=280, title="[Bivariate] Units vs discount (3,000 orders)"), width="stretch")
    c = st.columns(3)
    est = pd.DataFrame({"period": ["Full 2014–2025", "Out of time 2014–2023"], "pct": [full["pct"], oot["pct"]],
                        "low": [full["low"], oot["low"]], "high": [full["high"], oot["high"]]})
    eb = alt.Chart(est).encode(x=alt.X("period:N", title=None))
    c[0].altair_chart((eb.mark_bar(color=SKY).encode(y=alt.Y("pct:Q", title="% more units per point")) + eb.mark_rule(color="#334155").encode(y="low:Q", y2="high:Q"))
                      .properties(height=260, title="[Bivariate] Units response (95% CI)"), width="stretch")
    c[1].altair_chart(alt.Chart(sc.assign(m=sc.profit_change / 1e6, sign=np.where(sc.profit_change >= 0, "gain", "loss"))).mark_bar().encode(
        x=alt.X("volume case:N", sort=None, title="Volume response lost"), y=alt.Y("m:Q", title="KES million"),
        color=alt.Color("sign:N", scale=alt.Scale(domain=["gain", "loss"], range=[TEA, ROS]), legend=None), tooltip=["volume case", alt.Tooltip("m:Q", format=",.1f")])
        .properties(height=260, title="[Univariate] Profit change from a 5% cap"), width="stretch")
    c[2].altair_chart(alt.Chart(live).mark_bar(color=IND).encode(x=alt.X("Discount_Pct:O", title="Discount (%)"), y=alt.Y("count():Q", title="Orders"))
                      .properties(height=260, title="[Univariate] Discount levels"), width="stretch")

# ------------------------------------------------------------------ 7 inventory and forecast
@st.cache_data(show_spinner="Back-testing 60 demand series...")
def forecast_holdout(_live):
    m = _live.groupby(["Branch_Code", "Product_ID", _live.Sale_Date.dt.to_period("M").dt.to_timestamp().rename("month")]).Quantity.sum()
    top = m.groupby(level=[0, 1]).sum().nlargest(60).index                   # the 60 largest branch-product series
    months = pd.date_range("2014-01-01", "2025-12-01", freq="MS")
    rows = []
    for br, pid in top:
        y = m.loc[(br, pid)].reindex(months, fill_value=0).astype(float)
        train, hold = y[y.index.year <= 2024], y[y.index.year == 2025]
        naive = train.iloc[-12:].to_numpy()
        mean12 = np.repeat(train.iloc[-12:].mean(), 12)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            hw = ExponentialSmoothing(train.to_numpy(), trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit().forecast(12)
        except Exception:
            hw = naive * train.iloc[-12:].sum() / max(train.iloc[-24:-12].sum(), 1)
        for i, mo in enumerate(hold.index):
            rows.append([br, pid, mo, hold.iloc[i], mean12[i], naive[i], max(hw[i], 0)])
    return pd.DataFrame(rows, columns=["branch", "product", "month", "actual", "mean_12m", "seasonal_naive", "holt_winters"])


with T["Inventory & forecast"]:
    fp = forecast_holdout(live)
    w = lambda c_: (fp.actual - fp[c_]).abs().sum() / fp.actual.sum() * 100
    fh = pd.DataFrame({"method": ["Mean of last 12 months", "Seasonal naive", "Holt-Winters"], "wape": [w("mean_12m"), w("seasonal_naive"), w("holt_winters")]})
    canc_ids = s.loc[s.cancelled, "Sale_ID"]
    cso = inv[(inv.Movement_Type == "Sale_Out") & inv.Reference_ID.isin(canc_ids)].assign(value=lambda d: d.Quantity_Change.abs() * d.Unit_Cost_KES)
    par = cso.groupby("Product_ID").value.sum().sort_values(ascending=False)
    k = st.columns(4)
    kpi(k[0], "Chosen: mean of last 12 months", f"{fh.wape.iloc[0]:.1f}%", "WAPE, 2025 holdout", "indigo")
    kpi(k[1], "Seasonal naive", f"{fh.wape.iloc[1]:.1f}%", f"gain {(1 - fh.wape.iloc[0] / fh.wape.iloc[1]) * 100:.1f}%", "amber")
    kpi(k[2], "Cancelled stock value", f"KES {par.sum() / 1e6:,.1f} m", f"{len(cso):,} movements", "rose")
    kpi(k[3], "Top 5 products", f"{par.head(5).sum() / par.sum():.0%}", "of cancelled value", "violet", bar=par.head(5).sum() / par.sum() * 100)
    c = st.columns([1, 2])
    c[0].altair_chart(alt.Chart(fh).mark_bar(color=IND).encode(y=alt.Y("method:N", sort="x", title=None), x=alt.X("wape:Q", title="WAPE (%)"), tooltip=["method", alt.Tooltip("wape:Q", format=".1f")])
                      .properties(height=260, title="[Univariate] Forecast error, 2025 holdout"), width="stretch")
    mon = fp.groupby("month")[["actual", "mean_12m", "seasonal_naive"]].sum()
    c[1].line_chart(mon.rename(columns={"actual": "Actual", "mean_12m": "Mean of last 12 months", "seasonal_naive": "Seasonal naive"}), color=["#334155", IND, AMB])
    c = st.columns(2)
    pt = par.head(20).reset_index()
    pt["product"] = pt.Product_ID.map(mask)
    pt["cum"] = pt.value.cumsum() / par.sum() * 100
    pt["m"] = pt.value / 1e6
    base = alt.Chart(pt).encode(x=alt.X("product:N", sort=None, title=None))
    c[0].altair_chart(alt.layer(base.mark_bar(color=ROS).encode(y=alt.Y("m:Q", title="KES million")),
                                base.mark_line(color="#334155", point=True).encode(y=alt.Y("cum:Q", title="Cumulative (%)")))
                      .resolve_scale(y="independent").properties(height=280, title="[Univariate] Cancelled stock value: top 20 products"), width="stretch")
    mo = live.groupby(["Branch_Code", "Product_ID", live.Sale_Date.dt.to_period("M")]).Quantity.sum()
    stats = mo.groupby(level=[0, 1]).agg(["mean", "std"]).fillna(0)
    rp = pd.DataFrame({"monthly_mean": stats["mean"], "rop_30d": stats["mean"] + 1.65 * stats["std"],
                       "rop_60d": stats["mean"] * 2 + 1.65 * stats["std"] * np.sqrt(2)}).reset_index()
    c[1].altair_chart(alt.Chart(rp).mark_circle(size=12, opacity=0.5, color=IND).encode(x=alt.X("monthly_mean:Q", title="Monthly mean demand"), y=alt.Y("rop_60d:Q", title="Reorder point at 60 days"))
                      .properties(height=280, title="[Bivariate] Monthly demand vs reorder point (60 days)"), width="stretch")

# ------------------------------------------------------------------ 8 cost and suppliers
with T["Cost & suppliers"]:
    gr = inv[inv.Movement_Type == "Goods_Received"].assign(value=lambda d: d.Quantity_Change * d.Unit_Cost_KES)
    uc = gr.groupby("year").Unit_Cost_KES.median().reset_index()
    sh = gr.groupby(["year", "Supplier_ID"]).Quantity_Change.sum() / gr.groupby("year").Quantity_Change.sum()  # share of units received
    hhi = (sh.mul(100) ** 2).groupby(level=0).sum().reset_index(name="hhi")
    rows = []
    for cat, g in live.groupby("Product_Category"):
        mp = g.groupby(g.Sale_Date.dt.to_period("M")).Unit_Price_KES.median()
        mc = gr[gr.Product_ID.isin(g.Product_ID.unique())].groupby(gr.Movement_Date.dt.to_period("M")).Unit_Cost_KES.median()
        j = pd.concat([np.log(mp), np.log(mc)], axis=1, keys=["p", "c"]).dropna()
        x, y = j.c - j.c.mean(), j.p - j.p.mean()
        beta = (x * y).sum() / (x ** 2).sum()
        se = np.sqrt(((y - beta * x) ** 2).sum() / (len(j) - 2) / (x ** 2).sum())
        rows.append([cat, beta, beta - 1.96 * se, beta + 1.96 * se])
    ptb = pd.DataFrame(rows, columns=["category", "elasticity", "low", "high"])
    k = st.columns(3)
    kpi(k[0], "Median unit cost of receipts", f"KES {uc.Unit_Cost_KES.iloc[0]:,.0f} → {uc.Unit_Cost_KES.iloc[-1]:,.0f}", f"{int(uc.year.iloc[0])} → {int(uc.year.iloc[-1])}", "amber")
    kpi(k[1], "Supplier concentration (HHI)", f"{hhi.hhi.iloc[-1]:,.0f}", "below 1,500" if hhi.hhi.iloc[-1] < 1500 else "above 1,500", "teal")
    kpi(k[2], "Supplier premium", "0 of 30", "suppliers above reference price (notebook)", "indigo")
    c = st.columns(3)
    c[0].altair_chart(alt.Chart(uc).mark_line(point=True, color=IND).encode(x=alt.X("year:O", title=None), y=alt.Y("Unit_Cost_KES:Q", title="Median unit cost (KES)"))
                      .properties(height=260, title="[Bivariate] Unit cost of receipts by year"), width="stretch")
    c[1].altair_chart((alt.Chart(hhi).mark_line(point=True, color=ROS).encode(x=alt.X("year:O", title=None), y=alt.Y("hhi:Q", title="HHI"))
                       + alt.Chart(pd.DataFrame({"y": [1500]})).mark_rule(color="#334155", strokeDash=[4, 3]).encode(y="y:Q"))
                      .properties(height=260, title="[Bivariate] Supplier concentration (dashed: 1,500)"), width="stretch")
    pb = alt.Chart(ptb).encode(y=alt.Y("category:N", title=None))
    c[2].altair_chart((pb.mark_rule(color=SLA).encode(x=alt.X("low:Q", title="Pass-through elasticity"), x2="high:Q")
                       + pb.mark_point(filled=True, color=IND, size=60).encode(x="elasticity:Q", tooltip=["category", alt.Tooltip("elasticity:Q", format=".2f")])
                       + alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=ROS, strokeDash=[4, 3]).encode(x="x:Q"))
                      .properties(height=260, title="[Multivariate] Cost pass-through to price (95% CI)"), width="stretch")

# ------------------------------------------------------------------ 9 risk score
@st.cache_data
def risk_backtest(_b, prob_te):
    t = _b[TE].copy()
    cl, _, _ = process_tables(_b)
    n_ooc = 0
    for v in PROC:
        lim = cl[cl.variable == v].set_index("manufacturing_section")
        n_ooc = n_ooc + (~t[v].between(t.manufacturing_section.map(lim.lcl), t.manufacturing_section.map(lim.ucl)) & t[v].notna()).astype(int)
    q = pd.Series(prob_te, index=t.index).rank(pct=True)
    age = (t.equipment_age_years - t.equipment_age_years.min()) / (t.equipment_age_years.max() - t.equipment_age_years.min())
    eqc = 0.5 * age + 0.5 * t.maintenance_status.map({"Current": 0, "Due Soon": 0.5, "Critical": 1}).fillna(0)
    pc = n_ooc / 5
    y = t.batch_failed.to_numpy()
    scen = {"Base 0.6/0.2/0.2": (0.6, 0.2, 0.2), "Quality only": (1, 0, 0), "Equal weights": (1 / 3, 1 / 3, 1 / 3),
            "Equipment heavy": (0.2, 0.6, 0.2), "Process heavy": (0.2, 0.2, 0.6)}
    rows = []
    for name, (wq, we, wp) in scen.items():
        sc_ = wq * q + we * eqc + wp * pc
        f = (sc_ >= sc_.quantile(0.9)).to_numpy()
        rows.append([name, wq, we, wp, f.mean(), y[f].sum() / y.sum(), y[f].mean() / y.mean()])
    return pd.DataFrame(rows, columns=["scenario", "w_quality", "w_equipment", "w_process", "flagged", "caught", "lift"])


with T["Risk score"]:
    rte = 1 / (1 + np.exp(-raw[TE]))
    yte = b.batch_failed.to_numpy()[TE]
    ws = risk_backtest(b, prob[TE])
    flag = rte >= thr
    critf = (b.maintenance_status.to_numpy()[TE] == "Critical")
    rules = pd.DataFrame([["Classifier alert", flag.mean(), yte[flag].sum() / yte.sum()],
                          ["High risk tier (base weights)", ws.flagged.iloc[0], ws.caught.iloc[0]],
                          ["Critical maintenance only", critf.mean(), yte[critf].sum() / yte.sum()]], columns=["rule", "flagged", "caught"])
    k = st.columns(3)
    kpi(k[0], "Classifier alert", f"{rules.caught.iloc[0]:.0%} caught", f"from {rules.flagged.iloc[0]:.0%} of batches", "indigo")
    kpi(k[1], "High risk tier", f"{rules.caught.iloc[1]:.0%} caught", f"from 10% of batches · lift {ws.lift.iloc[0]:.2f}", "teal")
    kpi(k[2], "Critical only", f"{rules.caught.iloc[2]:.0%} caught", f"from {rules.flagged.iloc[2]:.1%} of batches", "rose")
    c = st.columns(2)
    rr = rules.assign(fx=rules.flagged * 100, cy=rules.caught * 100)
    c[0].altair_chart((alt.Chart(rr).mark_circle(size=200, color=IND).encode(x=alt.X("fx:Q", title="Batches flagged (%)"), y=alt.Y("cy:Q", title="Failures caught (%)"), tooltip=["rule", alt.Tooltip("fx:Q", format=".1f"), alt.Tooltip("cy:Q", format=".1f")])
                       + alt.Chart(rr).mark_text(dx=10, align="left").encode(x="fx:Q", y="cy:Q", text="rule:N")
                       + alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(color=SLA, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q"))
                      .properties(height=300, title="[Bivariate] Alert rules: flagged vs caught (test 2024–26)"), width="stretch")
    wl_ = ws.melt(id_vars="scenario", value_vars=["flagged", "caught"], var_name="measure", value_name="v")
    wl_["v"] *= 100
    c[1].altair_chart(alt.Chart(wl_).mark_bar().encode(x=alt.X("scenario:N", title=None), xOffset="measure:N", y=alt.Y("v:Q", title="%"),
                      color=alt.Color("measure:N", scale=alt.Scale(range=[SKY, ROS]), legend=alt.Legend(orient="bottom", title=None)),
                      tooltip=["scenario", "measure", alt.Tooltip("v:Q", format=".1f")]).properties(height=300, title="[Multivariate] Weight sensitivity: top 10% flagged"), width="stretch")
    wt = ws.melt(id_vars="scenario", value_vars=["w_quality", "w_equipment", "w_process"], var_name="component", value_name="weight")
    st.altair_chart(alt.Chart(wt).mark_bar().encode(x=alt.X("scenario:N", title=None), y=alt.Y("weight:Q", stack="zero"),
                    color=alt.Color("component:N", scale=alt.Scale(range=[IND, TEA, AMB]), legend=alt.Legend(orient="bottom", title=None)))
                    .properties(height=220, title="[Multivariate] Weights in each scenario"), width="stretch")

# ------------------------------------------------------------------ executive summary
with T["Executive summary"]:
    crit_rate = b.loc[b.maintenance_status == "Critical", "batch_failed"].mean()
    cur_rate = b.loc[b.maintenance_status == "Current", "batch_failed"].mean()
    cl, wl, cap = process_tables(b)
    k = st.columns(5)
    kpi(k[0], "Batches passed", f"{1 - b.batch_failed.mean():.1%}", f"{len(b):,} batches", "indigo", bar=(1 - b.batch_failed.mean()) * 100)
    kpi(k[1], "Critical vs Current", f"{crit_rate:.1%} vs {cur_rate:.1%}", f"adjusted OR {odds_ratios(b).set_index('term').odds_ratio['Maintenance: Critical']:.2f}", "rose")
    kpi(k[2], "Classifier ROC-AUC", f"{auc(*roc_curve(b.batch_failed.to_numpy()[TE], raw[TE])[:2]):.3f}", "triage aid only", "amber")
    kpi(k[3], "Assay Cpk ≥ 1.33", f"{int((cap.cpk >= 1.33).sum())} of {len(cap)}", "sections capable", "violet")
    kpi(k[4], "Open credit", f"KES {live.loc[live.Payment_Method == 'Credit Account', 'Balance_Due_KES'].sum() / 1e6:,.0f} m", "unpaid", "teal")
    st.subheader("Questions and answers")
    st.dataframe(pd.DataFrame([
        ["Where do failures occur?", "Wilson intervals, chi-square", "Operator experience 2.4–4.3%; Bolus 5.3%", "Operators, dosage form"],
        ["Does maintenance matter?", "Adjusted logistic (+ machine FE)", "Critical OR about 2.3", "Yes"],
        ["Can failure be predicted?", "Calibrated logistic", "ROC-AUC 0.58 · recall 0.82 · precision 0.04", "Triage only"],
        ["Is the process capable?", "Control limits, Cpk", "Within limits 97.8–99.8% · Cpk 0.65–1.25", "Not capable"],
        ["Are receivables collected?", "Credit ageing", "85% of open balance over a year old", "Data gap"],
        ["Do discounts pay?", "Within-product units model", "+8.6% units per point · margin 28% → 20%", "Cap at 5%"],
        ["Can demand be forecast?", "Mean 12 m vs naive vs Holt-Winters", "Mean of 12 m beats seasonal naive", "Yes (simple)"],
        ["Supplier and cost risk?", "Cost trend, HHI, pass-through", "Cost nearly doubled · HHI low", "Low concentration"],
        ["Which batches to check?", "Composite risk score", "Top 10% tier, lift about 1.7", "Weights need sign-off"],
    ], columns=["Question", "Method", "Key value", "Answer"]), hide_index=True, width="stretch")
    c = st.columns(2)
    c[0].subheader("Recommendations")
    c[0].dataframe(pd.DataFrame([
        ["High", "Engineering", "Service before Critical", "0 Critical runs"],
        ["High", "QA", "Assay process capability (Cpk)", "Cpk ≥ 1.33"],
        ["High", "Commercial", "Cap discounts at 5%", "Margin up"],
        ["Medium", "Training", "Mentor operators with 0–2 years", "Failure rate down"],
        ["Medium", "Finance", "Fix the balance field, then collect", "Valid ageing"],
        ["Low", "QA", "Risk tier for extra checks", "Weights agreed"],
    ], columns=["Priority", "Owner", "Action", "Target"]), hide_index=True, width="stretch")
    c[1].subheader("Notebook check")
    c[1].dataframe(pd.DataFrame([
        ["Batch failure rate (%)", "3.12", f"{b.batch_failed.mean() * 100:.2f}"],
        ["Test ROC-AUC", "0.581", f"{auc(*roc_curve(b.batch_failed.to_numpy()[TE], raw[TE])[:2]):.3f}"],
        ["Critical adjusted OR", "2.33", f"{odds_ratios(b).set_index('term').odds_ratio['Maintenance: Critical']:.2f}"],
        ["Margin at 0% / deepest (%)", "28.0 / 20.0", f"{(live.groupby('Discount_Pct').gp.sum() / live.groupby('Discount_Pct').Net_Amount_Excl_VAT_KES.sum() * 100).iloc[0]:.1f} / "
                                                      f"{(live.groupby('Discount_Pct').gp.sum() / live.groupby('Discount_Pct').Net_Amount_Excl_VAT_KES.sum() * 100).iloc[-1]:.1f}"],
        ["Open credit (KES m)", "1,804", f"{live.loc[live.Payment_Method == 'Credit Account', 'Balance_Due_KES'].sum() / 1e6:,.0f}"],
    ], columns=["Figure", "Notebook", "Dashboard"]), hide_index=True, width="stretch")
    st.caption("Repair times, risk weights and the 5% cap are assumptions · no opening stock or lead-time data.")
