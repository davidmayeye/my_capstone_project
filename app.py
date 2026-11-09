"""POI99999 — Pharmaceutical Operations Intelligence System (individual capstone) — Streamlit dashboard.

Run locally:   streamlit run app.py
Needs the four CSV files in the same folder as this file.
"""
import altair as alt                                                        # interactive charts (ships with Streamlit)
import numpy as np                                                          # numbers
import pandas as pd                                                         # tables
import streamlit as st                                                      # dashboard

st.set_page_config(page_title="POI Capstone Dashboard", page_icon="🧬", layout="wide")  # page settings
alt.data_transformers.disable_max_rows()                                    # allow charts with many points
TEAL, NAVY, CORAL, GOLD, GREY = "#0e7c7b", "#17324d", "#e4572e", "#f3a712", "#8d99ae"  # chart colours
TONES = {"teal": ("#0b5d5c", "#17a398"), "navy": ("#17324d", "#2f6690"), "coral": ("#a4243b", "#e4572e"),  # card gradients
         "gold": ("#b9770e", "#f3a712"), "violet": ("#4a2c6d", "#8e5ac8"), "green": ("#1b5e20", "#43a047")}

st.markdown("""
<style>
.banner {background: linear-gradient(120deg, #0b2a3d 0%, #0e7c7b 60%, #17a398 100%); color: #fff;
         padding: 22px 28px; border-radius: 16px; margin-bottom: 18px; box-shadow: 0 6px 18px rgba(14,124,123,.25);}
.banner h1 {color: #fff; font-size: 1.85rem; margin: 0; padding: 0;}
.banner p {color: #d9f2ef; margin: 6px 0 0 0; font-size: .95rem;}
.kpi {border-radius: 16px; padding: 16px 18px 14px 18px; color: #fff; min-height: 132px; margin-bottom: 12px;
      box-shadow: 0 6px 16px rgba(0,0,0,.14); transition: transform .15s ease, box-shadow .15s ease;}
.kpi:hover {transform: translateY(-3px); box-shadow: 0 10px 22px rgba(0,0,0,.20);}
.kpi .top {display: flex; justify-content: space-between; align-items: center;}
.kpi .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .92; font-weight: 600;}
.kpi .icon {font-size: 1.5rem; background: rgba(255,255,255,.18); border-radius: 50%; width: 42px; height: 42px;
            display: flex; align-items: center; justify-content: center;}
.kpi .value {font-size: clamp(1.25rem, 2vw, 1.95rem); font-weight: 800; line-height: 1.15; margin-top: 6px;}
.kpi .sub {font-size: .8rem; opacity: .92; margin-top: 2px;}
.kpi .bar {height: 6px; background: rgba(255,255,255,.25); border-radius: 4px; margin-top: 10px; overflow: hidden;}
.kpi .bar > div {height: 100%; background: #fff; border-radius: 4px;}
.chosen {background: #e8f6f5; border-left: 5px solid #0e7c7b; padding: 10px 14px; border-radius: 8px; margin: 6px 0 12px 0;}
</style>""", unsafe_allow_html=True)                                         # page styles: banner, cards, notes


def kpi(col, icon, label, value, sub="", tone="teal", bar=None):            # one coloured KPI card
    a, z = TONES[tone]                                                      # gradient colours
    meter = f'<div class="bar"><div style="width:{min(max(bar, 0), 100):.0f}%"></div></div>' if bar is not None else ""
    col.markdown(f'''<div class="kpi" style="background: linear-gradient(135deg, {a}, {z});">
      <div class="top"><div class="label">{label}</div><div class="icon">{icon}</div></div>
      <div class="value">{value}</div><div class="sub">{sub}</div>{meter}</div>''', unsafe_allow_html=True)


def note(text):                                                             # highlighted "chosen model" note
    st.markdown(f'<div class="chosen">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ data (cached: runs once per session)
@st.cache_data(show_spinner="Loading and cleaning the four data files...")
def load_data():
    prod = pd.read_csv("LAL_Batch_Manufacturing_Production.csv")            # batch production records
    maint = pd.read_csv("LAL_Equipment_Maintenance_Records.csv")            # equipment maintenance records
    sales = pd.read_csv("sales_records.csv", dtype={"Supplier_ID": str})    # sales transactions
    inv = pd.read_csv("inventory_records.csv", dtype={"Supplier_ID": str})  # stock movements
    missing = prod.isna().sum()                                             # missing values before cleaning
    data = prod.merge(maint[["batch_id", "maintenance_status"]], on="batch_id", how="left")  # same join as the notebook (SQL left join)
    data["production_start"] = pd.to_datetime(data["production_start"], errors="coerce")  # text -> date
    data["year"], data["month"] = data["production_start"].dt.year, data["production_start"].dt.month  # calendar fields
    for c in data.select_dtypes("number").columns.drop(["batch_passed"]):   # fill numeric gaps with the column mean
        data[c] = data[c].fillna(data[c].mean())
    data["batch_outcome"] = np.where(data["batch_passed"] == 1, "Passed", "Failed")  # readable outcome
    data["failed"] = (data["batch_passed"] == 0).astype(int)                # 1 = failed QC
    data["equipment_age_years"] = data["equipment_age_days"] / 365.25       # age in years
    sales.columns, inv.columns = sales.columns.str.lower(), inv.columns.str.lower()  # lower-case names as in the notebook
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])                 # text -> date
    inv["movement_date"] = pd.to_datetime(inv["movement_date"])             # text -> date
    sales["is_cancelled"] = sales["order_status"].eq("Cancelled")           # void sales
    inv["is_cancelled_sale"] = inv["reference_id"].isin(sales.loc[sales["is_cancelled"], "sale_id"])  # their stock movements
    inv["category"] = inv["product_id"].map(sales.groupby("product_id")["product_category"].first())  # product category
    moves = inv[~inv["is_cancelled_sale"]].sort_values(["movement_date", "movement_id"]).copy()  # valid movements in time order
    moves["stock_on_hand"] = moves.groupby(["branch_code", "product_id"])["quantity_change"].cumsum()  # stock rebuilt from 0
    return data, sales, moves, missing


@st.cache_data
def reorder_policy(moves, lead_days=14, z=1.65, over_days=180):              # notebook Section 17
    week = moves["movement_date"].dt.to_period("W-SUN").dt.start_time      # week of each movement
    last52 = pd.date_range(end=week.max(), periods=52, freq="7D")           # last 52 weeks
    sold = (-moves["quantity_change"]).clip(lower=0)                        # units out
    weekly = sold.groupby([moves["branch_code"], moves["product_id"], week]).sum()  # units out per week
    pairs = moves[["branch_code", "product_id"]].drop_duplicates()          # every branch-product item
    grid = pd.MultiIndex.from_tuples([(b, p, w) for b, p in pairs.itertuples(index=False) for w in last52])
    recent = weekly.reindex(grid, fill_value=0).groupby(level=[0, 1])       # weekly sales incl. zero weeks
    keys = ["branch_code", "product_id"]
    pol = pd.DataFrame({"avg_weekly": recent.mean(), "sd_weekly": recent.std(),
                        "stock_now": moves.groupby(keys)["quantity_change"].sum(),  # stock rebuilt from zero
                        "category": moves.groupby(keys)["category"].first()})
    pol.index.names = keys
    pol["avg_daily"] = pol["avg_weekly"] / 7                                # average daily demand
    pol["safety_stock"] = z * pol["sd_weekly"] / np.sqrt(7) * np.sqrt(lead_days)  # z x sd(daily) x sqrt(lead time)
    pol["reorder_point"] = pol["avg_daily"] * lead_days + pol["safety_stock"]  # lead-time demand + buffer
    pol["days_of_cover"] = pol["stock_now"] / pol["avg_daily"].replace(0, np.nan)  # how long stock lasts
    cost = moves.groupby("product_id")["unit_cost_kes"].last()              # latest unit cost
    pol["excess_value_kes"] = ((pol["stock_now"] - over_days * pol["avg_daily"]).clip(lower=0)
                               * pol.index.get_level_values("product_id").map(cost))  # value beyond the overstock limit
    pol["status"] = np.select([pol["avg_daily"] == 0, pol["stock_now"] <= pol["reorder_point"], pol["days_of_cover"] > over_days],
                              ["No recent demand", "Reorder now", "Overstock"], default="OK")
    return pol.round(1).reset_index()


data, sales, moves, missing = load_data()                                   # load once
valid = sales[~sales["is_cancelled"]]                                       # delivered orders only

# ------------------------------------------------------------------ results recorded in POI99999.ipynb (test sets)
RESULTS = {
    "1. Will a batch fail QC?": dict(metric="PR-AUC", best="Logistic Regression", table=pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "ANN", "Baseline (always Passed)"],
        "Accuracy": [0.603, 0.785, 0.805, 0.969], "Precision": [0.040, 0.053, 0.035, 0.0], "Recall": [0.515, 0.347, 0.197, 0.0],
        "F1": [0.075, 0.092, 0.059, 0.0], "ROC-AUC": [0.570, 0.600, 0.524, 0.500], "PR-AUC": [0.047, 0.045, 0.037, 0.031]}),
        why="Highest PR-AUC (0.047, 1.5× random). Catches 193 of 375 failed test batches (recall 51.5%), but only 4% of alerts are real: a weak screen."),
    "2. What yield will a batch achieve?": dict(metric="MAE", best="Random Forest", table=pd.DataFrame({
        "Model": ["Random Forest", "ANN", "Baseline (mean yield)"], "MAE": [0.858, 0.859, 1.102], "RMSE": [1.095, 1.096, 1.338],
        "R2": [0.330, 0.328, 0.0]}), why="Lowest error: 0.858 yield points, 22% better than the mean, explaining 33% of yield variation."),
    "3. What maintenance state is the equipment in?": dict(metric="Macro F1", best="Random Forest", table=pd.DataFrame({
        "Model": ["Random Forest", "ANN", "Logistic Regression", "Baseline (always Current)"], "Accuracy": [0.818, 0.632, 0.549, 0.903],
        "Balanced accuracy": [0.734, 0.503, 0.620, 0.333], "Macro F1": [0.557, 0.362, 0.354, 0.316],
        "Recall Critical": [0.690, 0.207, 0.563, 0.0]}), why="Best on every balanced measure; finds 60 of 87 Critical records (69%)."),
    "4. Are monthly quality indicators deteriorating?": dict(metric="WAPE (%)", best="Linear Regression", table=pd.DataFrame({
        "Model": ["Linear Regression", "Moving average (12 months)", "GRU", "Baseline (seasonal-naive)"],
        "WAPE (%)": [7.00, 7.24, 7.32, 8.82], "WAPE failure_rate": [26.26, 26.80, 24.60, 36.11],
        "WAPE batch_volume": [7.47, 7.63, 10.85, 7.17], "Gain vs seasonal-naive (%)": [20.62, 17.95, 17.06, 0.0]}),
        why="Lowest average WAPE (7.0%), 20.6% better than seasonal-naive, meeting the 15% target."),
    "5. Which batches ran under abnormal conditions?": dict(metric="PR-AUC", best="Z-score rule", table=pd.DataFrame({
        "Model": ["Z-score rule", "PCA reconstruction error", "KNN distance", "Baseline (random score)"],
        "ROC-AUC": [0.522, 0.502, 0.503, 0.506], "PR-AUC": [0.036, 0.036, 0.035, 0.033],
        "Failures in top 5%": [0.072, 0.072, 0.085, 0.056], "Lift": [1.44, 1.44, 1.71, 1.12]}),
        why="Ties for the best PR-AUC and is the simplest to explain; abnormal conditions are only weakly linked to failure."),
    "6. What are the main failure causes? (NLP)": dict(metric="Silhouette", best="TF-IDF + K-Means (k=6)", table=pd.DataFrame({
        "Model": [f"TF-IDF + K-Means (k={k})" for k in range(2, 7)], "Silhouette": [0.443, 0.543, 0.631, 0.734, 0.788]}),
        why="Six clearly separated cause groups (silhouette 0.788). Assay 37%, dissolution 18% and weight variation 15% make 70% of failures."),
    "7. What will weekly demand be?": dict(metric="WAPE (%)", best="Croston-SBA", table=pd.DataFrame({
        "Model": ["Linear Regression", "Croston-SBA", "Moving average (12 weeks)", "Baseline (seasonal-naive)"],
        "WAPE (%)": [104.44, 115.01, 120.19, 143.22], "Bias (%)": [-28.44, -5.90, -1.00, -6.87],
        "Gain vs seasonal-naive (%)": [27.07, 19.70, 16.08, 0.0], "Company-level WAPE (%)": [29.08, 12.42, 11.29, 16.54]}),
        why="Lowest WAPE among models with bias within ±10%: 19.7% better than seasonal-naive. Linear Regression under-forecasts by 28%."),
    "8. Which items will run out within 14 days?": dict(metric="PR-AUC", best="Gradient Boosting", table=pd.DataFrame({
        "Model": ["Gradient Boosting", "Logistic Regression", "Random Forest", "Baseline (always in stock)"],
        "Precision": [0.095, 0.048, 0.122, 0.0], "Recall": [0.865, 0.949, 0.821, 0.0], "ROC-AUC": [0.994, 0.994, 0.993, 0.5],
        "PR-AUC": [0.278, 0.265, 0.243, 0.001]}),
        why="Highest PR-AUC (0.278, about 206× random). At threshold 0.690: recall 0.80, 1,186 alerts for 156 stock-outs."),
    "9. Which product groups need different stock policies?": dict(metric="Silhouette", best="K-Means (k=2)", table=pd.DataFrame({
        "Model": ["K-Means (k=2)"], "Silhouette": [0.471]}),
        why="Two groups: 250 fast, cheap items (76% of value) and 50 slow, expensive items, mostly Equipment."),
}
LOWER_IS_BETTER = {"MAE", "WAPE (%)"}                                       # metrics where smaller wins
GAINS = {'1. Will a batch fail QC?': 50.0, '2. What yield will a batch achieve?': 22.1, '3. What maintenance state is the equipment in?': 76.2, '4. Are monthly quality indicators deteriorating?': 20.6, '5. Which batches ran under abnormal conditions?': 9.9, '7. What will weekly demand be?': 19.7, '8. Which items will run out within 14 days?': 20479.7}  # % better than baseline, from unrounded notebook scores

# ------------------------------------------------------------------ header and navigation
st.markdown('''<div class="banner"><h1>🧬 Pharmaceutical Operations Intelligence System</h1>
<p>Individual capstone (POI99999) · production · maintenance · sales · inventory — LAL data, 2007–2026</p></div>''',
            unsafe_allow_html=True)                                         # page banner
page = st.sidebar.radio("Go to", ["Overview", "Data cleaning", "EDA: production", "EDA: sales & inventory",
                                  "Models", "Failure causes", "Inventory policy", "Equipment screening", "Executive summary"])
st.sidebar.caption("Model results are the test-set results recorded in POI99999.ipynb. All other figures are computed live from the CSV files.")

# ------------------------------------------------------------------ overview
if page == "Overview":
    fr = data["failed"].mean()                                              # failure rate
    pol = reorder_policy(moves)                                             # stock status
    c = st.columns(5)
    kpi(c[0], "🧪", "Batches", f"{len(data):,}", f"{1 - fr:.1%} passed QC", "teal", bar=(1 - fr) * 100)
    kpi(c[1], "⚠️", "Failed batches", f"{data['failed'].sum():,}", f"{fr:.2%} failure rate", "coral")
    kpi(c[2], "💰", "Net sales", f"KES {valid['net_amount_excl_vat_kes'].sum() / 1e9:.1f} bn", f"{len(valid):,} delivered orders", "green")
    kpi(c[3], "🧾", "Unpaid balances", f"KES {sales['balance_due_kes'].sum() / 1e6:,.0f} m", "all on credit accounts", "violet")
    kpi(c[4], "📦", "Overstocked items", f"{(pol['status'] == 'Overstock').sum():,}", f"of {len(pol):,} branch-product items", "gold",
        bar=(pol["status"] == "Overstock").mean() * 100)
    left, right = st.columns([3, 2])
    left.subheader("Bottom line")
    left.markdown("""
- **Quality is high:** 96.9% of 60,000 batches pass; assay, dissolution and weight variation make up **70%** of failures.
- **Overdue maintenance doubles the risk:** Critical equipment fails at **6.9%** vs 3.1% for Current.
- **Failure prediction is weak** (best PR-AUC 0.047): keep laboratory release testing.
- **Yield and maintenance state can be predicted** (Random Forest: yield 22% better than average; macro F1 0.557).
- **Stock is bought without regard to sales:** 4.1m units received vs 2.1m sold; 1,033 of 1,200 items overstocked.
- **Stock-outs can be flagged early:** Gradient Boosting catches 80% of stock-outs two weeks ahead.""")
    yr = data.groupby("year").agg(batches=("batch_id", "size"), failure_rate=("failed", "mean")).reset_index()
    yr["failure_rate"] *= 100
    right.subheader("Production and failure rate by year")
    base = alt.Chart(yr).encode(x=alt.X("year:O", title=None))
    right.altair_chart(alt.layer(base.mark_bar(color=TEAL, opacity=0.6).encode(y=alt.Y("batches:Q", title="Batches")),
                                 base.mark_line(color=CORAL, point=True).encode(y=alt.Y("failure_rate:Q", title="Failure rate (%)")))
                       .resolve_scale(y="independent").properties(height=300), width="stretch")

# ------------------------------------------------------------------ data cleaning
elif page == "Data cleaning":
    c = st.columns(4)
    kpi(c[0], "🏭", "Production + maintenance", "60,000", "batches joined, 0 duplicates", "teal")
    kpi(c[1], "🛒", "Sales", f"{len(sales):,}", f"{sales['is_cancelled'].sum():,} cancelled (4.0%)", "navy")
    kpi(c[2], "🔄", "Stock movements", f"{len(moves) + sales['is_cancelled'].sum():,}", "12,000 receipts + 150,000 sale-outs", "violet")
    kpi(c[3], "✅", "Missing cells after cleaning", "0", "means for numbers, mode for text", "green")
    st.subheader("Missing values before cleaning (production file)")
    m = missing[missing > 0].drop(labels=["rejection_reason"], errors="ignore").sort_values().reset_index()
    m.columns = ["column", "missing"]
    st.altair_chart(alt.Chart(m).mark_bar(color=TEAL).encode(y=alt.Y("column:N", sort="-x", title=None), x="missing:Q",
                                                             tooltip=["column", "missing"]).properties(height=280), width="stretch")
    st.caption("rejection_reason is empty for the 58,125 passed batches (expected) and is left out of the chart.")
    st.subheader("Cleaning log")
    st.dataframe(pd.DataFrame([
        ["Cancelled orders separated from demand and revenue", f"{sales['is_cancelled'].sum():,}", "Void sales: nothing paid, nothing owed"],
        ["Stock movements of cancelled orders excluded", f"{sales['is_cancelled'].sum():,}", "Goods did not leave stock"],
        ["Supplier filled on sale-out movements", "150,000", "Supplier is recorded on the linked sale"],
        ["Margin outliers flagged (kept)", "15,970", "IQR rule; values are real"],
        ["Bulk orders flagged (kept)", "15,629", "Large institutional orders are genuine demand"],
        ["Rows removed", "0", "No duplicates, invalid dates or impossible values"]], columns=["Rule", "Rows affected", "Reason"]),
        hide_index=True, width="stretch")
    st.subheader("Stock rebuilt from zero: reconciliation by branch")
    rec = pd.DataFrame({"received": moves[moves["quantity_change"] > 0].groupby("branch_code")["quantity_change"].sum(),
                        "sold": -moves[moves["quantity_change"] < 0].groupby("branch_code")["quantity_change"].sum()})
    rec["closing stock"] = rec["received"] - rec["sold"]
    st.dataframe(rec, width="stretch")
    neg = (moves["stock_on_hand"] < 0).groupby(moves["movement_date"].dt.year).mean().mul(100).rename("% of movements")
    st.subheader("Movements leaving stock below zero, by year")
    st.line_chart(neg, color=CORAL)
    st.caption(f"Overall {(moves['stock_on_hand'] < 0).mean():.1%}: goods recorded as sold before enough stock was received.")

# ------------------------------------------------------------------ EDA production
elif page == "EDA: production":
    yrs = st.sidebar.slider("Production years", 2007, 2026, (2007, 2026))  # year filter
    d = data[data["year"].between(*yrs)]                                    # filtered batches
    c = st.columns(4)
    kpi(c[0], "🧪", "Batches", f"{len(d):,}", f"{yrs[0]}–{yrs[1]}", "teal")
    kpi(c[1], "📉", "Failure rate", f"{d['failed'].mean():.2%}", f"{d['failed'].sum():,} failed", "coral")
    kpi(c[2], "🎯", "Average yield", f"{d['actual_yield'].mean():.2f}%", "actual yield", "navy")
    kpi(c[3], "🔬", "Average purity", f"{d['assay_result_purity'].mean():.2f}%", "assay result", "violet")
    factor = st.selectbox("Failure rate by", ["maintenance_status", "manufacturing_section", "dosage_form", "equipment_id", "drug_name"])
    t = d.groupby(factor)["failed"].agg(rate="mean", batches="size").reset_index()
    t["rate"] *= 100
    st.altair_chart((alt.Chart(t).mark_bar(color=TEAL).encode(x=alt.X(f"{factor}:N", sort="-y", title=None), y=alt.Y("rate:Q", title="Failure rate (%)"),
                                                              tooltip=[factor, alt.Tooltip("rate:Q", format=".2f"), "batches"])
                     + alt.Chart(pd.DataFrame({"y": [d["failed"].mean() * 100]})).mark_rule(color=CORAL, strokeDash=[5, 3]).encode(y="y:Q"))
                    .properties(height=300), width="stretch")
    c = st.columns(2)
    c[0].altair_chart(alt.Chart(d[["actual_yield", "batch_outcome"]]).mark_bar(opacity=0.7).encode(
        x=alt.X("actual_yield:Q", bin=alt.Bin(maxbins=50), title="Actual yield (%)"), y=alt.Y("count():Q", title="Batches", stack=None),
        color=alt.Color("batch_outcome:N", scale=alt.Scale(domain=["Passed", "Failed"], range=[TEAL, CORAL]))).properties(
        height=280, title="Yield distribution by outcome"), width="stretch")
    s = d.sample(min(4000, len(d)), random_state=42)                        # sample keeps the scatter fast
    c[1].altair_chart(alt.Chart(s).mark_circle(size=14, opacity=0.5).encode(
        x=alt.X("production_hours:Q", title="Production hours"), y=alt.Y("actual_yield:Q", scale=alt.Scale(zero=False), title="Actual yield (%)"),
        color=alt.Color("batch_outcome:N", scale=alt.Scale(domain=["Passed", "Failed"], range=[TEAL, CORAL])),
        tooltip=["batch_id", "production_hours", "actual_yield", "batch_outcome"]).properties(height=280, title="Production hours vs yield (sample)"),
        width="stretch")
    st.subheader("Correlation of numeric production variables")
    cols = ["actual_yield", "assay_result_purity", "impurity_level", "production_hours", "batch_size", "temp_avg", "pressure_avg",
            "humidity_avg", "raw_material_quality", "equipment_age_years", "operator_experience_years", "failed"]
    corr = d[cols].corr().round(2).reset_index().melt("index")
    st.altair_chart(alt.Chart(corr).mark_rect().encode(x=alt.X("index:N", title=None), y=alt.Y("variable:N", title=None),
                                                       color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                                                       tooltip=["index", "variable", "value"]).properties(height=420), width="stretch")
    st.subheader("Quality summary by outcome")
    st.dataframe(d.groupby("batch_outcome").agg(batches=("batch_id", "count"), average_yield=("actual_yield", "mean"),
                                                average_purity=("assay_result_purity", "mean"), average_impurity=("impurity_level", "mean")).round(3),
                 width="stretch")

# ------------------------------------------------------------------ EDA sales and inventory
elif page == "EDA: sales & inventory":
    c = st.columns(4)
    kpi(c[0], "💰", "Net sales", f"KES {valid['net_amount_excl_vat_kes'].sum() / 1e9:.1f} bn", "2014–2025", "green")
    kpi(c[1], "🧾", "Orders delivered", f"{len(valid):,}", f"{sales['is_cancelled'].mean():.1%} cancelled", "navy")
    rec_u, sold_u = moves.loc[moves["quantity_change"] > 0, "quantity_change"].sum(), valid["quantity"].sum()
    kpi(c[2], "📥", "Units received", f"{rec_u / 1e6:.2f} m", f"{rec_u / sold_u:.1f}× units sold", "gold")
    kpi(c[3], "🏬", "Branch BR01", f"{(valid['branch_code'] == 'BR01').mean():.0%}", "of delivered orders", "violet",
        bar=(valid["branch_code"] == "BR01").mean() * 100)
    st.subheader("Net sales per month (KES million)")
    st.area_chart(valid.groupby(valid["sale_date"].dt.to_period("M").dt.to_timestamp())["net_amount_excl_vat_kes"].sum().div(1e6)
                  .rename("Net sales (KES m)"), color=TEAL)
    c = st.columns(2)
    bc = valid.pivot_table(index="product_category", columns="branch_code", values="net_amount_excl_vat_kes", aggfunc="sum").div(1e6)
    c[0].altair_chart(alt.Chart(bc.reset_index().melt("product_category")).mark_bar().encode(
        y=alt.Y("product_category:N", title=None), x=alt.X("value:Q", title="Net sales (KES m)"), color=alt.Color("branch_code:N", title="Branch"),
        tooltip=["product_category", "branch_code", alt.Tooltip("value:Q", format=",.0f")]).properties(height=300, title="Net sales by category and branch"),
        width="stretch")
    canc = sales.groupby("payment_method")["is_cancelled"].mean().mul(100).reset_index(name="cancel_%")
    c[1].altair_chart(alt.Chart(canc).mark_bar(color=CORAL).encode(y=alt.Y("payment_method:N", sort="-x", title=None),
                                                                   x=alt.X("cancel_%:Q", title="Cancellation rate (%)"))
                      .properties(height=300, title="Cancellation rate by payment method"), width="stretch")
    c = st.columns(2)
    c[0].subheader("Payment status by method (% of delivered orders)")
    c[0].dataframe((pd.crosstab(valid["payment_method"], valid["payment_status"], normalize="index") * 100).round(1), width="stretch")
    c[1].subheader("Units received vs sold")
    flow = pd.DataFrame({"received": moves[moves["quantity_change"] > 0].groupby("category")["quantity_change"].sum(),
                         "sold": valid.groupby("product_category")["quantity"].sum()})
    flow["received per unit sold"] = (flow["received"] / flow["sold"]).round(2)
    c[1].dataframe(flow, width="stretch")
    st.subheader("Seasonality: units sold by year and month")
    h = valid.groupby([valid["sale_date"].dt.year.rename("year"), valid["sale_date"].dt.strftime("%b").rename("month")])["quantity"].sum().reset_index()
    st.altair_chart(alt.Chart(h).mark_rect().encode(
        x=alt.X("month:O", sort=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], title=None),
        y=alt.Y("year:O", title=None), color=alt.Color("quantity:Q", scale=alt.Scale(scheme="teals"), title="Units"),
        tooltip=["year", "month", alt.Tooltip("quantity:Q", format=",")]).properties(height=340), width="stretch")

# ------------------------------------------------------------------ models
elif page == "Models":
    prob = st.selectbox("Problem", list(RESULTS))                           # pick a business question
    r = RESULTS[prob]
    t, metric = r["table"], r["metric"]
    best_score = t.set_index("Model").loc[r["best"], metric]
    base = t[t["Model"].str.startswith("Baseline")]                         # baseline row, if any
    c = st.columns(3)
    kpi(c[0], "🏆", "Chosen model", r["best"], f"decided on {metric}", "teal")
    kpi(c[1], "📏", metric, f"{best_score:,.3f}", "test-set score", "navy")
    if len(base):
        b = base[metric].iloc[0]                                            # baseline score
        kpi(c[2], "📈", "Gain over baseline", f"{GAINS[prob]:,.1f}%", f"baseline {b:,.3f}", "green")
    else:
        kpi(c[2], "🧩", "Models compared", f"{len(t)}", "no baseline for clustering", "violet")
    note(f"<b>Why {r['best']}:</b> {r['why']}")
    st.altair_chart(alt.Chart(t).mark_bar().encode(
        y=alt.Y("Model:N", sort="-x" if metric not in LOWER_IS_BETTER else "x", title=None), x=alt.X(f"{metric}:Q"),
        color=alt.condition(alt.datum.Model == r["best"], alt.value(TEAL), alt.value(GREY)),
        tooltip=list(t.columns)).properties(height=60 + 45 * len(t), title=f"Model comparison on {metric} (chosen in teal)"), width="stretch")
    st.dataframe(t, hide_index=True, width="stretch")

# ------------------------------------------------------------------ failure causes
elif page == "Failure causes":
    f = data[data["failed"] == 1]                                           # failed batches
    themes = pd.DataFrame({"Cause (TF-IDF + K-Means theme)": ["Assay out of specification", "Dissolution failure", "Weight variation",
                                                               "Microbial limit", "Related substances", "Blend uniformity"],
                           "Failed batches": [699, 341, 277, 217, 178, 163]})
    themes["Share (%)"] = (themes["Failed batches"] / themes["Failed batches"].sum() * 100).round(1)
    themes["Cumulative (%)"] = themes["Share (%)"].cumsum().round(1)
    c = st.columns(3)
    kpi(c[0], "🔬", "Largest cause", "Assay", "37% of failures", "coral", bar=37.3)
    kpi(c[1], "📊", "Top 3 causes", "70%", "assay, dissolution, weight", "gold", bar=70.2)
    kpi(c[2], "🏭", "Oral Solids", f"{(f['manufacturing_section'] == 'Oral Solids').mean():.0%}", "of failed batches", "navy",
        bar=(f["manufacturing_section"] == "Oral Solids").mean() * 100)
    base = alt.Chart(themes).encode(x=alt.X("Cause (TF-IDF + K-Means theme):N", sort=None, title=None))
    st.altair_chart(alt.layer(base.mark_bar(color=CORAL).encode(y=alt.Y("Failed batches:Q")),
                              base.mark_line(color=NAVY, point=True).encode(y=alt.Y("Cumulative (%):Q", scale=alt.Scale(domain=[0, 100]))))
                    .resolve_scale(y="independent").properties(height=320, title="Pareto of failure causes (NLP themes)"), width="stretch")
    st.dataframe(themes, hide_index=True, width="stretch")
    st.subheader("Raw rejection reasons by manufacturing section")
    st.dataframe(pd.crosstab(f["rejection_reason"], f["manufacturing_section"], margins=True, margins_name="Total")
                 .sort_values("Total", ascending=False), width="stretch")

# ------------------------------------------------------------------ inventory policy
elif page == "Inventory policy":
    c = st.columns(3)
    lead = c[0].slider("Lead time (days)", 7, 60, 14)                       # assumption: no lead-time field in the data
    z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[c[1].select_slider("Service level", ["90%", "95%", "99%"], value="95%")]
    over = c[2].slider("Overstock limit (days of cover)", 60, 365, 180)
    pol = reorder_policy(moves, lead, z, over)                              # recomputed with the chosen assumptions
    k = st.columns(4)
    for col, (s, icon, tone) in zip(k, [("Reorder now", "🛒", "coral"), ("Overstock", "📦", "gold"),
                                        ("No recent demand", "💤", "violet"), ("OK", "✅", "green")]):
        n = (pol["status"] == s).sum()
        kpi(col, icon, s, f"{n:,}", f"{n / len(pol):.0%} of {len(pol):,} items", tone, bar=n / len(pol) * 100)
    st.metric("Stock value beyond the overstock limit", f"KES {pol['excess_value_kes'].sum() / 1e9:,.1f} bn")
    c = st.columns(2)
    sc = pol.groupby(["category", "status"]).size().reset_index(name="items")
    c[0].altair_chart(alt.Chart(sc).mark_bar().encode(
        y=alt.Y("category:N", title=None), x=alt.X("items:Q", stack="normalize", title="Share of items"),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Reorder now", "Overstock", "No recent demand", "OK"], range=[CORAL, GOLD, GREY, TEAL])),
        tooltip=["category", "status", "items"]).properties(height=280, title="Stock status by category"), width="stretch")
    ev = pol.groupby("category")["excess_value_kes"].sum().div(1e6).reset_index(name="KES_m")
    c[1].altair_chart(alt.Chart(ev).mark_bar(color=CORAL).encode(y=alt.Y("category:N", sort="-x", title=None), x=alt.X("KES_m:Q", title="KES million"),
                                                                 tooltip=["category", alt.Tooltip("KES_m:Q", format=",.0f")])
                      .properties(height=280, title="Value held beyond the overstock limit"), width="stretch")
    show = st.radio("Show", ["Reorder now", "Overstock", "All"], horizontal=True)
    st.dataframe(pol if show == "All" else pol[pol["status"] == show], hide_index=True, width="stretch")
    st.download_button("Download list (CSV)", pol.to_csv(index=False), "reorder_policy.csv")

# ------------------------------------------------------------------ equipment screening
elif page == "Equipment screening":
    rate = data.groupby("maintenance_status")["failed"].mean().mul(100)
    c = st.columns(3)
    kpi(c[0], "🔴", "Critical", f"{rate.get('Critical', 0):.1f}%", f"failure rate · {(data['maintenance_status'] == 'Critical').sum()} batches", "coral")
    kpi(c[1], "🟡", "Due Soon", f"{rate.get('Due Soon', 0):.1f}%", "failure rate", "gold")
    kpi(c[2], "🟢", "Current", f"{rate.get('Current', 0):.1f}%", "failure rate", "green")
    eq = data.groupby("equipment_id").agg(batches=("batch_id", "size"), failure_rate=("failed", "mean"),
                                          age_years=("equipment_age_years", "max"), critical_batches=("maintenance_status", lambda s: (s == "Critical").sum()),
                                          max_days_since_service=("maintenance_days_ago", "max")).reset_index()
    eq["failure_rate"] = (eq["failure_rate"] * 100).round(2)
    st.altair_chart(alt.Chart(eq).mark_circle(opacity=0.8).encode(
        x=alt.X("age_years:Q", title="Equipment age (years)"), y=alt.Y("failure_rate:Q", title="Failure rate (%)"),
        size=alt.Size("batches:Q", title="Batches"), color=alt.Color("critical_batches:Q", scale=alt.Scale(scheme="orangered"), title="Critical batches"),
        tooltip=list(eq.columns)).properties(height=340, title="Machines: age vs failure rate (size = batches)"), width="stretch")
    st.subheader("Machines ranked by Critical batches and days since service")
    st.dataframe(eq.sort_values(["critical_batches", "max_days_since_service"], ascending=False).round(1), hide_index=True, width="stretch")
    st.caption("A screening list, not a failure probability: service the top machines before they reach Critical status.")

# ------------------------------------------------------------------ executive summary
else:
    rate = data.groupby("maintenance_status")["failed"].mean().mul(100)
    pol = reorder_policy(moves)
    c = st.columns(4)
    kpi(c[0], "🧪", "Batches / failed", f"{len(data):,}", f"{data['failed'].sum():,} failed ({data['failed'].mean():.1%})", "teal")
    kpi(c[1], "🔧", "Critical vs Current", f"{rate['Critical']:.1f}% vs {rate['Current']:.1f}%", "failure rate", "coral")
    kpi(c[2], "💰", "Net sales 2014–2025", f"KES {valid['net_amount_excl_vat_kes'].sum() / 1e9:.1f} bn", f"KES {sales['balance_due_kes'].sum() / 1e6:,.0f} m unpaid", "green")
    kpi(c[3], "📦", "Reorder / overstock", f"{(pol['status'] == 'Reorder now').sum()} / {(pol['status'] == 'Overstock').sum()}", "items", "gold")
    st.subheader("Model scorecard")
    rows = []
    for name, r in RESULTS.items():                                         # one row per problem
        t = r["table"].set_index("Model")[r["metric"]]
        b = [m for m in t.index if m.startswith("Baseline")]
        rows.append([name, r["best"], r["metric"], t[r["best"]], t[b[0]] if b else None, GAINS.get(name)])
    st.dataframe(pd.DataFrame(rows, columns=["Problem", "Chosen model", "Metric", "Score", "Baseline", "Gain over baseline (%)"]).round(3),
                 hide_index=True, width="stretch")
    st.subheader("Recommendations")
    st.dataframe(pd.DataFrame([
        ["High", "QA / CAPA", "CAPA on assay, dissolution and weight-variation failures, starting with Oral Solids", "70% of failures; Oral Solids = 74%"],
        ["High", "Engineering", "Service equipment before it becomes Critical; fixed schedule for the most-overdue machines", "6.9% vs 3.1% failure rate"],
        ["High", "Procurement", "Buy to the reorder-point policy; stop receipts for overstocked items, especially Equipment", "Receipts about 2× sales; KES 64 bn stock"],
        ["Medium", "Branch stores", "Weekly reorder list and stock-out watch-list", "Catches 80% of stock-outs"],
        ["Medium", "Planning", "Plan volumes with the Croston-SBA weekly forecast", "20% better than last year's pattern"],
        ["Medium", "Finance", "Review credit-account terms and collections", "KES 1.8 bn unpaid, all on credit accounts"],
        ["Low", "QA", "Use the failure ranking only for extra in-process checks", "Failure model is weak"]],
        columns=["Priority", "Owner", "Action", "Evidence"]), hide_index=True, width="stretch")
    st.caption("Limitations: no opening stock, expiry dates or supplier lead times; batches cannot be linked to sales; no savings are claimed.")
