"""POI99999 — Pharmaceutical Operations Intelligence System (individual capstone) — Streamlit dashboard.

Run locally:   streamlit run app.py
Needs the four CSV files in the same folder as this file.
"""
import altair as alt                                                        # interactive charts (ships with Streamlit)
import numpy as np                                                          # numbers
import pandas as pd                                                         # tables
import streamlit as st                                                      # dashboard

st.set_page_config(page_title="POI Capstone Dashboard", layout="wide")  # page settings
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
      <div class="top"><div class="label">{label}</div>{f'<div class="icon">{icon}</div>' if icon.strip() else ''}</div>
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


# ------------------------------------------------------------------ commercial, finance and forecasting helpers
@st.cache_data(show_spinner=False)
def hw_forecast(y, h=12):                                                   # Holt-Winters: damped trend + monthly season
    y = y.astype(float)                                                     # numbers as floats
    try:
        import warnings                                                     # hide fitting notes
        from statsmodels.tsa.holtwinters import ExponentialSmoothing        # the forecasting model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = ExponentialSmoothing(y.to_numpy(), trend="add", damped_trend=True, seasonal="add",
                                     seasonal_periods=12).fit().forecast(h)  # fit and forecast h months
    except Exception:                                                       # statsmodels missing: last year x growth
        growth = y.iloc[-12:].sum() / y.iloc[-24:-12].sum()
        f = np.tile(y.iloc[-12:].to_numpy(), int(np.ceil(h / 12)))[:h] * growth
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")  # the next h months
    return pd.Series(np.asarray(f), index=idx)


def wape(actual, forecast):                                                 # weighted absolute percentage error
    return float(np.abs(np.asarray(actual) - np.asarray(forecast)).sum() / np.abs(np.asarray(actual)).sum())


def backtest(y):                                                            # hold out the last 12 months
    hold = y.iloc[-12:]                                                     # months to predict
    fc = pd.DataFrame({"Actual": hold.to_numpy(),
                       "Seasonal naive": y.iloc[-24:-12].to_numpy(),        # same month last year
                       "Moving average (12 m)": np.repeat(y.iloc[-24:-12].mean(), 12),  # average of the year before
                       "Holt-Winters": hw_forecast(y.iloc[:-12], 12).to_numpy()}, index=hold.index)
    acc = pd.DataFrame({"WAPE (%)": [wape(fc["Actual"], fc[m]) * 100 for m in fc.columns[1:]],
                        "Bias (%)": [(fc[m].sum() / fc["Actual"].sum() - 1) * 100 for m in fc.columns[1:]]}, index=fc.columns[1:])
    acc["Gain vs seasonal naive (%)"] = (1 - acc["WAPE (%)"] / acc.loc["Seasonal naive", "WAPE (%)"]) * 100
    return fc, acc.round(1)


@st.cache_data(show_spinner="Building the commercial tables...")
def finance_tables(sales, moves):
    s, m = sales.rename(columns=str.lower), moves.rename(columns=str.lower)  # one naming style
    cancelled = s["order_status"].eq("Cancelled")                           # void orders
    d = s[~cancelled].copy()                                                # delivered orders only
    unit_cost = m[m["movement_type"].eq("Sale_Out")].drop_duplicates("reference_id").set_index("reference_id")["unit_cost_kes"]
    d["cogs_kes"] = d["quantity"] * d["sale_id"].map(unit_cost)             # cost of goods sold per order
    d["gross_profit_kes"] = d["net_amount_excl_vat_kes"] - d["cogs_kes"]    # revenue less cost
    d["month"] = d["sale_date"].dt.to_period("M").dt.to_timestamp()        # month of sale
    d["age_days"] = (s["sale_date"].max() - d["sale_date"]).dt.days         # invoice age at the data end date
    stock = m.groupby(["branch_code", "product_id"])["quantity_change"].sum().clip(lower=0)  # units on hand (rebuilt from 0)
    last_cost = m.groupby("product_id")["unit_cost_kes"].last()             # latest unit cost
    stock_value = float((stock * stock.index.get_level_values("product_id").map(last_cost)).sum())  # stock at cost
    return d, s[cancelled], stock_value


def finance_page(sales, moves, pal, tn):                                    # the Commercial & finance view
    c1, c2, c3, grey = pal                                                  # chart colours for this app
    d, canc, stock_value = finance_tables(sales, moves)
    rev, gp, cogs = d["net_amount_excl_vat_kes"].sum(), d["gross_profit_kes"].sum(), d["cogs_kes"].sum()
    disc, gross = d["discount_amount_kes"].sum(), d["gross_amount_kes"].sum()
    due, billed = d["balance_due_kes"].sum(), d["total_amount_kes"].sum()
    cogs12 = d.loc[d["month"] > d["month"].max() - pd.DateOffset(months=12), "cogs_kes"].sum()  # last 12 months cost of sales
    k = st.columns(4)
    kpi(k[0], "", "Net revenue", f"KES {rev / 1e9:.2f} bn", f"{len(d):,} delivered orders, 2014–2025", tn[0])
    kpi(k[1], "", "Gross profit", f"KES {gp / 1e9:.2f} bn", f"{gp / rev:.1%} gross margin", tn[1], bar=gp / rev * 100)
    kpi(k[2], "", "Discounts given", f"KES {disc / 1e6:,.0f} m", f"{disc / gross:.1%} of gross sales", tn[2], bar=disc / gross * 100)
    kpi(k[3], "", "Receivables", f"KES {due / 1e6:,.0f} m", f"{1 - due / billed:.1%} of billed value collected", tn[3],
        bar=(1 - due / billed) * 100)
    k = st.columns(4)
    kpi(k[0], "", "VAT billed", f"KES {d['vat_amount_kes'].sum() / 1e9:.2f} bn", "at 16% on net sales", tn[4])
    kpi(k[1], "", "Cancelled orders", f"KES {canc['net_amount_excl_vat_kes'].sum() / 1e6:,.0f} m", f"{len(canc):,} orders not delivered", tn[2])
    kpi(k[2], "", "Stock at cost", f"KES {stock_value / 1e9:.1f} bn", f"{stock_value / (cogs12 / 12):,.0f} months of cost of sales", tn[3])
    kpi(k[3], "", "Average order", f"KES {rev / len(d):,.0f}", "net, excluding VAT", tn[5])

    st.subheader("Revenue, gross profit and margin by year")
    yr = d.groupby(d["sale_date"].dt.year.rename("year")).agg(revenue=("net_amount_excl_vat_kes", "sum"),
                                                              gross_profit=("gross_profit_kes", "sum")).reset_index()
    yr["margin_pct"] = yr["gross_profit"] / yr["revenue"] * 100
    yl = yr.melt(["year", "margin_pct"], ["revenue", "gross_profit"], "measure", "kes")
    yl["kes_m"] = yl["kes"] / 1e6
    bars = alt.Chart(yl).mark_bar().encode(x=alt.X("year:O", title=None), xOffset="measure:N",
                                           y=alt.Y("kes_m:Q", title="KES million"),
                                           color=alt.Color("measure:N", scale=alt.Scale(range=[c2, c1]), title=None),
                                           tooltip=["year", "measure", alt.Tooltip("kes_m:Q", format=",.0f")])
    line = alt.Chart(yr).mark_line(color=c3, point=True).encode(x="year:O", y=alt.Y("margin_pct:Q", title="Gross margin (%)"),
                                                                tooltip=["year", alt.Tooltip("margin_pct:Q", format=".1f")])
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=320), width="stretch")

    c = st.columns(2)
    cat = d.groupby("product_category").agg(revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                            gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                            discount_kes_m=("discount_amount_kes", lambda x: x.sum() / 1e6),
                                            orders=("sale_id", "size"))
    cat["margin_pct"] = cat["gross_profit_kes_m"] / cat["revenue_kes_m"] * 100
    cat = cat.sort_values("gross_profit_kes_m", ascending=False).round(1)
    c[0].altair_chart(alt.Chart(cat.reset_index()).mark_bar().encode(
        y=alt.Y("product_category:N", sort="-x", title=None), x=alt.X("gross_profit_kes_m:Q", title="Gross profit (KES m)"),
        color=alt.Color("margin_pct:Q", scale=alt.Scale(scheme="greens"), title="Margin %"),
        tooltip=list(cat.reset_index().columns)).properties(height=260, title="Gross profit by category (colour = margin)"), width="stretch")
    c[0].dataframe(cat, width="stretch")
    br = d.groupby("branch_code").agg(revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                      gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                      orders=("sale_id", "size"), avg_order_kes=("net_amount_excl_vat_kes", "mean"),
                                      unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6))
    br["margin_pct"] = br["gross_profit_kes_m"] / br["revenue_kes_m"] * 100
    br["share_of_revenue_pct"] = br["revenue_kes_m"] / br["revenue_kes_m"].sum() * 100
    c[1].altair_chart(alt.Chart(br.reset_index()).mark_arc(innerRadius=60).encode(
        theta="revenue_kes_m:Q", color=alt.Color("branch_code:N", title="Branch"),
        tooltip=["branch_code", alt.Tooltip("revenue_kes_m:Q", format=",.0f"), alt.Tooltip("margin_pct:Q", format=".1f")])
        .properties(height=260, title="Revenue share by branch"), width="stretch")
    c[1].dataframe(br.round(1), width="stretch")

    st.subheader("Profitability")
    last = d["month"] > d["month"].max() - pd.DateOffset(months=12)         # last 12 months of sales
    rev12, gp12, cogs12b = (d.loc[last, c].sum() for c in ["net_amount_excl_vat_kes", "gross_profit_kes", "cogs_kes"])
    k = st.columns(4)
    kpi(k[0], "", "Gross profit margin", f"{gp / rev:.1%}", f"last 12 months {gp12 / rev12:.1%}", tn[1], bar=gp / rev * 100)
    kpi(k[1], "", "Markup on cost", f"{gp / cogs:.1%}", "gross profit ÷ cost of goods sold", tn[4])
    kpi(k[2], "", "Gross profit per order", f"KES {gp / len(d):,.0f}", f"on an average order of KES {rev / len(d):,.0f}", tn[5])
    kpi(k[3], "", "Discount share of profit", f"{disc / gp:.1%}", "discounts given ÷ gross profit", tn[2], bar=disc / gp * 100)
    mm = d.groupby("month").agg(revenue=("net_amount_excl_vat_kes", "sum"), gp=("gross_profit_kes", "sum"))
    mm["Monthly margin (%)"] = mm["gp"] / mm["revenue"] * 100               # gross margin each month
    mm["12-month rolling margin (%)"] = mm["gp"].rolling(12).sum() / mm["revenue"].rolling(12).sum() * 100
    st.markdown("**Gross profit margin over time**")
    st.line_chart(mm[["Monthly margin (%)", "12-month rolling margin (%)"]], color=[grey, c1])
    c = st.columns(2)
    dl = d.groupby("discount_pct").agg(orders=("sale_id", "size"), revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                       gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6))
    dl["margin_pct"] = dl["gross_profit_kes_m"] / dl["revenue_kes_m"] * 100
    c[0].altair_chart(alt.Chart(dl.reset_index()).mark_bar(color=c3).encode(
        x=alt.X("discount_pct:O", title="Discount (%)"), y=alt.Y("margin_pct:Q", title="Gross margin (%)"),
        tooltip=["discount_pct", "orders", alt.Tooltip("revenue_kes_m:Q", format=",.0f"), alt.Tooltip("margin_pct:Q", format=".1f")])
        .properties(height=260, title="Margin falls as the discount rises"), width="stretch")
    c[0].dataframe(dl.round(1), width="stretch")
    rep = d.groupby("sales_rep_id").agg(orders=("sale_id", "size"), revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                        gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                        discount_pct=("discount_pct", "mean"), unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6))
    rep["margin_pct"] = rep["gross_profit_kes_m"] / rep["revenue_kes_m"] * 100
    c[1].altair_chart(alt.Chart(rep.reset_index()).mark_bar().encode(
        y=alt.Y("sales_rep_id:N", sort="-x", title=None), x=alt.X("gross_profit_kes_m:Q", title="Gross profit (KES m)"),
        color=alt.Color("margin_pct:Q", scale=alt.Scale(scheme="greens"), title="Margin %"),
        tooltip=["sales_rep_id", "orders", alt.Tooltip("gross_profit_kes_m:Q", format=",.1f"), alt.Tooltip("margin_pct:Q", format=".1f")])
        .properties(height=260, title="Gross profit by sales rep"), width="stretch")
    c[1].dataframe(rep.sort_values("gross_profit_kes_m", ascending=False).round(1), width="stretch")
    prod = d.groupby(["product_id", "product_category"]).agg(revenue_kes_m=("net_amount_excl_vat_kes", lambda x: x.sum() / 1e6),
                                                             gross_profit_kes_m=("gross_profit_kes", lambda x: x.sum() / 1e6),
                                                             units=("quantity", "sum")).reset_index()
    prod["margin_pct"] = prod["gross_profit_kes_m"] / prod["revenue_kes_m"] * 100
    c = st.columns(2)
    c[0].markdown("**Top 10 products by gross profit**")
    c[0].dataframe(prod.nlargest(10, "gross_profit_kes_m").round(2), hide_index=True, width="stretch")
    c[1].markdown("**Lowest-margin 10 products**")
    c[1].dataframe(prod.nsmallest(10, "margin_pct").round(2), hide_index=True, width="stretch")

    st.subheader("Working capital and customers")
    billed12 = d.loc[last, "total_amount_kes"].sum()                         # value invoiced in the last 12 months
    dso, turns = due / billed12 * 365, cogs12b / stock_value                # days sales outstanding, stock turns a year
    top10 = d.groupby("customer_id")["net_amount_excl_vat_kes"].sum().nlargest(10).sum() / rev  # customer concentration
    k = st.columns(4)
    kpi(k[0], "", "Days sales outstanding", f"{dso:,.0f} days", "unpaid ÷ last 12 months billed × 365", tn[3])
    kpi(k[1], "", "Inventory turnover", f"{turns:.2f}× a year", f"about {365 / turns:,.0f} days of stock on hand", tn[2])
    kpi(k[2], "", "GMROI", f"{gp12 / stock_value:.3f}", "gross profit per KES 1 of stock at cost", tn[5])
    kpi(k[3], "", "Top 10 customers", f"{top10:.1%}", f"of revenue · {d['customer_id'].nunique():,} customers", tn[4], bar=top10 * 100)
    st.caption("Stock rebuilt from zero, mostly Equipment. Net profit not available (no operating costs).")
    st.subheader("Receivables (unpaid balances)")
    ag = d[d["balance_due_kes"] > 0].copy()                                 # invoices with money owed
    ag["age"] = pd.cut(ag["age_days"], [-1, 30, 90, 180, 365, 100000], labels=["0–30 days", "31–90", "91–180", "181–365", "over 1 year"])
    aging = ag.groupby("age", observed=False)["balance_due_kes"].sum().div(1e6).reset_index(name="kes_m")
    c = st.columns(2)
    c[0].altair_chart(alt.Chart(aging).mark_bar(color=c3).encode(
        x=alt.X("age:N", sort=None, title="Invoice age at the data end date"), y=alt.Y("kes_m:Q", title="Unpaid (KES m)"),
        tooltip=["age", alt.Tooltip("kes_m:Q", format=",.1f")]).properties(height=260, title="Unpaid balance by invoice age"), width="stretch")
    pm = d.groupby("payment_method").agg(billed_kes_m=("total_amount_kes", lambda x: x.sum() / 1e6),
                                         unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6))
    pm["unpaid_pct"] = pm["unpaid_kes_m"] / pm["billed_kes_m"] * 100
    c[1].markdown("**Billed and unpaid by payment method**")
    c[1].dataframe(pm.round(1), width="stretch")
    c[1].markdown("**Top 10 customers by unpaid balance**")
    c[1].dataframe(d.groupby("customer_id").agg(unpaid_kes_m=("balance_due_kes", lambda x: x.sum() / 1e6), orders=("sale_id", "size"))
                   .sort_values("unpaid_kes_m", ascending=False).head(10).round(2), width="stretch")

    st.subheader("Revenue forecast (next 12 months)")
    choice = st.selectbox("Revenue series", ["Total"] + sorted(d["product_category"].unique()), key="fin_series")
    dd = d if choice == "Total" else d[d["product_category"] == choice]     # chosen category
    y = dd.groupby("month")["net_amount_excl_vat_kes"].sum().asfreq("MS", fill_value=0) / 1e6  # monthly revenue, KES m
    fc, acc = backtest(y)                                                   # accuracy on the last 12 months
    future = hw_forecast(y, 12)                                             # next 12 months
    m12 = dd.loc[dd["month"] > dd["month"].max() - pd.DateOffset(months=12)]
    margin12 = m12["gross_profit_kes"].sum() / m12["net_amount_excl_vat_kes"].sum()  # trailing 12-month margin
    k = st.columns(4)
    growth = future.sum() / y.iloc[-12:].sum() - 1
    kpi(k[0], "", "Forecast revenue", f"KES {future.sum():,.0f} m", "next 12 months", tn[0])
    kpi(k[1], "" if growth >= 0 else "", "Change vs last 12 months", f"{growth:+.1%}", f"last 12 months KES {y.iloc[-12:].sum():,.0f} m",
        tn[1] if growth >= 0 else tn[2])
    kpi(k[2], "", "Projected gross profit", f"KES {future.sum() * margin12:,.0f} m", f"at the last 12 months' margin ({margin12:.1%})", tn[4])
    kpi(k[3], "", "Forecast error", f"{acc.loc['Holt-Winters', 'WAPE (%)']:.1f}%",
        f"WAPE on last 12 months vs {acc.loc['Seasonal naive', 'WAPE (%)']:.1f}% seasonal naive", tn[5])
    st.line_chart(pd.DataFrame({"Actual (KES m)": y, "Forecast (KES m)": future}), color=[c1, c3])
    c = st.columns(2)
    c[0].markdown("**Back-test on the last 12 months (KES m)**")
    c[0].line_chart(fc)
    c[1].markdown("**Accuracy of each method**")
    c[1].dataframe(acc, width="stretch")
    out = pd.DataFrame({"forecast_revenue_kes_m": future.round(1), "projected_gross_profit_kes_m": (future * margin12).round(1)})
    c[1].markdown("**Monthly forecast**")
    c[1].dataframe(out, width="stretch")
    st.download_button("Download revenue forecast (CSV)", out.to_csv(), "revenue_forecast.csv", key="fin_dl")
    st.caption("Gross profit at recorded unit cost · Holt-Winters forecast.")
def finance_headline(sales, moves):                                         # three numbers for the executive summary
    d, canc, stock_value = finance_tables(sales, moves)
    rev, gp = d["net_amount_excl_vat_kes"].sum(), d["gross_profit_kes"].sum()
    y = d.groupby("month")["net_amount_excl_vat_kes"].sum().asfreq("MS", fill_value=0) / 1e6
    return dict(revenue=rev, margin=gp / rev, unpaid=d["balance_due_kes"].sum(), stock_value=stock_value,
                forecast=hw_forecast(y, 12).sum(), last12=y.iloc[-12:].sum(), cancelled=canc["net_amount_excl_vat_kes"].sum())


def exec_commercial(sales, moves, tn):                                      # commercial picture for the executive summary
    h = finance_headline(sales, moves)
    st.subheader("Commercial picture")
    k = st.columns(4)
    kpi(k[0], "", "Revenue & margin", f"KES {h['revenue'] / 1e9:.1f} bn", f"{h['margin']:.1%} gross margin, 2014–2025", tn[0], bar=h["margin"] * 100)
    kpi(k[1], "", "Next 12 months", f"KES {h['forecast'] / 1e3:.2f} bn", f"{h['forecast'] / h['last12'] - 1:+.1%} vs last 12 months", tn[1])
    kpi(k[2], "", "Receivables", f"KES {h['unpaid'] / 1e6:,.0f} m", "unpaid, all on credit accounts", tn[3])
    kpi(k[3], "", "Stock at cost", f"KES {h['stock_value'] / 1e9:.1f} bn", "cash tied up in inventory", tn[2])
    st.caption(f"Cancelled orders: KES {h['cancelled'] / 1e6:,.0f} m of revenue lost.")


# ------------------------------------------------------------------ what-if predictor (enter values, see the predicted outcome)
from sklearn.compose import make_column_transformer                         # treat text and number inputs differently
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # tree models
from sklearn.impute import SimpleImputer as _Imputer                        # fill gaps
from sklearn.linear_model import LogisticRegression as _Logit               # failure-risk model
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score    # how good each model is
from sklearn.model_selection import train_test_split                        # hold-out test set
from sklearn.pipeline import make_pipeline as _pipe                         # chain the steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler as _Scaler  # encode text, scale numbers

P_NUM = ["batch_size", "production_hours", "target_yield", "temp_avg", "temp_variance", "pressure_avg", "pressure_variance",
         "humidity_avg", "raw_material_quality", "equipment_age_years", "maintenance_days_ago", "operator_experience_years"]
P_CAT = ["manufacturing_section", "dosage_form", "maintenance_status"]
S_NUM = ["equipment_age_years", "maintenance_days_ago", "operator_experience_years", "batch_size", "production_hours"]
S_CAT = ["manufacturing_section"]
LABELS = {"batch_size": "Batch size", "production_hours": "Production hours", "target_yield": "Target yield (%)",
          "temp_avg": "Average temperature (°C)", "temp_variance": "Temperature variance", "pressure_avg": "Average pressure",
          "pressure_variance": "Pressure variance", "humidity_avg": "Average humidity (%)", "raw_material_quality": "Raw material quality",
          "equipment_age_years": "Equipment age (years)", "maintenance_days_ago": "Days since maintenance",
          "operator_experience_years": "Operator experience (years)"}


def _prep(num, cat, scale=False):                                           # imputing, encoding (and scaling) step
    num_steps = [_Imputer(strategy="median")] + ([_Scaler()] if scale else [])
    return make_column_transformer((_pipe(*num_steps), num),
                                   (_pipe(_Imputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat))


@st.cache_resource(show_spinner="Training the what-if models (about 30 seconds, once)...")
def whatif_models(_df, target, key):                                        # key names the dataset for the cache
    df = _df.dropna(subset=["maintenance_status"])
    X, y = df[P_NUM + P_CAT], df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  # 80/20 split
    risk = _pipe(_prep(P_NUM, P_CAT, scale=True), _Logit(class_weight="balanced", max_iter=2000)).fit(Xtr, ytr)  # failure risk
    yv = df["actual_yield"]
    ytr_y, yte_y = yv.loc[Xtr.index], yv.loc[Xte.index]
    ymodel = _pipe(_prep(P_NUM, P_CAT), RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5,
                                                               n_jobs=-1, random_state=42)).fit(Xtr[ytr_y.notna()], ytr_y.dropna())
    ok = yte_y.notna()
    scores = {"risk_auc": roc_auc_score(yte, risk.predict_proba(Xte)[:, 1]),                  # failure model ranking quality
              "yield_mae": mean_absolute_error(yte_y[ok], ymodel.predict(Xte[ok])),           # average yield error
              "yield_base": mean_absolute_error(yte_y[ok], np.repeat(ytr_y.mean(), ok.sum()))}
    train_risk = risk.predict_proba(Xtr)[:, 1]                              # past scores, for the risk bands
    return risk, ymodel, scores, np.percentile(train_risk, [50, 90]), train_risk.mean()


def predictor_tab(df, target, key, pal, tn, sales, moves, policy):          # the What-if predictor view
    c1, c2, c3, grey = pal
    risk, ymodel, scores, (p50, p90), avg_risk = whatif_models(df, target, key)
    part = st.radio("What do you want to predict?", ["Batch outcome", "Equipment maintenance status", "Stock and reorder",
                                                     "Price and discount scenario"], horizontal=True, key=f"wi_{key}")
    q = lambda col, p: float(np.nanpercentile(df[col], p))                  # a percentile of a column, for input limits
    med = lambda col: float(df[col].median())

    if part == "Batch outcome":
        st.caption(f"Failure risk: logistic regression (ROC-AUC {scores['risk_auc']:.2f}) · Yield: random forest (MAE {scores['yield_mae']:.2f})")
        c = st.columns(3)
        v = {"manufacturing_section": c[0].selectbox("Manufacturing section", sorted(df["manufacturing_section"].unique()), key=f"s_{key}"),
             "dosage_form": c[0].selectbox("Dosage form", sorted(df["dosage_form"].unique()), key=f"d_{key}"),
             "maintenance_status": c[0].selectbox("Maintenance status", ["Current", "Due Soon", "Critical"], key=f"m_{key}")}
        for i, col in enumerate(P_NUM):                                      # one number box per input, prefilled with the median
            v[col] = c[(i % 2) + 1].number_input(LABELS[col], q(col, 0.5), q(col, 99.5), round(med(col), 2), key=f"n_{col}_{key}")
        row = pd.DataFrame([v])
        p, yl = risk.predict_proba(row)[0, 1], ymodel.predict(row)[0]
        band = "High" if p >= p90 else ("Medium" if p >= p50 else "Low")
        k = st.columns(4)
        kpi(k[0], "", "Failure risk score", f"{p:.2f}", f"{p / avg_risk:.1f}× the average batch", {"High": tn[2], "Medium": tn[5], "Low": tn[1]}[band], bar=p * 100)
        kpi(k[1], "", "Risk band", band, "High = top 10% of past batches", {"High": tn[2], "Medium": tn[5], "Low": tn[1]}[band])
        kpi(k[2], "", "Predicted yield", f"{yl:.2f}%", f"target {v['target_yield']:.2f}%", tn[0])
        gap = yl - v["target_yield"]
        kpi(k[3], "", "Yield vs target", f"{gap:+.2f} pts", "above target" if gap >= 0 else "below target", tn[1] if gap >= 0 else tn[2])
        st.markdown("**See how the prediction moves when you change one input** (all other inputs stay as entered)")
        vary = st.selectbox("Input to vary", [LABELS[c] for c in P_NUM] + ["Maintenance status"], key=f"vary_{key}")
        if vary == "Maintenance status":
            grid = pd.concat([row.assign(maintenance_status=s) for s in ["Current", "Due Soon", "Critical"]], ignore_index=True)
            xs = ["Current", "Due Soon", "Critical"]
        else:
            col = next(c for c in P_NUM if LABELS[c] == vary)
            xs = np.linspace(q(col, 2), q(col, 98), 25)
            grid = pd.concat([row] * len(xs), ignore_index=True)
            grid[col] = xs
        curve = pd.DataFrame({"value": xs, "Failure risk score": risk.predict_proba(grid)[:, 1], "Predicted yield (%)": ymodel.predict(grid)})
        cc = st.columns(2)
        for col_, measure, colour in [(cc[0], "Failure risk score", c3), (cc[1], "Predicted yield (%)", c1)]:
            ch = alt.Chart(curve).mark_line(point=True, color=colour).encode(
                x=alt.X("value:" + ("N" if vary == "Maintenance status" else "Q"), title=vary, sort=None),
                y=alt.Y(f"{measure}:Q", scale=alt.Scale(zero=False)), tooltip=["value", alt.Tooltip(f"{measure}:Q", format=".3f")])
            col_.altair_chart(ch.properties(height=260, title=measure), width="stretch")

    elif part == "Equipment maintenance status":
        lim = df.groupby("maintenance_status")["maintenance_days_ago"].agg(["min", "max"])  # the service rule seen in the data
        due, crit = int(lim.loc["Due Soon", "min"]), int(lim.loc["Critical", "min"])
        st.caption(f"Rule in the data: Current < {due} days · Due Soon {due}–{crit - 1} · Critical {crit}+")
        c = st.columns(3)
        days = c[0].number_input("Days since maintenance", 0, 400, 60, key=f"ed_{key}")
        age = c[1].number_input("Equipment age (years)", 0.0, 40.0, round(med("equipment_age_years"), 1), key=f"ea_{key}")
        sec = c[2].selectbox("Manufacturing section", sorted(df["manufacturing_section"].unique()), key=f"es_{key}")
        pred = "Critical" if days >= crit else ("Due Soon" if days >= due else "Current")
        nxt = (due - days) if pred == "Current" else ((crit - days) if pred == "Due Soon" else 0)
        hist = df.groupby("maintenance_status")[target].mean()              # failure rate seen for each status
        near = df[(df["manufacturing_section"] == sec) & (df["equipment_age_years"].sub(age).abs() <= 2)]  # similar machines
        tone = {"Current": tn[1], "Due Soon": tn[5], "Critical": tn[2]}
        k = st.columns(4)
        kpi(k[0], "", "Status", pred, f"{days} days since the last service", tone[pred])
        kpi(k[1], "", "Days to next stage", "now" if pred == "Critical" else f"{nxt}",
            "Critical: service immediately" if pred == "Critical" else ("until Due Soon" if pred == "Current" else "until Critical"), tone[pred])
        kpi(k[2], "", "Failure rate at this status", f"{hist[pred]:.1%}", f"vs {hist['Current']:.1%} when Current", tone[pred])
        kpi(k[3], "", "Similar machines", f"{near[target].mean():.1%}" if len(near) else "–",
            f"failure rate, {sec}, age ±2 years ({len(near):,} batches)", tn[4])
        bins = pd.cut(df["maintenance_days_ago"], range(0, 200, 10))       # 10-day bands
        fr = df.groupby(bins, observed=True)[target].agg(rate="mean", batches="size").reset_index()
        fr["days"] = fr["maintenance_days_ago"].apply(lambda b: b.left + 5)
        fr["rate"] *= 100
        bars = alt.Chart(fr).mark_bar(color=c1, opacity=0.8).encode(
            x=alt.X("days:Q", title="Days since maintenance"), y=alt.Y("rate:Q", title="Failure rate (%)"),
            tooltip=["days", alt.Tooltip("rate:Q", format=".2f"), "batches"])
        marks = alt.Chart(pd.DataFrame({"x": [due, crit], "label": ["Due Soon", "Critical"]})).mark_rule(color=c3, strokeDash=[5, 3]).encode(x="x:Q")
        you = alt.Chart(pd.DataFrame({"x": [days]})).mark_rule(color="black", strokeWidth=3).encode(x="x:Q")
        st.altair_chart((bars + marks + you).properties(height=280, title="Failure rate by days since maintenance (black line = your machine)"),
                        width="stretch")

    elif part == "Stock and reorder":
        pol = policy.rename(columns=str.lower)                              # one naming style for both dashboards
        c = st.columns(2)
        br = c[0].selectbox("Branch", sorted(pol["branch_code"].unique()), key=f"rb_{key}")
        pid = c[1].selectbox("Product", sorted(pol.loc[pol["branch_code"] == br, "product_id"]), key=f"rp_{key}")
        it = pol[(pol["branch_code"] == br) & (pol["product_id"] == pid)].iloc[0]
        st.caption(f"{pid} · {it['category']} · {br}")
        c = st.columns(5)
        stock = c[0].number_input("Stock on hand (units)", 0.0, 1e7, float(max(it["stock_now"], 0)), key=f"rs_{key}")
        wk = c[1].number_input("Average weekly demand", 0.0, 1e5, float(it["avg_weekly"]), key=f"rw_{key}")
        sd = c[2].number_input("Weekly demand spread (SD)", 0.0, 1e5, float(it["sd_weekly"]), key=f"rd_{key}")
        lead = c[3].number_input("Lead time (days)", 1, 120, 14, key=f"rl_{key}")
        cover = c[4].number_input("Cover to order up to (weeks)", 1, 52, 8, key=f"rc_{key}")
        z = 1.65                                                            # 95% service level
        daily = wk / 7
        safety = z * sd / np.sqrt(7) * np.sqrt(lead)                         # buffer for the lead time
        rop = daily * lead + safety                                         # reorder point
        days = stock / daily if daily > 0 else np.inf                       # days until stock runs out
        order = max(0.0, rop + cover * wk - stock)                          # order up to reorder point + cover
        status = "No demand" if daily == 0 else ("Reorder now" if stock <= rop else ("Overstock" if days > 180 else "OK"))
        k = st.columns(4)
        kpi(k[0], "", "Days of cover", "∞" if np.isinf(days) else f"{days:,.0f}", "at the entered demand", tn[0])
        kpi(k[1], "", "Reorder point", f"{rop:,.0f}", f"incl. safety stock {safety:,.0f}", tn[4])
        kpi(k[2], "", "Suggested order", f"{order:,.0f} units", f"to cover lead time + {cover} weeks", tn[5])
        kpi(k[3], "", "Status", status, "95% service level", {"Reorder now": tn[2], "Overstock": tn[5], "OK": tn[1]}.get(status, tn[3]))
        weeks = np.arange(0, 27)
        proj = pd.DataFrame({"week": weeks, "Stock if no order": np.maximum(stock - wk * weeks, 0),
                             "Reorder point": rop})
        st.line_chart(proj.set_index("week"), color=[c1, c3])
    else:
        d, _, _ = finance_tables(sales, moves)
        cat = st.selectbox("Category", ["All"] + sorted(d["product_category"].unique()), key=f"pc_{key}")
        base = d[d["month"] > d["month"].max() - pd.DateOffset(months=12)]   # last 12 months as the base
        base = base if cat == "All" else base[base["product_category"] == cat]
        R, C = base["net_amount_excl_vat_kes"].sum(), base["cogs_kes"].sum()  # base revenue and cost
        c = st.columns(4)
        price = c[0].slider("Price change (%)", -30, 30, 0, key=f"pp_{key}")
        disc = c[1].slider("Extra discount (points)", -5, 15, 0, key=f"pd_{key}")
        vol = c[2].slider("Volume change (%)", -50, 50, 0, key=f"pv_{key}")
        cost = c[3].slider("Unit cost change (%)", -20, 30, 0, key=f"pk_{key}")
        R2 = R * (1 + price / 100) * (1 - disc / 100) * (1 + vol / 100)     # scenario revenue
        C2 = C * (1 + cost / 100) * (1 + vol / 100)                         # scenario cost
        G, G2 = R - C, R2 - C2
        per_unit = R * (1 + price / 100) * (1 - disc / 100) - C * (1 + cost / 100)  # margin per base volume
        be = (G / per_unit - 1) * 100 if per_unit > 0 else np.nan            # volume change that keeps gross profit flat
        k = st.columns(4)
        kpi(k[0], "", "Revenue (12 months)", f"KES {R2 / 1e6:,.0f} m", f"{R2 / R - 1:+.1%} vs KES {R / 1e6:,.0f} m", tn[0])
        kpi(k[1], "", "Gross profit", f"KES {G2 / 1e6:,.0f} m", f"{G2 / G - 1:+.1%} vs KES {G / 1e6:,.0f} m", tn[1] if G2 >= G else tn[2])
        kpi(k[2], "", "Gross margin", f"{G2 / R2:.1%}" if R2 else "–", f"base {G / R:.1%}", tn[4])
        kpi(k[3], "", "Break-even volume change", "n/a" if np.isnan(be) else f"{be:+.1f}%",
            "needed to keep gross profit flat", tn[5])
        cmp = pd.DataFrame({"case": ["Base", "Base", "Scenario", "Scenario"], "measure": ["Revenue", "Gross profit"] * 2,
                            "kes_m": [R / 1e6, G / 1e6, R2 / 1e6, G2 / 1e6]})
        st.altair_chart(alt.Chart(cmp).mark_bar().encode(
            x=alt.X("measure:N", title=None), xOffset="case:N", y=alt.Y("kes_m:Q", title="KES million"),
            color=alt.Color("case:N", scale=alt.Scale(range=[grey, c1])), tooltip=["case", "measure", alt.Tooltip("kes_m:Q", format=",.0f")])
            .properties(height=280, title="Last 12 months: base vs your scenario"), width="stretch")
        st.caption("Base: last 12 months · no price elasticity.")
def _clean(name):                                                           # readable feature names from the model pipeline
    n = name.split("__", 1)[-1]
    for cat in P_CAT:
        if n.startswith(cat + "_"):
            return f"{cat.replace('_', ' ').capitalize()}: {n[len(cat) + 1:]}"
    return LABELS.get(n, n)


def _band_effect(df, target, col, label):                                   # failure rate in the lowest vs highest quarter
    lo, hi = df[col].quantile([0.25, 0.75])
    a, b = df.loc[df[col] <= lo, target].mean(), df.loc[df[col] >= hi, target].mean()
    return [label, f"≤ {lo:,.1f}", f"{a:.2%}", f"≥ {hi:,.1f}", f"{b:.2%}", (b - a) * 100]


def growth_driver(vol, price):                                              # which part of revenue growth dominates
    if abs(vol - price) / max(vol, price) < 0.15:
        return "volume and price contributed **almost equally**"
    return "growth came mainly from **" + ("volume" if vol > price else "price") + "**"


def exec_drivers(df, target, key, pal, tn, sales, moves, policy):
    c1, c2, c3, grey = pal
    risk, ymodel, scores, _, _ = whatif_models(df, target, key)             # the same models as the What-if tab
    st.subheader("How the features drove each finding")

    # ---- A. batch failure
    st.markdown("**A. Batch failure drivers**")
    rate = df.groupby("maintenance_status")[target].mean()
    age_band = pd.cut(df["equipment_age_years"], [-1, 5, 10, 100], labels=["0–5 yrs", "5–10 yrs", "10+ yrs"])
    exp_band = pd.cut(df["operator_experience_years"], [-1, 2, 5, 10, 100], labels=["0–2 yrs", "3–5 yrs", "6–10 yrs", "10+ yrs"])
    ar, er = df.groupby(age_band, observed=True)[target].mean(), df.groupby(exp_band, observed=True)[target].mean()
    dose = df.groupby("dosage_form")[target].mean().sort_values(ascending=False)
    k = st.columns(4)
    kpi(k[0], "", "Critical vs Current", f"{rate['Critical']:.1%} vs {rate['Current']:.1%}", "failure rate", tn[2])
    kpi(k[1], "", "Equipment 10+ vs 0–5 yrs", f"{ar.iloc[-1]:.1%} vs {ar.iloc[0]:.1%}", "failure rate", tn[5])
    kpi(k[2], "", "Operators 0–2 vs 10+ yrs", f"{er.iloc[0]:.1%} vs {er.iloc[-1]:.1%}", "failure rate", tn[3])
    kpi(k[3], "", f"Riskiest form: {dose.index[0]}", f"{dose.iloc[0]:.1%}", f"plant average {df[target].mean():.1%}", tn[0])
    names = [_clean(n) for n in risk[0].get_feature_names_out()]
    coef = pd.Series(risk[-1].coef_.ravel(), index=names)                   # push on the log-odds
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).reset_index()
    top.columns = ["feature", "weight"]
    top["direction"] = np.where(top["weight"] > 0, "raises risk", "lowers risk")
    eff = pd.DataFrame([_band_effect(df, target, col, LABELS[col]) for col in
                        ["equipment_age_years", "operator_experience_years", "maintenance_days_ago", "raw_material_quality",
                         "temp_variance", "production_hours", "batch_size"]],
                       columns=["feature", "low", "low_rate", "high", "high_rate", "diff"])
    c = st.columns(2)
    c[0].altair_chart(alt.Chart(top).mark_bar().encode(
        y=alt.Y("feature:N", sort=None, title=None), x=alt.X("weight:Q", title="Model weight"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[c3, c1]), title=None, legend=alt.Legend(orient="bottom")),
        tooltip=["feature", alt.Tooltip("weight:Q", format=".3f")]).properties(height=320, title="Failure model: top 10 weights"), width="stretch")
    c[1].altair_chart(alt.Chart(eff).mark_bar().encode(
        y=alt.Y("feature:N", sort=alt.EncodingSortField("diff", op="sum", order="descending"), title=None),
        x=alt.X("diff:Q", title="Failure rate, top vs bottom quarter (points)"),
        color=alt.condition(alt.datum.diff > 0, alt.value(c3), alt.value(c1)),
        tooltip=["feature", "low", "low_rate", "high", "high_rate", alt.Tooltip("diff:Q", format=".2f")])
        .properties(height=320, title="Effect of each feature on failure rate"), width="stretch")

    # ---- B. yield
    st.markdown("**B. Yield drivers**")
    imp = pd.Series(ymodel[-1].feature_importances_, index=[_clean(n) for n in ymodel[0].get_feature_names_out()])
    imp = imp.groupby(lambda n: n.split(":")[0]).sum().sort_values(ascending=False).head(8)
    yi = (imp / imp.sum() * 100).reset_index()
    yi.columns = ["feature", "share"]
    c = st.columns([1, 3])
    kpi(c[0], "", "Yield error", f"{scores['yield_mae']:.2f} pts", f"vs {scores['yield_base']:.2f} using the average", tn[0])
    kpi(c[0], "", "Better than average", f"{1 - scores['yield_mae'] / scores['yield_base']:.0%}", "random forest", tn[1],
        bar=(1 - scores["yield_mae"] / scores["yield_base"]) * 100)
    c[1].altair_chart(alt.Chart(yi).mark_bar(color=c1).encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("share:Q", title="Share of importance (%)"),
        tooltip=["feature", alt.Tooltip("share:Q", format=".1f")]).properties(height=270, title="Yield model: importance"), width="stretch")

    # ---- C. rise since 2019
    st.markdown("**C. Rise since 2019: ageing equipment**")
    yr = df.groupby("year").agg(failure_rate=(target, "mean"),
                                old_share=("equipment_age_years", lambda s: (s > 10).mean())).reset_index()
    yr[["failure_rate", "old_share"]] *= 100
    early, late = yr[yr["year"].between(2014, 2018)], yr[yr["year"].between(2020, 2025)]
    c = st.columns([1, 3])
    kpi(c[0], "", "Equipment 10+ yrs", f"{early['old_share'].mean():.0f}% → {late['old_share'].mean():.0f}%", "share of batches, 2014–18 → 2020–25", tn[5])
    kpi(c[0], "", "Failure rate", f"{early['failure_rate'].mean():.2f}% → {late['failure_rate'].mean():.2f}%", "2014–18 → 2020–25", tn[2])
    base = alt.Chart(yr).encode(x=alt.X("year:O", title=None))
    c[1].altair_chart(alt.layer(
        base.mark_area(color=grey, opacity=0.35).encode(y=alt.Y("old_share:Q", title="Equipment 10+ yrs (% of batches)")),
        base.mark_line(color=c3, point=True).encode(y=alt.Y("failure_rate:Q", title="Failure rate (%)"),
                                                    tooltip=["year", alt.Tooltip("failure_rate:Q", format=".2f"), alt.Tooltip("old_share:Q", format=".0f")]))
        .resolve_scale(y="independent").properties(height=270, title="Old-equipment share (area) vs failure rate (line)"), width="stretch")

    # ---- D. sales
    st.markdown("**D. Revenue and margin drivers**")
    d, _, _ = finance_tables(sales, moves)                                  # delivered orders with cost and gross profit
    yr = d.groupby(d["sale_date"].dt.year.rename("year")).agg(revenue=("net_amount_excl_vat_kes", "sum"), units=("quantity", "sum"))
    yr["price_per_unit"] = yr["revenue"] / yr["units"]
    first, last_ = yr.iloc[0], yr.iloc[-1]
    dm = d.groupby("discount_pct").agg(gp=("gross_profit_kes", "sum"), rev=("net_amount_excl_vat_kes", "sum"), units=("quantity", "median"))
    dm["margin"] = dm["gp"] / dm["rev"] * 100
    cm = d.groupby("product_category").agg(gp=("gross_profit_kes", "sum"), rev=("net_amount_excl_vat_kes", "sum"))
    cm["margin"] = cm["gp"] / cm["rev"] * 100
    br = d.groupby("branch_code").agg(gp=("gross_profit_kes", "sum"), rev=("net_amount_excl_vat_kes", "sum"))
    br["margin"] = br["gp"] / br["rev"] * 100
    unpaid = d.groupby("payment_method")["balance_due_kes"].sum()
    k = st.columns(4)
    kpi(k[0], "", "Revenue growth", f"{last_['revenue'] / first['revenue']:.1f}×", f"{int(yr.index[0])}–{int(yr.index[-1])}", tn[0])
    kpi(k[1], "", "Volume vs price", f"{last_['units'] / first['units']:.1f}× / {last_['price_per_unit'] / first['price_per_unit']:.1f}×",
        "units sold / net price per unit", tn[4])
    kpi(k[2], "", "Margin: 0% vs max discount", f"{dm['margin'].iloc[0]:.1f}% vs {dm['margin'].iloc[-1]:.1f}%", f"at {dm.index[-1]:g}% discount", tn[2])
    kpi(k[3], "", "Unpaid on credit accounts", f"{unpaid.get('Credit Account', 0) / unpaid.sum():.0%}", f"KES {unpaid.sum() / 1e6:,.0f} m unpaid", tn[3])
    idx = (yr[["revenue", "units", "price_per_unit"]] / yr[["revenue", "units", "price_per_unit"]].iloc[0] * 100).reset_index()
    idx = idx.rename(columns={"revenue": "Revenue", "units": "Units sold", "price_per_unit": "Net price per unit"}).melt("year", var_name="measure", value_name="index")
    spread = pd.DataFrame({"driver": ["Discount level", "Product category", "Branch"],
                           "points": [abs(dm["margin"].iloc[-1] - dm["margin"].iloc[0]), cm["margin"].max() - cm["margin"].min(),
                                      br["margin"].max() - br["margin"].min()]})
    c = st.columns(3)
    c[0].altair_chart(alt.Chart(idx).mark_line(point=True).encode(
        x=alt.X("year:O", title=None), y=alt.Y("index:Q", title=f"Index ({int(yr.index[0])} = 100)"),
        color=alt.Color("measure:N", scale=alt.Scale(range=[c3, c1, grey]), title=None, legend=alt.Legend(orient="bottom")),
        tooltip=["year", "measure", alt.Tooltip("index:Q", format=".0f")]).properties(height=280, title="Revenue = volume × price"), width="stretch")
    c[1].altair_chart(alt.Chart(dm.reset_index()).mark_bar(color=c3).encode(
        x=alt.X("discount_pct:O", title="Discount (%)"), y=alt.Y("margin:Q", title="Gross margin (%)", scale=alt.Scale(zero=False)),
        tooltip=["discount_pct", alt.Tooltip("margin:Q", format=".1f"), alt.Tooltip("units:Q", title="median units")])
        .properties(height=280, title="Margin by discount level"), width="stretch")
    c[2].altair_chart(alt.Chart(spread).mark_bar(color=c1).encode(
        y=alt.Y("driver:N", sort="-x", title=None), x=alt.X("points:Q", title="Margin spread (points)"),
        tooltip=["driver", alt.Tooltip("points:Q", format=".1f")]).properties(height=280, title="What moves the margin"), width="stretch")

    # ---- E. inventory
    st.markdown("**E. Stock drivers**")
    m = moves.rename(columns=str.lower)
    pol = policy.rename(columns=str.lower)
    flow = pd.DataFrame({"received": m[m["quantity_change"] > 0].groupby("category" if "category" in m else "product_category")["quantity_change"].sum(),
                         "sold": d.groupby("product_category")["quantity"].sum()})
    flow["ratio"] = flow["received"] / flow["sold"]
    stat = pol.groupby("status").agg(items=("product_id", "size"), weekly_demand=("avg_weekly", "median"),
                                     days_cover=("days_of_cover", "median")).reset_index()
    ro = stat.set_index("status").get("weekly_demand", pd.Series()).get("Reorder now", np.nan)
    ov = stat.set_index("status").loc["Overstock"] if "Overstock" in set(stat["status"]) else None
    k = st.columns(3)
    kpi(k[0], "", f"{flow['ratio'].idxmax()} received / sold", f"{flow['ratio'].max():.1f}×",
        f"others {flow.drop(flow['ratio'].idxmax())['ratio'].min():.1f}–{flow.drop(flow['ratio'].idxmax())['ratio'].max():.1f}×", tn[2])
    kpi(k[1], "", "Overstock: days of cover", f"{ov['days_cover']:,.0f}" if ov is not None else "–", "median per item", tn[5])
    kpi(k[2], "", "Weekly demand: reorder vs overstock", f"{ro:.1f} vs {ov['weekly_demand']:.1f}" if ov is not None else "–",
        "median units per week", tn[0])
    c = st.columns(2)
    c[0].altair_chart(alt.Chart(flow.reset_index(names="category")).mark_bar().encode(
        y=alt.Y("category:N", sort="-x", title=None), x=alt.X("ratio:Q", title="Units received per unit sold"),
        color=alt.condition(alt.datum.ratio > 3, alt.value(c3), alt.value(c1)),
        tooltip=["category", alt.Tooltip("ratio:Q", format=".2f"), alt.Tooltip("received:Q", format=","), alt.Tooltip("sold:Q", format=",")])
        .properties(height=260, title="Receipts vs sales by category"), width="stretch")
    c[1].altair_chart(alt.Chart(stat).mark_bar().encode(
        x=alt.X("status:N", sort=["Reorder now", "OK", "Overstock", "No recent demand"], title=None),
        y=alt.Y("weekly_demand:Q", title="Median weekly demand (units)"),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Reorder now", "OK", "Overstock", "No recent demand"], range=[c3, c1, c2, grey]), legend=None),
        tooltip=["status", "items", alt.Tooltip("weekly_demand:Q", format=".1f"), alt.Tooltip("days_cover:Q", format=",.0f")])
        .properties(height=260, title="Stock-outs hit fast movers"), width="stretch")

    # ---- F. value of acting
    st.markdown("**F. Value of acting (estimates)**")
    years = df["year"].nunique()
    crit = df[df["maintenance_status"] == "Critical"]
    old = df[df["equipment_age_years"] > 10]
    young_rate = df.loc[df["equipment_age_years"] <= 10, target].mean()
    novice = df[df["operator_experience_years"] <= 2]
    exp_rate = df.loc[df["operator_experience_years"] > 2, target].mean()
    lever = pd.DataFrame({"action": ["Renew / overhaul 10+ yr equipment", "Mentor 0–2 yr operators", "No runs on Critical equipment"],
                          "per_year": [len(old) * (old[target].mean() - young_rate) / years,
                                       len(novice) * (novice[target].mean() - exp_rate) / years,
                                       len(crit) * (crit[target].mean() - rate["Current"]) / years]})
    last12 = d[d["month"] > d["month"].max() - pd.DateOffset(months=12)]
    cap = last12[last12["discount_pct"] > 5]
    disc_gain = (cap["gross_amount_kes"] * (cap["discount_pct"] - 5) / 100).sum()  # profit kept if discounts capped at 5%
    last_cost = m.groupby("product_id")["unit_cost_kes"].last()
    excess_units = (pol["stock_now"] - 180 * pol["avg_weekly"] / 7).clip(lower=0)
    excess_value = (excess_units * pol["product_id"].map(last_cost)).sum()  # stock value beyond 180 days of cover
    old_debt = d.loc[(d["balance_due_kes"] > 0) & (d["age_days"] > 365), "balance_due_kes"].sum()
    c = st.columns([2, 1, 1, 1])
    c[0].altair_chart(alt.Chart(lever).mark_bar(color=c1).encode(
        y=alt.Y("action:N", sort="-x", title=None), x=alt.X("per_year:Q", title="Failed batches avoided per year"),
        tooltip=["action", alt.Tooltip("per_year:Q", format=".1f")]).properties(height=200, title="Quality levers"), width="stretch")
    kpi(c[1], "", "Cap discounts at 5%", f"KES {disc_gain / 1e6:,.1f} m", "gross profit a year", tn[1])
    kpi(c[2], "", "Stock to 180 days", f"KES {excess_value / 1e9:,.1f} bn", "cash released", tn[4])
    kpi(c[3], "", "Collect 1 yr+ debt", f"KES {old_debt / 1e6:,.0f} m", "cash", tn[3])
    st.caption("Estimates; quality levers overlap and are not additive.")



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
st.markdown('''<div class="banner"><h1>Pharmaceutical Operations Intelligence System</h1>
<p>Individual capstone (POI99999) · production · maintenance · sales · inventory — LAL data, 2007–2026</p></div>''',
            unsafe_allow_html=True)                                         # page banner
NAMES = ["Overview", "Data cleaning", "Sales", "Batch quality", "Failure causes", "Equipment", "Models",
         "What-if predictor", "Demand forecast", "Inventory", "Commercial & finance", "Executive summary"]
T = dict(zip(NAMES, st.tabs(NAMES)))                                    # tabs addressed by name  # same order as the team dashboard
st.sidebar.caption("Model scores: POI99999 test sets · other figures: live from the CSVs.")

# ------------------------------------------------------------------ overview
with T["Overview"]:                                                   # Overview
    fr = data["failed"].mean()                                              # failure rate
    pol = reorder_policy(moves)                                             # stock status
    c = st.columns(5)
    kpi(c[0], "", "Batches", f"{len(data):,}", f"{1 - fr:.1%} passed QC", "teal", bar=(1 - fr) * 100)
    kpi(c[1], "", "Failed batches", f"{data['failed'].sum():,}", f"{fr:.2%} failure rate", "coral")
    kpi(c[2], "", "Net sales", f"KES {valid['net_amount_excl_vat_kes'].sum() / 1e9:.1f} bn", f"{len(valid):,} delivered orders", "green")
    kpi(c[3], "", "Unpaid balances", f"KES {sales['balance_due_kes'].sum() / 1e6:,.0f} m", "all on credit accounts", "violet")
    kpi(c[4], "", "Overstocked items", f"{(pol['status'] == 'Overstock').sum():,}", f"of {len(pol):,} branch-product items", "gold",
        bar=(pol["status"] == "Overstock").mean() * 100)
    left, right = st.columns([3, 2])
    left.subheader("Key findings")
    left.dataframe(pd.DataFrame([["Batches passing QC", "96.9%"], ["Top 3 causes' share of failures", "70%"],
                                 ["Failure rate: Critical vs Current", "6.9% vs 3.1%"], ["Failure model PR-AUC", "0.047 (weak)"],
                                 ["Yield model vs average", "22% better"], ["Overstocked items", "1,033 of 1,200"],
                                 ["Stock-outs caught", "80%"]], columns=["Finding", "Value"]), hide_index=True, width="stretch")
    yr = data.groupby("year").agg(batches=("batch_id", "size"), failure_rate=("failed", "mean")).reset_index()
    yr["failure_rate"] *= 100
    right.subheader("Production and failure rate by year")
    base = alt.Chart(yr).encode(x=alt.X("year:O", title=None))
    right.altair_chart(alt.layer(base.mark_bar(color=TEAL, opacity=0.6).encode(y=alt.Y("batches:Q", title="Batches")),
                                 base.mark_line(color=CORAL, point=True).encode(y=alt.Y("failure_rate:Q", title="Failure rate (%)")))
                       .resolve_scale(y="independent").properties(height=300), width="stretch")

# ------------------------------------------------------------------ data cleaning
with T["Data cleaning"]:                                                   # Data cleaning
    c = st.columns(4)
    kpi(c[0], "", "Production + maintenance", "60,000", "batches joined, 0 duplicates", "teal")
    kpi(c[1], "", "Sales", f"{len(sales):,}", f"{sales['is_cancelled'].sum():,} cancelled (4.0%)", "navy")
    kpi(c[2], "", "Stock movements", f"{len(moves) + sales['is_cancelled'].sum():,}", "12,000 receipts + 150,000 sale-outs", "violet")
    kpi(c[3], "", "Missing cells after cleaning", "0", "means for numbers, mode for text", "green")
    st.subheader("Missing values before cleaning (production file)")
    m = missing[missing > 0].drop(labels=["rejection_reason"], errors="ignore").sort_values().reset_index()
    m.columns = ["column", "missing"]
    st.altair_chart(alt.Chart(m).mark_bar(color=TEAL).encode(y=alt.Y("column:N", sort="-x", title=None), x="missing:Q",
                                                             tooltip=["column", "missing"]).properties(height=280), width="stretch")
    st.caption("rejection_reason excluded (empty for passed batches).")
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
    st.caption(f"Overall: {(moves['stock_on_hand'] < 0).mean():.1%}")

# ------------------------------------------------------------------ EDA production
with T["Batch quality"]:                                                   # Batch quality
    yrs = st.sidebar.slider("Production years (Batch quality tab)", 2007, 2026, (2007, 2026))  # year filter
    d = data[data["year"].between(*yrs)]                                    # filtered batches
    c = st.columns(4)
    kpi(c[0], "", "Batches", f"{len(d):,}", f"{yrs[0]}–{yrs[1]}", "teal")
    kpi(c[1], "", "Failure rate", f"{d['failed'].mean():.2%}", f"{d['failed'].sum():,} failed", "coral")
    kpi(c[2], "", "Average yield", f"{d['actual_yield'].mean():.2f}%", "actual yield", "navy")
    kpi(c[3], "", "Average purity", f"{d['assay_result_purity'].mean():.2f}%", "assay result", "violet")
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
with T["Sales"]:                                                   # Sales
    c = st.columns(4)
    kpi(c[0], "", "Net sales", f"KES {valid['net_amount_excl_vat_kes'].sum() / 1e9:.1f} bn", "2014–2025", "green")
    kpi(c[1], "", "Orders delivered", f"{len(valid):,}", f"{sales['is_cancelled'].mean():.1%} cancelled", "navy")
    rec_u, sold_u = moves.loc[moves["quantity_change"] > 0, "quantity_change"].sum(), valid["quantity"].sum()
    kpi(c[2], "", "Units received", f"{rec_u / 1e6:.2f} m", f"{rec_u / sold_u:.1f}× units sold", "gold")
    kpi(c[3], "", "Branch BR01", f"{(valid['branch_code'] == 'BR01').mean():.0%}", "of delivered orders", "violet",
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
with T["Models"]:                                                   # Models
    prob = st.selectbox("Problem", list(RESULTS))                           # pick a business question
    r = RESULTS[prob]
    t, metric = r["table"], r["metric"]
    best_score = t.set_index("Model").loc[r["best"], metric]
    base = t[t["Model"].str.startswith("Baseline")]                         # baseline row, if any
    c = st.columns(3)
    kpi(c[0], "", "Chosen model", r["best"], f"decided on {metric}", "teal")
    kpi(c[1], "", metric, f"{best_score:,.3f}", "test-set score", "navy")
    if len(base):
        b = base[metric].iloc[0]                                            # baseline score
        kpi(c[2], "", "Gain over baseline", f"{GAINS[prob]:,.1f}%", f"baseline {b:,.3f}", "green")
    else:
        kpi(c[2], "", "Models compared", f"{len(t)}", "no baseline for clustering", "violet")
    note(f"<b>Why {r['best']}:</b> {r['why']}")
    st.altair_chart(alt.Chart(t).mark_bar().encode(
        y=alt.Y("Model:N", sort="-x" if metric not in LOWER_IS_BETTER else "x", title=None), x=alt.X(f"{metric}:Q"),
        color=alt.condition(alt.datum.Model == r["best"], alt.value(TEAL), alt.value(GREY)),
        tooltip=list(t.columns)).properties(height=60 + 45 * len(t), title=f"Model comparison on {metric} (chosen in teal)"), width="stretch")
    st.dataframe(t, hide_index=True, width="stretch")

# ------------------------------------------------------------------ failure causes
with T["Failure causes"]:                                                   # Failure causes
    f = data[data["failed"] == 1]                                           # failed batches
    themes = pd.DataFrame({"Cause (TF-IDF + K-Means theme)": ["Assay out of specification", "Dissolution failure", "Weight variation",
                                                               "Microbial limit", "Related substances", "Blend uniformity"],
                           "Failed batches": [699, 341, 277, 217, 178, 163]})
    themes["Share (%)"] = (themes["Failed batches"] / themes["Failed batches"].sum() * 100).round(1)
    themes["Cumulative (%)"] = themes["Share (%)"].cumsum().round(1)
    c = st.columns(3)
    kpi(c[0], "", "Largest cause", "Assay", "37% of failures", "coral", bar=37.3)
    kpi(c[1], "", "Top 3 causes", "70%", "assay, dissolution, weight", "gold", bar=70.2)
    kpi(c[2], "", "Oral Solids", f"{(f['manufacturing_section'] == 'Oral Solids').mean():.0%}", "of failed batches", "navy",
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
with T["Inventory"]:                                                   # Inventory
    c = st.columns(3)
    lead = c[0].slider("Lead time (days)", 7, 60, 14)                       # assumption: no lead-time field in the data
    z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[c[1].select_slider("Service level", ["90%", "95%", "99%"], value="95%")]
    over = c[2].slider("Overstock limit (days of cover)", 60, 365, 180)
    pol = reorder_policy(moves, lead, z, over)                              # recomputed with the chosen assumptions
    k = st.columns(4)
    for col, (s, icon, tone) in zip(k, [("Reorder now", "", "coral"), ("Overstock", "", "gold"),
                                        ("No recent demand", "", "violet"), ("OK", "", "green")]):
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
with T["Equipment"]:                                                   # Equipment
    rate = data.groupby("maintenance_status")["failed"].mean().mul(100)
    c = st.columns(3)
    kpi(c[0], "", "Critical", f"{rate.get('Critical', 0):.1f}%", f"failure rate · {(data['maintenance_status'] == 'Critical').sum()} batches", "coral")
    kpi(c[1], "", "Due Soon", f"{rate.get('Due Soon', 0):.1f}%", "failure rate", "gold")
    kpi(c[2], "", "Current", f"{rate.get('Current', 0):.1f}%", "failure rate", "green")
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
    st.caption("Screening list, not a probability.")

# ------------------------------------------------------------------ executive summary
# ------------------------------------------------------------------ forecasting
with T["Demand forecast"]:                                                   # Demand forecast
    st.subheader("Monthly demand forecast (units)")
    cats = ["Total"] + sorted(valid["product_category"].unique())
    choice = st.selectbox("Series", cats)                                   # pick a category
    vv = valid if choice == "Total" else valid[valid["product_category"] == choice]
    y = vv.groupby(vv["sale_date"].dt.to_period("M").dt.to_timestamp())["quantity"].sum().asfreq("MS", fill_value=0)  # units per month
    fc, acc = backtest(y)                                                   # accuracy on the last 12 months
    future = hw_forecast(y, 12)                                             # next 12 months
    growth = future.sum() / y.iloc[-12:].sum() - 1
    c = st.columns(4)
    kpi(c[0], "", "Next 12 months", f"{future.sum():,.0f}", "units forecast", "teal")
    kpi(c[1], "", "Last 12 months", f"{y.iloc[-12:].sum():,.0f}", "units sold", "navy")
    kpi(c[2], "" if growth >= 0 else "", "Expected change", f"{growth:+.1%}", "next year vs last year", "green" if growth >= 0 else "coral")
    kpi(c[3], "", "Forecast error", f"{acc.loc['Holt-Winters', 'WAPE (%)']:.1f}%",
        f"WAPE vs {acc.loc['Seasonal naive', 'WAPE (%)']:.1f}% seasonal naive", "violet")
    st.line_chart(pd.DataFrame({"Actual": y, "Forecast (Holt-Winters)": future}), color=[NAVY, TEAL])
    c = st.columns(2)
    c[0].markdown("**Back-test on the last 12 months**")
    c[0].line_chart(fc)
    c[1].markdown("**Accuracy of each method**")
    c[1].dataframe(acc, width="stretch")
    c[1].markdown("**Monthly forecast (units)**")
    c[1].dataframe(future.round(0).rename("units").to_frame(), width="stretch")
    st.download_button("Download forecast (CSV)", future.rename("units").to_csv(), "demand_forecast.csv")
    st.subheader("Forecasting models from the notebook")
    c = st.columns(2)
    for col, key in zip(c, ["7. What will weekly demand be?", "4. Are monthly quality indicators deteriorating?"]):
        r = RESULTS[key]
        col.markdown(f"**{key}** — chosen: **{r['best']}**")
        col.dataframe(r["table"], hide_index=True, width="stretch")
        col.caption(r["why"])

# ------------------------------------------------------------------ what-if predictor
with T["What-if predictor"]:
    predictor_tab(data, "failed", "poi", (TEAL, GOLD, CORAL, GREY), ["teal", "green", "coral", "violet", "navy", "gold"], sales, moves, reorder_policy(moves))

# ------------------------------------------------------------------ commercial and finance
with T["Commercial & finance"]:                                                   # Commercial & finance
    finance_page(sales, moves, (TEAL, NAVY, CORAL, GREY), ["green", "teal", "coral", "violet", "navy", "gold"])

with T["Executive summary"]:                                                  # Executive summary
    rate = data.groupby("maintenance_status")["failed"].mean().mul(100)    # failure rate per equipment status
    pol = reorder_policy(moves)                                             # stock status
    c = st.columns(5)
    kpi(c[0], "", "Batches passed", f"{1 - data['failed'].mean():.1%}", f"{len(data):,} batches · {data['failed'].sum():,} failed", "teal",
        bar=(1 - data["failed"].mean()) * 100)
    kpi(c[1], "", "Critical vs Current", f"{rate['Critical']:.1f}% vs {rate['Current']:.1f}%", "failure rate", "coral")
    kpi(c[2], "", "Top 3 failure causes", "70%", "assay, dissolution, weight", "gold", bar=70)
    kpi(c[3], "", "Questions answered", "6 of 9", "useful model found", "green", bar=6 / 9 * 100)
    kpi(c[4], "", "Reorder / overstock", f"{(pol['status'] == 'Reorder now').sum()} / {(pol['status'] == 'Overstock').sum()}",
        f"of {len(pol):,} items", "violet", bar=(pol["status"] == "Overstock").mean() * 100)
    exec_commercial(sales, moves, ["green", "teal", "gold", "violet"])     # revenue, forecast, receivables, stock value

    st.subheader("Questions and answers")
    st.dataframe(pd.DataFrame([
        ["1. Batch fails QC?", "Logistic regression", "PR-AUC 0.047 vs 0.031 · recall 51.5%", "Weak"],
        ["2. Batch yield?", "Random forest", "MAE 0.858 vs 1.102 · R² 0.33", "Yes"],
        ["3. Maintenance state?", "Random forest", "Macro F1 0.557 · 69% of Critical found", "Yes"],
        ["4. Quality trend?", "Linear regression", "WAPE 7.0% · 20.6% gain", "Yes"],
        ["5. Abnormal conditions?", "Z-score rule", "PR-AUC 0.036 · lift 1.4", "Weak"],
        ["6. Failure causes?", "TF-IDF + K-Means (k=6)", "Silhouette 0.788 · top 3 = 70%", "Yes"],
        ["7. Weekly demand?", "Croston-SBA", "WAPE 115% · 19.7% gain · bias −5.9%", "Yes (totals)"],
        ["8. Stock-outs in 14 days?", "Gradient boosting", "PR-AUC 0.278 · recall 0.80", "Yes"],
        ["9. Product groups?", "K-Means (k=2)", "Silhouette 0.471 · 250 fast / 50 slow", "Yes"],
    ], columns=["Question", "Model", "Key value", "Answer"]), hide_index=True, width="stretch")

    exec_drivers(data, "failed", "poi", (TEAL, GOLD, CORAL, GREY), ["teal", "green", "coral", "violet", "navy", "gold"],
                 sales, moves, reorder_policy(moves))

    st.subheader("Model results")
    rows = []
    for name, r in RESULTS.items():                                         # one row per problem
        t = r["table"].set_index("Model")[r["metric"]]
        b = [m for m in t.index if m.startswith("Baseline")]
        rows.append([name, r["best"], r["metric"], t[r["best"]], t[b[0]] if b else None, GAINS.get(name)])
    score = pd.DataFrame(rows, columns=["Problem", "Chosen model", "Metric", "Score", "Baseline", "Gain over baseline (%)"])
    c = st.columns(2)
    g = score.dropna(subset=["Gain over baseline (%)"])
    g = g[g["Gain over baseline (%)"] < 1000]                               # stock-out gain (about 206x) left out of the chart
    g = g.assign(level=np.where(g["Gain over baseline (%)"] >= 15, "15%+ better", "under 15%"))
    c[0].altair_chart(alt.Chart(g).mark_bar().encode(
        y=alt.Y("Problem:N", sort="-x", title=None), x=alt.X("Gain over baseline (%):Q"),
        color=alt.Color("level:N", scale=alt.Scale(domain=["15%+ better", "under 15%"], range=[TEAL, GOLD]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Problem", "Chosen model", "Gain over baseline (%)"]).properties(height=280, title="Chosen model vs baseline (stock-out: 206×)"),
        width="stretch")
    c[1].dataframe(score[["Problem", "Chosen model", "Metric", "Score"]].round(3), hide_index=True, width="stretch", height=280)
    c = st.columns(3)
    themes = pd.DataFrame({"cause": ["Assay", "Dissolution", "Weight variation", "Microbial limit", "Related substances", "Blend uniformity"],
                           "share": [37.3, 18.2, 14.8, 11.6, 9.5, 8.7]})
    themes["cumulative"] = themes["share"].cumsum()
    base = alt.Chart(themes).encode(x=alt.X("cause:N", sort=None, title=None))
    c[0].altair_chart(alt.layer(base.mark_bar(color=CORAL).encode(y=alt.Y("share:Q", title="Share of failures (%)")),
                                base.mark_line(color=NAVY, point=True).encode(y=alt.Y("cumulative:Q", title="Cumulative %", scale=alt.Scale(domain=[0, 100]))))
                      .resolve_scale(y="independent").properties(height=260, title="Failure causes: top 3 = 70%"), width="stretch")
    dm = pd.DataFrame({"Model": ["Seasonal-naive", "Moving average", "Croston-SBA", "Linear Regression"] * 2,
                       "Level": ["Product-week"] * 4 + ["Company total"] * 4,
                       "WAPE (%)": [143.2, 120.2, 115.0, 104.4, 16.5, 11.3, 12.4, 29.1]})
    c[1].altair_chart(alt.Chart(dm).mark_bar().encode(
        x=alt.X("Model:N", sort=None, title=None), y=alt.Y("WAPE (%):Q"), xOffset="Level:N",
        color=alt.Color("Level:N", scale=alt.Scale(range=[NAVY, TEAL]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "Level", "WAPE (%)"]).properties(height=260, title="Weekly demand error"), width="stretch")
    c[2].altair_chart(alt.Chart(pol["status"].value_counts().reset_index(name="items")).mark_arc(innerRadius=55).encode(
        theta="items:Q", color=alt.Color("status:N", scale=alt.Scale(domain=["Overstock", "Reorder now", "No recent demand", "OK"],
                                                                     range=[GOLD, CORAL, GREY, TEAL]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["status", "items"]).properties(height=260, title=f"Stock status · KES {pol['excess_value_kes'].sum() / 1e9:,.1f} bn excess"),
        width="stretch")

    c = st.columns(2)
    c[0].subheader("How we handled it")
    c[0].dataframe(pd.DataFrame([
        ["3.1% failure rate", "PR-AUC, recall, macro F1 · baselines"], ["Weak failure model", "Ranking only, never release"],
        ["Biased forecasts", "Bias ≤ ±10% rule"], ["Future leakage", "Time-ordered test sets"],
        ["Rare stock-outs (0.14%)", "Threshold for recall ≥ 0.80"], ["Free-text reasons", "TF-IDF + K-Means"],
        ["Cancelled orders", "6,045 removed · stock rebuilt"], ["Outliers", "Flagged, kept"], ["No lead times", "14 days assumed"],
    ], columns=["Issue", "Handling"]), hide_index=True, width="stretch")
    c[1].subheader("Recommendations")
    c[1].dataframe(pd.DataFrame([
        ["High", "QA", "CAPA on top 3 causes, Oral Solids first", "Failure rate < 3%"],
        ["High", "Engineering", "Service before Critical", "0 Critical runs"],
        ["High", "Procurement", "Buy to reorder point", "Cover < 180 days"],
        ["Medium", "Stores", "Weekly stock-out list", "Fewer stock-outs"],
        ["Medium", "Planning", "Croston weekly plan", "Error tracked"],
        ["Medium", "Finance", "Credit terms and collections", "Unpaid share down"],
        ["Low", "QA", "Risk ranking for extra checks", "Failures per check"],
    ], columns=["Priority", "Owner", "Action", "Target"]), hide_index=True, width="stretch")
    st.caption("Limits: no opening stock, expiry or lead-time data · batches not linked to sales · single test sets · no savings claimed.")
