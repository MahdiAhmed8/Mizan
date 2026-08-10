"""Streamlit user interface for the Tunisia Profit Simulator."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulator import (
    BusinessInputs,
    UncertaintyInputs,
    calculate_metrics,
    inputs_as_dict,
    run_monte_carlo,
    scenario_summary,
)


st.set_page_config(
    page_title="Mizan | Tunisia Profit Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
  --ink:#e7f7ff; --muted:#86a4b4; --navy:#050d18; --panel:#0a1725;
  --blue:#25a7ff; --cyan:#30e1ff; --green:#27e09c; --line:rgba(80,202,255,.17);
}
.stApp {
  background:
    linear-gradient(rgba(37,167,255,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(37,167,255,.035) 1px,transparent 1px),
    radial-gradient(circle at 82% 2%,rgba(37,167,255,.16),transparent 31rem),
    radial-gradient(circle at 5% 42%,rgba(39,224,156,.09),transparent 26rem),
    #050d18;
  background-size:32px 32px,32px 32px,auto,auto,auto;
  color:var(--ink);
}
.block-container { max-width:1460px; padding-top:2rem; padding-bottom:1.5rem; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
h1, h2, h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:var(--ink); }
h2, h3 { margin-top:1.25rem; }
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 20% 5%,rgba(39,224,156,.08),transparent 18rem),
    linear-gradient(180deg,rgba(8,21,34,.99),rgba(5,14,24,.99));
  border-right:1px solid var(--line);
  box-shadow:12px 0 40px rgba(0,0,0,.25);
}
[data-testid="stSidebar"] h2 { color:var(--green); font-size:1.65rem; text-transform:uppercase; letter-spacing:.03em; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background:rgba(14,32,49,.72); border:1px solid var(--line);
  border-radius:10px; overflow:hidden; margin-bottom:.5rem;
}
[data-testid="stMetric"] {
  position:relative; overflow:hidden; min-height:128px;
  background:linear-gradient(145deg,rgba(13,33,51,.95),rgba(7,20,33,.96));
  border:1px solid var(--line); border-top:2px solid var(--cyan);
  padding:19px 20px; border-radius:14px;
  box-shadow:0 12px 32px rgba(0,0,0,.24),inset 0 1px 0 rgba(255,255,255,.025);
  transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;
}
[data-testid="stMetric"]:hover { transform:translateY(-3px); border-color:rgba(48,225,255,.42); box-shadow:0 15px 38px rgba(0,0,0,.32),0 0 24px rgba(37,167,255,.08); }
[data-testid="stMetric"]:after {
  content:""; position:absolute; width:80px; height:80px; right:-32px; top:-36px;
  border:1px solid rgba(48,225,255,.18); transform:rotate(45deg); background:rgba(37,167,255,.035);
}
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:700; letter-spacing:.01em; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; color:#f2fbff; letter-spacing:-.04em; }
.hero {
  position:relative; overflow:hidden; display:grid; grid-template-columns:minmax(0,1fr) 210px;
  align-items:center; gap:24px; padding:38px 42px; border-radius:18px; color:white; margin-bottom:26px;
  background:
    linear-gradient(rgba(48,225,255,.05) 1px,transparent 1px),
    linear-gradient(90deg,rgba(48,225,255,.05) 1px,transparent 1px),
    radial-gradient(circle at 84% 45%,rgba(39,224,156,.16),transparent 15rem),
    linear-gradient(125deg,#081625 0%,#0a2032 58%,#07332f 125%);
  background-size:22px 22px,22px 22px,auto,auto;
  border:1px solid rgba(48,225,255,.24);
  box-shadow:0 20px 58px rgba(0,0,0,.34),inset 0 0 60px rgba(37,167,255,.035);
}
.hero:after { content:""; position:absolute; left:0; right:0; top:0; height:1px; background:linear-gradient(90deg,transparent,var(--cyan),var(--green),transparent); box-shadow:0 0 18px var(--cyan); }
.hero-copy { position:relative; z-index:2; }
.hero-kicker { display:flex; align-items:center; gap:9px; text-transform:uppercase; letter-spacing:.19em; font-family:'Space Grotesk',sans-serif; font-size:.69rem; font-weight:700; color:var(--green); }
.hero-kicker:before { content:""; width:7px; height:7px; background:var(--green); border-radius:50%; box-shadow:0 0 12px var(--green); }
.hero h1 { color:#f2fbff; margin:.5rem 0 .7rem; font-size:clamp(2.2rem,4vw,3.45rem); line-height:1.02; max-width:780px; text-shadow:0 0 30px rgba(37,167,255,.13); }
.hero h1 span { color:var(--cyan); }
.hero p { max-width:720px; color:#a9c2ce; margin:0; font-size:1.03rem; line-height:1.65; }
.hero-badge { display:inline-flex; align-items:center; gap:8px; margin-top:19px; padding:8px 12px; border-radius:8px; background:rgba(37,167,255,.08); border:1px solid rgba(48,225,255,.20); color:#c7f6ff; font-family:'Space Grotesk',sans-serif; font-size:.75rem; font-weight:700; letter-spacing:.04em; }
.hero-badge:before { content:""; width:6px; height:6px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); }
.radar { position:relative; z-index:2; width:174px; height:174px; margin:auto; border-radius:50%; border:1px solid rgba(48,225,255,.38); box-shadow:inset 0 0 32px rgba(37,167,255,.11),0 0 24px rgba(37,167,255,.08); background:repeating-radial-gradient(circle,transparent 0 26px,rgba(48,225,255,.15) 27px 28px),linear-gradient(90deg,transparent 49.5%,rgba(48,225,255,.16) 50%,transparent 50.5%),linear-gradient(transparent 49.5%,rgba(48,225,255,.16) 50%,transparent 50.5%); }
.radar:before { content:""; position:absolute; inset:8px; border-radius:50%; background:conic-gradient(from 18deg,rgba(39,224,156,.32),transparent 22%,transparent); animation:sweep 4s linear infinite; }
.radar:after { content:""; position:absolute; width:9px; height:9px; border-radius:50%; right:38px; top:48px; background:var(--green); box-shadow:0 0 13px var(--green); }
@keyframes sweep { to { transform:rotate(360deg); } }
@media (prefers-reduced-motion:reduce) { .radar:before { animation:none; } [data-testid="stMetric"] { transition:none; } }
.section-note { color:var(--muted); margin-top:-8px; }
.status-positive,.status-negative { padding:15px 18px; border-radius:10px; font-weight:700; margin:10px 0 22px; }
.status-positive { background:linear-gradient(90deg,rgba(39,224,156,.13),rgba(39,224,156,.04)); color:#72f7c3; border:1px solid rgba(39,224,156,.28); }
.status-negative { background:linear-gradient(90deg,rgba(255,100,116,.13),rgba(255,100,116,.04)); color:#ff9aa5; border:1px solid rgba(255,100,116,.25); }
div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:7px; background:rgba(8,23,37,.78); padding:6px; border-radius:12px; border:1px solid var(--line); }
div[data-testid="stTabs"] button { height:42px; padding:0 18px; border-radius:11px; font-weight:700; color:var(--muted); }
div[data-testid="stTabs"] button[aria-selected="true"] { background:linear-gradient(110deg,rgba(37,167,255,.22),rgba(39,224,156,.14)); color:#eaffff; border:1px solid rgba(48,225,255,.25); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],div[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }
.stButton>button,.stDownloadButton>button { border-radius:9px; border:1px solid rgba(48,225,255,.35); background:linear-gradient(100deg,#0875bc,#079b82); color:white; font-family:'Space Grotesk',sans-serif; font-weight:700; box-shadow:0 7px 22px rgba(0,143,204,.18); transition:transform .18s ease,box-shadow .18s ease; }
.stButton>button:hover,.stDownloadButton>button:hover { color:white; border-color:var(--cyan); transform:translateY(-1px); box-shadow:0 10px 28px rgba(0,173,210,.28); }
div[data-baseweb="slider"] [role="slider"] { background:var(--green); border-color:#071521; box-shadow:0 0 0 2px var(--green),0 0 12px rgba(39,224,156,.45); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:0 10px 28px rgba(0,0,0,.20); }
[data-testid="stAlert"] { border-radius:10px; border-color:var(--line); }
.footer { color:#587889; text-align:center; padding:34px 0 12px; font-size:.79rem; letter-spacing:.04em; text-transform:uppercase; }
@media (max-width:800px) {
  .block-container { padding-top:1rem; }
  .hero { grid-template-columns:1fr; padding:28px 25px; border-radius:14px; }
  .radar { display:none; }
  div[data-testid="stTabs"] button { padding:0 10px; font-size:.78rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def tnd(value: float, decimals: int = 0) -> str:
    if not np.isfinite(value):
        return "Not achievable"
    return f"{value:,.{decimals}f} TND"


def integer(value: float) -> str:
    return "Not achievable" if not np.isfinite(value) else f"{math.ceil(value):,}"


def style_chart(figure: go.Figure) -> None:
    """Apply the dashboard's technical dark theme to a Plotly chart."""
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,33,.52)",
        font=dict(color="#9ab5c3", family="Inter"),
        title_font=dict(color="#e7f7ff", family="Space Grotesk"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    figure.update_xaxes(gridcolor="rgba(80,202,255,.09)", zerolinecolor="rgba(80,202,255,.18)")
    figure.update_yaxes(gridcolor="rgba(80,202,255,.09)", zerolinecolor="rgba(80,202,255,.18)")


with st.sidebar:
    st.markdown("## ◈ MIZAN")
    st.caption("Monthly business assumptions")
    with st.expander("Sales & demand", expanded=True):
        price = st.slider("Product price (TND)", 1.0, 500.0, 25.0, 0.5)
        customers = st.slider("Customers per month", 0, 5_000, 450, 10)
        units_per_customer = st.slider("Average units per customer", 0.1, 10.0, 1.2, 0.1)
        seasonal_demand = st.slider(
            "Seasonal demand adjustment", -80, 100, 0, 5, format="%d%%",
            help="Use a negative value for a quiet month and a positive value for a peak month.",
        )
    with st.expander("Costs", expanded=True):
        material_cost = st.slider("Material cost per unit (TND)", 0.0, 400.0, 8.0, 0.5)
        rent = st.slider("Monthly rent (TND)", 0, 20_000, 1_500, 100)
        employees = st.slider("Number of employees", 0, 30, 2)
        salary = st.slider("Salary per employee (TND)", 0, 10_000, 900, 50)
        marketing = st.slider("Marketing budget (TND)", 0, 20_000, 600, 100)
        other_costs = st.slider("Other fixed costs (TND)", 0, 20_000, 350, 50)
    with st.expander("Tax & goal", expanded=False):
        tax_rate = st.slider("Tax on positive profit", 0, 50, 15, 1, format="%d%%")
        profit_target = st.slider("After-tax profit target (TND)", 0, 50_000, 3_000, 250)
    st.divider()
    st.caption("Uncertainty ranges (± from the base case)")
    demand_uncertainty = st.slider("Demand uncertainty", 0, 80, 20, 5, format="±%d%%")
    price_uncertainty = st.slider("Price uncertainty", 0, 40, 5, 1, format="±%d%%")
    material_uncertainty = st.slider("Material-cost uncertainty", 0, 60, 12, 2, format="±%d%%")
    simulations_count = st.select_slider(
        "Simulation runs", options=[1_000, 2_500, 5_000, 10_000, 20_000], value=5_000
    )
    if "seed" not in st.session_state:
        st.session_state.seed = 42
    if st.button("↻ Resample uncertainty", use_container_width=True):
        st.session_state.seed += 1


inputs = BusinessInputs(
    price=price,
    customers=customers,
    units_per_customer=units_per_customer,
    rent=rent,
    employees=employees,
    salary_per_employee=salary,
    material_cost_per_unit=material_cost,
    marketing_budget=marketing,
    other_fixed_costs=other_costs,
    seasonal_demand_percent=seasonal_demand,
    tax_rate_percent=tax_rate,
    profit_target=profit_target,
)
uncertainty = UncertaintyInputs(
    demand_percent=demand_uncertainty,
    price_percent=price_uncertainty,
    material_cost_percent=material_uncertainty,
    simulations=simulations_count,
    seed=st.session_state.seed,
)
metrics = calculate_metrics(inputs)
simulations = run_monte_carlo(inputs, uncertainty)
scenarios = scenario_summary(simulations)
loss_probability = (simulations["Net profit"] < 0).mean() * 100

st.markdown(
    f"""
<div class="hero">
  <div class="hero-copy">
    <div class="hero-kicker">Tunisian small-business planning</div>
    <h1>Model the future.<br><span>Control the outcome.</span></h1>
    <p>Explore your monthly economics, stress-test uncertainty, and discover the sales level your next profit milestone requires.</p>
    <div class="hero-badge">Live plan · {simulations_count:,} possible months</div>
  </div>
  <div class="radar" aria-hidden="true"></div>
</div>
""",
    unsafe_allow_html=True,
)

headline = st.columns(4)
headline[0].metric("Monthly revenue", tnd(metrics["revenue"]), help="Price × units sold")
headline[1].metric(
    "Net monthly cash flow", tnd(metrics["net_profit"]),
    delta=tnd(metrics["net_profit"] - profit_target) + " vs target",
    delta_color="normal",
)
headline[2].metric("Gross margin", f'{metrics["gross_margin_percent"]:.1f}%')
headline[3].metric(
    "Chance of losing money", f"{loss_probability:.1f}%",
    delta="Lower is safer", delta_color="inverse",
)

if metrics["net_profit"] >= profit_target:
    st.markdown(
        f'<div class="status-positive">On track: the base case exceeds your monthly profit target by {tnd(metrics["net_profit"] - profit_target)}.</div>',
        unsafe_allow_html=True,
    )
else:
    gap = profit_target - metrics["net_profit"]
    st.markdown(
        f'<div class="status-negative">Action needed: the base case is {tnd(gap)} below your monthly profit target.</div>',
        unsafe_allow_html=True,
    )

overview_tab, uncertainty_tab, target_tab, details_tab = st.tabs(
    ["Business overview", "Uncertainty", "Profit target", "Calculation details"]
)

with overview_tab:
    st.subheader("Where the money goes")
    st.markdown('<p class="section-note">A monthly profit-and-loss waterfall based on the current inputs.</p>', unsafe_allow_html=True)
    waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
            x=["Revenue", "Materials", "Rent", "Salaries", "Marketing", "Other costs", "Profit before tax"],
            y=[metrics["revenue"], -metrics["material_costs"], -rent, -metrics["salaries"], -marketing, -other_costs, metrics["profit_before_tax"]],
            connector={"line": {"color": "#466576"}},
            increasing={"marker": {"color": "#27e09c"}},
            decreasing={"marker": {"color": "#ff6474"}},
            totals={"marker": {"color": "#25a7ff"}},
            hovertemplate="%{x}<br>%{y:,.0f} TND<extra></extra>",
        )
    )
    waterfall.update_layout(height=440, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="TND / month")
    style_chart(waterfall)
    st.plotly_chart(waterfall, use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Break-even customers", integer(metrics["break_even_customers"]))
    b.metric("Base-case customers", integer(metrics["customers"]))
    c.metric("Contribution per unit", tnd(metrics["contribution_per_unit"], 2))

with uncertainty_tab:
    st.subheader("Profit is a range, not a promise")
    st.markdown(
        '<p class="section-note">The chart combines all selected uncertainty ranges across thousands of possible months.</p>',
        unsafe_allow_html=True,
    )
    histogram = px.histogram(
        simulations, x="Net profit", nbins=55, color_discrete_sequence=["#25a7ff"],
        labels={"Net profit": "Monthly net profit (TND)"},
    )
    histogram.add_vline(x=0, line_dash="dash", line_color="#ff6474", annotation_text="Loss line")
    histogram.add_vline(x=profit_target, line_dash="dot", line_color="#27e09c", annotation_text="Profit target")
    histogram.update_layout(height=420, margin=dict(l=20, r=20, t=25, b=20), bargap=0.04, yaxis_title="Simulated months")
    style_chart(histogram)
    st.plotly_chart(histogram, use_container_width=True)

    st.markdown("#### Scenario range")
    scenario_display = scenarios.copy()
    scenario_display["Revenue"] = scenario_display["Revenue"].map(tnd)
    scenario_display["Net profit"] = scenario_display["Net profit"].map(tnd)
    scenario_display["Customers"] = scenario_display["Customers"].map(lambda x: f"{x:,.0f}")
    st.dataframe(scenario_display, use_container_width=True)
    st.caption("P10 means only 10% of simulated months are lower; P90 means only 10% are higher.")

with target_tab:
    st.subheader("Sales needed to reach the goal")
    st.markdown(
        '<p class="section-note">Required sales include fixed costs, unit economics, and tax on positive profit.</p>',
        unsafe_allow_html=True,
    )
    if metrics["contribution_per_unit"] <= 0:
        st.error("The selling price does not cover the material cost per unit. Raise the price or lower the unit cost before a break-even point can exist.")
    else:
        x, y, z = st.columns(3)
        x.metric("Required customers", integer(metrics["required_customers"]))
        y.metric("Required units", integer(metrics["required_units"]))
        additional = max(0, math.ceil(metrics["required_customers"] - metrics["customers"]))
        z.metric("Additional customers needed", f"{additional:,}")

        max_customers = max(metrics["required_customers"] * 1.35, metrics["customers"] * 1.35, 100)
        customer_range = np.linspace(0, max_customers, 120)
        unit_range = customer_range * units_per_customer
        pretax = unit_range * metrics["contribution_per_unit"] - metrics["fixed_costs"]
        profit_range = pretax - np.maximum(pretax, 0) * tax_rate / 100
        goal_chart = go.Figure()
        goal_chart.add_trace(go.Scatter(x=customer_range, y=profit_range, mode="lines", name="Net profit", line=dict(color="#30e1ff", width=4)))
        goal_chart.add_hline(y=0, line_dash="dash", line_color="#8a9692", annotation_text="Break-even")
        goal_chart.add_hline(y=profit_target, line_dash="dot", line_color="#27e09c", annotation_text="Target")
        goal_chart.add_vline(x=metrics["customers"], line_dash="dot", line_color="#25a7ff", annotation_text="Current demand")
        goal_chart.update_layout(height=430, margin=dict(l=20, r=20, t=25, b=20), xaxis_title="Customers / month", yaxis_title="Net profit (TND)", hovermode="x unified")
        style_chart(goal_chart)
        st.plotly_chart(goal_chart, use_container_width=True)

with details_tab:
    st.subheader("Transparent calculation")
    detail_rows = [
        ("Adjusted customers", metrics["customers"], "Customers × seasonal adjustment"),
        ("Units sold", metrics["units"], "Adjusted customers × units per customer"),
        ("Revenue", metrics["revenue"], "Units × price"),
        ("Material costs", metrics["material_costs"], "Units × material cost per unit"),
        ("Gross profit", metrics["gross_profit"], "Revenue − material costs"),
        ("Fixed costs", metrics["fixed_costs"], "Rent + salaries + marketing + other fixed costs"),
        ("Profit before tax", metrics["profit_before_tax"], "Gross profit − fixed costs"),
        ("Estimated tax", metrics["taxes"], "Tax rate × positive profit before tax"),
        ("Net monthly cash flow", metrics["net_profit"], "Profit before tax − estimated tax"),
    ]
    detail_df = pd.DataFrame(detail_rows, columns=["Line item", "Amount", "Formula"])
    detail_df["Amount"] = detail_df["Amount"].map(tnd)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    st.info("This is a planning model, not accounting or tax advice. Confirm Tunisian tax treatment, social charges, VAT, depreciation, financing, and working-capital timing with a qualified local adviser.")

    export = pd.DataFrame([inputs_as_dict(inputs)])
    st.download_button(
        "Download current assumptions (CSV)", export.to_csv(index=False).encode("utf-8"),
        "mizan_assumptions.csv", "text/csv", use_container_width=True,
    )

st.markdown('<div class="footer">Mizan Profit Simulator · Built for clearer, uncertainty-aware business decisions</div>', unsafe_allow_html=True)
