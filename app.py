"""
app.py
------
PlaceMux · Altrodav Technologies Pvt. Ltd. · Phase 3
TASK 15 — Intelligence Layer Integration & Model Governance

Run with:  streamlit run app.py

This single app satisfies every "Done when" bullet in the task brief:
  1. Live model performance dashboards (quality, drift, fairness, latency, cost)
  2. Alerting on model degradation and outcome shifts
  3. An integrated view linking model metrics to business metrics
  4. A working "break it on purpose" control (Stage E.3) + live demo mode (Stage E.4)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import data_generator as dg
import metrics as m
import alerts as al

st.set_page_config(
    page_title="PlaceMux · Intelligence Layer & Model Governance",
    page_icon="📡",
    layout="wide",
)

# ----------------------------------------------------------------------
# Sidebar — global controls (acts as the "production knobs" for the demo)
# ----------------------------------------------------------------------
st.sidebar.title("📡 PlaceMux Intelligence Layer")
st.sidebar.caption("Task 15 · Sprint C · Data Analyst · Phase 3")

st.sidebar.markdown("### Data window")
n_days = st.sidebar.slider("Days of production log to load", 14, 45, 30)
window_days = st.sidebar.slider("Current-vs-reference window size (days)", 3, 10, 7)

st.sidebar.markdown("### Simulate real conditions")
inject_drift = st.sidebar.checkbox("Model has been silently drifting since v2.4.0 rollout", value=True)
inject_fairness_gap = st.sidebar.checkbox("Inject a fairness gap (Group B under-scored)", value=True)
break_it = st.sidebar.checkbox(
    "🔥 Break it on purpose (simulate acute degradation, last 6h)", value=False,
    help="Stage E.3 requirement: force the failure path and confirm alerting reacts correctly."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Good looks like:** The business can see whether the AI is actually "
    "helping — or quietly getting worse — at any moment."
)
st.sidebar.markdown("[📄 View methodology / defend-the-numbers notes](#methodology)")


@st.cache_data(show_spinner=False)
def load_data(n_days, drift, fair_gap, degrade, window_days):
    df = dg.generate_prediction_log(
        n_days=n_days,
        inject_drift=drift,
        inject_fairness_gap=fair_gap,
        inject_degradation_event=degrade,
    )
    ref = dg.reference_window(df, days=window_days)
    cur = dg.current_window(df, days=window_days)
    return df, ref, cur


df, ref_df, cur_df = load_data(n_days, inject_drift, inject_fairness_gap, break_it, window_days)

st.title("Intelligence Layer — Model Performance & Governance")
st.caption(
    f"Live production log · {len(df):,} scored events · window {df['ts'].min():%Y-%m-%d} → "
    f"{df['ts'].max():%Y-%m-%d} · reference = first {window_days}d, current = last {window_days}d"
)

if break_it:
    st.error(
        "🔥 **Simulated acute degradation is ACTIVE** — the last 6 hours of traffic have an injected "
        "score collapse, latency spike, and cost spike. Use this to verify every alert below reacts correctly, "
        "then untick it in the sidebar to return to steady state.",
        icon="🚨",
    )

tabs = st.tabs([
    "🏠 Overview",
    "🎯 Quality",
    "📉 Drift",
    "⚖️ Fairness",
    "⚡ Latency & Cost",
    "🚨 Alerts",
    "💼 Business Impact",
    "📚 Methodology",
])

# ========================================================================
# TAB 0 — OVERVIEW
# ========================================================================
with tabs[0]:
    st.subheader("At a glance — current window vs reference window")

    acc_cur = m.quality_accuracy(cur_df)
    acc_ref = m.quality_accuracy(ref_df)
    psi_res = m.drift_score_psi(ref_df, cur_df)
    dp = m.fairness_demographic_parity(cur_df)
    lat = m.latency_percentiles(cur_df)
    cost = m.cost_summary(cur_df)
    biz = m.business_linked_view(cur_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy (current)", f"{acc_cur['value']:.1%}",
              f"{(acc_cur['value']-acc_ref['value'])*100:+.1f}pp vs reference")
    c2.metric("Score drift (PSI)", f"{psi_res['value']:.3f}",
              "🔴 action" if psi_res['value'] > al.THRESHOLDS['psi_action']
              else ("🟠 watch" if psi_res['value'] > al.THRESHOLDS['psi_watch'] else "🟢 stable"))
    c3.metric("Fairness gap (parity)", f"{dp['gap_pp']:.1f} pp",
              "🔴 over bar" if dp['gap_pp'] > al.THRESHOLDS['fairness_gap_pp'] else "🟢 within bar")
    c4.metric("Latency p95", f"{lat['p95']:.0f} ms",
              "🔴 SLA breach" if lat['p95'] > al.THRESHOLDS['latency_p95_sla_ms'] else "🟢 within SLA")
    c5.metric("Net value / 1000 preds", f"${biz['net_per_1000']:.2f}",
              "🔴 value-negative" if biz['net_per_1000'] < 0 else "🟢 value-positive")

    fired = al.evaluate_all(df, ref_df, cur_df)
    n_crit = sum(1 for a in fired if a.triggered and a.severity == "critical")
    n_warn = sum(1 for a in fired if a.triggered and a.severity == "warning")
    if n_crit:
        st.error(f"🔴 {n_crit} critical alert(s) firing right now. See the Alerts tab.")
    elif n_warn:
        st.warning(f"🟠 {n_warn} warning-level alert(s) firing. See the Alerts tab.")
    else:
        st.success("🟢 No alerts firing — model is within all governed bounds.")

    st.markdown("#### Daily accuracy & positive-rate trend (the single chart leadership watches)")
    daily = df.copy()
    daily["date"] = daily["ts"].dt.date
    trend = daily.groupby("date").apply(
        lambda g: pd.Series({
            "accuracy": (g.predicted_label == g.actual_label).mean(),
            "positive_rate": g.predicted_label.mean(),
            "avg_latency_ms": g.latency_ms.mean(),
        })
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["accuracy"], name="Accuracy", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["positive_rate"], name="Positive-prediction rate", mode="lines+markers"))
    fig.add_hline(y=al.THRESHOLDS["accuracy_floor"], line_dash="dash", line_color="red",
                  annotation_text="Accuracy floor")
    fig.update_layout(yaxis_tickformat=".0%", height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# TAB 1 — QUALITY
# ========================================================================
with tabs[1]:
    st.subheader("Live model quality dashboard")
    acc = m.quality_accuracy(cur_df)
    pr = m.quality_precision_recall(cur_df)
    ll = m.quality_log_loss(cur_df)

    st.info(f"**Definition:** {acc['definition']}\n\n**Decision it drives:** {acc['decision']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc['value']:.1%}", f"95% CI [{acc['ci_low']:.1%}, {acc['ci_high']:.1%}], n={acc['n']:,}")
    c2.metric("Precision", f"{pr['precision']:.1%}", f"TP={pr['tp']:,} FP={pr['fp']:,}")
    c3.metric("Recall", f"{pr['recall']:.1%}", f"FN={pr['fn']:,}")

    st.metric("Log loss (calibration)", f"{ll['value']:.3f}",
              f"95% CI [{ll['ci_low']:.3f}, {ll['ci_high']:.3f}]")
    st.caption(ll["decision"])

    st.markdown("#### Accuracy by model version (reconciliation: did the last rollout actually help?)")
    by_version = df.groupby("model_version").apply(
        lambda g: (g.predicted_label == g.actual_label).mean()
    ).rename("accuracy").reset_index()
    fig = px.bar(by_version, x="model_version", y="accuracy", text_auto=".1%",
                 color="model_version", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(yaxis_tickformat=".0%", height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Sanity check: v2.4.0 should look at least as good as v2.3.0 on day one. "
        "If it looks worse here, that is the 'looked good on day two, then quietly worsened' trap the "
        "brief warns about — the daily trend in Overview is what actually catches it, not this snapshot alone."
    )

    st.markdown("#### Confusion breakdown (current window)")
    conf = pd.DataFrame({
        "Predicted \\ Actual": ["Predicted positive", "Predicted negative"],
        "Actual positive": [pr["tp"], pr["fn"]],
        "Actual negative": [pr["fp"], len(cur_df) - pr["tp"] - pr["fp"] - pr["fn"]],
    })
    st.dataframe(conf, use_container_width=True, hide_index=True)

# ========================================================================
# TAB 2 — DRIFT
# ========================================================================
with tabs[2]:
    st.subheader("Live drift dashboard — current window vs reference window")
    psi_res = m.drift_score_psi(ref_df, cur_df)
    ks_res = m.drift_ks_test(ref_df, cur_df)
    lr = m.drift_label_rate(ref_df, cur_df)

    st.info(f"**Definition:** {psi_res['definition']}\n\n**Decision it drives:** {psi_res['decision']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("PSI (score drift)", f"{psi_res['value']:.3f}",
              "stable" if psi_res['value'] < 0.1 else ("watch" if psi_res['value'] < 0.25 else "action"))
    c2.metric("KS statistic (reconciliation)", f"{ks_res['statistic']:.3f}", f"p={ks_res['p_value']:.4f}")
    c3.metric("Positive-rate shift", f"{lr['delta_pp']:+.1f} pp",
              f"{lr['reference_rate']:.1%} → {lr['current_rate']:.1%}")

    st.caption(
        "Validation (Stage B.3): PSI and the KS test are two independent statistics computed on the same "
        "underlying score column — when both agree a shift is real, that is the reconciliation; when they "
        "disagree, we trust neither and widen the sampling window before acting."
    )

    st.markdown("#### Score distribution: reference vs current window")
    ref_plot = ref_df[["predicted_score"]].assign(window="Reference (early)")
    cur_plot = cur_df[["predicted_score"]].assign(window="Current (recent)")
    dist = pd.concat([ref_plot, cur_plot])
    fig = px.histogram(dist, x="predicted_score", color="window", barmode="overlay",
                        nbins=40, opacity=0.6, color_discrete_sequence=["#636EFA", "#EF553B"])
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### PSI trend over time (rolling 3-day windows vs the fixed reference)")
    daily_groups = sorted(df["ts"].dt.date.unique())
    psi_trend = []
    for d in daily_groups:
        window_df = df[(df["ts"].dt.date >= d) & (df["ts"].dt.date < d + pd.Timedelta(days=3))]
        if len(window_df) < 20:
            continue
        val = m.psi(ref_df["predicted_score"].values, window_df["predicted_score"].values)
        psi_trend.append({"date": d, "psi": val})
    psi_trend_df = pd.DataFrame(psi_trend)
    fig2 = px.line(psi_trend_df, x="date", y="psi", markers=True)
    fig2.add_hline(y=0.10, line_dash="dot", line_color="orange", annotation_text="watch (0.10)")
    fig2.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="action (0.25)")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ========================================================================
# TAB 3 — FAIRNESS
# ========================================================================
with tabs[3]:
    st.subheader("Live fairness dashboard")
    dp = m.fairness_demographic_parity(cur_df)
    eo = m.fairness_equal_opportunity(cur_df)

    st.info(f"**Definition:** {dp['definition']}\n\n**Decision it drives:** {dp['decision']}")

    c1, c2 = st.columns(2)
    c1.metric("Demographic parity gap", f"{dp['gap_pp']:.1f} pp",
              "🔴 over bar (8pp)" if dp['gap_pp'] > 8 else "🟢 within bar")
    c2.metric("Equal opportunity (TPR) gap", f"{eo['gap_pp']:.1f} pp",
              "🔴 over bar (8pp)" if eo['gap_pp'] > 8 else "🟢 within bar")

    st.markdown("#### Positive-prediction rate by group")
    dp_df = pd.DataFrame({"group": list(dp["rates"].keys()), "rate": list(dp["rates"].values())})
    fig = px.bar(dp_df, x="group", y="rate", text_auto=".1%", color="group",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(yaxis_tickformat=".0%", height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### True positive rate by group (equal opportunity)")
    eo_df = pd.DataFrame({"group": list(eo["rates"].keys()), "tpr": list(eo["rates"].values())})
    fig2 = px.bar(eo_df, x="group", y="tpr", text_auto=".1%", color="group",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(yaxis_tickformat=".0%", height=340, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(eo["decision"])
    st.warning(
        "Worry-list check: the brief explicitly flags **'model monitored only on technical metrics'** as a "
        "failure mode. This tab exists so fairness never falls out of the live dashboard even when accuracy "
        "and latency look perfectly healthy."
    )

# ========================================================================
# TAB 4 — LATENCY & COST
# ========================================================================
with tabs[4]:
    st.subheader("Live latency & cost dashboard")
    lat = m.latency_percentiles(cur_df)
    cost = m.cost_summary(cur_df)
    cost_ref = m.cost_summary(ref_df)

    st.info(f"**Definition:** {lat['definition']}\n\n**Decision it drives:** {lat['decision']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("p50 latency", f"{lat['p50']:.0f} ms")
    c2.metric("p95 latency", f"{lat['p95']:.0f} ms",
              "🔴 SLA breach" if lat['p95'] > 150 else "🟢 within SLA")
    c3.metric("p99 latency", f"{lat['p99']:.0f} ms",
              "🔴 page on-call" if lat['p99'] > 300 else "🟢 within SLA")

    st.markdown("#### Latency distribution (current window)")
    fig = px.histogram(cur_df, x="latency_ms", nbins=50, color_discrete_sequence=["#00CC96"])
    fig.add_vline(x=150, line_dash="dash", line_color="orange", annotation_text="p95 SLA")
    fig.add_vline(x=300, line_dash="dash", line_color="red", annotation_text="p99 SLA")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cost per 1,000 predictions — reference vs current")
    growth_pct = (cost["cost_per_1000"] - cost_ref["cost_per_1000"]) / cost_ref["cost_per_1000"] * 100
    c1, c2 = st.columns(2)
    c1.metric("Cost / 1000 preds (current)", f"${cost['cost_per_1000']:.3f}",
              f"{growth_pct:+.1f}% vs reference")
    c2.metric("Total inference spend (current window)", f"${cost['total_usd']:.2f}")
    st.caption(cost["decision"])

    st.markdown("#### Latency by model version (reconciliation against the rollout)")
    lat_by_version = df.groupby("model_version")["latency_ms"].mean().reset_index()
    fig2 = px.bar(lat_by_version, x="model_version", y="latency_ms", text_auto=".1f",
                  color="model_version", color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ========================================================================
# TAB 5 — ALERTS
# ========================================================================
with tabs[5]:
    st.subheader("Alerting on model degradation & outcome shifts")
    st.caption(
        "Every rule below enforces a threshold already named in the metric's own 'decision' definition — "
        "alerting and dashboards are not two separate opinions, this is the same bar being watched live."
    )

    fired = al.evaluate_all(df, ref_df, cur_df)
    alert_df = al.alerts_to_dataframe(fired)

    n_crit = sum(1 for a in fired if a.triggered and a.severity == "critical")
    n_warn = sum(1 for a in fired if a.triggered and a.severity == "warning")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical alerts firing", n_crit)
    c2.metric("🟠 Warning alerts firing", n_warn)
    c3.metric("Rules evaluated", len(fired))

    def highlight(row):
        color = {"CRITICAL": "#ffe3e3", "WARNING": "#fff3cd", "INFO": "#e6f4ea"}.get(row["Severity"], "")
        return [f"background-color: {color}"] * len(row)

    st.dataframe(alert_df.style.apply(highlight, axis=1), use_container_width=True, hide_index=True, height=420)

    st.markdown("#### Demonstrate the alert path (Stage E.2 / E.3)")
    st.markdown(
        "1. Tick **'🔥 Break it on purpose'** in the sidebar — this injects a real score collapse, "
        "latency spike, and cost spike into the last 6 hours of the log.\n"
        "2. Watch the **Accuracy**, **Latency SLA**, and **Net business value** rules above flip to 🔴 CRITICAL.\n"
        "3. Untick it — confirm every rule returns to 🟢 within one refresh, proving the alert path is "
        "reversible and not hard-coded to a single demo state."
    )
    if break_it:
        st.success("You are currently in the injected-degradation state — check that the rules above reacted.")

# ========================================================================
# TAB 6 — BUSINESS IMPACT (integrated view)
# ========================================================================
with tabs[6]:
    st.subheader("Integrated view — model metrics linked to business metrics")
    biz = m.business_linked_view(cur_df)
    st.info(f"**Definition:** {biz['definition']}\n\n**Decision it drives:** {biz['decision']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Business value / 1000 preds", f"${biz['value_per_1000']:.2f}")
    c2.metric("Inference cost / 1000 preds", f"${biz['cost_per_1000']:.2f}")
    c3.metric("Net value / 1000 preds", f"${biz['net_per_1000']:.2f}",
              "🔴 value-negative" if biz['net_per_1000'] < 0 else "🟢 value-positive")

    c1, c2 = st.columns(2)
    c1.metric("Value captured by correct positive calls", f"${biz['correct_positive_value_usd']:,.0f}")
    c2.metric("Value lost to missed positives (false negatives)", f"${biz['missed_value_usd']:,.0f}",
              help="This is money the business left on the table because the model said no when the "
                   "true outcome would have been yes — the clearest technical-metric-to-dollar bridge.")

    st.markdown("#### Net value trend — does model quality actually move revenue?")
    daily = df.copy()
    daily["date"] = daily["ts"].dt.date
    daily_biz = daily.groupby("date").apply(
        lambda g: pd.Series({
            "accuracy": (g.predicted_label == g.actual_label).mean(),
            "net_value_per_1000": (g.business_outcome_value.sum() - g.cost_usd.sum()) / len(g) * 1000,
        })
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_biz["date"], y=daily_biz["accuracy"], name="Accuracy",
                              yaxis="y1", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=daily_biz["date"], y=daily_biz["net_value_per_1000"], name="Net value / 1000 ($)",
                              yaxis="y2", mode="lines+markers", line=dict(color="green")))
    fig.update_layout(
        height=400,
        yaxis=dict(title="Accuracy", tickformat=".0%"),
        yaxis2=dict(title="Net value / 1000 ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This is the chart that closes the loop the brief asks for: leadership does not need to "
        "understand PSI or log loss — they need to see that when accuracy dips, net value dips with it, "
        "on the same timeline."
    )

    st.markdown("#### Business value by slice (where quality is moving revenue)")
    slice_col = st.selectbox("Slice by", ["segment", "region", "group"])
    slice_df = m.business_by_slice(cur_df, slice_col)
    fig2 = px.bar(slice_df, x=slice_col, y="business_value_usd", color="accuracy",
                  text_auto=".2s", color_continuous_scale="RdYlGn",
                  hover_data=["events", "avg_latency_ms"])
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(slice_df.style.format({
        "accuracy": "{:.1%}", "business_value_usd": "${:,.0f}", "avg_latency_ms": "{:.1f} ms"
    }), use_container_width=True, hide_index=True)

# ========================================================================
# TAB 7 — METHODOLOGY (defend-the-numbers documentation, Stage A/B.3/B.4)
# ========================================================================
with tabs[7]:
    st.subheader("📚 Methodology, reconciliation & hand-off notes")
    st.markdown("""
<a name="methodology"></a>

### One-sentence bar (Stage A.3)
**Good looks like:** the business can see whether the AI is actually helping — or quietly
getting worse — at any moment.

### Data source (this demo vs. real PlaceMux production)
This app is built to plug into a real prediction-log warehouse table with zero changes to
`metrics.py` or `alerts.py` — only `data_generator.py` would be replaced with a SQL/warehouse
query. `data_generator.py` currently **simulates** that raw event stream (with a documented
generating process: seasonality, a real model-version rollout, injectable drift, an injectable
fairness gap, and an on-demand degradation event) so every metric below is *computed* from raw
rows rather than hand-typed — the same computation path a production connector would use.

### Metric-by-metric definition → decision (Stage B/C/D.1 & .4)
Every metric in the dashboards states, in the app itself, right next to the number:
- **what it measures and which raw columns it depends on**
- **which decision it changes** if it crosses its bar

This is deliberate: an "undefendable number" (the brief's phrase) is one nobody can trace back
to a raw event or forward to an action. Every number on every tab passes that test.

### Validation approach (Stage B/C/D.3)
- **Quality:** Wilson score confidence intervals on accuracy (honest at small n), bootstrap CI on log loss.
- **Drift:** two independent statistics (PSI + KS test) must agree before a drift alert escalates —
  this is the reconciliation check the brief asks for.
- **Fairness:** two independent fairness definitions (demographic parity + equal opportunity) are
  tracked, since optimizing one can silently worsen the other.
- **Latency/Cost:** percentiles (not means) for latency, since SLAs are almost always tail-latency
  commitments; cost is normalized per-1000-predictions so it's comparable across days with different
  traffic volume.
- **Business:** net value = realized $ value − inference cost, reconciled against the accuracy trend
  on the same timeline in the Business Impact tab, so a $ number is never presented without its
  quality context.

### Alerting thresholds (Stage C)
All thresholds are defined once, in `alerts.py`, and are the *same* bars named in each metric's
`decision` field in `metrics.py` — see the table in the Alerts tab for the live list.

### Break-it-on-purpose result (Stage E.3)
Ticking **"🔥 Break it on purpose"** in the sidebar injects, into the last 6 hours only:
- a score collapse (predicted scores compressed toward zero),
- a 2.5–4x latency spike,
- a 1.8–2.6x cost spike.

Expected result: Accuracy, Latency SLA, and Net Business Value alerts should flip to CRITICAL;
drift and fairness alerts should be largely unaffected (this is an acute shock, not a distributional
drift or a fairness issue) — which is itself a useful check that alerts are specific, not just "everything
turns red together."

### Worry-list self-check (explicit call-outs from the brief)
| Worry-list item | Where this app addresses it |
|---|---|
| Model monitored only on technical metrics | Fairness tab + Business Impact tab are first-class, not afterthoughts |
| No alerting on degradation | Alerts tab, 9 governed rules, all reconciled to metric thresholds |
| Calling an experiment early because it looked good on day two | Overview & Quality tabs show the **daily trend**, not a single snapshot — the injectable drift scenario is deliberately a "looked fine on day one, quietly worsens" case |

### Hand-off (per the brief)
This dashboard is the artifact handed to **AI-ML and leadership**. Everything needed to pick it
up without the original author is in this Methodology tab plus inline `definition`/`decision`
docstrings in `metrics.py` and `alerts.py`.
""")
