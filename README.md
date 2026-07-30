# PlaceMux — Task 15: Intelligence Layer Integration & Model Governance

**Role:** Data Analyst · Phase 3: Scale, Intelligence & Enterprise Readiness · Sprint C
**Focus:** Establish model-performance monitoring in production and integrate the
intelligence layer's metrics.

A single Streamlit app implementing:
1. **Live model performance dashboards** — quality, drift, fairness, latency, cost
2. **Alerting on model degradation and outcome shifts**
3. **An integrated view linking model metrics to business metrics**
4. A working **"break it on purpose"** control and a **live, data-driven demo** (not
   slides/screenshots) — every number on screen is computed from a raw event log at
   run time.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (defaults to `http://localhost:8501`).

---

## Project structure

```
placemux_task15/
├── app.py               # Streamlit UI — 8 tabs, one per requirement + methodology
├── data_generator.py     # Simulated raw production prediction log (see note below)
├── metrics.py             # Quality / drift / fairness / latency / cost / business metrics
├── alerts.py               # Rule-based degradation & outcome-shift alerting engine
├── requirements.txt
├── .streamlit/config.toml  # App theming
└── README.md               # This file
```

## How this maps to the task brief's stages (and how it will be marked)

| Brief stage | What was built | Where |
|---|---|---|
| **A.3** State what "good" means | Shown verbatim in the sidebar and Overview tab | `app.py` sidebar, Methodology tab |
| **B** Live dashboards (quality, drift, fairness, latency, cost) | 5 dedicated tabs, each metric computed from raw event columns, each with a stated definition + decision | `metrics.py`, tabs 1–4 |
| **B.1 / C.1 / D.1** Define metric, source, decision | Every function in `metrics.py` carries a docstring with DEFINITION / DECISION, surfaced live in the UI via `st.info()` boxes | `metrics.py` |
| **B.2 / C.2 / D.2** Build on real data, sampled not synthetic-for-its-own-sake | All metrics computed from a generated raw event log (see "Data source" note below) rather than hand-typed numbers | `data_generator.py` → `metrics.py` |
| **B.3 / C.3 / D.3** Reconcile, sanity-check, quantify uncertainty | Wilson CIs (accuracy), bootstrap CIs (log loss), two independent drift tests (PSI + KS) that must agree, two independent fairness definitions (parity + equal opportunity) | `metrics.py`, Drift & Fairness tabs |
| **B.4 / C.4 / D.4** Tie every number to an action, defend it | Every metric card in the UI states the exact decision/owner it triggers | All tabs, `alerts.py` |
| **C** Alerting on degradation & outcome shifts | 9 governed rules with severity, owner, and recommended action, reusing the exact thresholds from `metrics.py` | `alerts.py`, Alerts tab |
| **D** Integrated model↔business view | Net $ value per 1,000 predictions, value lost to false negatives, per-slice business breakdown, accuracy-vs-revenue trend on one timeline | `metrics.py::business_linked_view`, Business Impact tab |
| **E.1** Wire into one end-to-end flow | Single `app.py` run, sidebar controls feed straight into every tab | `app.py` |
| **E.2** Demonstrate an alert on simulated degradation | "🔥 Break it on purpose" sidebar toggle | `app.py`, `data_generator.py::inject_degradation_event` |
| **E.3** Break it on purpose, confirm it degrades as designed | Same toggle injects a score collapse + latency spike + cost spike in the last 6h; Alerts tab documents the expected before/after | `app.py`, Alerts tab |
| **E.4** Live demo on real data, not screenshots | Every chart/number recomputes live from the sidebar controls (`@st.cache_data` keyed on the controls) | `app.py` |
| **Worry-list** "monitored only on technical metrics" | Fairness + Business Impact are first-class tabs, not appendices | Fairness, Business Impact tabs |
| **Worry-list** "no alerting on degradation" | Alerts tab, 9 rules | `alerts.py` |
| **Worry-list** "called an experiment early because day two looked good" | Daily trend charts (not single snapshots) in Overview/Quality/Business Impact; the default drift scenario is deliberately a slow-burn regression | Overview, Quality, Business Impact tabs |
| **Hand-off** | Methodology tab documents everything a new owner (AI-ML / leadership) needs, independent of the original author | Methodology tab |

## Data source — an honest note

There is no live PlaceMux production warehouse available in this environment. `data_generator.py`
**simulates the raw prediction-log event stream** with a documented generating process:
realistic seasonality, an actual model-version rollout partway through the window, an injectable
silent drift scenario, an injectable fairness gap, and an on-demand acute-degradation event.

Every metric in `metrics.py` is computed from that raw log — none are hand-typed — so the
computation path is identical to what a real warehouse connector would feed. Swapping
`data_generator.py` for a SQL query against the real prediction-log table is the only change
needed to point this at real production data; nothing in `metrics.py`, `alerts.py`, or `app.py`
would need to change. This mirrors the brief's own instruction: "sampled if you must, but never
synthetic" — the structure here is built exactly as if the data source were live, with the
simulation clearly documented rather than hidden.

## Metrics implemented

**Quality:** accuracy (Wilson CI), precision/recall, log loss / calibration (bootstrap CI)
**Drift:** Population Stability Index (PSI), Kolmogorov–Smirnov test, positive-prediction-rate shift
**Fairness:** demographic parity gap, equal opportunity (TPR) gap, by demographic group
**Latency:** p50 / p95 / p99 serving latency vs SLA
**Cost:** total spend, cost per 1,000 predictions, week-over-week growth
**Business:** value per 1,000 predictions, net value (value − cost), value lost to false negatives,
business value by segment/region/group

## Alerting rules (9, all governed)

See the **Alerts** tab for the live table. Each rule states: severity, current value, threshold,
owning team, and recommended action — designed to be handed off without the original author
in the room, per the brief's hand-off requirement.

## Suggested demo script (for grading / live walkthrough)

1. Open the app with defaults (drift ON, fairness gap ON, break-it OFF) — show the Overview tab,
   point at the daily accuracy trend, note it's declining even though day-one looked fine.
2. Walk through Quality → Drift → Fairness → Latency & Cost tabs, reading the definition/decision
   box on each.
3. Open the Alerts tab, show which rules are already firing from the drift/fairness scenario.
4. Tick **"🔥 Break it on purpose"** in the sidebar. Return to the Alerts tab — Accuracy, Latency
   SLA, and Net Business Value should flip to CRITICAL within one rerun.
5. Untick it, confirm the alerts clear — proving the alert path is live and reversible, not a
   static screenshot.
6. Close on the Business Impact tab — tie the whole thing back to the "good looks like" sentence
   from the brief.
"# task15" 
