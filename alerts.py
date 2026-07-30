"""
alerts.py
---------
PlaceMux · Task 15 — Intelligence Layer Integration & Model Governance

Stage C: "Alerting on model degradation and outcome shifts"

DEFINITION: An alert fires when a metric from metrics.py crosses a documented
threshold. Every threshold below is the same one named in that metric's
`decision` string in metrics.py — alerts.py does not invent new bars, it
just watches the ones metrics.py already committed to. This is the
reconciliation link between Stage B/C/D: alerting is not a separate
opinion, it is the enforcement of the thresholds the metrics already defined.

Each alert record carries: severity, the metric name, current value,
threshold, the owning team, and the recommended action — so this can be
handed off (per the brief's "Hand-off" section) to AI-ML / leadership
without needing the author in the room.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

import metrics as m


@dataclass
class Alert:
    severity: str          # "critical" | "warning" | "info"
    metric: str
    current_value: str
    threshold: str
    owner: str
    action: str
    triggered: bool = True


# Documented thresholds (the same bars named in metrics.py's `decision` fields)
THRESHOLDS = {
    "accuracy_floor": 0.55,
    "log_loss_ceiling": 0.85,
    "psi_watch": 0.10,
    "psi_action": 0.25,
    "ks_p_value": 0.01,
    "label_rate_shift_pp": 5.0,
    "fairness_gap_pp": 8.0,
    "latency_p95_sla_ms": 150.0,
    "latency_p99_sla_ms": 300.0,
    "cost_growth_pct": 20.0,
    "net_value_floor": 0.0,
}


def evaluate_all(df: pd.DataFrame, ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> list:
    """
    Runs every governance rule against the current data and returns the
    list of Alert objects (fired or not), ordered by severity.
    This is the single function app.py calls to populate the Alerts tab —
    keeping the "which rule fired, why, what happens next" logic in one
    auditable place.
    """
    alerts = []

    # --- Quality: accuracy floor -----------------------------------------
    acc = m.quality_accuracy(cur_df)
    fired = acc["value"] < THRESHOLDS["accuracy_floor"]
    alerts.append(Alert(
        severity="critical" if fired else "info",
        metric="Accuracy (current window)",
        current_value=f"{acc['value']:.1%}",
        threshold=f"< {THRESHOLDS['accuracy_floor']:.0%}",
        owner="AI-ML on-call",
        action="Pause auto-actioning on model output; revert traffic to last stable version." if fired
               else "No action — accuracy within floor.",
        triggered=fired,
    ))

    # --- Quality: calibration (log loss) ---------------------------------
    ll = m.quality_log_loss(cur_df)
    fired = ll["value"] > THRESHOLDS["log_loss_ceiling"]
    alerts.append(Alert(
        severity="warning" if fired else "info",
        metric="Log loss / calibration (current window)",
        current_value=f"{ll['value']:.3f}",
        threshold=f"> {THRESHOLDS['log_loss_ceiling']:.2f}",
        owner="AI-ML modeling team",
        action="Open a probability-recalibration job." if fired else "No action — calibration within bounds.",
        triggered=fired,
    ))

    # --- Drift: PSI --------------------------------------------------------
    psi_res = m.drift_score_psi(ref_df, cur_df)
    psi_val = psi_res["value"]
    critical = psi_val > THRESHOLDS["psi_action"]
    watch = (not critical) and psi_val > THRESHOLDS["psi_watch"]
    alerts.append(Alert(
        severity="critical" if critical else ("warning" if watch else "info"),
        metric="Score drift (PSI, current vs reference window)",
        current_value=f"{psi_val:.3f}",
        threshold=f"watch > {THRESHOLDS['psi_watch']}, action > {THRESHOLDS['psi_action']}",
        owner="AI-ML on-call",
        action="Block auto-promotion; open a mandatory drift investigation ticket." if critical else
               ("Log a watch-list entry; monitor next window." if watch else "No action — distribution stable."),
        triggered=critical or watch,
    ))

    # --- Drift: KS reconciliation -------------------------------------------
    ks_res = m.drift_ks_test(ref_df, cur_df)
    fired = (ks_res["p_value"] < THRESHOLDS["ks_p_value"]) and watch or critical
    alerts.append(Alert(
        severity="critical" if fired else "info",
        metric="Score drift reconciliation (KS test)",
        current_value=f"stat={ks_res['statistic']:.3f}, p={ks_res['p_value']:.4f}",
        threshold=f"p < {THRESHOLDS['ks_p_value']}",
        owner="AI-ML on-call",
        action="Confirms drift is real (not sampling noise) — escalate drift ticket to P1." if fired
               else "No independent confirmation of drift.",
        triggered=fired,
    ))

    # --- Drift: business-facing label rate shift --------------------------
    lr = m.drift_label_rate(ref_df, cur_df)
    fired = abs(lr["delta_pp"]) > THRESHOLDS["label_rate_shift_pp"]
    alerts.append(Alert(
        severity="warning" if fired else "info",
        metric="Positive-prediction rate shift",
        current_value=f"{lr['delta_pp']:+.1f} pp",
        threshold=f"beyond +/-{THRESHOLDS['label_rate_shift_pp']} pp",
        owner="Product owner",
        action="Business-side sanity check before next release." if fired else "No action — rate stable.",
        triggered=fired,
    ))

    # --- Fairness: demographic parity --------------------------------------
    dp = m.fairness_demographic_parity(cur_df)
    fired = dp["gap_pp"] > THRESHOLDS["fairness_gap_pp"]
    alerts.append(Alert(
        severity="critical" if fired else "info",
        metric="Fairness — demographic parity gap",
        current_value=f"{dp['gap_pp']:.1f} pp",
        threshold=f"> {THRESHOLDS['fairness_gap_pp']} pp",
        owner="Model governance board",
        action="Mandatory fairness review before this version stays in production." if fired
               else "No action — parity gap within tolerance.",
        triggered=fired,
    ))

    # --- Fairness: equal opportunity ----------------------------------------
    eo = m.fairness_equal_opportunity(cur_df)
    fired = eo["gap_pp"] > THRESHOLDS["fairness_gap_pp"]
    alerts.append(Alert(
        severity="critical" if fired else "info",
        metric="Fairness — equal opportunity (TPR) gap",
        current_value=f"{eo['gap_pp']:.1f} pp",
        threshold=f"> {THRESHOLDS['fairness_gap_pp']} pp",
        owner="Model governance board",
        action="Flag as harm-of-omission; route to fairness review board." if fired
               else "No action — TPR gap within tolerance.",
        triggered=fired,
    ))

    # --- Latency SLA --------------------------------------------------------
    lat = m.latency_percentiles(cur_df)
    p95_fired = lat["p95"] > THRESHOLDS["latency_p95_sla_ms"]
    p99_fired = lat["p99"] > THRESHOLDS["latency_p99_sla_ms"]
    alerts.append(Alert(
        severity="critical" if p99_fired else ("warning" if p95_fired else "info"),
        metric="Latency SLA (p95 / p99)",
        current_value=f"p95={lat['p95']:.0f}ms, p99={lat['p99']:.0f}ms",
        threshold=f"p95 > {THRESHOLDS['latency_p95_sla_ms']:.0f}ms, p99 > {THRESHOLDS['latency_p99_sla_ms']:.0f}ms",
        owner="Infra on-call",
        action=("Page on-call — p99 breach." if p99_fired else
                "Open autoscaling review — p95 breach." if p95_fired else
                "No action — latency within SLA."),
        triggered=p95_fired or p99_fired,
    ))

    # --- Business: net value floor -------------------------------------------
    biz = m.business_linked_view(cur_df)
    fired = biz["net_per_1000"] < THRESHOLDS["net_value_floor"]
    alerts.append(Alert(
        severity="critical" if fired else "info",
        metric="Net business value per 1,000 predictions",
        current_value=f"${biz['net_per_1000']:.2f}",
        threshold=f"< ${THRESHOLDS['net_value_floor']:.2f}",
        owner="Executive sponsor",
        action="Go/no-go review on whether the model stays live." if fired
               else "No action — model is net value-positive.",
        triggered=fired,
    ))

    order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: order[a.severity])
    return alerts


def alerts_to_dataframe(alerts: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "Severity": a.severity.upper(),
        "Metric": a.metric,
        "Current value": a.current_value,
        "Threshold": a.threshold,
        "Fired": "🔴 YES" if a.triggered else "🟢 no",
        "Owner": a.owner,
        "Recommended action": a.action,
    } for a in alerts])
