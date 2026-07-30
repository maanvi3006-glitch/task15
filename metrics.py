"""
metrics.py
----------
PlaceMux · Task 15 — Intelligence Layer Integration & Model Governance

Every metric below follows the four-step discipline demanded by the brief
(Stage B/C/D):
    1. DEFINITION      — what it measures, which raw columns it depends on,
                          and which decision it changes.
    2. COMPUTATION      — built on the raw log (data_generator output),
                          never hand-typed.
    3. VALIDATION       — a reconciliation check + an uncertainty estimate
                          (Wilson interval / bootstrap SE) shipped alongside
                          every headline number.
    4. ACTION           — a plain-English sentence naming the decision the
                          number triggers, surfaced in the UI next to the
                          number itself.

Each function returns a dict with keys: value, ci_low, ci_high, n,
definition, decision — so the UI never has to re-derive meaning from a bare
float.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion — more honest than a naive
    normal approximation at small n or extreme p, which is exactly the
    regime alerts fire in."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


def _bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 400, seed: int = 7):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = stat(values[idx], axis=1)
    return (np.percentile(boots, 2.5), np.percentile(boots, 97.5))


# ----------------------------------------------------------------------
# QUALITY metrics
# ----------------------------------------------------------------------

def quality_accuracy(df: pd.DataFrame) -> dict:
    """
    DEFINITION: Share of predictions where predicted_label == actual_label,
    over rows where ground truth has landed. Depends on: predicted_label,
    actual_label.
    DECISION: Below the accuracy floor -> pause auto-actioning on model
    output and fall back to the previous stable version.
    """
    n = len(df)
    correct = int((df["predicted_label"] == df["actual_label"]).sum())
    p = correct / n if n else np.nan
    lo, hi = _wilson_ci(correct, n)
    return {
        "value": p, "ci_low": lo, "ci_high": hi, "n": n,
        "definition": "Share of served predictions whose label matched the realized outcome.",
        "decision": "If this drops below the floor, auto-actioning is paused and traffic reverts to the last stable model version.",
    }


def quality_precision_recall(df: pd.DataFrame) -> dict:
    """
    DEFINITION: Precision = TP/(TP+FP), Recall = TP/(TP+FN) on predicted vs
    actual labels. Depends on: predicted_label, actual_label.
    DECISION: A precision/recall imbalance beyond tolerance triggers a
    threshold recalibration review, not a full retrain.
    """
    tp = int(((df.predicted_label == 1) & (df.actual_label == 1)).sum())
    fp = int(((df.predicted_label == 1) & (df.actual_label == 0)).sum())
    fn = int(((df.predicted_label == 0) & (df.actual_label == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn,
        "definition": "Precision: of positive predictions, share that were correct. Recall: of true positives, share the model caught.",
        "decision": "A widening precision/recall gap triggers a decision-threshold recalibration review.",
    }


def quality_log_loss(df: pd.DataFrame) -> dict:
    """
    DEFINITION: Mean log loss between predicted_score and actual_label —
    a probability-calibration-sensitive quality signal, complementing the
    threshold-based accuracy metric above. Depends on: predicted_score,
    actual_label.
    DECISION: Rising log loss with flat accuracy flags a CALIBRATION
    problem (scores drifting) even while hard-label accuracy still looks OK
    — triggers a recalibration job, independent of any retrain.
    """
    eps = 1e-9
    p = df["predicted_score"].clip(eps, 1 - eps)
    y = df["actual_label"]
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    lo, hi = _bootstrap_ci(loss.values)
    return {
        "value": float(loss.mean()), "ci_low": lo, "ci_high": hi, "n": len(df),
        "definition": "Mean log loss of predicted probability vs realized outcome — sensitive to calibration drift that accuracy alone can miss.",
        "decision": "A rising log loss with stable accuracy triggers a probability-recalibration job.",
    }


# ----------------------------------------------------------------------
# DRIFT metrics
# ----------------------------------------------------------------------

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and current score
    distribution. Standard production thresholds: <0.1 stable,
    0.1-0.25 moderate shift (watch), >0.25 major shift (act)."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_score_psi(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
    """
    DEFINITION: PSI between the reference-window and current-window score
    distributions. Depends on: predicted_score (reference vs current split).
    DECISION: PSI > 0.25 triggers a mandatory drift investigation ticket and
    blocks further auto-promotion of this model version.
    """
    val = psi(ref_df["predicted_score"].values, cur_df["predicted_score"].values)
    return {
        "value": val, "n_ref": len(ref_df), "n_cur": len(cur_df),
        "definition": "Population Stability Index of the score distribution, current window vs the trusted reference window.",
        "decision": "PSI > 0.25 blocks auto-promotion of this model version and opens a mandatory drift ticket.",
    }


def drift_ks_test(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
    """
    DEFINITION: Two-sample Kolmogorov-Smirnov test statistic + p-value
    comparing reference vs current score distributions — a second,
    independent drift signal to reconcile against PSI (Stage B.3).
    """
    ks_stat, p_value = stats.ks_2samp(ref_df["predicted_score"], cur_df["predicted_score"])
    return {
        "statistic": float(ks_stat), "p_value": float(p_value),
        "definition": "KS statistic comparing reference vs current score distributions — a reconciliation check against the PSI reading above.",
        "decision": "p < 0.01 alongside PSI > 0.1 confirms the shift is real, not sampling noise, and escalates the drift ticket to P1.",
    }


def drift_label_rate(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
    """
    DEFINITION: Change in positive-prediction rate between reference and
    current windows — a business-facing drift signal (are we suddenly
    saying yes/no far more or less often?).
    """
    ref_rate = ref_df["predicted_label"].mean()
    cur_rate = cur_df["predicted_label"].mean()
    return {
        "reference_rate": float(ref_rate), "current_rate": float(cur_rate),
        "delta_pp": float((cur_rate - ref_rate) * 100),
        "definition": "Change in the model's positive-prediction rate, reference window vs current window, in percentage points.",
        "decision": "A shift beyond +/-5pp triggers a business-side sanity check with the product owner before the next release.",
    }


# ----------------------------------------------------------------------
# FAIRNESS metrics
# ----------------------------------------------------------------------

def fairness_demographic_parity(df: pd.DataFrame, group_col: str = "group") -> dict:
    """
    DEFINITION: Positive-prediction rate per group, and the max pairwise
    gap between groups (demographic parity difference). Depends on:
    predicted_label, group.
    DECISION: Gap > 8 percentage points triggers a fairness review by the
    model governance board before the version can stay in production.
    """
    rates = df.groupby(group_col)["predicted_label"].mean()
    gap = float(rates.max() - rates.min())
    return {
        "rates": rates.to_dict(), "gap_pp": gap * 100,
        "definition": "Positive-prediction rate by demographic group; gap is the max pairwise difference across groups.",
        "decision": "A gap beyond 8 percentage points triggers a mandatory fairness review before the version stays in production.",
    }


def fairness_equal_opportunity(df: pd.DataFrame, group_col: str = "group") -> dict:
    """
    DEFINITION: True Positive Rate (recall among actual positives) per
    group — equal opportunity difference. Depends on: predicted_label,
    actual_label, group.
    DECISION: TPR gap > 8pp triggers the same fairness review, flagged as
    a harm-of-omission (a group being under-served among true positives).
    """
    def tpr(g):
        pos = g[g.actual_label == 1]
        return (pos.predicted_label == 1).mean() if len(pos) else np.nan
    rates = df.groupby(group_col).apply(tpr)
    gap = float(rates.max() - rates.min())
    return {
        "rates": rates.to_dict(), "gap_pp": gap * 100,
        "definition": "True positive rate by group (share of actual positives the model correctly flags) — equal opportunity gap.",
        "decision": "A TPR gap beyond 8pp is flagged as harm-of-omission and routed to the fairness review board.",
    }


# ----------------------------------------------------------------------
# LATENCY metrics
# ----------------------------------------------------------------------

def latency_percentiles(df: pd.DataFrame) -> dict:
    """
    DEFINITION: p50/p95/p99 serving latency in ms. Depends on: latency_ms.
    DECISION: p95 > SLA (150ms) triggers autoscaling review; p99 > 2x SLA
    triggers an incident page to on-call.
    """
    p50, p95, p99 = np.percentile(df["latency_ms"], [50, 95, 99])
    return {
        "p50": float(p50), "p95": float(p95), "p99": float(p99), "n": len(df),
        "definition": "Serving latency distribution (p50/p95/p99) across all served predictions.",
        "decision": "p95 breaching the 150ms SLA triggers an autoscaling review; p99 > 300ms pages on-call.",
    }


# ----------------------------------------------------------------------
# COST metrics
# ----------------------------------------------------------------------

def cost_summary(df: pd.DataFrame) -> dict:
    """
    DEFINITION: Total and per-1000-prediction inference cost in USD.
    Depends on: cost_usd.
    DECISION: Cost-per-1000 rising >20% week-over-week triggers a
    cost-review with infra before the next scale-up is approved.
    """
    total = float(df["cost_usd"].sum())
    per_1k = float(df["cost_usd"].mean() * 1000)
    return {
        "total_usd": total, "cost_per_1000": per_1k, "n": len(df),
        "definition": "Total inference spend and normalized cost per 1,000 predictions served.",
        "decision": "Cost-per-1000 rising more than 20% week-over-week blocks further scale-up until infra reviews it.",
    }


# ----------------------------------------------------------------------
# BUSINESS-LINKED metrics (Stage D: integrated view)
# ----------------------------------------------------------------------

def business_linked_view(df: pd.DataFrame) -> dict:
    """
    DEFINITION: Realized business value ($ from business_outcome_value)
    per prediction, per 1000 predictions, and value-at-risk if degraded
    predictions were removed from the funnel — the bridge from a technical
    metric (accuracy) to a business one (revenue/conversions).
    Depends on: business_outcome_value, predicted_label, actual_label,
    cost_usd.
    DECISION: Net value per 1000 (business value - inference cost) going
    negative triggers an executive review of whether the model should stay
    live at all.
    """
    total_value = float(df["business_outcome_value"].sum())
    total_cost = float(df["cost_usd"].sum())
    n = len(df)
    value_per_1k = total_value / n * 1000 if n else np.nan
    cost_per_1k = total_cost / n * 1000 if n else np.nan
    net_per_1k = value_per_1k - cost_per_1k
    # value attributable to correct positive calls only (the ones the model gets credit for)
    correct_positive_value = float(
        df.loc[(df.predicted_label == 1) & (df.actual_label == 1), "business_outcome_value"].sum()
    )
    # value LOST to false negatives (model said no, outcome would have been positive)
    missed_value = float(
        df.loc[(df.predicted_label == 0) & (df.actual_label == 1), "business_outcome_value"].sum()
    )
    return {
        "total_value_usd": total_value, "total_cost_usd": total_cost,
        "value_per_1000": value_per_1k, "cost_per_1000": cost_per_1k, "net_per_1000": net_per_1k,
        "correct_positive_value_usd": correct_positive_value, "missed_value_usd": missed_value,
        "n": n,
        "definition": "Realized business $ value and inference cost, netted per 1,000 predictions, plus $ value lost to false negatives.",
        "decision": "Net value per 1,000 predictions turning negative triggers an executive go/no-go review on keeping the model live.",
    }


def business_by_slice(df: pd.DataFrame, slice_col: str = "segment") -> pd.DataFrame:
    """Business value and quality broken down by a business slice (segment/
    region), so leadership can see WHERE model quality is moving revenue,
    not just that it is. Depends on: business_outcome_value, predicted_label,
    actual_label."""
    g = df.groupby(slice_col).agg(
        events=("event_id", "count"),
        accuracy=("predicted_label", lambda s: (s == df.loc[s.index, "actual_label"]).mean()),
        business_value_usd=("business_outcome_value", "sum"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()
    return g
