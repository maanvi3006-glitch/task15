"""
data_generator.py
------------------
PlaceMux · Task 15 — Intelligence Layer Integration & Model Governance

WHY THIS FILE EXISTS
=====================
Stage B/C/D of the brief require every metric to be built on real production
data, "sampled if you must, but never synthetic-only-for-the-sake-of-it."
In a real deployment this module would be a read connector to:
    - the model's prediction log table (Kafka topic -> warehouse table)
    - the outcome/label table (ground truth arriving later, e.g. after a
      recommendation is accepted/rejected, a listing converts, etc.)
    - the request trace table (latency, cost per call from the gateway)

Since this is a course/task environment with no live PlaceMux production
warehouse to connect to, this module SIMULATES that raw event stream with a
realistic generating process (seasonality, drift injection, fairness gap
injection, cost/latency distributions, degradation events) so that every
downstream metric in metrics.py is COMPUTED from raw rows, not hand-typed.
This mirrors the real integration: swap this module for a real SQL/warehouse
query and nothing downstream changes.

All randomness is seeded for reproducibility across a demo run, but a
"simulate degradation" control (see alerts.py / app.py) can inject a fresh
failure at any point, on demand, to satisfy Stage E.3 ("break it on purpose").
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
N_DAYS = 30
EVENTS_PER_DAY = 900  # ~ hourly batches x events, kept small enough for a live demo to recompute fast

GROUPS = ["Group A (majority)", "Group B (minority)"]
REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["New user", "Returning user", "Power user"]


def _base_timestamp_range(n_days=N_DAYS, end=None):
    end = end or datetime.now()
    start = end - timedelta(days=n_days)
    return start, end


def generate_prediction_log(
    n_days: int = N_DAYS,
    events_per_day: int = EVENTS_PER_DAY,
    inject_drift: bool = True,
    inject_fairness_gap: bool = True,
    inject_degradation_event: bool = False,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Simulates the raw prediction log: one row per model inference served in
    production, with the ground-truth outcome attached once it is known.

    Columns map 1:1 to what a real event log would contain:
        event_id, ts, model_version, group, region, segment,
        predicted_score, predicted_label, actual_label (ground truth),
        latency_ms, cost_usd, business_outcome_value

    Returns
    -------
    pd.DataFrame sorted by ts ascending.
    """
    rng = np.random.default_rng(seed)
    start, end = _base_timestamp_range(n_days)
    n = n_days * events_per_day

    # timestamps: spread across the window, mild daily seasonality (busier midday)
    day_offsets = rng.integers(0, n_days, size=n)
    hour_of_day = rng.normal(loc=14, scale=4, size=n).clip(0, 23).astype(int)
    minute = rng.integers(0, 60, size=n)
    ts = [start + timedelta(days=int(d), hours=int(h), minutes=int(m))
          for d, h, m in zip(day_offsets, hour_of_day, minute)]

    df = pd.DataFrame({"ts": ts})
    df = df.sort_values("ts").reset_index(drop=True)
    day_index = (df["ts"] - start).dt.days  # 0 .. n_days-1, used to phase-in drift

    # model version rolls once mid-window (a real retrain/redeploy event)
    df["model_version"] = np.where(day_index < n_days * 0.6, "v2.3.0", "v2.4.0")

    df["group"] = rng.choice(GROUPS, size=n, p=[0.72, 0.28])
    df["region"] = rng.choice(REGIONS, size=n)
    df["segment"] = rng.choice(SEGMENTS, size=n, p=[0.35, 0.45, 0.20])

    # ---- Ground-truth generating process -----------------------------
    # Base propensity depends on segment (power users convert/accept more)
    segment_lift = df["segment"].map({"New user": -0.10, "Returning user": 0.0, "Power user": 0.15})
    base_p = 0.42 + segment_lift

    # Model score = true propensity + noise; degrades (more noise, negative bias)
    # as day_index increases IF inject_drift, simulating silent quality decay
    # after the v2.4.0 rollout (a realistic "looked fine on day two" trap).
    drift_phase = (day_index / n_days).clip(0, 1)
    noise_scale = 0.15 + (0.35 * drift_phase if inject_drift else 0.0)
    score_bias = (-0.18 * drift_phase) if inject_drift else 0.0

    # Fairness gap: Group B systematically under-scored relative to true propensity
    fairness_penalty = np.where(
        (df["group"] == "Group B (minority)") & inject_fairness_gap, -0.12, 0.0
    )

    true_propensity = (base_p + fairness_penalty).clip(0.02, 0.98)
    predicted_score = (true_propensity + score_bias + rng.normal(0, noise_scale, size=n)).clip(0.01, 0.99)
    df["predicted_score"] = predicted_score
    df["predicted_label"] = (predicted_score >= 0.5).astype(int)

    # actual outcome realized from the TRUE propensity (independent of what model predicted)
    df["actual_label"] = rng.binomial(1, true_propensity)

    # ---- Latency -------------------------------------------------------
    base_latency = rng.lognormal(mean=4.2, sigma=0.35, size=n)  # ~p50 ~ 65ms
    version_latency_penalty = np.where(df["model_version"] == "v2.4.0", 18, 0)  # bigger model, slower
    df["latency_ms"] = (base_latency + version_latency_penalty).round(1)

    # ---- Cost ------------------------------------------------------------
    # cost scales with model version (v2.4.0 is a larger/more expensive model)
    base_cost = np.where(df["model_version"] == "v2.4.0", 0.00041, 0.00024)
    df["cost_usd"] = (base_cost * rng.normal(1.0, 0.08, size=n)).clip(min=0.00005)

    # ---- Business outcome value -------------------------------------------
    # $ value realized only when actual_label == 1 (e.g. successful match/lease/sale)
    outcome_value = rng.normal(loc=185, scale=45, size=n).clip(min=20)
    df["business_outcome_value"] = np.where(df["actual_label"] == 1, outcome_value, 0.0)

    # ---- Optional on-demand degradation injection (Stage E.3: break it) ----
    if inject_degradation_event:
        cutoff = end - timedelta(hours=6)
        mask = df["ts"] >= cutoff
        # sudden score collapse + latency spike + cost spike in the last 6 hours
        df.loc[mask, "predicted_score"] = (df.loc[mask, "predicted_score"] * 0.4).clip(0.01, 0.99)
        df.loc[mask, "predicted_label"] = (df.loc[mask, "predicted_score"] >= 0.5).astype(int)
        df.loc[mask, "latency_ms"] = df.loc[mask, "latency_ms"] * rng.uniform(2.5, 4.0, size=mask.sum())
        df.loc[mask, "cost_usd"] = df.loc[mask, "cost_usd"] * rng.uniform(1.8, 2.6, size=mask.sum())

    df["event_id"] = np.arange(1, n + 1)
    return df[[
        "event_id", "ts", "model_version", "group", "region", "segment",
        "predicted_score", "predicted_label", "actual_label",
        "latency_ms", "cost_usd", "business_outcome_value",
    ]]


def reference_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """The earliest `days` of the log — used as the drift REFERENCE distribution
    (i.e. 'what the model looked like when we trusted it'), per Stage B.3
    (reconcile against a source: here, the model's own accepted baseline)."""
    cutoff = df["ts"].min() + timedelta(days=days)
    return df[df["ts"] < cutoff]


def current_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Most recent `days` of the log — the CURRENT distribution compared
    against the reference window to detect drift."""
    cutoff = df["ts"].max() - timedelta(days=days)
    return df[df["ts"] >= cutoff]
