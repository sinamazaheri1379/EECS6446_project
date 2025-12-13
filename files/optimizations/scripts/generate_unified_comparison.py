#!/usr/bin/env python3
"""
generate_unified_comparison_academic.py
========================================================
Academic Statistical Analysis for CAPA+ v3 CLEAN Experiments
(Follows Jain, 1991. Designed for publishable rigor.)

Implements (per Audit):
1) Transient removal (default: first 10%, configurable)
2) Batch means CI for autocorrelated observations (Jain 25.5.2)
3) Paired comparison on differences (Jain 13.4.1) + paired t-test
4) Mean selection:
   - Latency: Arithmetic
   - Throughput: Harmonic
   - Speedup ratios: Geometric (in log-space CI)
5) Sample size adequacy / precision check (Jain 13.9)
6) Per-service strict comparison + system-level aggregation
7) Saves summary CSV + plots

Inputs:
- baseline_raw_<runid>.csv
- capa_raw_<runid>.csv
Both are produced by unified_experiment_v3_clean.py

Pairing strategy:
- Observations are paired by keys: (pattern, step_idx, tick, service)
- This ensures same workload schedule point is compared across runs.

Usage:
  python generate_unified_comparison_academic.py --data-dir ./results --run-id 20251213_123000
  python generate_unified_comparison_academic.py --data-dir ./results --latest
"""

import os
import math
import json
import glob
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConfidenceInterval:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    confidence_level: float
    sample_size: int
    method: str

    def includes_zero(self) -> bool:
        return self.ci_lower <= 0 <= self.ci_upper


@dataclass
class BatchMeansResult:
    overall_mean: float
    batch_means: List[float]
    batch_size: int
    num_batches: int
    variance_of_batch_means: float
    autocovariance_lag1: float
    ci: ConfidenceInterval
    is_valid: bool
    warning: Optional[str] = None


@dataclass
class PairedComparisonResult:
    metric: str
    n: int

    baseline_mean: float
    capa_mean: float

    mean_type: str
    improvement_pct: float

    # paired difference baseline - capa
    diff_mean: float
    diff_ci_low: float
    diff_ci_high: float
    p_value: float
    significant: bool
    effect_size_d: float

    # speedup (baseline/capa) for time metrics
    speedup_geo_mean: float
    speedup_ci_low: float
    speedup_ci_high: float

    # precision check (Jain 13.9)
    precision_halfwidth_over_mean: float
    precision_target: float
    adequate_precision: bool

    metadata: Dict[str, Any]


# =============================================================================
# CORE STATISTICS
# =============================================================================

def compute_ci_naive(data: np.ndarray, confidence: float) -> ConfidenceInterval:
    n = int(len(data))
    if n < 2:
        return ConfidenceInterval(0, 0, 0, 0, 0, confidence, n, "naive_smallN")
    x = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    alpha = 1.0 - confidence
    tval = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half = tval * s / math.sqrt(n)
    return ConfidenceInterval(x, s, x - half, x + half, 2 * half, confidence, n, "naive")


def batch_means_analysis(data: np.ndarray, confidence: float, initial_batch_size: int = 10) -> BatchMeansResult:
    """
    Jain 25.5.2 batch means for autocorrelated observations.
    We increase batch size until lag-1 autocovariance is small relative to var.
    """
    data = np.asarray(data, dtype=float)
    N = int(len(data))
    if N < 20:
        ci = compute_ci_naive(data, confidence)
        return BatchMeansResult(ci.mean, [ci.mean], N, 1, 0.0, 0.0, ci, False, "N<20")

    batch_size = int(initial_batch_size)
    best: Optional[BatchMeansResult] = None

    for _ in range(10):
        m = N // batch_size
        if m < 5:
            break

        bmeans = [float(np.mean(data[i * batch_size:(i + 1) * batch_size])) for i in range(m)]
        overall = float(np.mean(bmeans))
        var_bm = float(np.var(bmeans, ddof=1))

        # lag-1 autocov of batch means
        acov = 0.0
        if m >= 2 and var_bm > 0:
            acov = sum((bmeans[i] - overall) * (bmeans[i + 1] - overall) for i in range(m - 1)) / (m - 1)

        alpha = 1.0 - confidence
        tval = float(stats.t.ppf(1 - alpha / 2, df=m - 1))
        half = tval * math.sqrt(var_bm / m) if var_bm > 0 else 0.0

        ci = ConfidenceInterval(
            mean=overall,
            std=math.sqrt(var_bm) if var_bm > 0 else 0.0,
            ci_lower=overall - half,
            ci_upper=overall + half,
            ci_width=2 * half,
            confidence_level=confidence,
            sample_size=N,
            method=f"batch(n={batch_size},m={m})",
        )

        is_valid = (var_bm == 0.0) or (abs(acov) < (0.1 * var_bm))
        res = BatchMeansResult(overall, bmeans, batch_size, m, var_bm, acov, ci, is_valid)
        best = res
        if is_valid:
            return res
        batch_size *= 2

    if best is None:
        ci = compute_ci_naive(data, confidence)
        return BatchMeansResult(ci.mean, [ci.mean], N, 1, 0.0, 0.0, ci, False, "BM failed; fallback naive")

    best.warning = "High autocorrelation remained after max batch scaling"
    return best


def remove_transient(data: np.ndarray, frac: float) -> Tuple[np.ndarray, int]:
    """
    Jain 25.3: discard transient. We use a conservative heuristic by default: first X%.
    (You can replace with variance-based detection later; keep deterministic and transparent.)
    """
    data = np.asarray(data, dtype=float)
    N = len(data)
    if N < 50:
        return data, 0
    k = int(frac * N)
    return data[k:], k


def harmonic_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if len(v) == 0:
        return 0.0
    return float(len(v) / np.sum(1.0 / v))


def geometric_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if len(v) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(v))))


def cohens_d_paired(b: np.ndarray, c: np.ndarray) -> float:
    """
    For paired designs, compute d on differences.
    """
    d = b - c
    if len(d) < 2:
        return 0.0
    sd = float(np.std(d, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(d) / sd)


def precision_check(mean_est: float, ci_half_width: float, target: float) -> Tuple[float, bool]:
    """
    Jain 13.9: adequate if (halfwidth/mean) <= target.
    """
    if mean_est == 0:
        return float("inf"), False
    ratio = abs(ci_half_width) / abs(mean_est)
    return float(ratio), bool(ratio <= target)


def speedup_geo_with_ci(b: np.ndarray, c: np.ndarray, confidence: float) -> Tuple[float, float, float]:
    """
    Speedup ratios are multiplicative => geometric mean (Jain 12.5).
    Compute CI in log-space:
      s_i = b_i / c_i
      log_s = log(s_i)
      CI(log) => exp(CI)
    """
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    mask = (b > 0) & (c > 0)
    s = b[mask] / c[mask]
    if len(s) < 2:
        gm = geometric_mean(s)
        return gm, gm, gm

    log_s = np.log(s)
    ci = compute_ci_naive(log_s, confidence)
    gm = float(np.exp(ci.mean))
    lo = float(np.exp(ci.ci_lower))
    hi = float(np.exp(ci.ci_upper))
    return gm, lo, hi


# =============================================================================
# LOADING + PAIRING
# =============================================================================

REQUIRED_COLS = [
    "run_id", "run_type", "pattern", "step_idx", "tick", "service",
    "latency_p95_ms", "latency_avg_ms", "arrival_rate_rps", "cpu_utilization"
]

def load_run(data_dir: str, run_id: str, run_type: str) -> pd.DataFrame:
    pat = os.path.join(data_dir, f"{run_type}_raw_{run_id}.csv")
    if not os.path.exists(pat):
        raise FileNotFoundError(f"Missing file: {pat}")
    df = pd.read_csv(pat)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Column {c} missing in {pat}")
    return df


def find_latest_run_id(data_dir: str) -> str:
    metas = sorted(glob.glob(os.path.join(data_dir, "run_metadata_*.json")))
    if not metas:
        raise FileNotFoundError("No run_metadata_*.json found")
    latest = metas[-1]
    base = os.path.basename(latest)
    run_id = base.replace("run_metadata_", "").replace(".json", "")
    return run_id


def pair_by_key(
    base_df: pd.DataFrame,
    capa_df: pd.DataFrame,
    metric_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Pair by (pattern, step_idx, tick, service). Returns aligned arrays.
    """
    key_cols = ["pattern", "step_idx", "tick", "service"]
    b = base_df[key_cols + [metric_col]].copy()
    c = capa_df[key_cols + [metric_col]].copy()
    b = b.rename(columns={metric_col: "baseline"})
    c = c.rename(columns={metric_col: "capa"})

    merged = b.merge(c, on=key_cols, how="inner")
    merged = merged.dropna(subset=["baseline", "capa"])
    return merged["baseline"].to_numpy(dtype=float), merged["capa"].to_numpy(dtype=float), merged


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

def compare_paired(
    baseline: np.ndarray,
    capa: np.ndarray,
    metric: str,
    metric_type: str,
    confidence: float,
    transient_frac: float,
    precision_target: float,
) -> PairedComparisonResult:
    """
    Jain 13.4.1 paired comparison:
      d_i = baseline_i - capa_i
      CI(d) via batch means, significance by paired t-test.
    Mean selection:
      latency => arithmetic
      throughput => harmonic (and "improvement" sign adjusted)
    """
    # transient removal on each series (same fraction). Then align by min length.
    b2, k1 = remove_transient(baseline, transient_frac)
    c2, k2 = remove_transient(capa, transient_frac)
    n = min(len(b2), len(c2))
    b2, c2 = b2[:n], c2[:n]

    if n < 2:
        return PairedComparisonResult(
            metric=metric, n=n,
            baseline_mean=float(np.mean(b2)) if n else 0.0,
            capa_mean=float(np.mean(c2)) if n else 0.0,
            mean_type="n/a", improvement_pct=0.0,
            diff_mean=0.0, diff_ci_low=0.0, diff_ci_high=0.0,
            p_value=1.0, significant=False, effect_size_d=0.0,
            speedup_geo_mean=1.0, speedup_ci_low=1.0, speedup_ci_high=1.0,
            precision_halfwidth_over_mean=float("inf"),
            precision_target=precision_target,
            adequate_precision=False,
            metadata={"transient_removed": int(max(k1, k2))}
        )

    # Mean selection
    if metric_type == "latency":
        b_mean = float(np.mean(b2))
        c_mean = float(np.mean(c2))
        mean_type = "Arithmetic (Time)"
        improvement = ((b_mean - c_mean) / b_mean) * 100.0 if b_mean > 0 else 0.0
        speed_gm, speed_lo, speed_hi = speedup_geo_with_ci(b2, c2, confidence)
    elif metric_type == "throughput":
        b_mean = harmonic_mean(b2)
        c_mean = harmonic_mean(c2)
        mean_type = "Harmonic (Rate)"
        # For throughput, higher is better: improvement is reversed
        improvement = ((c_mean - b_mean) / b_mean) * 100.0 if b_mean > 0 else 0.0
        # speedup concept not used; set to ratio of means (optional)
        speed_gm, speed_lo, speed_hi = (c_mean / b_mean if b_mean > 0 else 1.0,) * 3
    else:
        b_mean = float(np.mean(b2))
        c_mean = float(np.mean(c2))
        mean_type = "Arithmetic"
        improvement = ((b_mean - c_mean) / b_mean) * 100.0 if b_mean > 0 else 0.0
        speed_gm, speed_lo, speed_hi = speedup_geo_with_ci(b2, c2, confidence)

    # Paired diffs
    d = b2 - c2
    bm = batch_means_analysis(d, confidence=confidence, initial_batch_size=10)
    diff_mean = float(np.mean(d))

    # Paired t-test (for completeness). Primary decision is CI(d) includes zero or not.
    _, p = stats.ttest_rel(b2, c2)

    sig = (not (bm.ci.ci_lower <= 0 <= bm.ci.ci_upper)) and (p < 0.05)
    eff = cohens_d_paired(b2, c2)

    # Precision check on diffs CI
    half_width = bm.ci.ci_width / 2.0
    prec_ratio, ok = precision_check(bm.ci.mean, half_width, precision_target)

    return PairedComparisonResult(
        metric=metric,
        n=n,
        baseline_mean=b_mean,
        capa_mean=c_mean,
        mean_type=mean_type,
        improvement_pct=float(improvement),
        diff_mean=diff_mean,
        diff_ci_low=float(bm.ci.ci_lower),
        diff_ci_high=float(bm.ci.ci_upper),
        p_value=float(p),
        significant=bool(sig),
        effect_size_d=float(eff),
        speedup_geo_mean=float(speed_gm),
        speedup_ci_low=float(speed_lo),
        speedup_ci_high=float(speed_hi),
        precision_halfwidth_over_mean=float(prec_ratio),
        precision_target=float(precision_target),
        adequate_precision=bool(ok),
        metadata={
            "batch_valid": bool(bm.is_valid),
            "batch_size": int(bm.batch_size),
            "num_batches": int(bm.num_batches),
            "autocov_lag1": float(bm.autocovariance_lag1),
            "transient_removed_each": int(max(int(transient_frac * len(baseline)), int(transient_frac * len(capa)))),
        },
    )


# =============================================================================
# REPORTING + PLOTS
# =============================================================================

def plot_box(output_dir: str, title: str, baseline: np.ndarray, capa: np.ndarray, fname: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.boxplot([baseline, capa], labels=["Baseline", "CAPA+"], showfliers=False)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Academic analysis for CAPA+ v3 CLEAN")
    ap.add_argument("--data-dir", default="./results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--output-dir", default="./analysis_output")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--transient-frac", type=float, default=0.10)
    ap.add_argument("--precision-target", type=float, default=0.10)  # ±10% default
    args = ap.parse_args()

    if args.latest or not args.run_id:
        run_id = find_latest_run_id(args.data_dir)
    else:
        run_id = args.run_id

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    base_df = load_run(args.data_dir, run_id, "baseline")
    capa_df = load_run(args.data_dir, run_id, "capa")

    # Metrics to analyze (extendable)
    metrics = [
        ("latency_p95_ms", "latency"),
        ("latency_avg_ms", "latency"),
        ("arrival_rate_rps", "throughput"),
        ("cpu_utilization", "other"),
    ]

    services = sorted(set(base_df["service"].unique()).intersection(set(capa_df["service"].unique())))
    patterns = sorted(set(base_df["pattern"].unique()).intersection(set(capa_df["pattern"].unique())))

    results_rows: List[Dict[str, Any]] = []

    # ---- System-level: pool across services by pairing key (service included) ----
    print("=" * 78)
    print(f"CAPA+ ACADEMIC REPORT (Jain 1991) | run_id={run_id}")
    print("=" * 78)
    print(f"Services: {len(services)} | Patterns: {len(patterns)}")
    print(f"Confidence: {args.confidence:.2f} | Transient: {args.transient_frac:.0%} | Precision target: ±{args.precision_target:.0%}\n")

    for metric_col, mtype in metrics:
        b, c, merged = pair_by_key(base_df, capa_df, metric_col)
        res = compare_paired(
            b, c,
            metric=metric_col,
            metric_type=mtype,
            confidence=args.confidence,
            transient_frac=args.transient_frac,
            precision_target=args.precision_target,
        )

        print(f"--- SYSTEM (pooled) | {metric_col} | {res.mean_type} ---")
        print(f"n(paired after transient): {res.n}")
        print(f"Baseline mean: {res.baseline_mean:.4f}")
        print(f"CAPA+ mean:    {res.capa_mean:.4f}")
        print(f"Improvement:   {res.improvement_pct:+.2f}%")
        print(f"Diff (B-C):    {res.diff_mean:.4f}")
        print(f"CI(diff):      [{res.diff_ci_low:.4f}, {res.diff_ci_high:.4f}]  (batch_valid={res.metadata['batch_valid']})")
        print(f"p-value:       {res.p_value:.3e} | significant={res.significant}")
        print(f"Effect size d: {res.effect_size_d:.3f}")
        if mtype == "latency":
            print(f"Speedup (geo): {res.speedup_geo_mean:.3f}x  CI=[{res.speedup_ci_low:.3f},{res.speedup_ci_high:.3f}]")
        print(f"Precision:     halfwidth/mean={res.precision_halfwidth_over_mean:.3f}  adequate={res.adequate_precision}")
        print()

        results_rows.append({
            "scope": "system",
            "service": "ALL",
            **asdict(res),
        })

        # plot (system)
        if metric_col == "latency_p95_ms":
            b2, _ = remove_transient(b, args.transient_frac)
            c2, _ = remove_transient(c, args.transient_frac)
            n = min(len(b2), len(c2))
            plot_box(out_dir, "System P95 Latency (Steady-State)", b2[:n], c2[:n], "system_p95_latency_box.png")

    # ---- Per-service strict comparison ----
    for svc in services:
        bsvc = base_df[base_df["service"] == svc]
        csvc = capa_df[capa_df["service"] == svc]

        for metric_col, mtype in metrics:
            b, c, merged = pair_by_key(bsvc, csvc, metric_col)
            res = compare_paired(
                b, c,
                metric=f"{metric_col}",
                metric_type=mtype,
                confidence=args.confidence,
                transient_frac=args.transient_frac,
                precision_target=args.precision_target,
            )
            results_rows.append({
                "scope": "service",
                "service": svc,
                **asdict(res),
            })

        # plot per service for p95 latency
        b, c, _ = pair_by_key(bsvc, csvc, "latency_p95_ms")
        b2, _ = remove_transient(b, args.transient_frac)
        c2, _ = remove_transient(c, args.transient_frac)
        n = min(len(b2), len(c2))
        if n >= 2:
            plot_box(out_dir, f"{svc} P95 Latency (Steady-State)", b2[:n], c2[:n], f"{svc}_p95_latency_box.png")

    # Save summary CSV
    summary = pd.DataFrame(results_rows)
    out_csv = os.path.join(out_dir, f"academic_summary_{run_id}.csv")
    summary.to_csv(out_csv, index=False)

    # Save JSON summary too
    out_json = os.path.join(out_dir, f"academic_summary_{run_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_rows, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    print(f"Figures: {out_dir}")


if __name__ == "__main__":
    main()
