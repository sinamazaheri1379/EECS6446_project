#!/usr/bin/env python3
"""
generate_unified_comparison_academic.py
========================================================
Academic / Statistically rigorous comparison for CAPA+ experiments
Following Jain (1991), Chapters 12, 13, 25.

Key features:
- Loads paired baseline_run_*.csv and capa_run_*.csv from results/<exp_name>/
- Aligns paired observations using tick + service (+ pattern/step_idx/users)
- Transient removal (default: first 10%)
- Batch Means CI for autocorrelated samples (Jain 25.5.2)
- Paired comparison CI on differences (Jain 13.4.1)
- Geometric mean speedup with CI in log-space (Jain 12.5)
- Throughput uses harmonic mean (rates)
- Sample size adequacy check (Jain 13.9) using CI precision
- Per-service comparison tables + optional plots

Usage:
  python generate_unified_comparison_academic.py --data-dir ./results/capa_experiment
"""

import os
import math
import glob
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Optional plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from collections import defaultdict

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
    confidence: float
    n: int
    method: str

    def includes_zero(self) -> bool:
        return self.ci_lower <= 0.0 <= self.ci_upper


@dataclass
class BatchMeansResult:
    overall_mean: float
    batch_means: List[float]
    batch_size: int
    num_batches: int
    var_batch_means: float
    autocov_lag1: float
    ci: ConfidenceInterval
    is_valid: bool
    warning: Optional[str] = None


@dataclass
class PairedComparison:
    metric: str
    baseline_mean: float
    capa_mean: float
    improvement_pct: float
    p_value: float
    significant: bool
    diff_ci: ConfidenceInterval
    effect_size_d: float
    notes: List[str]


# =============================================================================
# CORE STAT UTILS (Jain)
# =============================================================================

def compute_ci_naive(x: np.ndarray, confidence: float = 0.95, method: str = "naive") -> ConfidenceInterval:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return ConfidenceInterval(0, 0, 0, 0, 0, confidence, n, method)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    alpha = 1.0 - confidence
    tval = float(stats.t.ppf(1 - alpha/2, df=n-1))
    hw = tval * s / math.sqrt(n)
    return ConfidenceInterval(m, s, m-hw, m+hw, 2*hw, confidence, n, method)


def batch_means_ci(x: np.ndarray, confidence: float = 0.95, initial_batch: int = 10) -> BatchMeansResult:
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N < 20:
        ci = compute_ci_naive(x, confidence, method="naive_smallN")
        return BatchMeansResult(ci.mean, [ci.mean], N, 1, 0.0, 0.0, ci, False, "N<20 (naive)")

    batch = max(2, int(initial_batch))
    best = None

    for _ in range(10):
        m = N // batch
        if m < 5:
            break

        bmeans = [float(np.mean(x[i*batch:(i+1)*batch])) for i in range(m)]
        overall = float(np.mean(bmeans))
        var_bm = float(np.var(bmeans, ddof=1)) if m > 1 else 0.0

        # lag-1 autocovariance of batch means
        acov = 0.0
        if m >= 2:
            acov = sum((bmeans[i] - overall) * (bmeans[i+1] - overall) for i in range(m-1)) / (m-1)

        alpha = 1.0 - confidence
        tval = float(stats.t.ppf(1 - alpha/2, df=m-1))
        hw = tval * math.sqrt(var_bm / m) if m > 0 else 0.0
        ci = ConfidenceInterval(overall, math.sqrt(var_bm), overall-hw, overall+hw, 2*hw, confidence, N, f"batch(n={batch})")

        # Jain-style heuristic: batch means should be "less autocorrelated"
        is_valid = (var_bm > 0) and (abs(acov) < (0.1 * var_bm))
        res = BatchMeansResult(overall, bmeans, batch, m, var_bm, acov, ci, is_valid)
        best = res
        if is_valid:
            return res

        batch *= 2

    if best:
        best.warning = "High autocorrelation remained after batch growth"
        return best

    ci = compute_ci_naive(x, confidence, method="naive_fallback")
    return BatchMeansResult(ci.mean, [ci.mean], N, 1, 0.0, 0.0, ci, False, "fallback")


def remove_transient_fixed_fraction(x: np.ndarray, frac: float = 0.10) -> Tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=float)
    if len(x) < 50:
        return x, 0
    k = int(frac * len(x))
    return x[k:], k


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    pooled = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)) if (n1+n2-2) > 0 else 0.0
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def geometric_mean_ci_speedup(baseline: np.ndarray, capa: np.ndarray, confidence: float = 0.95) -> Tuple[float, ConfidenceInterval]:
    """
    Speedup s_i = baseline_i / capa_i
    Use geometric mean: exp(mean(log(s_i))) with CI in log-space (Jain 12.5).
    """
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(capa, dtype=float)
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]

    # avoid division by zero and invalids
    mask = (b > 0) & (c > 0)
    s = b[mask] / c[mask]
    if len(s) < 2:
        return 0.0, ConfidenceInterval(0, 0, 0, 0, 0, confidence, len(s), "geo_speedup_insufficient")

    logs = np.log(s)
    m = float(np.mean(logs))
    sd = float(np.std(logs, ddof=1))
    alpha = 1.0 - confidence
    tval = float(stats.t.ppf(1 - alpha/2, df=len(logs)-1))
    hw = tval * sd / math.sqrt(len(logs))

    gm = float(np.exp(m))
    ci_low = float(np.exp(m - hw))
    ci_up = float(np.exp(m + hw))
    ci = ConfidenceInterval(gm, sd, ci_low, ci_up, ci_up - ci_low, confidence, len(logs), "geo_speedup_logspace")
    return gm, ci


def harmonic_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    return float(len(x) / np.sum(1.0 / x))


def sample_size_adequacy(ci: ConfidenceInterval, desired_precision: float = 0.10,
                         abs_threshold: float = 0.01) -> Tuple[bool, float]:
    """
    Jain 13.9: Check if CI half-width / |mean| <= desired_precision.
    
    For near-zero means (e.g., failure_rate ≈ 0), uses absolute CI width
    instead of relative precision to avoid division by zero / inf.
    
    Args:
        ci: Confidence interval result
        desired_precision: Relative precision threshold (default 10%)
        abs_threshold: Absolute CI half-width threshold for near-zero means
    
    Returns:
        (adequate, current_precision) where precision is relative for
        normal means, or absolute half-width for near-zero means.
    """
    half = ci.ci_width / 2.0
    
    # Handle near-zero means: use absolute width instead of relative
    if abs(ci.mean) < 1e-6:
        # For metrics like failure_rate that are often ~0,
        # check if the CI is tight in absolute terms
        adequate = (half < abs_threshold)
        return adequate, half  # Return absolute width, not relative
    
    # Normal case: relative precision
    prec = abs(half / ci.mean)
    return (prec <= desired_precision), float(prec)


# =============================================================================
# CSV LOADING
# =============================================================================

def read_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != len(header):
                # tolerate commas in notes by truncation
                parts = parts[:len(header)]
                parts += [""] * (len(header) - len(parts))
            d = dict(zip(header, parts))
            rows.append(d)
    return rows


def latest_two_runs(data_dir: str) -> Tuple[str, str]:
    base = sorted(glob.glob(os.path.join(data_dir, "baseline_run_*.csv")))
    capa = sorted(glob.glob(os.path.join(data_dir, "capa_run_*.csv")))
    if not base or not capa:
        raise FileNotFoundError("Missing baseline_run_*.csv or capa_run_*.csv in data-dir.")
    return base[-1], capa[-1]


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


# =============================================================================
# ALIGNMENT (PAIRED)
# =============================================================================

def align_pairs(baseline_rows: List[Dict[str, Any]],
                capa_rows: List[Dict[str, Any]],
                key_fields: Tuple[str, ...] = ("service","pattern","step_idx","tick")) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Produces per-service aligned arrays for metrics.
    Key idea: align by (service, pattern, step_idx, tick) which is stable in runner.
    Returns:
      { service: { metric_name: np.array([... paired ...]) for baseline/capa } }
    """
    # index baseline by key
    idx_b = {}
    for r in baseline_rows:
        k = tuple(r.get(f, "") for f in key_fields)
        idx_b[k] = r

    per_service = defaultdict(lambda: defaultdict(list))

    for r in capa_rows:
        k = tuple(r.get(f, "") for f in key_fields)
        b = idx_b.get(k)
        if not b:
            continue
        svc = r.get("service", "unknown")

        # metrics to compare
        for metric in ["latency_p95_ms", "latency_avg_ms", "arrival_rate_rps", "cpu_utilization", "failure_rate"]:
            per_service[svc][f"baseline_{metric}"].append(to_float(b.get(metric, 0)))
            per_service[svc][f"capa_{metric}"].append(to_float(r.get(metric, 0)))

    # to arrays
    out = {}
    for svc, d in per_service.items():
        out[svc] = {}
        for k, v in d.items():
            out[svc][k] = np.asarray(v, dtype=float)
    return out


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

def paired_test_with_batch_ci(b: np.ndarray, c: np.ndarray, confidence: float = 0.95) -> Tuple[float, bool, ConfidenceInterval]:
    """
    Jain 13.4.1: treat as paired differences d_i = b_i - c_i, build CI for mean difference.
    Use batch means CI to handle autocorrelation of d.
    """
    n = min(len(b), len(c))
    if n < 2:
        return 1.0, False, ConfidenceInterval(0,0,0,0,0,confidence,n,"insufficient")

    d = b[:n] - c[:n]
    bm = batch_means_ci(d, confidence=confidence, initial_batch=10)
    # p-value from paired t-test (note: assumes approx normality; CI is batch-based)
    _, p = stats.ttest_rel(b[:n], c[:n])
    sig = bool(p < 0.05 and (not bm.ci.includes_zero()))
    return float(p), sig, bm.ci


def improvement_pct(b_mean: float, c_mean: float, higher_is_better: bool) -> float:
    if b_mean == 0:
        return 0.0
    if higher_is_better:
        # improvement means CAPA higher
        return float(((c_mean - b_mean) / b_mean) * 100.0)
    else:
        # improvement means CAPA lower
        return float(((b_mean - c_mean) / b_mean) * 100.0)


def summarize_metric(name: str, b: np.ndarray, c: np.ndarray, metric_type: str,
                     confidence: float = 0.95, transient_frac: float = 0.10) -> PairedComparison:
    """
    Summarize a metric comparison between baseline and CAPA.
    
    Key fix: Transient removal uses the SAME absolute index for both arrays
    to maintain paired alignment.
    """
    notes = []
    
    # Align to common length FIRST
    n_common = min(len(b), len(c))
    b = b[:n_common]
    c = c[:n_common]
    
    # Transient removal: same absolute count for both (maintains pairing)
    if n_common >= 50:
        k = int(transient_frac * n_common)
        b2 = b[k:]
        c2 = c[k:]
        notes.append(f"Removed {k} transient samples ({transient_frac*100:.0f}%)")
    else:
        b2 = b
        c2 = c
        k = 0
    
    n = len(b2)
    if n < 20:
        notes.append("N<20 after transient removal; batch-means validity may be weak.")

    # Mean selection based on metric type (Jain ch. 12)
    if metric_type == "latency":
        # Arithmetic mean for response times
        b_mean = float(np.mean(b2))
        c_mean = float(np.mean(c2))
        higher_is_better = False
    elif metric_type == "throughput":
        # Harmonic mean for rates (Jain 12.4)
        b_mean = harmonic_mean(b2)
        c_mean = harmonic_mean(c2)
        higher_is_better = True
    else:
        b_mean = float(np.mean(b2))
        c_mean = float(np.mean(c2))
        higher_is_better = True

    imp = improvement_pct(b_mean, c_mean, higher_is_better=higher_is_better)
    p, sig, diff_ci = paired_test_with_batch_ci(b2, c2, confidence=confidence)
    d = cohens_d(b2, c2)

    return PairedComparison(
        metric=name,
        baseline_mean=b_mean,
        capa_mean=c_mean,
        improvement_pct=imp,
        p_value=p,
        significant=sig,
        diff_ci=diff_ci,
        effect_size_d=d,
        notes=notes
    )

# =============================================================================
# REPORTING
# =============================================================================

def print_table(rows: List[List[str]], header: List[str]):
    colw = [len(h) for h in header]
    for r in rows:
        for i, cell in enumerate(r):
            colw[i] = max(colw[i], len(cell))
    def fmt_row(r):
        return " | ".join(r[i].ljust(colw[i]) for i in range(len(r)))
    print(fmt_row(header))
    print("-+-".join("-"*w for w in colw))
    for r in rows:
        print(fmt_row(r))


def plot_box(out_dir: str, svc: str, metric: str, b: np.ndarray, c: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.boxplot([b, c], labels=["Baseline", "CAPA"])
    plt.title(f"{svc}: {metric} (steady-state, after transient removal)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{svc}_{metric}_box.png"))
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing baseline_run_*.csv and capa_run_*.csv")
    ap.add_argument("--output-dir", default="./analysis_output")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--transient-frac", type=float, default=0.10)
    ap.add_argument("--precision", type=float, default=0.10, help="Desired CI half-width/mean threshold (Jain 13.9)")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    base_path, capa_path = latest_two_runs(args.data_dir)
    print("="*70)
    print("CAPA+ ACADEMIC COMPARISON REPORT (Jain 1991)")
    print("="*70)
    print(f"Baseline CSV: {os.path.basename(base_path)}")
    print(f"CAPA CSV:     {os.path.basename(capa_path)}")

    baseline_rows = read_csv(base_path)
    capa_rows = read_csv(capa_path)

    aligned = align_pairs(baseline_rows, capa_rows)

    if not aligned:
        print("No aligned paired observations found. Ensure both runs used same seed/patterns.")
        return

    # Global aggregate across services (concatenate)
    global_metrics = defaultdict(list)
    for svc, d in aligned.items():
        for k, arr in d.items():
            global_metrics[k].append(arr)
    for k in list(global_metrics.keys()):
        global_metrics[k] = np.concatenate(global_metrics[k]) if global_metrics[k] else np.array([])

    # Metrics to analyze
    metric_defs = [
        ("latency_p95_ms", "latency"),
        ("latency_avg_ms", "latency"),
        ("arrival_rate_rps", "throughput"),
        ("cpu_utilization", "other"),
        ("failure_rate", "other"),
    ]

    print("\n--- GLOBAL (All services pooled; interpret cautiously if latency is global aggregate) ---")
    global_rows = []
    for mname, mtype in metric_defs:
        b = global_metrics.get(f"baseline_{mname}", np.array([]))
        c = global_metrics.get(f"capa_{mname}", np.array([]))
        if len(b) < 2 or len(c) < 2:
            continue
        res = summarize_metric(mname, b, c, mtype, confidence=args.confidence, transient_frac=args.transient_frac)
        adequate, prec = sample_size_adequacy(res.diff_ci, desired_precision=args.precision)
        global_rows.append([
            mname,
            f"{res.baseline_mean:.4f}",
            f"{res.capa_mean:.4f}",
            f"{res.improvement_pct:+.2f}%",
            f"{res.p_value:.2e}",
            "YES" if res.significant else "NO",
            f"[{res.diff_ci.ci_lower:.4f},{res.diff_ci.ci_upper:.4f}]",
            f"d={res.effect_size_d:.2f}",
            f"prec={prec*100:.1f}% {'OK' if adequate else 'LOW'}"
        ])
    print_table(global_rows, ["Metric","Baseline","CAPA","Δ","p","Sig","CI(diff b-c)","Effect","CI precision"])

    # Speedup on p95 latency (geometric mean)
    b_p95 = global_metrics.get("baseline_latency_p95_ms", np.array([]))
    c_p95 = global_metrics.get("capa_latency_p95_ms", np.array([]))
    if len(b_p95) >= 2 and len(c_p95) >= 2:
        # transient removal applied before speedup
        b2, _ = remove_transient_fixed_fraction(b_p95, args.transient_frac)
        c2, _ = remove_transient_fixed_fraction(c_p95, args.transient_frac)
        gm, ci = geometric_mean_ci_speedup(b2, c2, confidence=args.confidence)
        print("\n--- SPEEDUP (Geometric mean on latency_p95_ms; Jain 12.5) ---")
        print(f"Speedup (Baseline/CAPA): {gm:.4f}x   CI[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]  (n={ci.n})")

    print("\n--- PER-SERVICE COMPARISON ---")
    svc_rows = []
    for svc, d in sorted(aligned.items()):
        b = d.get("baseline_latency_p95_ms", np.array([]))
        c = d.get("capa_latency_p95_ms", np.array([]))
        if len(b) < 2 or len(c) < 2:
            continue
        res = summarize_metric("latency_p95_ms", b, c, "latency",
                               confidence=args.confidence, transient_frac=args.transient_frac)
        adequate, prec = sample_size_adequacy(res.diff_ci, desired_precision=args.precision)
        svc_rows.append([
            svc,
            f"{len(b)}",
            f"{res.baseline_mean:.2f}",
            f"{res.capa_mean:.2f}",
            f"{res.improvement_pct:+.2f}%",
            f"{res.p_value:.2e}",
            "YES" if res.significant else "NO",
            f"[{res.diff_ci.ci_lower:.2f},{res.diff_ci.ci_upper:.2f}]",
            f"{prec*100:.1f}%",
            "OK" if adequate else "LOW"
        ])

        if args.plots:
            # plot after transient
            b2, _ = remove_transient_fixed_fraction(b, args.transient_frac)
            c2, _ = remove_transient_fixed_fraction(c, args.transient_frac)
            plot_box(args.output_dir, svc, "latency_p95_ms", b2, c2)

    print_table(
        svc_rows,
        ["Service","N","Base P95","CAPA P95","Δ","p","Sig","CI(diff b-c)","CI prec","Adeq"]
    )

    # Save JSON report
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "baseline_csv": os.path.basename(base_path),
        "capa_csv": os.path.basename(capa_path),
        "confidence": args.confidence,
        "transient_frac": args.transient_frac,
        "global": global_rows,
        "per_service": svc_rows,
        "notes": [
            "If Locust aggregate latency is used, latency is global and identical per service in the raw data.",
            "Sequential paired runs are not the same as independent replications; for publication-grade results, run multiple replications and compare distributions."
        ]
    }
    with open(os.path.join(args.output_dir, "academic_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved report: {os.path.join(args.output_dir, 'academic_report.json')}")
    if args.plots:
        print(f"Saved plots into: {args.output_dir}")

if __name__ == "__main__":
    main()
