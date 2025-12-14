#!/usr/bin/env python3
"""
meta_analysis_replications.py - Jain 25.5.1 Independent Replications
"""

import glob
import json
import math
import argparse
import numpy as np
from scipy import stats


def meta_analysis(differences, confidence=0.95):
    """
    Jain 25.5.1 Independent Replications Meta-Analysis
    differences: list of per-replication mean differences (Baseline − CAPA)
    """
    R = len(differences)
    if R < 2:
        raise ValueError(f"At least 2 replications required, got {R}")
    
    mu_bar = np.mean(differences)
    sB2 = np.var(differences, ddof=1)
    alpha = 1 - confidence
    tval = stats.t.ppf(1 - alpha / 2, df=R - 1)
    half_width = tval * math.sqrt(sB2 / R)
    
    return {
        "replications": R,
        "mean_difference": float(mu_bar),
        "std_between": float(math.sqrt(sB2)),
        "ci_low": float(mu_bar - half_width),
        "ci_high": float(mu_bar + half_width),
        "ci_width": float(2 * half_width),
        "significant": not (mu_bar - half_width <= 0 <= mu_bar + half_width),
        "confidence": confidence,
    }


def load_replication_results(path_pattern, metric="latency_p95_ms", scope="system"):
    """
    Load per-replication results from academic_summary_*.json files.
    
    Expected structure (from generate_unified_comparison_academic.py):
    {
        "global": [
            ["latency_p95_ms", "123.45", "100.23", "+18.82%", ...],
            ...
        ],
        ...
    }
    """
    diffs = []
    files_found = sorted(glob.glob(path_pattern))
    
    if not files_found:
        raise FileNotFoundError(f"No files matching: {path_pattern}")
    
    print(f"Found {len(files_found)} replication files")
    
    for f in files_found:
        with open(f) as fh:
            data = json.load(fh)
        
        # Handle the actual output format from generate_unified_comparison_academic.py
        global_rows = data.get("global", [])
        
        for row in global_rows:
            if isinstance(row, list) and len(row) > 0:
                row_metric = row[0]
                if row_metric == metric:
                    # row format: [metric, baseline_mean, capa_mean, delta%, ...]
                    baseline_mean = float(row[1])
                    capa_mean = float(row[2])
                    diff = baseline_mean - capa_mean  # Positive = CAPA better
                    diffs.append(diff)
                    print(f"  {f}: {metric} diff = {diff:.3f} ms")
                    break
    
    return diffs


def check_independence_assumptions(diffs):
    """
    Basic checks for independence assumption validity.
    """
    print("\n--- Independence Assumption Checks ---")
    
    # 1. Sample size adequacy
    R = len(diffs)
    if R < 5:
        print(f"⚠️  Only {R} replications (recommend ≥5 for robust CI)")
    else:
        print(f"✅ {R} replications (adequate)")
    
    # 2. Check for trend (simple linear regression)
    if R >= 3:
        x = np.arange(R)
        slope, _, r_value, p_value, _ = stats.linregress(x, diffs)
        if p_value < 0.05:
            print(f"⚠️  Significant trend detected (slope={slope:.3f}, p={p_value:.3f})")
            print("    This may indicate non-stationarity or learning effects")
        else:
            print(f"✅ No significant trend (p={p_value:.3f})")
    
    # 3. Normality check (Shapiro-Wilk, only meaningful for R >= 3)
    if R >= 3:
        stat, p_value = stats.shapiro(diffs)
        if p_value < 0.05:
            print(f"⚠️  Normality rejected (Shapiro p={p_value:.3f})")
            print("    Consider non-parametric methods for small R")
        else:
            print(f"✅ Normality not rejected (p={p_value:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="./results/meta_experiment/academic_summary_*.json",
                    help="Glob pattern for replication summary files")
    ap.add_argument("--metric", default="latency_p95_ms")
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()
    
    diffs = load_replication_results(args.path, metric=args.metric)
    
    if len(diffs) < 2:
        print(f"ERROR: Need at least 2 replications, found {len(diffs)}")
        raise SystemExit(1)
    
    check_independence_assumptions(diffs)
    
    res = meta_analysis(diffs, confidence=args.confidence)
    
    print("\n" + "=" * 60)
    print("META-ANALYSIS ACROSS INDEPENDENT REPLICATIONS (Jain 25.5.1)")
    print("=" * 60)
    print(f"Metric: {args.metric}")
    print(f"Replications: {res['replications']}")
    print(f"Mean diff (Baseline − CAPA): {res['mean_difference']:.3f} ms")
    print(f"Between-replication std: {res['std_between']:.3f} ms")
    print(f"{int(res['confidence']*100)}% CI: [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print(f"CI width: {res['ci_width']:.3f} ms")
    print(f"Statistically significant: {'YES ✅' if res['significant'] else 'NO'}")
    
    if res['mean_difference'] > 0:
        print(f"\nInterpretation: CAPA reduces {args.metric} by {res['mean_difference']:.1f} ms on average")
    else:
        print(f"\nInterpretation: CAPA increases {args.metric} by {-res['mean_difference']:.1f} ms on average")
