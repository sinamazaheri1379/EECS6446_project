#!/usr/bin/env python3
"""
generate_unified_comparison.py
========================================================
Unified Academic + Rigorous Statistical Analysis for CAPA+ Experiments

This module consolidates and corrects:
- statistical_analysis (rigorous CI, batch means, transient removal, paired comparison)
- generate_academic_analysis (full report, plots, overfitting, shadow mode)

Primary references:
- Jain (1991): The Art of Computer Systems Performance Analysis (Ch. 11-13, 25, 27)
- INTROD_1: Confidence intervals, steady-state and measurement basics
- Harchol-Balter (2013): Performance Modeling & Design of Computer Systems (supporting context)

Key guarantees:
1) Confidence Intervals computed correctly (t or z as appropriate)
2) Batch Means for autocorrelated observations (Jain 25.5.2) + autocov validity check
3) Transient Removal (Jain 25.3) with selectable heuristic
4) Paired Comparison (Jain 13.4.1) on paired observations + CI on mean difference
5) Correct mean selection (Jain Ch. 12): arithmetic/geometric/harmonic + weighted CPU utilization
6) Sample size adequacy / precision checking (Jain 13.9)
7) Academic-quality reporting + plots (matplotlib only)

Author: EECS6446 Cloud Computing Project
Date: December 2025
"""

import os
import sys
import math
import json
import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

# Plotting (matplotlib only; avoid seaborn dependency)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

SLA_THRESHOLDS = {
    "frontend": {"latency_ms": 200, "cpu_target": 0.5},
    "recommendationservice": {"latency_ms": 100, "cpu_target": 0.5},
    "productcatalogservice": {"latency_ms": 100, "cpu_target": 0.5},
    "cartservice": {"latency_ms": 100, "cpu_target": 0.5},
    "checkoutservice": {"latency_ms": 150, "cpu_target": 0.5},
}

# Keep colors stable but do not rely on seaborn
COLORS = {
    "baseline": "#2ecc71",
    "capa": "#3498db",
    "shadow": "#9b59b6",
    "warning": "#e74c3c",
    "neutral": "#95a5a6",
}


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

    def __str__(self) -> str:
        half = self.ci_width / 2
        return (
            f"{self.mean:.3f} ± {half:.3f} "
            f"({self.confidence_level*100:.0f}% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}])"
        )


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
    system_a_name: str
    system_b_name: str
    mean_difference: float  # A - B
    std_difference: float
    ci: ConfidenceInterval
    is_significant: bool
    t_statistic: float
    p_value: float
    n_pairs: int


@dataclass
class ComparisonResult:
    metric_name: str
    baseline_value: float
    capa_value: float
    improvement_pct: float
    mean_type: str
    p_value: Optional[float]
    significant: bool
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class OverfittingAnalysis:
    train_p95: float
    test_p95: float
    generalization_ratio: float
    is_overfitting: bool
    severity: str
    recommendations: List[str]


@dataclass
class SteadyStateAnalysis:
    is_steady_state: bool
    transient_length: int
    reason: str
    arrival_rate_cv: float
    response_time_cv: float
    utilization_trend: float


# =============================================================================
# CORE STATISTICAL UTILITIES (Jain Ch. 13)
# =============================================================================

def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
    method: str = "naive"
) -> ConfidenceInterval:
    """
    CI for mean: x̄ ± t_{1-α/2; n-1} * (s/√n) (Jain 13.2)
    Uses z for large n (>=30) as a practical convention.
    Assumes IID unless otherwise stated.
    """
    n = len(data)
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0.0, confidence, 0, method)
    if n < 2:
        x = float(data[0])
        return ConfidenceInterval(x, 0.0, x, x, 0.0, confidence, n, method)

    x_bar = float(np.mean(data))
    s = float(np.std(data, ddof=1))

    alpha = 1.0 - confidence
    if n >= 30:
        z = float(stats.norm.ppf(1 - alpha / 2))
        half_width = z * s / math.sqrt(n)
    else:
        t_val = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        half_width = t_val * s / math.sqrt(n)

    return ConfidenceInterval(
        mean=x_bar,
        std=s,
        ci_lower=x_bar - half_width,
        ci_upper=x_bar + half_width,
        ci_width=2 * half_width,
        confidence_level=confidence,
        sample_size=n,
        method=method
    )


# =============================================================================
# BATCH MEANS (Jain 25.5.2 + note on autocovariance vs covariance)
# =============================================================================

def batch_means_analysis(
    data: List[float],
    confidence: float = 0.95,
    initial_batch_size: int = 10,
    max_iterations: int = 10,
    min_batches: int = 5,
    autocov_threshold_ratio: float = 0.10
) -> BatchMeansResult:
    """
    Batch means CI for autocorrelated observations (Jain 25.5.2).
    We iteratively increase batch size until lag-1 autocovariance of batch means
    is small relative to variance of batch means.
    """
    N = len(data)
    if N < 20:
        ci = compute_confidence_interval(data, confidence, method="naive_smallN")
        return BatchMeansResult(
            overall_mean=ci.mean,
            batch_means=[ci.mean],
            batch_size=N,
            num_batches=1,
            variance_of_batch_means=ci.std ** 2,
            autocovariance_lag1=0.0,
            ci=ci,
            is_valid=False,
            warning="Batch means needs N>=20; used naive CI."
        )

    batch_size = max(2, int(initial_batch_size))
    best: Optional[BatchMeansResult] = None

    for _ in range(max_iterations):
        m = N // batch_size
        if m < min_batches:
            break

        # batch means
        bmeans = [
            float(np.mean(data[i * batch_size:(i + 1) * batch_size]))
            for i in range(m)
        ]

        overall_mean = float(np.mean(bmeans))
        var_bm = float(np.var(bmeans, ddof=1)) if m > 1 else 0.0

        # Jain 27.3 autocovariance at lag 1:
        # R1 = (1/(m-1)) * sum_{i=1..m-1} (x_i - x̄)(x_{i+1} - x̄)
        if m >= 2:
            acov = sum(
                (bmeans[i] - overall_mean) * (bmeans[i + 1] - overall_mean)
                for i in range(m - 1)
            ) / (m - 1)
        else:
            acov = 0.0

        # CI based on batch means treated as approximately independent
        alpha = 1.0 - confidence
        t_val = float(stats.t.ppf(1 - alpha / 2, df=m - 1))
        std_of_mean = math.sqrt(var_bm / m) if m > 0 else 0.0
        half_width = t_val * std_of_mean

        ci = ConfidenceInterval(
            mean=overall_mean,
            std=math.sqrt(var_bm) if var_bm >= 0 else 0.0,
            ci_lower=overall_mean - half_width,
            ci_upper=overall_mean + half_width,
            ci_width=2 * half_width,
            confidence_level=confidence,
            sample_size=N,
            method=f"batch_means(n={batch_size}, m={m})"
        )

        is_valid = True
        if var_bm > 0:
            is_valid = abs(acov) < (autocov_threshold_ratio * var_bm)

        res = BatchMeansResult(
            overall_mean=overall_mean,
            batch_means=bmeans,
            batch_size=batch_size,
            num_batches=m,
            variance_of_batch_means=var_bm,
            autocovariance_lag1=acov,
            ci=ci,
            is_valid=is_valid,
            warning=None
        )
        best = res

        if is_valid:
            return res

        batch_size *= 2

    if best is not None:
        best.warning = "Autocovariance still significant; returned best available batch size."
        return best

    ci = compute_confidence_interval(data, confidence, method="naive_fallback")
    return BatchMeansResult(
        overall_mean=ci.mean,
        batch_means=[ci.mean],
        batch_size=N,
        num_batches=1,
        variance_of_batch_means=ci.std ** 2,
        autocovariance_lag1=0.0,
        ci=ci,
        is_valid=False,
        warning="Fallback to naive CI."
    )


# =============================================================================
# TRANSIENT REMOVAL (Jain 25.3)
# =============================================================================

def detect_transient_period(data: List[float], method: str = "batch_variance") -> int:
    """
    Heuristic transient detection. For serious studies, you would usually
    also plot moving averages / use domain-specific warm-up logic.
    """
    N = len(data)
    if N == 0:
        return 0

    if method == "rule_of_thumb":
        return int(0.10 * N)

    if method == "moving_average":
        window = max(5, N // 20)
        ma = np.convolve(np.asarray(data), np.ones(window) / window, mode="valid")
        deriv = np.abs(np.diff(ma))
        thresh = float(np.median(deriv) * 0.5) if len(deriv) else 0.0
        for i, d in enumerate(deriv):
            if d < thresh:
                return i + window
        return int(0.10 * N)

    # batch_variance (Jain 25.3.6 heuristic)
    min_bs = max(2, N // 50)
    max_bs = max(min_bs, N // 5)

    variances = []
    batch_sizes = []
    for bs in range(min_bs, max_bs + 1, min_bs):
        m = N // bs
        if m < 3:
            break
        bmeans = [np.mean(data[i * bs:(i + 1) * bs]) for i in range(m)]
        variances.append(float(np.var(bmeans, ddof=1)))
        batch_sizes.append(bs)

    if not variances:
        return int(0.10 * N)

    peak = int(np.argmax(variances))
    return int(batch_sizes[peak]) if peak > 0 else int(0.10 * N)


def remove_transient(
    data: List[float],
    transient_length: Optional[int] = None,
    method: str = "batch_variance"
) -> Tuple[List[float], int]:
    if transient_length is None:
        transient_length = detect_transient_period(data, method=method)
    transient_length = max(0, min(transient_length, len(data)))
    return data[transient_length:], transient_length


# =============================================================================
# PAIRED COMPARISON (Jain 13.4.1)
# =============================================================================

def paired_comparison(
    system_a: List[float],
    system_b: List[float],
    system_a_name: str = "System A",
    system_b_name: str = "System B",
    confidence: float = 0.95
) -> PairedComparisonResult:
    if len(system_a) != len(system_b):
        raise ValueError(f"Paired comparison requires equal sizes: {len(system_a)} vs {len(system_b)}")
    n = len(system_a)
    if n < 2:
        raise ValueError("Need at least 2 paired observations.")

    diffs = np.asarray(system_a, dtype=float) - np.asarray(system_b, dtype=float)
    d_bar = float(np.mean(diffs))
    s_d = float(np.std(diffs, ddof=1))

    t_stat = d_bar / (s_d / math.sqrt(n)) if s_d > 0 else 0.0
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))

    alpha = 1 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half_width = t_crit * s_d / math.sqrt(n)

    ci = ConfidenceInterval(
        mean=d_bar,
        std=s_d,
        ci_lower=d_bar - half_width,
        ci_upper=d_bar + half_width,
        ci_width=2 * half_width,
        confidence_level=confidence,
        sample_size=n,
        method="paired_comparison"
    )

    return PairedComparisonResult(
        system_a_name=system_a_name,
        system_b_name=system_b_name,
        mean_difference=d_bar,
        std_difference=s_d,
        ci=ci,
        is_significant=not ci.includes_zero(),
        t_statistic=t_stat,
        p_value=p_val,
        n_pairs=n
    )


# =============================================================================
# MEAN SELECTION (Jain Ch. 12)
# =============================================================================

class MeanCalculator:
    @staticmethod
    def arithmetic_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        if len(values) == 0:
            return 0.0
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if np.sum(w) == 0:
                return float(np.mean(values))
            return float(np.average(values, weights=w))
        return float(np.mean(values))

    @staticmethod
    def geometric_mean(values: np.ndarray) -> float:
        v = np.asarray(values, dtype=float)
        v = v[v > 0]
        if len(v) == 0:
            return 0.0
        return float(np.exp(np.mean(np.log(v))))

    @staticmethod
    def harmonic_mean(values: np.ndarray) -> float:
        v = np.asarray(values, dtype=float)
        v = v[v > 0]
        if len(v) == 0:
            return 0.0
        return float(len(v) / np.sum(1.0 / v))

    @staticmethod
    def weighted_cpu_utilization(cpu_utils: np.ndarray, durations: np.ndarray) -> float:
        u = np.asarray(cpu_utils, dtype=float)
        t = np.asarray(durations, dtype=float)
        if len(u) == 0 or len(t) == 0:
            return 0.0
        if len(u) != len(t):
            # fallback: unweighted mean if durations mismatch
            return float(np.mean(u))
        total_time = float(np.sum(t))
        if total_time == 0:
            return 0.0
        return float(np.sum(u * t) / total_time)


# =============================================================================
# EFFECT SIZE (Cohen's d)
# =============================================================================

def cohens_d(baseline: np.ndarray, treatment: np.ndarray) -> float:
    b = np.asarray(baseline, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if len(b) < 2 or len(t) < 2:
        return 0.0
    var1 = float(np.var(b, ddof=1))
    var2 = float(np.var(t, ddof=1))
    n1, n2 = len(b), len(t)
    pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
    if pooled == 0:
        return 0.0
    return float((np.mean(b) - np.mean(t)) / pooled)


# =============================================================================
# SPEEDUP (Geometric mean in ratio space + CI in log space)
# =============================================================================

def compute_speedup_with_ci(
    baseline_times: np.ndarray,
    improved_times: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, ConfidenceInterval]:
    b = np.asarray(baseline_times, dtype=float)
    i = np.asarray(improved_times, dtype=float)
    if len(b) != len(i):
        raise ValueError("Speedup requires paired observations (same length).")
    if len(b) < 2:
        gm = float(b[0] / i[0]) if (len(b) == 1 and i[0] > 0) else 1.0
        ci = ConfidenceInterval(gm, 0.0, gm, gm, 0.0, confidence, len(b), "speedup_smallN")
        return gm, ci

    # speedup = baseline / improved
    speedups = np.where(i > 0, b / i, 1.0)
    speedups = speedups[speedups > 0]
    gm = float(np.exp(np.mean(np.log(speedups))))

    log_s = np.log(speedups)
    log_ci = compute_confidence_interval(list(log_s), confidence=confidence, method="log_speedup")
    ci = ConfidenceInterval(
        mean=gm,
        std=float(np.exp(log_ci.std) - 1.0),
        ci_lower=float(np.exp(log_ci.ci_lower)),
        ci_upper=float(np.exp(log_ci.ci_upper)),
        ci_width=float(np.exp(log_ci.ci_upper) - np.exp(log_ci.ci_lower)),
        confidence_level=confidence,
        sample_size=len(speedups),
        method="geometric_mean_speedup"
    )
    return gm, ci


# =============================================================================
# SAMPLE SIZE ADEQUACY (Jain 13.9)
# =============================================================================

def check_precision(
    data: np.ndarray,
    desired_precision: float = 0.10,
    confidence: float = 0.95,
    assume_iid: bool = True
) -> Tuple[bool, float, int]:
    """
    desired_precision is relative half-width: (CI_half_width / |mean|)
    For autocorrelated series you should compute CI via batch means before using this.
    Here we default to naive CI unless caller pre-processes.
    """
    x = np.asarray(data, dtype=float)
    if len(x) < 2:
        return False, float("inf"), 100

    ci = compute_confidence_interval(list(x), confidence=confidence, method="naive_precision")
    if ci.mean == 0:
        return False, float("inf"), 100

    current_precision = (ci.ci_width / 2) / abs(ci.mean)
    ok = current_precision <= desired_precision
    if ok:
        return True, current_precision, 0

    ratio = (current_precision / desired_precision) ** 2
    additional = int(len(x) * (ratio - 1))
    return False, current_precision, max(0, additional)


# =============================================================================
# STEADY-STATE CHECK (support utility)
# =============================================================================

def check_steady_state(
    arrival_rates: List[float],
    response_times: List[float],
    utilizations: List[float],
    cv_threshold: float = 0.30
) -> SteadyStateAnalysis:
    if len(arrival_rates) < 5 or len(response_times) < 5 or len(utilizations) < 5:
        return SteadyStateAnalysis(
            is_steady_state=False,
            transient_length=0,
            reason="Insufficient data (<5)",
            arrival_rate_cv=0.0,
            response_time_cv=0.0,
            utilization_trend=0.0
        )

    arr = np.asarray(arrival_rates, dtype=float)
    rsp = np.asarray(response_times, dtype=float)
    utl = np.asarray(utilizations, dtype=float)

    arr_mean = float(np.mean(arr))
    rsp_mean = float(np.mean(rsp))

    arr_cv = float(np.std(arr) / arr_mean) if arr_mean > 0 else float("inf")
    rsp_cv = float(np.std(rsp) / rsp_mean) if rsp_mean > 0 else float("inf")

    x = np.arange(len(utl))
    slope, _, _, _, _ = stats.linregress(x, utl)

    reasons = []
    if arr_cv > cv_threshold:
        reasons.append(f"Arrival CV={arr_cv:.2f} > {cv_threshold}")
    if rsp_cv > (cv_threshold + 0.10):
        reasons.append(f"Response CV={rsp_cv:.2f} too high")
    if slope > 0.01:
        reasons.append(f"Utilization trend slope={slope:.4f} > 0.01")

    return SteadyStateAnalysis(
        is_steady_state=(len(reasons) == 0),
        transient_length=0,
        reason="; ".join(reasons) if reasons else "Steady state plausible",
        arrival_rate_cv=arr_cv,
        response_time_cv=rsp_cv,
        utilization_trend=float(slope)
    )


# =============================================================================
# SYSTEM COMPARATOR (unified)
# =============================================================================

class SystemComparator:
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self.mean_calc = MeanCalculator()

    def compare_response_times(
        self,
        baseline_latencies: np.ndarray,
        capa_latencies: np.ndarray,
        use_batch_means_for_diff: bool = True
    ) -> ComparisonResult:
        """
        Response times are additive => arithmetic mean (Jain Ch. 12).
        Statistical inference is best done on paired differences (Jain 13.4.1).
        For autocorrelated data, CI for mean difference should use batch means.
        """
        b = np.asarray(baseline_latencies, dtype=float)
        c = np.asarray(capa_latencies, dtype=float)
        n = min(len(b), len(c))
        b = b[:n]
        c = c[:n]

        baseline_mean = self.mean_calc.arithmetic_mean(b)
        capa_mean = self.mean_calc.arithmetic_mean(c)
        improvement = ((baseline_mean - capa_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0.0

        # Paired t-test if paired, else independent
        p_value = None
        significant = False
        if n >= 2:
            if len(b) == len(c):
                t_stat, p_value = stats.ttest_rel(b, c)
                significant = bool(p_value < 0.05)
            else:
                t_stat, p_value = stats.ttest_ind(b, c)
                significant = bool(p_value < 0.05)

        # CI on difference (baseline - capa) so positive means CAPA+ improved
        ci_tuple = None
        meta: Dict[str, Any] = {}

        if n >= 2 and len(b) == len(c):
            diff = (b - c).tolist()

            if use_batch_means_for_diff and len(diff) >= 20:
                bm = batch_means_analysis(diff, confidence=self.confidence)
                ci_tuple = (bm.ci.ci_lower, bm.ci.ci_upper)
                meta["ci_method"] = "batch_means_on_diff"
                meta["batch_size"] = bm.batch_size
                meta["num_batches"] = bm.num_batches
                meta["autocovariance_lag1"] = bm.autocovariance_lag1
                meta["variance_of_batch_means"] = bm.variance_of_batch_means
                meta["batch_valid"] = bm.is_valid
                if bm.warning:
                    meta["batch_warning"] = bm.warning
            else:
                ci = compute_confidence_interval(diff, confidence=self.confidence, method="naive_on_diff")
                ci_tuple = (ci.ci_lower, ci.ci_upper)
                meta["ci_method"] = "naive_on_diff"

        effect = cohens_d(b, c)

        return ComparisonResult(
            metric_name="Response Time (P95 sample mean)",
            baseline_value=float(baseline_mean),
            capa_value=float(capa_mean),
            improvement_pct=float(improvement),
            mean_type="arithmetic",
            p_value=float(p_value) if p_value is not None else None,
            significant=significant,
            confidence_interval=ci_tuple,
            effect_size=float(effect),
            metadata=meta
        )

    def compare_throughput(self, baseline_tp: np.ndarray, capa_tp: np.ndarray) -> ComparisonResult:
        """
        Throughput is a rate => harmonic mean (Jain Ch. 12.6).
        """
        b = np.asarray(baseline_tp, dtype=float)
        c = np.asarray(capa_tp, dtype=float)
        n = min(len(b), len(c))
        b = b[:n]
        c = c[:n]

        b = np.where(b > 0, b, 0.001)
        c = np.where(c > 0, c, 0.001)

        baseline_mean = self.mean_calc.harmonic_mean(b)
        capa_mean = self.mean_calc.harmonic_mean(c)
        improvement = ((capa_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0.0

        p_value = None
        significant = False
        if n >= 2 and len(b) == len(c):
            _, p_value = stats.ttest_rel(b, c)
            significant = bool(p_value < 0.05)

        effect = cohens_d(b, c)

        return ComparisonResult(
            metric_name="Throughput (RPS)",
            baseline_value=float(baseline_mean),
            capa_value=float(capa_mean),
            improvement_pct=float(improvement),
            mean_type="harmonic",
            p_value=float(p_value) if p_value is not None else None,
            significant=significant,
            confidence_interval=None,
            effect_size=float(effect),
            metadata={}
        )

    def compare_speedup(self, baseline_lat: np.ndarray, capa_lat: np.ndarray) -> ComparisonResult:
        """
        Speedup ratio => geometric mean (Jain Ch. 12.5 + ratio games Ch. 11).
        """
        b = np.asarray(baseline_lat, dtype=float)
        c = np.asarray(capa_lat, dtype=float)
        n = min(len(b), len(c))
        b = b[:n]
        c = c[:n]

        gm, ci = compute_speedup_with_ci(b, c, confidence=self.confidence)
        # improvement_pct relative to 1.0
        return ComparisonResult(
            metric_name="Speedup (Baseline / CAPA+)",
            baseline_value=1.0,
            capa_value=float(gm),
            improvement_pct=float((gm - 1.0) * 100.0),
            mean_type="geometric",
            p_value=None,
            significant=bool(ci.ci_lower > 1.0),
            confidence_interval=(ci.ci_lower, ci.ci_upper),
            effect_size=None,
            metadata={"ci_method": ci.method}
        )

    def compare_cpu_utilization(
        self,
        baseline_cpu: np.ndarray,
        baseline_dur: np.ndarray,
        capa_cpu: np.ndarray,
        capa_dur: np.ndarray
    ) -> ComparisonResult:
        """
        CPU utilization must be weighted by duration (Jain Example 12.3).
        """
        b_u = np.asarray(baseline_cpu, dtype=float)
        b_t = np.asarray(baseline_dur, dtype=float)
        c_u = np.asarray(capa_cpu, dtype=float)
        c_t = np.asarray(capa_dur, dtype=float)

        baseline_w = self.mean_calc.weighted_cpu_utilization(b_u, b_t)
        capa_w = self.mean_calc.weighted_cpu_utilization(c_u, c_t)
        diff = capa_w - baseline_w

        return ComparisonResult(
            metric_name="CPU Utilization (weighted by duration)",
            baseline_value=float(baseline_w),
            capa_value=float(capa_w),
            improvement_pct=float(diff * 100.0),  # absolute percentage points scaled
            mean_type="weighted_arithmetic",
            p_value=None,
            significant=bool(abs(diff) > 0.10),
            confidence_interval=None,
            effect_size=None,
            metadata={"note": "Weighted by duration per Jain Ex. 12.3"}
        )

    def full_comparison(self, baseline_df: pd.DataFrame, capa_df: pd.DataFrame) -> Dict[str, ComparisonResult]:
        results: Dict[str, ComparisonResult] = {}

        if "p95_latency_ms" in baseline_df.columns and "p95_latency_ms" in capa_df.columns:
            results["response_time"] = self.compare_response_times(
                baseline_df["p95_latency_ms"].values,
                capa_df["p95_latency_ms"].values
            )
            results["speedup"] = self.compare_speedup(
                baseline_df["p95_latency_ms"].values,
                capa_df["p95_latency_ms"].values
            )

        if "throughput_rps" in baseline_df.columns and "throughput_rps" in capa_df.columns:
            results["throughput"] = self.compare_throughput(
                baseline_df["throughput_rps"].values,
                capa_df["throughput_rps"].values
            )

        if all(col in baseline_df.columns for col in ["cpu_util", "duration_sec"]) and \
           all(col in capa_df.columns for col in ["cpu_util", "duration_sec"]):
            results["cpu_utilization"] = self.compare_cpu_utilization(
                baseline_df["cpu_util"].values,
                baseline_df["duration_sec"].values,
                capa_df["cpu_util"].values,
                capa_df["duration_sec"].values
            )

        return results


# =============================================================================
# OVERFITTING DETECTOR
# =============================================================================

class OverfittingDetector:
    def __init__(self, threshold_mild: float = 1.2, threshold_severe: float = 1.5):
        self.threshold_mild = threshold_mild
        self.threshold_severe = threshold_severe

    def analyze(self, train_results: List[Dict], test_results: List[Dict]) -> OverfittingAnalysis:
        train_p95 = float(np.mean([r.get("p95_latency_ms", 0.0) for r in train_results])) if train_results else 0.0
        test_p95 = float(np.mean([r.get("p95_latency_ms", 0.0) for r in test_results])) if test_results else 0.0
        ratio = (test_p95 / train_p95) if train_p95 > 0 else 1.0

        if ratio >= self.threshold_severe:
            return OverfittingAnalysis(
                train_p95=train_p95,
                test_p95=test_p95,
                generalization_ratio=ratio,
                is_overfitting=True,
                severity="severe",
                recommendations=[
                    "Reduce state space further (fewer buckets)",
                    "Increase exploration (higher epsilon_min)",
                    "Add more training patterns",
                    "Consider stronger regularization / smoothing of Q-updates",
                    "Consider function approximation instead of pure tabular Q-learning",
                ],
            )
        if ratio >= self.threshold_mild:
            return OverfittingAnalysis(
                train_p95=train_p95,
                test_p95=test_p95,
                generalization_ratio=ratio,
                is_overfitting=True,
                severity="mild",
                recommendations=[
                    "Slightly reduce state space",
                    "Increase epsilon_min moderately",
                    "Add more diverse training patterns",
                    "Increase replay buffer size (if used)",
                ],
            )
        return OverfittingAnalysis(
            train_p95=train_p95,
            test_p95=test_p95,
            generalization_ratio=ratio,
            is_overfitting=False,
            severity="none",
            recommendations=["Model generalizes well; proceed with broader validation or shadow rollout."],
        )


# =============================================================================
# SHADOW MODE ANALYZER (kept simple & correct)
# =============================================================================

class ShadowModeAnalyzer:
    """
    decisions items: {"rl_action": int, "baseline_action": int, ...}
    Expect action coding consistent with your system (e.g., 2=scale_up).
    """
    def analyze(self, decisions: List[Dict]) -> Dict[str, Any]:
        if not decisions:
            return {
                "total_decisions": 0,
                "agreement_rate": 0.0,
                "rl_scale_up_rate": 0.0,
                "baseline_scale_up_rate": 0.0,
                "disagreements": [],
            }

        total = len(decisions)
        agreed = sum(1 for d in decisions if d.get("rl_action") == d.get("baseline_action"))

        rl_scale_up = sum(1 for d in decisions if d.get("rl_action") == 2)
        bl_scale_up = sum(1 for d in decisions if d.get("baseline_action") == 2)

        disagreements = [d for d in decisions if d.get("rl_action") != d.get("baseline_action")]

        return {
            "total_decisions": total,
            "agreement_rate": agreed / total,
            "rl_scale_up_rate": rl_scale_up / total,
            "baseline_scale_up_rate": bl_scale_up / total,
            "disagreements": disagreements[:10],
        }


# =============================================================================
# VISUALIZATION (matplotlib only)
# =============================================================================

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_latency_comparison(self, baseline: np.ndarray, capa: np.ndarray) -> str:
        b = np.asarray(baseline, dtype=float)
        c = np.asarray(capa, dtype=float)
        n = min(len(b), len(c))
        b, c = b[:n], c[:n]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Boxplot
        ax = axes[0]
        bp = ax.boxplot([b, c], labels=["Baseline HPA", "CAPA+"], patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORS["baseline"])
        bp["boxes"][1].set_facecolor(COLORS["capa"])
        ax.set_ylabel("P95 Latency (ms)")
        ax.set_title("Distribution (samples)")
        ax.grid(True, alpha=0.3)

        # Time series
        ax = axes[1]
        ax.plot(np.arange(n), b, label="Baseline HPA", color=COLORS["baseline"], alpha=0.7)
        ax.plot(np.arange(n), c, label="CAPA+", color=COLORS["capa"], alpha=0.7)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("P95 Latency (ms)")
        ax.set_title("Time Series (aligned)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.output_dir, "latency_comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        return out

    def plot_statistical_summary(self, comparisons: Dict[str, ComparisonResult]) -> str:
        metrics = list(comparisons.keys())
        imps = [comparisons[m].improvement_pct for m in metrics]
        pvals = [comparisons[m].p_value if comparisons[m].p_value is not None else 1.0 for m in metrics]
        sigs = [comparisons[m].significant for m in metrics]
        effs = [abs(comparisons[m].effect_size) if comparisons[m].effect_size is not None else 0.0 for m in metrics]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Improvements
        ax = axes[0, 0]
        colors = [COLORS["baseline"] if v >= 0 else COLORS["warning"] for v in imps]
        ax.barh(metrics, imps, color=colors, edgecolor="black")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Improvement (%)")
        ax.set_title("Performance Improvements")

        # Significance: -log10(p)
        ax = axes[0, 1]
        scores = [-math.log10(max(p, 1e-300)) for p in pvals]
        colors = [COLORS["baseline"] if s else COLORS["warning"] for s in sigs]
        ax.barh(metrics, scores, color=colors, edgecolor="black")
        ax.axvline(-math.log10(0.05), color=COLORS["warning"], linestyle="--", label="p=0.05")
        ax.set_xlabel("-log10(p-value)")
        ax.set_title("Statistical Significance")
        ax.legend()

        # Effect sizes
        ax = axes[1, 0]
        ax.barh(metrics, effs, color=COLORS["neutral"], edgecolor="black")
        ax.axvline(0.2, color="gray", linestyle=":", alpha=0.7)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(0.8, color="gray", linestyle="-", alpha=0.7)
        ax.set_xlabel("Cohen's d (absolute)")
        ax.set_title("Effect Sizes")

        # Table summary
        ax = axes[1, 1]
        ax.axis("off")
        rows = []
        for m in metrics:
            r = comparisons[m]
            rows.append([
                r.metric_name,
                r.mean_type,
                f"{r.baseline_value:.3f}",
                f"{r.capa_value:.3f}",
                "Yes" if r.significant else "No",
            ])
        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Mean Type", "Baseline", "CAPA+", "Sig?"],
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title("Summary Table (Jain Ch. 12)")

        plt.tight_layout()
        out = os.path.join(self.output_dir, "statistical_summary.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        return out

    def plot_overfitting(self, analysis: OverfittingAnalysis) -> str:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        xs = ["Train", "Test"]
        ys = [analysis.train_p95, analysis.test_p95]
        colors = [COLORS["baseline"], COLORS["warning"] if analysis.is_overfitting else COLORS["baseline"]]
        ax.bar(xs, ys, color=colors, edgecolor="black")
        ax.set_ylabel("P95 Latency (ms)")
        ax.set_title(f"Overfitting Check (ratio={analysis.generalization_ratio:.2f}x, {analysis.severity})")
        out = os.path.join(self.output_dir, "overfitting.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        return out

    def plot_shadow(self, shadow: Dict[str, Any]) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Agreement pie
        ax = axes[0]
        agree = shadow["agreement_rate"]
        ax.pie([agree, 1 - agree], labels=["Agree", "Disagree"], autopct="%1.1f%%",
               colors=[COLORS["baseline"], COLORS["warning"]], startangle=90)
        ax.set_title(f"Agreement (N={shadow['total_decisions']})")

        # Scale-up rate bars
        ax = axes[1]
        ax.bar(["RL scale-up", "Baseline scale-up"],
               [shadow["rl_scale_up_rate"], shadow["baseline_scale_up_rate"]],
               color=[COLORS["capa"], COLORS["baseline"]],
               edgecolor="black")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.set_title("Scale-up Rates")

        out = os.path.join(self.output_dir, "shadow_mode.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        return out


# =============================================================================
# REPORT GENERATOR (unified)
# =============================================================================

class ReportGenerator:
    def __init__(self, output_dir: str, confidence: float = 0.95):
        self.output_dir = output_dir
        self.fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        self.viz = Visualizer(self.fig_dir)
        self.confidence = confidence

    def generate(
        self,
        baseline_df: pd.DataFrame,
        capa_df: pd.DataFrame,
        train_results: Optional[List[Dict]] = None,
        test_results: Optional[List[Dict]] = None,
        shadow_decisions: Optional[List[Dict]] = None,
        remove_transient_enabled: bool = False,
        transient_method: str = "batch_variance"
    ) -> str:
        lines: List[str] = []
        lines.append("=" * 88)
        lines.append("CAPA+ UNIFIED COMPARISON REPORT (Jain 1991 compliant)")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 88)
        lines.append("")

        # Optional transient removal on latency series
        baseline_used = baseline_df.copy()
        capa_used = capa_df.copy()

        transient_removed = 0
        if remove_transient_enabled and "p95_latency_ms" in baseline_df.columns and len(baseline_df) > 20:
            steady_b, tlen = remove_transient(list(baseline_df["p95_latency_ms"].values), method=transient_method)
            transient_removed = tlen
            # align lengths after removal
            baseline_used = baseline_used.iloc[tlen:].reset_index(drop=True)
            capa_used = capa_used.iloc[tlen:].reset_index(drop=True)

        lines.append("## 1. DATA SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Baseline samples: {len(baseline_used)}")
        lines.append(f"CAPA+ samples: {len(capa_used)}")
        lines.append(f"Transient removed: {transient_removed} (enabled={remove_transient_enabled}, method={transient_method})")
        lines.append("")

        # System comparison
        lines.append("## 2. SYSTEM COMPARISON (Jain Ch. 12 + Ch. 13 + Ch. 25)")
        lines.append("-" * 40)

        comparator = SystemComparator(confidence=self.confidence)
        comparisons = comparator.full_comparison(baseline_used, capa_used)

        for key, r in comparisons.items():
            lines.append(f"\n### {r.metric_name}")
            lines.append(f"Mean Type: {r.mean_type}")
            lines.append(f"Baseline: {r.baseline_value:.4f}")
            lines.append(f"CAPA+: {r.capa_value:.4f}")
            lines.append(f"Improvement: {r.improvement_pct:+.2f}%")

            if r.p_value is not None:
                lines.append(f"p-value: {r.p_value:.6f}")
                lines.append(f"Significant (α=0.05): {'Yes' if r.significant else 'No'}")

            if r.effect_size is not None:
                lines.append(f"Effect size (Cohen's d): {r.effect_size:.4f} ({self._interpret_effect_size(r.effect_size)})")

            if r.confidence_interval is not None:
                lines.append(f"{int(self.confidence*100)}% CI: [{r.confidence_interval[0]:.4f}, {r.confidence_interval[1]:.4f}]")

            if r.metadata:
                lines.append("Metadata:")
                for mk, mv in r.metadata.items():
                    lines.append(f"  - {mk}: {mv}")

        # Precision check (naive; caller should use batch means CI if highly autocorrelated)
        if "p95_latency_ms" in baseline_used.columns and len(baseline_used) >= 10:
            ok, prec, add = check_precision(
                baseline_used["p95_latency_ms"].values,
                desired_precision=0.10,
                confidence=self.confidence
            )
            lines.append("\n### Sample Size Adequacy (Jain 13.9)")
            lines.append(f"Current precision: ±{prec*100:.2f}% of mean (naive CI)")
            lines.append("Target precision: ±10% of mean")
            lines.append("Adequate: " + ("Yes" if ok else f"No (need ~{add} more samples)"))

        # Figures
        if "p95_latency_ms" in baseline_used.columns and "p95_latency_ms" in capa_used.columns:
            fig1 = self.viz.plot_latency_comparison(
                baseline_used["p95_latency_ms"].values,
                capa_used["p95_latency_ms"].values
            )
            lines.append(f"\n[Figure] Latency comparison: {fig1}")

        fig2 = self.viz.plot_statistical_summary(comparisons)
        lines.append(f"[Figure] Statistical summary: {fig2}")

        # Overfitting
        if train_results is not None and test_results is not None and len(train_results) > 0 and len(test_results) > 0:
            lines.append("\n\n## 3. OVERFITTING ANALYSIS")
            lines.append("-" * 40)
            over = OverfittingDetector().analyze(train_results, test_results)
            lines.append(f"Train P95: {over.train_p95:.2f} ms")
            lines.append(f"Test P95: {over.test_p95:.2f} ms")
            lines.append(f"Generalization ratio: {over.generalization_ratio:.2f}x")
            lines.append(f"Overfitting: {'YES' if over.is_overfitting else 'NO'} (severity={over.severity})")
            lines.append("Recommendations:")
            for rec in over.recommendations:
                lines.append(f"  - {rec}")
            fig3 = self.viz.plot_overfitting(over)
            lines.append(f"[Figure] Overfitting: {fig3}")

        # Shadow mode
        if shadow_decisions is not None and len(shadow_decisions) > 0:
            lines.append("\n\n## 4. SHADOW MODE ANALYSIS")
            lines.append("-" * 40)
            shadow = ShadowModeAnalyzer().analyze(shadow_decisions)
            lines.append(f"Total decisions: {shadow['total_decisions']}")
            lines.append(f"Agreement rate: {shadow['agreement_rate']:.2%}")
            lines.append(f"RL scale-up rate: {shadow['rl_scale_up_rate']:.2%}")
            lines.append(f"Baseline scale-up rate: {shadow['baseline_scale_up_rate']:.2%}")
            fig4 = self.viz.plot_shadow(shadow)
            lines.append(f"[Figure] Shadow mode: {fig4}")

        # Methodology notes
        lines.append("\n\n## 5. METHODOLOGY NOTES")
        lines.append("-" * 40)
        lines.append(
            "This report follows Jain (1991) for mean selection (Ch. 12), paired comparisons (Ch. 13.4.1),\n"
            "sample size adequacy (Ch. 13.9), transient removal (Ch. 25.3), and batch means for autocorrelation\n"
            "(Ch. 25.5.2, with autocovariance per Ch. 27.3). Speedups use geometric mean (Ch. 12.5).\n"
        )

        # Executive summary
        lines.append("\n\n## 6. EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        if "response_time" in comparisons:
            r = comparisons["response_time"]
            if r.improvement_pct > 0 and r.significant:
                verdict = "CAPA+ shows statistically significant improvement in response time."
            elif r.improvement_pct > 0:
                verdict = "CAPA+ shows improvement, but not statistically significant."
            elif r.improvement_pct < 0 and r.significant:
                verdict = "CAPA+ shows statistically significant degradation."
            else:
                verdict = "No statistically significant difference detected."
            lines.append(verdict)
            lines.append(f"Response time improvement: {r.improvement_pct:+.2f}%")

        lines.append("\n" + "=" * 88)
        lines.append("END OF REPORT")
        lines.append("=" * 88)

        report_text = "\n".join(lines)
        report_path = os.path.join(self.output_dir, "unified_comparison_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        return report_path

    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        d = abs(d)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_experiment_data(directory: str) -> Tuple[pd.DataFrame, pd.DataFrame, List, List, List]:
    baseline_df = None
    capa_df = None
    train_results: List[Dict] = []
    test_results: List[Dict] = []
    shadow_decisions: List[Dict] = []

    baseline_file = os.path.join(directory, "baseline_results.csv")
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)

    capa_file = os.path.join(directory, "capa_results.csv")
    if os.path.exists(capa_file):
        capa_df = pd.read_csv(capa_file)

    train_file = os.path.join(directory, "train_results.json")
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            train_results = json.load(f)

    test_file = os.path.join(directory, "test_results.json")
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            test_results = json.load(f)

    decisions_file = os.path.join(directory, "decisions_history.json")
    if os.path.exists(decisions_file):
        with open(decisions_file, "r", encoding="utf-8") as f:
            shadow_decisions = json.load(f)

    return baseline_df, capa_df, train_results, test_results, shadow_decisions


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict], List[Dict], List[Dict]]:
    np.random.seed(42)
    n = 200
    baseline_df = pd.DataFrame({
        "timestamp": np.arange(n),
        "p95_latency_ms": np.random.normal(200, 35, n),
        "throughput_rps": np.random.normal(500, 50, n),
        "cpu_util": np.random.uniform(0.3, 0.8, n),
        "duration_sec": np.ones(n) * 10,
    })
    capa_df = pd.DataFrame({
        "timestamp": np.arange(n),
        "p95_latency_ms": np.random.normal(180, 30, n),
        "throughput_rps": np.random.normal(520, 40, n),
        "cpu_util": np.random.uniform(0.4, 0.7, n),
        "duration_sec": np.ones(n) * 10,
    })

    train_results = [{"p95_latency_ms": 240}, {"p95_latency_ms": 260}]
    test_results = [{"p95_latency_ms": 280}, {"p95_latency_ms": 300}]
    shadow_decisions = [{"rl_action": 1, "baseline_action": 1} for _ in range(400)] + \
                     [{"rl_action": 2, "baseline_action": 1} for _ in range(100)]

    return baseline_df, capa_df, train_results, test_results, shadow_decisions


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified comparison + academic statistical report for CAPA+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_unified_comparison.py --data-dir ./results --output-dir ./analysis_output
  python generate_unified_comparison.py --baseline-csv baseline.csv --capa-csv capa.csv --output-dir ./out
  python generate_unified_comparison.py --sample
"""
    )
    parser.add_argument("--data-dir", type=str, default="./results", help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="./analysis_output", help="Output directory")
    parser.add_argument("--baseline-csv", type=str, default=None, help="Path to baseline CSV")
    parser.add_argument("--capa-csv", type=str, default=None, help="Path to CAPA+ CSV")
    parser.add_argument("--sample", action="store_true", help="Use generated sample data")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default 0.95)")
    parser.add_argument("--remove-transient", action="store_true", help="Enable transient removal (Jain 25.3)")
    parser.add_argument("--transient-method", type=str, default="batch_variance",
                        choices=["batch_variance", "moving_average", "rule_of_thumb"],
                        help="Transient detection method")

    args = parser.parse_args()

    print("=" * 70)
    print("CAPA+ Unified Comparison Tool (Jain 1991 compliant)")
    print("=" * 70)

    if args.sample:
        baseline_df, capa_df, train_results, test_results, shadow_decisions = create_sample_data()
    elif args.baseline_csv and args.capa_csv:
        baseline_df = pd.read_csv(args.baseline_csv)
        capa_df = pd.read_csv(args.capa_csv)
        train_results, test_results, shadow_decisions = None, None, None
    else:
        baseline_df, capa_df, train_results, test_results, shadow_decisions = load_experiment_data(args.data_dir)

    if baseline_df is None or capa_df is None:
        print("\nERROR: Could not load baseline or CAPA+ data.")
        sys.exit(1)

    gen = ReportGenerator(args.output_dir, confidence=args.confidence)
    report_path = gen.generate(
        baseline_df=baseline_df,
        capa_df=capa_df,
        train_results=train_results,
        test_results=test_results,
        shadow_decisions=shadow_decisions,
        remove_transient_enabled=args.remove_transient,
        transient_method=args.transient_method
    )

    print("\n" + "=" * 70)
    print(f"Report saved: {report_path}")
    print(f"Figures saved: {os.path.join(args.output_dir, 'figures')}")
    print("=" * 70)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
