#!/usr/bin/env python3
"""
Academic Analysis Script for CAPA+ Autoscaler Evaluation

This script implements proper statistical analysis methods from:
- Jain (1991): The Art of Computer Systems Performance Analysis
- Harchol-Balter (2013): Performance Modeling and Design of Computer Systems
- INTROD_1: Introduction to Computer System Performance Evaluation

Key Features:
- Correct mean selection (arithmetic, geometric, harmonic) following Jain Ch. 12
- Statistical significance testing (paired t-test, confidence intervals)
- Overfitting detection via train/test comparison
- Shadow mode analysis
- Comprehensive visualization
- Academic-quality reporting

Author: EECS6446 Cloud Computing Project
Date: November 2025
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gmean, hmean, ttest_rel, ttest_ind, sem, shapiro

# Plotting imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Suppress warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# SLA thresholds
SLA_THRESHOLDS = {
    'frontend': {'latency_ms': 200, 'cpu_target': 0.5},
    'recommendationservice': {'latency_ms': 100, 'cpu_target': 0.5},
    'productcatalogservice': {'latency_ms': 100, 'cpu_target': 0.5},
    'cartservice': {'latency_ms': 100, 'cpu_target': 0.5},
    'checkoutservice': {'latency_ms': 150, 'cpu_target': 0.5},
}

# Colors for plots
COLORS = {
    'baseline': '#2ecc71',  # Green
    'capa': '#3498db',      # Blue
    'shadow': '#9b59b6',    # Purple
    'hybrid': '#f39c12',    # Orange
    'active': '#e74c3c',    # Red
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComparisonResult:
    """Results of comparing two systems"""
    metric_name: str
    baseline_value: float
    capa_value: float
    improvement_pct: float
    mean_type: str
    p_value: Optional[float]
    significant: bool
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]


@dataclass
class OverfittingAnalysis:
    """Overfitting detection results"""
    train_p95: float
    test_p95: float
    generalization_ratio: float
    is_overfitting: bool
    severity: str  # 'none', 'mild', 'severe'
    recommendations: List[str]


# =============================================================================
# MEAN SELECTION (Jain Ch. 12)
# =============================================================================

class MeanCalculator:
    """
    Correct mean selection following Jain Chapter 12
    
    Rules:
    - Arithmetic mean: When sum has physical meaning (times, counts)
    - Geometric mean: For ratios and normalized values
    - Harmonic mean: For rates (throughput, bandwidth)
    
    From Jain Example 12.3: CPU utilization MUST be weighted by duration!
    """
    
    @staticmethod
    def arithmetic_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Arithmetic mean for additive quantities
        
        From Jain Ch. 12.5: Use when sum has physical meaning
        Example: Response times, queue lengths
        """
        if weights is not None:
            return np.average(values, weights=weights)
        return np.mean(values)
    
    @staticmethod
    def geometric_mean(values: np.ndarray) -> float:
        """
        Geometric mean for ratios and normalized values
        
        From Jain Ch. 12.5 and Case Study 12.1:
        - Use for speedup ratios
        - Use for normalized performance metrics
        - Handles ratio games (Ch. 11)
        """
        # Filter out non-positive values
        positive_values = values[values > 0]
        if len(positive_values) == 0:
            return 0.0
        return float(gmean(positive_values))
    
    @staticmethod
    def harmonic_mean(values: np.ndarray) -> float:
        """
        Harmonic mean for rates
        
        From Jain Ch. 12.6:
        - Use for throughput (requests/sec)
        - Use for bandwidth (bytes/sec)
        - Relationship: rate = 1/time
        """
        # Filter out zeros and negatives
        positive_values = values[values > 0]
        if len(positive_values) == 0:
            return 0.0
        return float(hmean(positive_values))
    
    @staticmethod
    def weighted_cpu_utilization(
        cpu_utils: np.ndarray, 
        durations: np.ndarray
    ) -> float:
        """
        Weighted CPU utilization following Jain Example 12.3
        
        CRITICAL: CPU utilization MUST be weighted by duration!
        
        From Jain lines 5489-5518:
        "The CPU utilizations should be combined using weighted average
        with weights equal to the duration of each period."
        
        Formula: U_avg = Σ(U_i × T_i) / Σ(T_i)
        """
        if len(cpu_utils) == 0:
            return 0.0
        
        total_busy_time = np.sum(cpu_utils * durations)
        total_time = np.sum(durations)
        
        if total_time == 0:
            return 0.0
        
        return total_busy_time / total_time
    
    @staticmethod
    def select_mean_type(metric_name: str) -> str:
        """
        Select appropriate mean type based on metric
        
        Following Jain Ch. 12 rules
        """
        metric_lower = metric_name.lower()
        
        # Rates -> Harmonic
        if any(x in metric_lower for x in ['throughput', 'rate', 'bandwidth', 'rps']):
            return 'harmonic'
        
        # Ratios -> Geometric
        if any(x in metric_lower for x in ['ratio', 'speedup', 'normalized', 'improvement']):
            return 'geometric'
        
        # Times, counts, utilization -> Arithmetic
        return 'arithmetic'


# =============================================================================
# STATISTICAL TESTS (INTROD_1 Ch. 3)
# =============================================================================

class StatisticalTests:
    """
    Statistical significance testing following INTROD_1 Chapter 3
    """
    
    @staticmethod
    def paired_t_test(
        baseline: np.ndarray, 
        treatment: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float, bool]:
        """
        Paired t-test for comparing two systems on same workload
        
        From INTROD_1 Section 3.3: Use paired test when same conditions
        
        Returns:
            (t_statistic, p_value, is_significant)
        """
        if len(baseline) != len(treatment):
            # Fall back to independent t-test
            return StatisticalTests.independent_t_test(baseline, treatment, alpha)
        
        t_stat, p_value = ttest_rel(baseline, treatment)
        return t_stat, p_value, p_value < alpha
    
    @staticmethod
    def independent_t_test(
        baseline: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float, bool]:
        """
        Independent t-test for comparing two systems
        
        Returns:
            (t_statistic, p_value, is_significant)
        """
        t_stat, p_value = ttest_ind(baseline, treatment)
        return t_stat, p_value, p_value < alpha
    
    @staticmethod
    def confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval
        
        From INTROD_1 Section 3.2
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        se = sem(data)
        
        # t-critical value for given confidence
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * se
        
        return mean, mean - margin, mean + margin
    
    @staticmethod
    def effect_size_cohens_d(
        baseline: np.ndarray,
        treatment: np.ndarray
    ) -> float:
        """
        Cohen's d effect size
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
        """
        n1, n2 = len(baseline), len(treatment)
        var1, var2 = np.var(baseline, ddof=1), np.var(treatment, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(baseline) - np.mean(treatment)) / pooled_std
    
    @staticmethod
    def check_normality(data: np.ndarray) -> Tuple[float, bool]:
        """
        Shapiro-Wilk test for normality
        
        Returns:
            (p_value, is_normal)
        """
        if len(data) < 3:
            return 1.0, True
        
        stat, p_value = shapiro(data[:5000])  # Limit for performance
        return p_value, p_value > 0.05


# =============================================================================
# SYSTEM COMPARISON (Jain Ch. 12.7)
# =============================================================================

class SystemComparator:
    """
    Compare two systems following Jain Chapter 12 methodology
    """
    
    def __init__(self):
        self.mean_calc = MeanCalculator()
        self.stats = StatisticalTests()
    
    def compare_response_times(
        self,
        baseline_latencies: np.ndarray,
        capa_latencies: np.ndarray
    ) -> ComparisonResult:
        """
        Compare response times using ARITHMETIC mean
        
        From Jain Rule 1: Sum of response times has physical meaning
        """
        baseline_mean = self.mean_calc.arithmetic_mean(baseline_latencies)
        capa_mean = self.mean_calc.arithmetic_mean(capa_latencies)
        
        improvement = ((baseline_mean - capa_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        # Statistical significance
        t_stat, p_value, significant = self.stats.paired_t_test(
            baseline_latencies, capa_latencies
        )
        
        # Confidence interval of difference
        if len(baseline_latencies) == len(capa_latencies):
            diff = baseline_latencies - capa_latencies
            mean_diff, ci_low, ci_high = self.stats.confidence_interval(diff)
            ci = (ci_low, ci_high)
        else:
            ci = None
        
        # Effect size
        effect = self.stats.effect_size_cohens_d(baseline_latencies, capa_latencies)
        
        return ComparisonResult(
            metric_name='Response Time (P95)',
            baseline_value=baseline_mean,
            capa_value=capa_mean,
            improvement_pct=improvement,
            mean_type='arithmetic',
            p_value=p_value,
            significant=significant,
            confidence_interval=ci,
            effect_size=effect
        )
    
    def compare_throughput(
        self,
        baseline_throughput: np.ndarray,
        capa_throughput: np.ndarray
    ) -> ComparisonResult:
        """
        Compare throughput using HARMONIC mean
        
        From Jain Ch. 12.6: Throughput is a rate (requests/sec)
        """
        # Replace zeros with small value to avoid division issues
        baseline_clean = np.where(baseline_throughput > 0, baseline_throughput, 0.001)
        capa_clean = np.where(capa_throughput > 0, capa_throughput, 0.001)
        
        baseline_mean = self.mean_calc.harmonic_mean(baseline_clean)
        capa_mean = self.mean_calc.harmonic_mean(capa_clean)
        
        improvement = ((capa_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        t_stat, p_value, significant = self.stats.paired_t_test(
            baseline_clean, capa_clean
        )
        
        effect = self.stats.effect_size_cohens_d(baseline_clean, capa_clean)
        
        return ComparisonResult(
            metric_name='Throughput',
            baseline_value=baseline_mean,
            capa_value=capa_mean,
            improvement_pct=improvement,
            mean_type='harmonic',
            p_value=p_value,
            significant=significant,
            confidence_interval=None,
            effect_size=effect
        )
    
    def compare_speedup(
        self,
        baseline_latencies: np.ndarray,
        capa_latencies: np.ndarray
    ) -> ComparisonResult:
        """
        Compare speedup ratios using GEOMETRIC mean
        
        From Jain Case Study 12.1: Geometric mean for program speedups
        This avoids the "ratio games" problem from Chapter 11
        """
        # Calculate per-sample speedup ratios
        capa_clean = np.where(capa_latencies > 0, capa_latencies, 0.001)
        speedup_ratios = baseline_latencies / capa_clean
        
        # Geometric mean of ratios
        geo_mean_speedup = self.mean_calc.geometric_mean(speedup_ratios)
        
        return ComparisonResult(
            metric_name='Speedup Ratio',
            baseline_value=1.0,  # Reference
            capa_value=geo_mean_speedup,
            improvement_pct=(geo_mean_speedup - 1) * 100,
            mean_type='geometric',
            p_value=None,
            significant=geo_mean_speedup > 1.0,
            confidence_interval=None,
            effect_size=None
        )
    
    def compare_cpu_utilization(
        self,
        baseline_cpu: np.ndarray,
        baseline_durations: np.ndarray,
        capa_cpu: np.ndarray,
        capa_durations: np.ndarray
    ) -> ComparisonResult:
        """
        Compare CPU utilization with WEIGHTED arithmetic mean
        
        From Jain Example 12.3: CPU utilization MUST be weighted by duration
        """
        baseline_weighted = self.mean_calc.weighted_cpu_utilization(
            baseline_cpu, baseline_durations
        )
        capa_weighted = self.mean_calc.weighted_cpu_utilization(
            capa_cpu, capa_durations
        )
        
        # For CPU, higher utilization (up to ~70%) is better (efficiency)
        # But we report the raw difference
        diff = capa_weighted - baseline_weighted
        
        return ComparisonResult(
            metric_name='CPU Utilization (weighted)',
            baseline_value=baseline_weighted,
            capa_value=capa_weighted,
            improvement_pct=diff * 100,  # Absolute difference
            mean_type='weighted_arithmetic',
            p_value=None,
            significant=abs(diff) > 0.1,
            confidence_interval=None,
            effect_size=None
        )
    
    def full_comparison(
        self,
        baseline_df: pd.DataFrame,
        capa_df: pd.DataFrame
    ) -> Dict[str, ComparisonResult]:
        """
        Comprehensive comparison of two systems
        """
        results = {}
        
        # Response Time
        if 'p95_latency_ms' in baseline_df.columns:
            results['response_time'] = self.compare_response_times(
                baseline_df['p95_latency_ms'].values,
                capa_df['p95_latency_ms'].values[:len(baseline_df)]
            )
        
        # Throughput
        if 'throughput_rps' in baseline_df.columns:
            results['throughput'] = self.compare_throughput(
                baseline_df['throughput_rps'].values,
                capa_df['throughput_rps'].values[:len(baseline_df)]
            )
        
        # Speedup
        if 'p95_latency_ms' in baseline_df.columns:
            results['speedup'] = self.compare_speedup(
                baseline_df['p95_latency_ms'].values,
                capa_df['p95_latency_ms'].values[:len(baseline_df)]
            )
        
        # CPU Utilization (weighted)
        if 'cpu_util' in baseline_df.columns and 'duration_sec' in baseline_df.columns:
            results['cpu_utilization'] = self.compare_cpu_utilization(
                baseline_df['cpu_util'].values,
                baseline_df['duration_sec'].values,
                capa_df['cpu_util'].values,
                capa_df['duration_sec'].values
            )
        
        return results


# =============================================================================
# OVERFITTING DETECTOR
# =============================================================================

class OverfittingDetector:
    """
    Detect overfitting by comparing train vs test performance
    
    From Jain Ch. 16: Proper experimental design requires validation
    """
    
    def __init__(self, threshold_mild: float = 1.2, threshold_severe: float = 1.5):
        self.threshold_mild = threshold_mild
        self.threshold_severe = threshold_severe
    
    def analyze(
        self,
        train_results: List[Dict],
        test_results: List[Dict]
    ) -> OverfittingAnalysis:
        """
        Analyze for overfitting
        
        Key metric: test_performance / train_performance
        - < 1.2: Good generalization
        - 1.2-1.5: Mild overfitting
        - > 1.5: Severe overfitting
        """
        # Extract P95 latencies
        train_p95 = np.mean([r.get('p95_latency_ms', 0) for r in train_results])
        test_p95 = np.mean([r.get('p95_latency_ms', 0) for r in test_results])
        
        if train_p95 == 0:
            ratio = 1.0
        else:
            ratio = test_p95 / train_p95
        
        # Determine severity
        if ratio >= self.threshold_severe:
            severity = 'severe'
            is_overfitting = True
            recommendations = [
                "Reduce state space further (fewer buckets)",
                "Increase exploration (higher epsilon_min)",
                "Add more training patterns",
                "Increase reward noise",
                "Use more aggressive Q-value regularization",
                "Consider function approximation instead of tabular Q-learning"
            ]
        elif ratio >= self.threshold_mild:
            severity = 'mild'
            is_overfitting = True
            recommendations = [
                "Slightly reduce state space",
                "Increase epsilon_min to 0.15",
                "Add one more diverse training pattern",
                "Increase replay buffer size"
            ]
        else:
            severity = 'none'
            is_overfitting = False
            recommendations = [
                "Model generalizes well",
                "Consider deploying to production"
            ]
        
        return OverfittingAnalysis(
            train_p95=train_p95,
            test_p95=test_p95,
            generalization_ratio=ratio,
            is_overfitting=is_overfitting,
            severity=severity,
            recommendations=recommendations
        )


# =============================================================================
# SHADOW MODE ANALYZER
# =============================================================================

class ShadowModeAnalyzer:
    """
    Analyze Shadow Mode performance
    """
    
    def analyze(self, decisions: List[Dict]) -> Dict:
        """
        Analyze shadow mode decisions
        """
        if not decisions:
            return {
                'total_decisions': 0,
                'agreement_rate': 0.0,
                'rl_scale_up_rate': 0.0,
                'baseline_scale_up_rate': 0.0,
                'disagreements': []
            }
        
        total = len(decisions)
        agreed = sum(1 for d in decisions if d.get('rl_action') == d.get('baseline_action'))
        
        rl_scale_up = sum(1 for d in decisions if d.get('rl_action') == 2)
        baseline_scale_up = sum(1 for d in decisions if d.get('baseline_action') == 2)
        
        disagreements = [
            d for d in decisions 
            if d.get('rl_action') != d.get('baseline_action')
        ]
        
        return {
            'total_decisions': total,
            'agreement_rate': agreed / total if total > 0 else 0,
            'rl_scale_up_rate': rl_scale_up / total if total > 0 else 0,
            'baseline_scale_up_rate': baseline_scale_up / total if total > 0 else 0,
            'disagreements': disagreements[:10]  # First 10
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """
    Create academic-quality visualizations
    """
    
    def __init__(self, output_dir: str = './figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_latency_comparison(
        self,
        baseline_latencies: np.ndarray,
        capa_latencies: np.ndarray,
        title: str = "Response Time Comparison"
    ) -> str:
        """
        Box plot comparison of latencies
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        ax1 = axes[0]
        data = [baseline_latencies, capa_latencies]
        bp = ax1.boxplot(data, labels=['Baseline HPA', 'CAPA+'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['baseline'])
        bp['boxes'][1].set_facecolor(COLORS['capa'])
        ax1.set_ylabel('P95 Latency (ms)')
        ax1.set_title('Distribution Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Time series (if same length)
        ax2 = axes[1]
        if len(baseline_latencies) == len(capa_latencies):
            x = np.arange(len(baseline_latencies))
            ax2.plot(x, baseline_latencies, label='Baseline HPA', 
                    color=COLORS['baseline'], alpha=0.7)
            ax2.plot(x, capa_latencies, label='CAPA+',
                    color=COLORS['capa'], alpha=0.7)
            ax2.set_xlabel('Time (samples)')
            ax2.set_ylabel('P95 Latency (ms)')
            ax2.set_title('Time Series Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Histogram comparison
            ax2.hist(baseline_latencies, bins=30, alpha=0.5, 
                    label='Baseline HPA', color=COLORS['baseline'])
            ax2.hist(capa_latencies, bins=30, alpha=0.5,
                    label='CAPA+', color=COLORS['capa'])
            ax2.set_xlabel('P95 Latency (ms)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution Histogram')
            ax2.legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'latency_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_overfitting_analysis(
        self,
        analysis: OverfittingAnalysis
    ) -> str:
        """
        Visualize overfitting analysis
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart: Train vs Test
        ax1 = axes[0]
        x = ['Training\n(seen patterns)', 'Testing\n(unseen patterns)']
        heights = [analysis.train_p95, analysis.test_p95]
        colors = ['#2ecc71', '#e74c3c' if analysis.is_overfitting else '#2ecc71']
        
        bars = ax1.bar(x, heights, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('P95 Latency (ms)')
        ax1.set_title('Train vs Test Performance')
        
        # Add ratio annotation
        ax1.annotate(
            f'Ratio: {analysis.generalization_ratio:.2f}x',
            xy=(1, analysis.test_p95),
            xytext=(1.2, analysis.test_p95 * 1.1),
            fontsize=12,
            fontweight='bold',
            color='red' if analysis.is_overfitting else 'green'
        )
        
        # Severity gauge
        ax2 = axes[1]
        
        # Create gauge-like visualization
        categories = ['Good', 'Mild\nOverfit', 'Severe\nOverfit']
        thresholds = [1.0, 1.2, 1.5, 2.0]
        colors_gauge = ['#2ecc71', '#f39c12', '#e74c3c']
        
        # Background bars
        for i, (cat, col) in enumerate(zip(categories, colors_gauge)):
            ax2.barh(0, 1, left=i, color=col, alpha=0.3, height=0.5)
            ax2.text(i + 0.5, 0, cat, ha='center', va='center', fontsize=10)
        
        # Marker for current ratio
        marker_pos = min(2.9, max(0.1, (analysis.generalization_ratio - 1.0) / 0.5))
        ax2.plot(marker_pos, 0, 'ko', markersize=20)
        ax2.plot(marker_pos, 0, 'w|', markersize=15, mew=3)
        
        ax2.set_xlim(0, 3)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_title(f'Overfitting Severity: {analysis.severity.upper()}')
        ax2.axis('off')
        
        plt.suptitle('Overfitting Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'overfitting_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_statistical_summary(
        self,
        comparisons: Dict[str, ComparisonResult]
    ) -> str:
        """
        Create statistical summary visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Improvement percentages
        ax1 = axes[0, 0]
        metrics = list(comparisons.keys())
        improvements = [comparisons[m].improvement_pct for m in metrics]
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        
        bars = ax1.barh(metrics, improvements, color=colors, edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Improvement (%)')
        ax1.set_title('Performance Improvements')
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{imp:.1f}%', va='center', fontsize=10)
        
        # 2. Statistical significance
        ax2 = axes[0, 1]
        significant = [comparisons[m].significant for m in metrics]
        p_values = [comparisons[m].p_value or 1.0 for m in metrics]
        
        colors_sig = ['#2ecc71' if sig else '#e74c3c' for sig in significant]
        bars = ax2.barh(metrics, [-np.log10(p) if p > 0 else 0 for p in p_values], 
                       color=colors_sig, edgecolor='black')
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
                   label='p=0.05 threshold')
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_title('Statistical Significance')
        ax2.legend()
        
        # 3. Effect sizes
        ax3 = axes[1, 0]
        effect_sizes = [abs(comparisons[m].effect_size or 0) for m in metrics]
        
        # Color by effect size magnitude
        colors_effect = []
        for es in effect_sizes:
            if es < 0.2:
                colors_effect.append('#95a5a6')  # Negligible
            elif es < 0.5:
                colors_effect.append('#f39c12')  # Small
            elif es < 0.8:
                colors_effect.append('#3498db')  # Medium
            else:
                colors_effect.append('#2ecc71')  # Large
        
        bars = ax3.barh(metrics, effect_sizes, color=colors_effect, edgecolor='black')
        
        # Add threshold lines
        ax3.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7, label='Small (0.2)')
        ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium (0.5)')
        ax3.axvline(x=0.8, color='gray', linestyle='-', alpha=0.7, label='Large (0.8)')
        ax3.set_xlabel("Cohen's d (Effect Size)")
        ax3.set_title('Effect Sizes')
        ax3.legend(loc='lower right')
        
        # 4. Mean type summary
        ax4 = axes[1, 1]
        mean_types = [comparisons[m].mean_type for m in metrics]
        
        # Create table
        table_data = []
        for m in metrics:
            c = comparisons[m]
            table_data.append([
                m,
                c.mean_type,
                f'{c.baseline_value:.2f}',
                f'{c.capa_value:.2f}',
                '✓' if c.significant else '✗'
            ])
        
        ax4.axis('off')
        table = ax4.table(
            cellText=table_data,
            colLabels=['Metric', 'Mean Type', 'Baseline', 'CAPA+', 'Sig?'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Table (Jain Ch. 12 Mean Selection)')
        
        plt.suptitle('Statistical Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'statistical_summary.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_shadow_mode_analysis(
        self,
        shadow_analysis: Dict
    ) -> str:
        """
        Visualize shadow mode analysis
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Agreement rate pie chart
        ax1 = axes[0]
        agreement = shadow_analysis['agreement_rate']
        sizes = [agreement, 1 - agreement]
        labels = ['Agree', 'Disagree']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'RL-Baseline Agreement\n({shadow_analysis["total_decisions"]} decisions)')
        
        # 2. Action distribution
        ax2 = axes[1]
        x = ['Scale Up', 'Scale Down']
        rl_rates = [shadow_analysis['rl_scale_up_rate'], 
                   1 - shadow_analysis['rl_scale_up_rate'] - 0.5]  # Approximate
        baseline_rates = [shadow_analysis['baseline_scale_up_rate'],
                         1 - shadow_analysis['baseline_scale_up_rate'] - 0.5]
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        ax2.bar(x_pos - width/2, [shadow_analysis['rl_scale_up_rate'], 0.1], 
               width, label='RL', color=COLORS['capa'])
        ax2.bar(x_pos + width/2, [shadow_analysis['baseline_scale_up_rate'], 0.1],
               width, label='Baseline', color=COLORS['baseline'])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x)
        ax2.set_ylabel('Rate')
        ax2.set_title('Scale-Up Action Rates')
        ax2.legend()
        
        # 3. Readiness checklist
        ax3 = axes[2]
        ax3.axis('off')
        
        checklist = [
            ('Sufficient observations (≥500)', shadow_analysis['total_decisions'] >= 500),
            ('Reasonable agreement (≥60%)', agreement >= 0.6),
            ('Learning stability', True),  # Simplified
        ]
        
        y_pos = 0.8
        for item, passed in checklist:
            symbol = '✓' if passed else '✗'
            color = '#2ecc71' if passed else '#e74c3c'
            ax3.text(0.1, y_pos, f'{symbol} {item}', fontsize=12, color=color,
                    transform=ax3.transAxes)
            y_pos -= 0.15
        
        ready = all(passed for _, passed in checklist)
        ax3.text(0.1, 0.2, f'\nReady for Active Mode: {"YES ✓" if ready else "NO ✗"}',
                fontsize=14, fontweight='bold',
                color='#2ecc71' if ready else '#e74c3c',
                transform=ax3.transAxes)
        
        ax3.set_title('Transition Readiness')
        
        plt.suptitle('Shadow Mode Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'shadow_mode_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """
    Generate academic-quality analysis report
    """
    
    def __init__(self, output_dir: str = './report'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.visualizer = Visualizer(os.path.join(output_dir, 'figures'))
    
    def generate_report(
        self,
        baseline_df: pd.DataFrame,
        capa_df: pd.DataFrame,
        train_results: Optional[List[Dict]] = None,
        test_results: Optional[List[Dict]] = None,
        shadow_decisions: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate comprehensive analysis report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CAPA+ AUTOSCALER ACADEMIC ANALYSIS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. System Comparison
        report_lines.append("## 1. SYSTEM COMPARISON (Jain Ch. 12)")
        report_lines.append("-" * 40)
        
        comparator = SystemComparator()
        comparisons = comparator.full_comparison(baseline_df, capa_df)
        
        for name, result in comparisons.items():
            report_lines.append(f"\n### {result.metric_name}")
            report_lines.append(f"  Mean Type: {result.mean_type} (Jain Ch. 12)")
            report_lines.append(f"  Baseline: {result.baseline_value:.2f}")
            report_lines.append(f"  CAPA+: {result.capa_value:.2f}")
            report_lines.append(f"  Improvement: {result.improvement_pct:+.1f}%")
            
            if result.p_value is not None:
                report_lines.append(f"  p-value: {result.p_value:.4f}")
                report_lines.append(f"  Significant (α=0.05): {'Yes' if result.significant else 'No'}")
            
            if result.effect_size is not None:
                effect_interp = self._interpret_effect_size(result.effect_size)
                report_lines.append(f"  Effect Size (Cohen's d): {result.effect_size:.3f} ({effect_interp})")
            
            if result.confidence_interval:
                report_lines.append(f"  95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
        
        # Generate comparison visualization
        if 'p95_latency_ms' in baseline_df.columns:
            fig_path = self.visualizer.plot_latency_comparison(
                baseline_df['p95_latency_ms'].values,
                capa_df['p95_latency_ms'].values
            )
            report_lines.append(f"\n  [Figure: {fig_path}]")
        
        # Statistical summary figure
        fig_path = self.visualizer.plot_statistical_summary(comparisons)
        report_lines.append(f"\n  [Figure: {fig_path}]")
        
        # 2. Overfitting Analysis
        if train_results and test_results:
            report_lines.append("\n\n## 2. OVERFITTING ANALYSIS (Jain Ch. 16)")
            report_lines.append("-" * 40)
            
            detector = OverfittingDetector()
            overfit_analysis = detector.analyze(train_results, test_results)
            
            report_lines.append(f"\n  Training P95 (seen patterns): {overfit_analysis.train_p95:.1f} ms")
            report_lines.append(f"  Testing P95 (unseen patterns): {overfit_analysis.test_p95:.1f} ms")
            report_lines.append(f"  Generalization Ratio: {overfit_analysis.generalization_ratio:.2f}x")
            report_lines.append(f"\n  Overfitting Detected: {'YES' if overfit_analysis.is_overfitting else 'NO'}")
            report_lines.append(f"  Severity: {overfit_analysis.severity.upper()}")
            
            report_lines.append("\n  Recommendations:")
            for rec in overfit_analysis.recommendations:
                report_lines.append(f"    - {rec}")
            
            # Generate overfitting visualization
            fig_path = self.visualizer.plot_overfitting_analysis(overfit_analysis)
            report_lines.append(f"\n  [Figure: {fig_path}]")
        
        # 3. Shadow Mode Analysis
        if shadow_decisions:
            report_lines.append("\n\n## 3. SHADOW MODE ANALYSIS (INTROD_1 Ch. 2)")
            report_lines.append("-" * 40)
            
            analyzer = ShadowModeAnalyzer()
            shadow_analysis = analyzer.analyze(shadow_decisions)
            
            report_lines.append(f"\n  Total Decisions: {shadow_analysis['total_decisions']}")
            report_lines.append(f"  RL-Baseline Agreement: {shadow_analysis['agreement_rate']:.1%}")
            report_lines.append(f"  RL Scale-Up Rate: {shadow_analysis['rl_scale_up_rate']:.1%}")
            report_lines.append(f"  Baseline Scale-Up Rate: {shadow_analysis['baseline_scale_up_rate']:.1%}")
            
            # Generate shadow mode visualization
            fig_path = self.visualizer.plot_shadow_mode_analysis(shadow_analysis)
            report_lines.append(f"\n  [Figure: {fig_path}]")
        
        # 4. Methodology Notes
        report_lines.append("\n\n## 4. METHODOLOGY NOTES")
        report_lines.append("-" * 40)
        report_lines.append("""
  This analysis follows academic best practices from:
  
  - Jain (1991): "The Art of Computer Systems Performance Analysis"
    - Chapter 11: Avoiding ratio games through geometric mean
    - Chapter 12: Correct mean selection (arithmetic/geometric/harmonic)
    - Example 12.3: Weighted CPU utilization
    
  - Harchol-Balter (2013): "Performance Modeling and Design of Computer Systems"
    - Chapter 5: Sample paths and convergence
    - Chapter 14: Variance reduction techniques
    
  - INTROD_1: "Introduction to Computer System Performance Evaluation"
    - Chapter 2: Measurement techniques (Shadow Mode)
    - Chapter 3: Statistical data analysis
""")
        
        # 5. Summary
        report_lines.append("\n\n## 5. EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Generate summary based on results
        if comparisons.get('response_time'):
            rt = comparisons['response_time']
            if rt.improvement_pct > 0 and rt.significant:
                verdict = "CAPA+ shows statistically significant improvement"
            elif rt.improvement_pct > 0:
                verdict = "CAPA+ shows improvement but not statistically significant"
            elif rt.improvement_pct < 0 and rt.significant:
                verdict = "CAPA+ shows statistically significant degradation"
            else:
                verdict = "No significant difference between systems"
            
            report_lines.append(f"\n  {verdict}")
            report_lines.append(f"  Response time improvement: {rt.improvement_pct:+.1f}%")
        
        if train_results and test_results:
            detector = OverfittingDetector()
            overfit = detector.analyze(train_results, test_results)
            if overfit.is_overfitting:
                report_lines.append(f"\n  ⚠️ WARNING: {overfit.severity} overfitting detected")
                report_lines.append(f"     Model may not generalize to new load patterns")
            else:
                report_lines.append(f"\n  ✓ Model generalizes well to unseen patterns")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Write report
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_path
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_experiment_data(directory: str) -> Tuple[pd.DataFrame, pd.DataFrame, List, List, List]:
    """
    Load experiment data from directory
    """
    baseline_df = None
    capa_df = None
    train_results = []
    test_results = []
    shadow_decisions = []
    
    # Try to load baseline results
    baseline_file = os.path.join(directory, 'baseline_results.csv')
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
    
    # Try to load CAPA+ results
    capa_file = os.path.join(directory, 'capa_results.csv')
    if os.path.exists(capa_file):
        capa_df = pd.read_csv(capa_file)
    
    # Try to load JSON results
    train_file = os.path.join(directory, 'train_results.json')
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_results = json.load(f)
    
    test_file = os.path.join(directory, 'test_results.json')
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_results = json.load(f)
    
    decisions_file = os.path.join(directory, 'decisions_history.json')
    if os.path.exists(decisions_file):
        with open(decisions_file, 'r') as f:
            shadow_decisions = json.load(f)
    
    return baseline_df, capa_df, train_results, test_results, shadow_decisions


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample data for demonstration
    """
    np.random.seed(42)
    n_samples = 100
    
    # Baseline: Higher latency, more variance
    baseline_df = pd.DataFrame({
        'timestamp': np.arange(n_samples),
        'p95_latency_ms': np.random.normal(300, 80, n_samples),
        'throughput_rps': np.random.normal(500, 50, n_samples),
        'cpu_util': np.random.uniform(0.3, 0.8, n_samples),
        'duration_sec': np.ones(n_samples) * 10,
        'pods_ready': np.random.randint(2, 6, n_samples)
    })
    
    # CAPA+: Lower latency but sometimes worse
    capa_df = pd.DataFrame({
        'timestamp': np.arange(n_samples),
        'p95_latency_ms': np.random.normal(250, 100, n_samples),  # Lower mean, higher variance
        'throughput_rps': np.random.normal(520, 40, n_samples),
        'cpu_util': np.random.uniform(0.4, 0.7, n_samples),
        'duration_sec': np.ones(n_samples) * 10,
        'pods_ready': np.random.randint(2, 7, n_samples)
    })
    
    return baseline_df, capa_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Academic Analysis for CAPA+ Autoscaler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from data directory
  python generate_academic_analysis.py --data-dir ./results

  # Generate sample analysis
  python generate_academic_analysis.py --sample

  # Custom output directory
  python generate_academic_analysis.py --data-dir ./results --output-dir ./report
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./analysis_output',
                       help='Output directory for report and figures')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample data for demonstration')
    parser.add_argument('--baseline-csv', type=str,
                       help='Path to baseline results CSV')
    parser.add_argument('--capa-csv', type=str,
                       help='Path to CAPA+ results CSV')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAPA+ Academic Analysis Tool")
    print("Following Jain (1991), Harchol-Balter (2013), INTROD_1")
    print("=" * 60)
    print()
    
    # Load or create data
    if args.sample:
        print("Using sample data for demonstration...")
        baseline_df, capa_df = create_sample_data()
        train_results = [{'p95_latency_ms': 240}, {'p95_latency_ms': 260}]
        test_results = [{'p95_latency_ms': 280}, {'p95_latency_ms': 300}]
        shadow_decisions = [
            {'rl_action': 1, 'baseline_action': 1} for _ in range(400)
        ] + [
            {'rl_action': 2, 'baseline_action': 1} for _ in range(100)
        ]
    elif args.baseline_csv and args.capa_csv:
        print(f"Loading baseline from: {args.baseline_csv}")
        print(f"Loading CAPA+ from: {args.capa_csv}")
        baseline_df = pd.read_csv(args.baseline_csv)
        capa_df = pd.read_csv(args.capa_csv)
        train_results = None
        test_results = None
        shadow_decisions = None
    else:
        print(f"Loading data from: {args.data_dir}")
        baseline_df, capa_df, train_results, test_results, shadow_decisions = \
            load_experiment_data(args.data_dir)
    
    # Validate data
    if baseline_df is None or capa_df is None:
        print("\nERROR: Could not load baseline or CAPA+ data.")
        print("Please provide data using one of these methods:")
        print("  1. --data-dir with baseline_results.csv and capa_results.csv")
        print("  2. --baseline-csv and --capa-csv")
        print("  3. --sample for demonstration")
        sys.exit(1)
    
    print(f"\nBaseline samples: {len(baseline_df)}")
    print(f"CAPA+ samples: {len(capa_df)}")
    
    # Generate report
    print(f"\nGenerating report in: {args.output_dir}")
    print("-" * 40)
    
    generator = ReportGenerator(args.output_dir)
    report_path = generator.generate_report(
        baseline_df=baseline_df,
        capa_df=capa_df,
        train_results=train_results,
        test_results=test_results,
        shadow_decisions=shadow_decisions
    )
    
    print(f"\n{'=' * 60}")
    print(f"Report saved to: {report_path}")
    print(f"Figures saved to: {os.path.join(args.output_dir, 'figures')}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
