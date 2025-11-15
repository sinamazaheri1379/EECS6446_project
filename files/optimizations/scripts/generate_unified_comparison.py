#!/usr/bin/env python3
"""
EECS6446 Project - Unified Visualization Generator
Creates comparison charts matching baseline report format
Baseline HPA vs Elascale-Optimized HPA
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

# Configuration
RESULTS_DIR = Path("/home/common/EECS6446_project/optimizations/files/results")
OUTPUT_DIR = Path("/home/common/EECS6446_project/optimizations/files/results/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style to match baseline report
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Services from baseline report
SERVICES = [
    "frontend",
    "cartservice", 
    "checkoutservice",
    "currencyservice",
    "recommendationservice",
    "productcatalogservice"
]

# ============================================================
# Data Loading
# ============================================================

def load_latest_results(config_name):
    """Load most recent complete results for a configuration"""
    pattern = f"{config_name}_complete_*.csv"
    files = list(RESULTS_DIR.glob(pattern))
    
    if not files:
        print(f"❌ No results found for {config_name}")
        print(f"   Looking for: {RESULTS_DIR}/{pattern}")
        return None
    
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"✓ Loading {config_name}: {latest.name}")
    
    df = pd.read_csv(latest)
    return df

# ============================================================
# Figure 1: User Load and Throughput (Matching Baseline Report)
# ============================================================

def plot_user_load_and_throughput(baseline_df, elascale_df):
    """
    Create user load and throughput plots matching baseline report
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Throughput over time
    ax1 = axes[0]
    for label, df, color in [("Baseline", baseline_df, 'C0'), 
                              ("Elascale", elascale_df, 'C1')]:
        if df is not None:
            ax1.plot(df['elapsed_minutes'], df['throughput_rps'], 
                    label=label, linewidth=2, alpha=0.8, color=color)
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Throughput (req/s)')
    ax1.set_title('Throughput over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: User load over time
    ax2 = axes[1]
    if baseline_df is not None:
        ax2.plot(baseline_df['elapsed_minutes'], baseline_df['scenario_users'],
                linewidth=2, color='C2', label='User Load')
    elif elascale_df is not None:
        ax2.plot(elascale_df['elapsed_minutes'], elascale_df['scenario_users'],
                linewidth=2, color='C2', label='User Load')
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('User Load Pattern')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = OUTPUT_DIR / 'fig1_user_load_throughput.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# ============================================================
# Figure 2: Fault Rate and Response Time (Matching Baseline)
# ============================================================

def plot_fault_rate_and_response_time(baseline_df, elascale_df):
    """
    Create fault rate and response time plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Fault Rate
    ax1 = axes[0]
    for label, df, color in [("Baseline", baseline_df, 'C0'),
                              ("Elascale", elascale_df, 'C1')]:
        if df is not None:
            ax1.plot(df['elapsed_minutes'], df['fault_rate_percent'],
                    label=label, linewidth=2, alpha=0.8, color=color)
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Fault % (errors)')
    ax1.set_title('Fault Rate Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=-0.05)  # Start at 0 like baseline
    
    # Plot 2: Response Time (avg and p95)
    ax2 = axes[1]
    for label, df, color_avg, color_p95 in [
        ("Baseline", baseline_df, 'C0', 'C3'),
        ("Elascale", elascale_df, 'C1', 'C4')
    ]:
        if df is not None:
            ax2.plot(df['elapsed_minutes'], df['avg_response_time_ms'],
                    label=f'{label} avg', linewidth=2, alpha=0.7, 
                    color=color_avg, linestyle='-')
            ax2.plot(df['elapsed_minutes'], df['p95_response_time_ms'],
                    label=f'{label} 95th', linewidth=2, alpha=0.7,
                    color=color_p95, linestyle='--')
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Average Response Time (ms)')
    ax2.set_title('Response Time Over Time')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = OUTPUT_DIR / 'fig2_fault_rate_response_time.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# ============================================================
# Figure 3-8: Service-Specific Metrics (Matching Baseline)
# ============================================================

def plot_service_cpu_and_pods(baseline_df, elascale_df, service):
    """
    Create CPU and pod count plots for a specific service
    Matches baseline report format
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    service_title = service.replace('service', ' Service').title()
    fig.suptitle(f'{service_title} - Baseline vs Elascale', 
                 fontsize=14, fontweight='bold')
    
    cpu_col = f"{service}_cpu_millicores"
    cpu_pct_col = f"{service}_cpu_percent"
    ordered_col = f"{service}_replicas_ordered"
    ready_col = f"{service}_replicas_ready"
    
    # Plot 1: CPU Usage (millicores)
    ax1 = axes[0, 0]
    for label, df, color in [("Baseline", baseline_df, 'C0'),
                              ("Elascale", elascale_df, 'C1')]:
        if df is not None and cpu_col in df.columns:
            ax1.plot(df['elapsed_minutes'], df[cpu_col],
                    label=label, linewidth=2, alpha=0.8, color=color)
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('CPU Usage (millicores)')
    ax1.set_title('CPU Usage Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: CPU Usage per Container
    ax2 = axes[0, 1]
    for label, df, color in [("Baseline", baseline_df, 'C0'),
                              ("Elascale", elascale_df, 'C1')]:
        if df is not None and ordered_col in df.columns and cpu_col in df.columns:
            # Calculate CPU per pod
            cpu_per_pod = df[cpu_col] / df[ordered_col].replace(0, 1)
            ax2.plot(df['elapsed_minutes'], cpu_per_pod,
                    label=label, linewidth=2, alpha=0.8, color=color)
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('CPU Usage Per Container (millicores)')
    ax2.set_title('CPU Per Pod')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Pod Count (ordered vs ready)
    ax3 = axes[1, 0]
    for label, df, color_ordered, color_ready in [
        ("Baseline", baseline_df, 'C0', 'C3'),
        ("Elascale", elascale_df, 'C1', 'C4')
    ]:
        if df is not None and ordered_col in df.columns:
            ax3.plot(df['elapsed_minutes'], df[ordered_col],
                    label=f'{label} ordered', linewidth=2,
                    color=color_ordered, linestyle='-')
            ax3.plot(df['elapsed_minutes'], df[ready_col],
                    label=f'{label} ready', linewidth=2,
                    color=color_ready, linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Pod Count')
    ax3.set_title('Pod Counts (Ordered vs Ready)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Memory Usage
    ax4 = axes[1, 1]
    mem_col = f"{service}_memory_bytes"
    for label, df, color in [("Baseline", baseline_df, 'C0'),
                              ("Elascale", elascale_df, 'C1')]:
        if df is not None and mem_col in df.columns:
            # Convert to MB
            mem_mb = df[mem_col] / (1024 * 1024)
            ax4.plot(df['elapsed_minutes'], mem_mb,
                    label=label, linewidth=2, alpha=0.8, color=color)
    
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Memory Usage (MB)')
    ax4.set_title('Memory Usage Over Time')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = OUTPUT_DIR / f'service_{service}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# ============================================================
# Summary Comparison Table
# ============================================================

def generate_summary_comparison(baseline_df, elascale_df):
    """
    Generate summary statistics table comparing configurations
    """
    summary_rows = []
    
    # Overall metrics
    for label, df in [("Baseline", baseline_df), ("Elascale", elascale_df)]:
        if df is not None:
            row = {
                "Configuration": label,
                "Avg Response Time (ms)": f"{df['avg_response_time_ms'].mean():.1f}",
                "P95 Response Time (ms)": f"{df['p95_response_time_ms'].mean():.1f}",
                "Max P95 Response (ms)": f"{df['p95_response_time_ms'].max():.1f}",
                "Avg Throughput (rps)": f"{df['throughput_rps'].mean():.1f}",
                "Max Fault Rate (%)": f"{df['fault_rate_percent'].max():.2f}",
                "Avg Frontend Pods": f"{df['frontend_replicas_ordered'].mean():.1f}",
                "Max Frontend Pods": f"{df['frontend_replicas_ordered'].max():.0f}",
            }
            summary_rows.append(row)
    
    # Calculate improvements if both exist
    if baseline_df is not None and elascale_df is not None:
        baseline_p95_avg = baseline_df['p95_response_time_ms'].mean()
        elascale_p95_avg = elascale_df['p95_response_time_ms'].mean()
        improvement = ((baseline_p95_avg - elascale_p95_avg) / baseline_p95_avg) * 100
        
        row = {
            "Configuration": "Improvement",
            "Avg Response Time (ms)": f"{improvement:.1f}%",
            "P95 Response Time (ms)": f"{improvement:.1f}%",
            "Max P95 Response (ms)": f"{((baseline_df['p95_response_time_ms'].max() - elascale_df['p95_response_time_ms'].max()) / baseline_df['p95_response_time_ms'].max() * 100):.1f}%",
            "Avg Throughput (rps)": f"{((elascale_df['throughput_rps'].mean() - baseline_df['throughput_rps'].mean()) / baseline_df['throughput_rps'].mean() * 100):.1f}%",
            "Max Fault Rate (%)": "N/A",
            "Avg Frontend Pods": f"{((elascale_df['frontend_replicas_ordered'].mean() - baseline_df['frontend_replicas_ordered'].mean()) / baseline_df['frontend_replicas_ordered'].mean() * 100):.1f}%",
            "Max Frontend Pods": "N/A",
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save as CSV
    filename = OUTPUT_DIR / 'summary_comparison.csv'
    summary_df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")
    
    # Print to console
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON - Baseline vs Elascale")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    return summary_df

# ============================================================
# Service-Specific Summary
# ============================================================

def generate_service_summary(baseline_df, elascale_df):
    """
    Generate per-service summary table
    """
    service_rows = []
    
    for service in SERVICES:
        cpu_col = f"{service}_cpu_millicores"
        ordered_col = f"{service}_replicas_ordered"
        
        row = {"Service": service}
        
        for label, df in [("Baseline", baseline_df), ("Elascale", elascale_df)]:
            if df is not None and cpu_col in df.columns:
                row[f"{label} Avg CPU (mc)"] = f"{df[cpu_col].mean():.0f}"
                row[f"{label} Max CPU (mc)"] = f"{df[cpu_col].max():.0f}"
                row[f"{label} Avg Pods"] = f"{df[ordered_col].mean():.1f}"
                row[f"{label} Max Pods"] = f"{df[ordered_col].max():.0f}"
        
        service_rows.append(row)
    
    service_df = pd.DataFrame(service_rows)
    
    # Save as CSV
    filename = OUTPUT_DIR / 'service_summary.csv'
    service_df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")
    
    # Print to console
    print(f"\n{'='*80}")
    print("SERVICE-LEVEL COMPARISON")
    print(f"{'='*80}")
    print(service_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    return service_df

# ============================================================
# Main Execution
# ============================================================

def main():
    print(f"\n{'='*70}")
    print("EECS6446 Project - Unified Visualization Generator")
    print("Generating Comparison Charts: Baseline vs Elascale")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading experiment results...\n")
    baseline_df = load_latest_results("baseline")
    elascale_df = load_latest_results("elascale")
    
    if baseline_df is None and elascale_df is None:
        print("\n❌ No experiment data found!")
        print("\nPlease run experiments first:")
        print("  python3 /home/claude/unified_experiment.py")
        sys.exit(1)
    
    if baseline_df is not None:
        print(f"  Baseline: {len(baseline_df)} data points")
    if elascale_df is not None:
        print(f"  Elascale: {len(elascale_df)} data points")
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("Generating Comparison Visualizations")
    print(f"{'='*70}\n")
    
    print("Creating figures matching baseline report format...")
    
    # Figure 1: User load and throughput
    plot_user_load_and_throughput(baseline_df, elascale_df)
    
    # Figure 2: Fault rate and response time
    plot_fault_rate_and_response_time(baseline_df, elascale_df)
    
    # Figures 3-8: Per-service comparisons
    for service in SERVICES:
        print(f"  Generating {service} comparison...")
        plot_service_cpu_and_pods(baseline_df, elascale_df, service)
    
    # Generate summary tables
    print(f"\n{'='*70}")
    print("Generating Summary Statistics")
    print(f"{'='*70}\n")
    
    generate_summary_comparison(baseline_df, elascale_df)
    generate_service_summary(baseline_df, elascale_df)
    
    # Final summary
    print(f"\n{'='*70}")
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    
    print(f"Output directory: {OUTPUT_DIR}/\n")
    
    print("Generated files:")
    for img_file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  📊 {img_file.name}")
    for csv_file in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  📄 {csv_file.name}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("\n1. Review comparison charts:")
    print(f"   - All charts saved to: {OUTPUT_DIR}/")
    print("\n2. Analyze improvements:")
    print("   - Check summary_comparison.csv for overall metrics")
    print("   - Review service_summary.csv for per-service details")
    print("\n3. Include in your report:")
    print("   - Use charts to demonstrate performance improvements")
    print("   - Reference specific metrics from summary tables")
    print("   - Explain why Elascale performs better (or different)")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
