#!/usr/bin/env python3
"""
EECS6446 Elascale Optimization - Diagram Generator

This script generates visualizations for:
1. MAPE-K Loop diagram
2. Service dependency graph with replication factors
3. Multi-factor scaling formula visualization
4. Performance comparison charts (baseline vs. Elascale)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os
import pandas as pd
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("/home/common/EECS6446_project/files/optimizations/results/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'monitor': '#3498db',    # Blue
    'analyze': '#e74c3c',    # Red
    'plan': '#f39c12',       # Orange
    'execute': '#2ecc71',    # Green
    'knowledge': '#9b59b6',  # Purple
    'service': '#34495e',    # Dark gray
    'arrow': '#7f8c8d',      # Gray
}

# ============================================================
# 1. MAPE-K Loop Diagram
# ============================================================
def draw_mapek_loop():
    """
    Draw the MAPE-K (Monitor-Analyze-Plan-Execute-Knowledge) control loop
    as implemented in the Elascale optimization
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'MAPE-K Autonomic Control Loop\nfor Elascale HPA Optimization', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Define positions for MAPE-K components (circular layout)
    center_x, center_y = 5, 5
    radius = 2.5
    
    components = {
        'Monitor': {'angle': 90, 'color': COLORS['monitor']},
        'Analyze': {'angle': 18, 'color': COLORS['analyze']},
        'Plan': {'angle': -54, 'color': COLORS['plan']},
        'Execute': {'angle': -126, 'color': COLORS['execute']},
    }
    
    # Knowledge base in the center
    knowledge_box = FancyBboxPatch(
        (center_x - 1.2, center_y - 0.6), 2.4, 1.2,
        boxstyle="round,pad=0.1",
        edgecolor=COLORS['knowledge'],
        facecolor=COLORS['knowledge'],
        alpha=0.3,
        linewidth=2
    )
    ax.add_patch(knowledge_box)
    ax.text(center_x, center_y, 'Knowledge\n(Results DB)\nCSV Storage', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw MAPE components
    box_positions = {}
    for comp_name, comp_info in components.items():
        angle_rad = np.radians(comp_info['angle'])
        x = center_x + radius * np.cos(angle_rad)
        y = center_y + radius * np.sin(angle_rad)
        
        box = FancyBboxPatch(
            (x - 0.8, y - 0.5), 1.6, 1.0,
            boxstyle="round,pad=0.1",
            edgecolor=comp_info['color'],
            facecolor=comp_info['color'],
            alpha=0.3,
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, y, comp_name, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        box_positions[comp_name] = (x, y)
    
    # Draw arrows between components (circular flow)
    arrow_props = dict(arrowstyle='->', lw=2.5, color=COLORS['arrow'])
    
    # Monitor -> Analyze
    ax.annotate('', xy=box_positions['Analyze'], xytext=box_positions['Monitor'],
                arrowprops=arrow_props)
    
    # Analyze -> Plan
    ax.annotate('', xy=box_positions['Plan'], xytext=box_positions['Analyze'],
                arrowprops=arrow_props)
    
    # Plan -> Execute
    ax.annotate('', xy=box_positions['Execute'], xytext=box_positions['Plan'],
                arrowprops=arrow_props)
    
    # Execute -> Monitor (completing the loop)
    ax.annotate('', xy=box_positions['Monitor'], xytext=box_positions['Execute'],
                arrowprops=arrow_props)
    
    # Knowledge connections (bidirectional)
    knowledge_arrow_props = dict(arrowstyle='<->', lw=1.5, color=COLORS['knowledge'], 
                                 linestyle='--', alpha=0.7)
    
    for comp_name, (x, y) in box_positions.items():
        ax.annotate('', xy=(center_x, center_y), xytext=(x, y),
                   arrowprops=knowledge_arrow_props)
    
    # Add descriptions for each component
    descriptions = {
        'Monitor': 'Collect metrics:\n• CPU utilization\n• Memory usage\n• Replica counts\n(Prometheus + kubectl)',
        'Analyze': 'Compute scaling score:\nf = α·CPU + β·MEM\n   + γ·NET + λ·REP\n(Multi-factor formula)',
        'Plan': 'Determine action:\n• Scale up (>70%)\n• Scale down (<40%)\n• Maintain\n(Threshold-based)',
        'Execute': 'Apply HPA config:\n• Baseline (CPU-only)\n• Elascale (multi-factor)\n(kubectl apply)',
    }
    
    desc_offset = 1.3
    for comp_name, (x, y) in box_positions.items():
        angle_rad = np.radians(components[comp_name]['angle'])
        desc_x = x + desc_offset * np.cos(angle_rad)
        desc_y = y + desc_offset * np.sin(angle_rad)
        
        ax.text(desc_x, desc_y, descriptions[comp_name],
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add managed system at the bottom
    system_box = FancyBboxPatch(
        (1, 0.3), 8, 1.2,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor='lightblue',
        alpha=0.3,
        linewidth=2
    )
    ax.add_patch(system_box)
    ax.text(5, 0.9, 'Managed System: Kubernetes Cluster + Online Boutique Microservices', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Connection from Execute to Managed System
    ax.annotate('', xy=(5, 1.5), xytext=box_positions['Execute'],
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(4, 2.2, 'HPA\nConfigs', ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Connection from Managed System to Monitor
    ax.annotate('', xy=box_positions['Monitor'], xytext=(5, 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(6, 2.2, 'Metrics\nData', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mapek_loop.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'mapek_loop.png'}")
    plt.close()

# ============================================================
# 2. Service Dependency Graph
# ============================================================
def draw_service_dependencies():
    """
    Draw service dependency graph with replication factors
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, 'Online Boutique Service Dependencies\nwith Elascale Replication Factors', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Define service positions and properties
    services = {
        'Frontend': {
            'pos': (5, 8),
            'formula': 'f = 0.4·CPU + 0.3·MEM + 0.3·NET',
            'replicas': '2-20',
            'color': '#e74c3c'
        },
        'Cart Service': {
            'pos': (2, 6),
            'formula': 'f = 0.3·CPU + 0.4·MEM + 0.1·NET + 0.2·REP',
            'replicas': '2-15',
            'color': '#3498db',
            'dep': 'Redis Cart'
        },
        'Checkout Service': {
            'pos': (5, 6),
            'formula': 'f = 0.5·CPU + 0.3·MEM + 0.2·REP',
            'replicas': '2-12',
            'color': '#f39c12'
        },
        'Product Catalog': {
            'pos': (8, 6),
            'formula': 'f = 0.3·CPU + 0.5·MEM + 0.2·REP',
            'replicas': '2-10',
            'color': '#2ecc71'
        },
        'Recommendation': {
            'pos': (2, 4),
            'formula': 'f = 0.6·CPU + 0.4·MEM',
            'replicas': '1-8',
            'color': '#9b59b6'
        },
        'Redis Cart': {
            'pos': (2, 2),
            'formula': 'Stateful (no HPA)',
            'replicas': '1',
            'color': '#95a5a6'
        }
    }
    
    # Draw service boxes
    for service_name, props in services.items():
        x, y = props['pos']
        
        box = FancyBboxPatch(
            (x - 0.8, y - 0.6), 1.6, 1.2,
            boxstyle="round,pad=0.1",
            edgecolor=props['color'],
            facecolor=props['color'],
            alpha=0.3,
            linewidth=2
        )
        ax.add_patch(box)
        
        # Service name
        ax.text(x, y + 0.3, service_name, ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        # Replica range
        ax.text(x, y, f"Replicas: {props['replicas']}", ha='center', va='center',
                fontsize=8)
        
        # Formula (below the box)
        ax.text(x, y - 0.9, props['formula'], ha='center', va='top',
                fontsize=7, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw dependencies
    arrow_props = dict(arrowstyle='->', lw=2, color='gray')
    
    # Frontend -> Cart Service
    ax.annotate('', xy=(2.8, 6.6), xytext=(4.2, 7.4),
                arrowprops=arrow_props)
    ax.text(3.5, 7.2, 'cart ops', fontsize=7, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Frontend -> Checkout Service
    ax.annotate('', xy=(5, 6.6), xytext=(5, 7.4),
                arrowprops=arrow_props)
    ax.text(5.5, 7, 'checkout\n2:1 ratio', fontsize=7, ha='left',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Frontend -> Product Catalog
    ax.annotate('', xy=(7.2, 6.6), xytext=(5.8, 7.4),
                arrowprops=arrow_props)
    ax.text(6.5, 7.2, 'products', fontsize=7, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Frontend -> Recommendation
    ax.annotate('', xy=(2.8, 4.6), xytext=(4.2, 7.4),
                arrowprops=arrow_props)
    ax.text(3, 5.5, 'recommendations', fontsize=7, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Cart Service -> Redis Cart
    ax.annotate('', xy=(2, 2.6), xytext=(2, 5.4),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(2.5, 4, '3:1 ratio\n(λ=0.2)', fontsize=8, ha='left', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Checkout -> Product Catalog
    ax.annotate('', xy=(7.2, 5.4), xytext=(5.8, 6.6),
                arrowprops=arrow_props)
    ax.text(6.5, 6, '2:1 ratio', fontsize=7, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#e74c3c', alpha=0.3, label='User-facing (high priority)'),
        mpatches.Patch(color='#3498db', alpha=0.3, label='Stateful dependency'),
        mpatches.Patch(color='#f39c12', alpha=0.3, label='Business logic'),
        mpatches.Patch(color='#2ecc71', alpha=0.3, label='Data service'),
        mpatches.Patch(color='#9b59b6', alpha=0.3, label='ML/AI service'),
        mpatches.Patch(color='#95a5a6', alpha=0.3, label='External dependency'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'service_dependencies.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'service_dependencies.png'}")
    plt.close()

# ============================================================
# 3. Multi-Factor Formula Visualization
# ============================================================
def draw_scaling_formula():
    """
    Visualize the multi-factor scaling formula for each service
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Elascale Multi-Factor Scaling Formulas\nf = α·CPU + β·MEM + γ·NET + λ·REP',
                 fontsize=16, fontweight='bold')
    
    services_data = {
        'Frontend': {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3, 'lambda': 0.0},
        'Cart Service': {'alpha': 0.3, 'beta': 0.4, 'gamma': 0.1, 'lambda': 0.2},
        'Checkout': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.0, 'lambda': 0.2},
        'Product Catalog': {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.0, 'lambda': 0.2},
        'Recommendation': {'alpha': 0.6, 'beta': 0.4, 'gamma': 0.0, 'lambda': 0.0},
        'Baseline HPA': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'lambda': 0.0},
    }
    
    axes = axes.flatten()
    
    for idx, (service_name, weights) in enumerate(services_data.items()):
        ax = axes[idx]
        
        factors = ['CPU\n(α)', 'Memory\n(β)', 'Network\n(γ)', 'Replication\n(λ)']
        values = [weights['alpha'], weights['beta'], weights['gamma'], weights['lambda']]
        colors_list = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
        
        bars = ax.bar(factors, values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Weight', fontsize=10)
        ax.set_title(service_name, fontsize=12, fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight dominant factor
        max_idx = values.index(max(values))
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scaling_formulas.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'scaling_formulas.png'}")
    plt.close()

# ============================================================
# 4. Scaling Behavior Comparison
# ============================================================
def draw_scaling_behavior():
    """
    Compare baseline vs. Elascale scaling behavior
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('HPA Scaling Behavior: Baseline vs. Elascale', fontsize=16, fontweight='bold')
    
    # Time axis
    time = np.arange(0, 300, 1)
    
    # Simulate load pattern
    load = 30 + 40 * np.sin(time / 50) + 10 * np.random.randn(len(time))
    load = np.clip(load, 0, 100)
    
    # Baseline scaling (slower, CPU-only)
    baseline_replicas = np.ones(len(time))
    for i in range(1, len(time)):
        if load[i] > 70 and baseline_replicas[i-1] < 10:
            # Slow scale-up: +1 pod every 60s
            if i % 60 == 0:
                baseline_replicas[i] = min(baseline_replicas[i-1] + 1, 10)
            else:
                baseline_replicas[i] = baseline_replicas[i-1]
        elif load[i] < 40 and baseline_replicas[i-1] > 1:
            # Slow scale-down: -1 pod every 120s
            if i % 120 == 0:
                baseline_replicas[i] = max(baseline_replicas[i-1] - 1, 1)
            else:
                baseline_replicas[i] = baseline_replicas[i-1]
        else:
            baseline_replicas[i] = baseline_replicas[i-1]
    
    # Elascale scaling (faster, multi-factor)
    elascale_replicas = np.ones(len(time)) * 2  # Start with min=2
    for i in range(1, len(time)):
        if load[i] > 65 and elascale_replicas[i-1] < 15:
            # Fast scale-up: +3 pods every 30s
            if i % 30 == 0:
                elascale_replicas[i] = min(elascale_replicas[i-1] + 3, 15)
            else:
                elascale_replicas[i] = elascale_replicas[i-1]
        elif load[i] < 45 and elascale_replicas[i-1] > 2:
            # Conservative scale-down: -1 pod every 120s
            if i % 120 == 0:
                elascale_replicas[i] = max(elascale_replicas[i-1] - 1, 2)
            else:
                elascale_replicas[i] = elascale_replicas[i-1]
        else:
            elascale_replicas[i] = elascale_replicas[i-1]
    
    # Plot 1: Baseline
    ax1_load = ax1.twinx()
    ax1.plot(time, baseline_replicas, 'b-', linewidth=2, label='Replicas')
    ax1_load.plot(time, load, 'r--', alpha=0.5, label='Load (%)')
    ax1_load.axhline(y=70, color='orange', linestyle=':', label='Scale-up threshold')
    ax1_load.axhline(y=40, color='green', linestyle=':', label='Scale-down threshold')
    
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Number of Replicas', fontsize=11, color='b')
    ax1_load.set_ylabel('CPU Utilization (%)', fontsize=11, color='r')
    ax1.set_title('Baseline HPA (CPU-only, slow)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_load.tick_params(axis='y', labelcolor='r')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_load.legend(loc='upper right')
    
    # Plot 2: Elascale
    ax2_load = ax2.twinx()
    ax2.plot(time, elascale_replicas, 'b-', linewidth=2, label='Replicas')
    ax2_load.plot(time, load, 'r--', alpha=0.5, label='Load (%)')
    ax2_load.axhline(y=65, color='orange', linestyle=':', label='Scale-up threshold')
    ax2_load.axhline(y=45, color='green', linestyle=':', label='Scale-down threshold')
    
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Number of Replicas', fontsize=11, color='b')
    ax2_load.set_ylabel('Multi-factor Score (%)', fontsize=11, color='r')
    ax2.set_title('Elascale HPA (multi-factor, aggressive)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_load.tick_params(axis='y', labelcolor='r')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_load.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scaling_behavior.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'scaling_behavior.png'}")
    plt.close()

# ============================================================
# 5. Performance Comparison (if data exists)
# ============================================================
def draw_performance_comparison():
    """
    Draw performance comparison charts if experimental data exists
    """
    results_dir = Path("/home/common/EECS6446_project/files/optimizations/results")
    
    # Check if any result files exist
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print("ℹ  No experimental data found. Skipping performance comparison charts.")
        print("   Run experiments first: python3 elascale_mape_k_experiment.py")
        return
    
    # Load the most recent complete experiment
    complete_files = [f for f in csv_files if 'complete_experiment' in f.name]
    
    if not complete_files:
        print("ℹ  No complete experiment data found.")
        return
    
    latest_file = max(complete_files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    print(f"📊 Loading data from: {latest_file.name}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline vs. Elascale Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Response Time by User Count
    ax1 = axes[0, 0]
    for hpa_config in df['hpa_config'].unique():
        subset = df[df['hpa_config'] == hpa_config].groupby('scenario_users')['response_time_p95'].mean()
        ax1.plot(subset.index, subset.values, marker='o', label=hpa_config.capitalize(), linewidth=2)
    
    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('P95 Response Time (ms)')
    ax1.set_title('Response Time vs. Load')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Replica Count by User Count
    ax2 = axes[0, 1]
    for hpa_config in df['hpa_config'].unique():
        for service in df['service'].unique()[:3]:  # Top 3 services
            subset = df[(df['hpa_config'] == hpa_config) & (df['service'] == service)]
            subset = subset.groupby('scenario_users')['replicas'].mean()
            ax2.plot(subset.index, subset.values, marker='o', 
                    label=f'{hpa_config}-{service}', linewidth=2)
    
    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Average Replicas')
    ax2.set_title('Scaling Behavior')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Plot 3: CPU Utilization
    ax3 = axes[1, 0]
    for hpa_config in df['hpa_config'].unique():
        subset = df[df['hpa_config'] == hpa_config].groupby('scenario_users')['cpu_util'].mean()
        ax3.plot(subset.index, subset.values, marker='o', label=hpa_config.capitalize(), linewidth=2)
    
    ax3.set_xlabel('Number of Users')
    ax3.set_ylabel('Average CPU Utilization (%)')
    ax3.set_title('CPU Utilization')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    # Plot 4: Memory Utilization
    ax4 = axes[1, 1]
    for hpa_config in df['hpa_config'].unique():
        subset = df[df['hpa_config'] == hpa_config].groupby('scenario_users')['mem_util'].mean()
        ax4.plot(subset.index, subset.values, marker='o', label=hpa_config.capitalize(), linewidth=2)
    
    ax4.set_xlabel('Number of Users')
    ax4.set_ylabel('Average Memory Utilization (%)')
    ax4.set_title('Memory Utilization')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'performance_comparison.png'}")
    plt.close()

# ============================================================
# Main Execution
# ============================================================
def main():
    print("\n" + "="*60)
    print("EECS6446 Elascale Optimization - Diagram Generator")
    print("="*60 + "\n")
    
    print("Generating diagrams...\n")
    
    # Generate all diagrams
    draw_mapek_loop()
    draw_service_dependencies()
    draw_scaling_formula()
    draw_scaling_behavior()
    draw_performance_comparison()
    
    print("\n" + "="*60)
    print("All diagrams generated successfully!")
    print(f"Location: {OUTPUT_DIR}")
    print("="*60)
    
    print("\nGenerated files:")
    for img_file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {img_file.name}")
    
    print("\nUse these diagrams in your:")
    print("  1. Academic report")
    print("  2. Presentation slides")
    print("  3. Jupyter notebook analysis")

if __name__ == "__main__":
    main()
