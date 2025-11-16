#!/usr/bin/env python3
"""
EECS6446 Project - Interactive Unified Experiment (Notebook-Style)
Combines optimization framework automation with notebook-style real-time visualization

Features:
- Continuous progressive load pattern (like notebook)
- Real-time chart updates during execution
- Automated comparative testing (baseline vs elascale)
- Live monitoring with matplotlib
- PACSLoadTester-style continuous data collection
"""

import time
import requests
import subprocess
import pandas as pd
from datetime import datetime
import json
import sys
import threading
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

# ============================================================
# Configuration
# ============================================================
PROMETHEUS_URL = "http://localhost:9090"
LOCUST_URL = "http://localhost:8089"
NAMESPACE = "default"
OUTPUT_DIR = Path("/home/common/EECS6446_project/files/optimizations/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Continuous load pattern (like notebook - 10 minutes total)
LOAD_PHASES = [
    {"name": "baseline", "users": 50, "duration": 60, "start_time": 0},      # 0-1 min
    {"name": "increase", "users": 100, "duration": 60, "start_time": 60},    # 1-2 min
    {"name": "moderate", "users": 500, "duration": 60, "start_time": 120},   # 2-3 min
    {"name": "peak", "users": 1000, "duration": 210, "start_time": 180},     # 3-6.5 min
    {"name": "decrease", "users": 500, "duration": 90, "start_time": 390},   # 6.5-8 min
    {"name": "return", "users": 100, "duration": 120, "start_time": 480},    # 8-10 min
]

TOTAL_DURATION = 600  # 10 minutes
COLLECTION_INTERVAL = 2  # Collect every 2 seconds (like notebook WorkerThread)

SERVICES = ["frontend", "cartservice", "checkoutservice", 
            "currencyservice", "recommendationservice", "productcatalogservice"]

# ============================================================
# Live Data Collector (like PACSLoadTester WorkerThread)
# ============================================================
class LiveDataCollector(threading.Thread):
    """Background thread that continuously collects metrics"""
    
    def __init__(self, config_name, collection_interval=2):
        super().__init__()
        self.daemon = True
        self.stop_signal = False
        self.collection_interval = collection_interval
        self.config_name = config_name
        self.data_points = []
        self.lock = threading.Lock()
        self.start_time = None
        
    def run(self):
        """Continuous collection loop"""
        self.start_time = time.time()
        
        while not self.stop_signal:
            try:
                # Collect snapshot
                snapshot = self.collect_snapshot()
                
                if snapshot:
                    with self.lock:
                        self.data_points.append(snapshot)
                
                # Sleep for interval
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Collection error: {e}")
                time.sleep(1)
    
    def collect_snapshot(self):
        """Collect single snapshot of all metrics"""
        try:
            # Get Locust stats
            locust_response = requests.get(f"{LOCUST_URL}/stats/requests", timeout=5)
            if locust_response.status_code != 200:
                return None
            
            locust_stats = locust_response.json()
            
            # Only collect if test is running
            if locust_stats['state'] not in ['running', 'spawning']:
                return None
            
            elapsed = time.time() - self.start_time
            
            snapshot = {
                'config': self.config_name,
                'elapsed_sec': elapsed,
                'elapsed_min': elapsed / 60,
                'timestamp': time.time(),
                
                # Locust metrics
                'user_count': locust_stats.get('user_count', 0),
                'total_rps': locust_stats.get('total_rps', 0),
                'fail_ratio': locust_stats.get('fail_ratio', 0),
                'current_response_time_average': locust_stats.get('current_response_time_average', 0),
                'current_response_time_percentile_50': locust_stats.get('current_response_time_percentile_50', 0),
                'current_response_time_percentile_95': locust_stats.get('current_response_time_percentile_95', 0),
            }
            
            # Aggregate stats (index -1 in stats array)
            if locust_stats.get('stats') and len(locust_stats['stats']) > 0:
                agg_stats = locust_stats['stats'][-1]
                snapshot.update({
                    'avg_response_time': agg_stats.get('avg_response_time', 0),
                    'median_response_time': agg_stats.get('median_response_time', 0),
                    'max_response_time': agg_stats.get('max_response_time', 0),
                    'min_response_time': agg_stats.get('min_response_time', 0),
                    'num_requests': agg_stats.get('num_requests', 0),
                    'num_failures': agg_stats.get('num_failures', 0),
                })
            
            # Collect CPU and pod counts for each service
            for service in SERVICES:
                # CPU usage
                cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*", namespace="{NAMESPACE}"}}[1m])) * 100'
                try:
                    cpu_response = requests.get(
                        f"{PROMETHEUS_URL}/api/v1/query",
                        params={"query": cpu_query},
                        timeout=3
                    )
                    if cpu_response.status_code == 200:
                        result = cpu_response.json()['data']['result']
                        snapshot[f'{service}_cpu'] = float(result[0]['value'][1]) if result else 0
                    else:
                        snapshot[f'{service}_cpu'] = 0
                except:
                    snapshot[f'{service}_cpu'] = 0
                
                # Pod counts
                try:
                    result = subprocess.run(
                        ['kubectl', 'get', 'pods', '-n', NAMESPACE, '-l', f'app={service}', '-o', 'json'],
                        capture_output=True, text=True, timeout=3
                    )
                    if result.returncode == 0:
                        pods_data = json.loads(result.stdout)
                        total_pods = len(pods_data.get('items', []))
                        ready_pods = sum(1 for pod in pods_data.get('items', [])
                                       if pod.get('status', {}).get('phase') == 'Running')
                        snapshot[f'{service}_pods_total'] = total_pods
                        snapshot[f'{service}_pods_ready'] = ready_pods
                    else:
                        snapshot[f'{service}_pods_total'] = 0
                        snapshot[f'{service}_pods_ready'] = 0
                except:
                    snapshot[f'{service}_pods_total'] = 0
                    snapshot[f'{service}_pods_ready'] = 0
            
            return snapshot
            
        except Exception as e:
            print(f"Error collecting snapshot: {e}")
            return None
    
    def get_data(self):
        """Thread-safe data retrieval"""
        with self.lock:
            return self.data_points.copy()
    
    def stop(self):
        """Stop the collector"""
        self.stop_signal = True

# ============================================================
# Locust Control Functions
# ============================================================
def start_load_test(users, spawn_rate=10):
    """Start Locust load test"""
    try:
        response = requests.post(
            f"{LOCUST_URL}/swarm",
            data={"user_count": users, "spawn_rate": spawn_rate}
        )
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Could not start load test: {e}")
        return False

def stop_load_test():
    """Stop Locust load test"""
    try:
        requests.get(f"{LOCUST_URL}/stop", timeout=5)
        return True
    except:
        return False

def reset_locust_stats():
    """Reset Locust statistics"""
    try:
        response = requests.get(f"{LOCUST_URL}/stats/reset", timeout=5)
        return response.text == 'ok'
    except:
        return False

# ============================================================
# HPA Configuration Management
# ============================================================
def apply_hpa_config(config_type):
    """Apply HPA configuration"""
    scaling_dir = Path("/home/common/EECS6446_project/files/optimizations/scaling")
    
    print(f"\n{'='*60}")
    print(f"Applying {config_type.upper()} HPA Configuration")
    print(f"{'='*60}\n")
    
    # Remove existing HPAs
    print("Removing existing HPAs...")
    subprocess.run(
        ["kubectl", "delete", "hpa", "--all", "-n", NAMESPACE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)
    
    if config_type == "baseline":
        hpa_file = scaling_dir / "hpa_backup.yaml"
        if hpa_file.exists():
            result = subprocess.run(
                ["kubectl", "apply", "-f", str(hpa_file)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✓ Baseline HPA applied (CPU-only, 70% threshold)")
            else:
                print(f"❌ Failed to apply baseline HPA: {result.stderr}")
                return False
    
    elif config_type == "elascale":
        files = [
            scaling_dir / "cartservice-elascale-hpa.yaml",
            scaling_dir / "services-elascale-hpa.yaml"
        ]
        
        for hpa_file in files:
            if hpa_file.exists():
                result = subprocess.run(
                    ["kubectl", "apply", "-f", str(hpa_file)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"✓ Applied {hpa_file.name}")
                else:
                    print(f"❌ Failed: {result.stderr}")
                    return False
    
    print("\nWaiting 30s for HPA to stabilize...")
    time.sleep(30)
    print(f"{'='*60}\n")
    return True

# ============================================================
# Real-Time Visualization
# ============================================================
def create_live_plots():
    """Create matplotlib figure with live updating plots"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Live Experiment Monitoring', fontsize=16, fontweight='bold')
    
    # Configure each subplot
    plots = {
        'throughput': axes[0, 0],
        'users': axes[0, 1],
        'response_time': axes[1, 0],
        'fail_rate': axes[1, 1],
        'cpu': axes[2, 0],
        'pods': axes[2, 1],
    }
    
    # Throughput (RPS)
    plots['throughput'].set_title('Throughput (Requests/sec)')
    plots['throughput'].set_xlabel('Time (minutes)')
    plots['throughput'].set_ylabel('RPS')
    plots['throughput'].grid(True, alpha=0.3)
    
    # User Load
    plots['users'].set_title('User Load')
    plots['users'].set_xlabel('Time (minutes)')
    plots['users'].set_ylabel('Number of Users')
    plots['users'].grid(True, alpha=0.3)
    
    # Response Time
    plots['response_time'].set_title('Response Time')
    plots['response_time'].set_xlabel('Time (minutes)')
    plots['response_time'].set_ylabel('Time (ms)')
    plots['response_time'].grid(True, alpha=0.3)
    
    # Fail Rate
    plots['fail_rate'].set_title('Fault Rate')
    plots['fail_rate'].set_xlabel('Time (minutes)')
    plots['fail_rate'].set_ylabel('Fail Ratio')
    plots['fail_rate'].grid(True, alpha=0.3)
    
    # CPU Usage (Frontend)
    plots['cpu'].set_title('CPU Usage per Container (Frontend)')
    plots['cpu'].set_xlabel('Time (minutes)')
    plots['cpu'].set_ylabel('CPU (millicores)')
    plots['cpu'].grid(True, alpha=0.3)
    
    # Pod Counts (Frontend)
    plots['pods'].set_title('Pod Count (Frontend)')
    plots['pods'].set_xlabel('Time (minutes)')
    plots['pods'].set_ylabel('Number of Pods')
    plots['pods'].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, plots

def update_plots(collector, plots, experiment_name):
    """Update plots with latest data"""
    data_points = collector.get_data()
    
    if len(data_points) < 2:
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data_points)
    
    # Clear all plots
    for plot in plots.values():
        plot.clear()
    
    # Throughput
    plots['throughput'].plot(df['elapsed_min'], df['total_rps'], 'b-', linewidth=2)
    plots['throughput'].set_title('Throughput (Requests/sec)')
    plots['throughput'].set_xlabel('Time (minutes)')
    plots['throughput'].set_ylabel('RPS')
    plots['throughput'].grid(True, alpha=0.3)
    
    # User Load
    plots['users'].plot(df['elapsed_min'], df['user_count'], 'r-', linewidth=2)
    plots['users'].set_title('User Load')
    plots['users'].set_xlabel('Time (minutes)')
    plots['users'].set_ylabel('Number of Users')
    plots['users'].grid(True, alpha=0.3)
    
    # Response Time
    plots['response_time'].plot(df['elapsed_min'], df['avg_response_time'], 
                                'g-', linewidth=2, label='Average')
    plots['response_time'].plot(df['elapsed_min'], df['current_response_time_percentile_95'], 
                                'orange', linewidth=2, label='95th Percentile')
    plots['response_time'].set_title('Response Time')
    plots['response_time'].set_xlabel('Time (minutes)')
    plots['response_time'].set_ylabel('Time (ms)')
    plots['response_time'].legend()
    plots['response_time'].grid(True, alpha=0.3)
    
    # Fail Rate
    plots['fail_rate'].plot(df['elapsed_min'], df['fail_ratio'], 'r-', linewidth=2)
    plots['fail_rate'].set_title('Fault Rate')
    plots['fail_rate'].set_xlabel('Time (minutes)')
    plots['fail_rate'].set_ylabel('Fail Ratio')
    plots['fail_rate'].grid(True, alpha=0.3)
    
    # CPU Usage (Frontend)
    if 'frontend_cpu' in df.columns:
        plots['cpu'].plot(df['elapsed_min'], df['frontend_cpu'], 'b-', linewidth=2)
        plots['cpu'].set_title(f'CPU Usage per Container (Frontend) - {experiment_name}')
        plots['cpu'].set_xlabel('Time (minutes)')
        plots['cpu'].set_ylabel('CPU (millicores)')
        plots['cpu'].grid(True, alpha=0.3)
    
    # Pod Counts (Frontend)
    if 'frontend_pods_total' in df.columns:
        plots['pods'].plot(df['elapsed_min'], df['frontend_pods_total'], 
                          'b-', linewidth=2, label='Ordered')
        plots['pods'].plot(df['elapsed_min'], df['frontend_pods_ready'], 
                          'orange', linewidth=2, label='Ready')
        plots['pods'].set_title(f'Pod Count (Frontend) - {experiment_name}')
        plots['pods'].set_xlabel('Time (minutes)')
        plots['pods'].set_ylabel('Number of Pods')
        plots['pods'].legend()
        plots['pods'].grid(True, alpha=0.3)
    
    plt.tight_layout()

# ============================================================
# Progressive Load Pattern Execution
# ============================================================
def run_progressive_load_experiment(config_name):
    """
    Run single experiment with continuous progressive load pattern
    (Like the notebook approach)
    """
    print(f"\n{'='*70}")
    print(f"RUNNING {config_name.upper()} EXPERIMENT")
    print(f"Progressive Load Pattern (10 minutes)")
    print(f"{'='*70}\n")
    
    # Apply HPA configuration
    if not apply_hpa_config(config_name):
        print("❌ Failed to apply HPA configuration")
        return None
    
    # Reset Locust stats
    reset_locust_stats()
    
    # Start live data collector
    collector = LiveDataCollector(config_name, COLLECTION_INTERVAL)
    collector.start()
    
    # Create live plots
    fig, plots = create_live_plots()
    plt.ion()  # Interactive mode
    plt.show()
    
    # Execute load phases
    experiment_start = time.time()
    current_phase_idx = 0
    
    print("\nStarting progressive load test...\n")
    
    # Start with first phase
    phase = LOAD_PHASES[0]
    print(f"Phase 1/{len(LOAD_PHASES)}: {phase['name']} - {phase['users']} users")
    start_load_test(phase['users'], spawn_rate=10)
    
    try:
        while True:
            elapsed = time.time() - experiment_start
            
            # Check if we should transition to next phase
            if current_phase_idx < len(LOAD_PHASES) - 1:
                next_phase = LOAD_PHASES[current_phase_idx + 1]
                if elapsed >= next_phase['start_time']:
                    current_phase_idx += 1
                    phase = next_phase
                    print(f"\nPhase {current_phase_idx + 1}/{len(LOAD_PHASES)}: {phase['name']} - {phase['users']} users")
                    start_load_test(phase['users'], spawn_rate=20)
            
            # Update plots every 5 seconds
            if int(elapsed) % 5 == 0:
                update_plots(collector, plots, config_name)
                plt.pause(0.1)
            
            # Print status every 30 seconds
            if int(elapsed) % 30 == 0 and elapsed > 0:
                data = collector.get_data()
                if data:
                    latest = data[-1]
                    print(f"  [{elapsed/60:.1f}min] Users: {latest['user_count']:4d} | "
                          f"RPS: {latest['total_rps']:6.1f} | "
                          f"Avg RT: {latest['avg_response_time']:6.1f}ms | "
                          f"Fails: {latest['fail_ratio']:.3f}")
            
            # Check if experiment complete
            if elapsed >= TOTAL_DURATION:
                print("\n✓ Experiment complete!")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    
    finally:
        # Stop load test and collector
        stop_load_test()
        collector.stop()
        time.sleep(2)
        
        # Final plot update
        update_plots(collector, plots, config_name)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = OUTPUT_DIR / f"{config_name}_live_plots_{timestamp}.png"
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plots saved to: {plot_filename}")
        
        plt.ioff()
        plt.close(fig)
        
        # Get collected data
        data_points = collector.get_data()
        
        return data_points

# ============================================================
# Results Saving
# ============================================================
def save_results(data_points, config_name):
    """Save collected data to CSV"""
    if not data_points:
        print("No data to save")
        return None
    
    df = pd.DataFrame(data_points)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"{config_name}_results_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    print(f"✓ Data saved to: {filename}")
    
    return filename

# ============================================================
# Prerequisites Check
# ============================================================
def check_prerequisites():
    """Check if environment is ready"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60 + "\n")
    
    checks_passed = True
    
    # Check Prometheus
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", 
                               params={"query": "up"}, timeout=5)
        if response.status_code == 200:
            print("✓ Prometheus accessible")
        else:
            print("❌ Prometheus not responding correctly")
            checks_passed = False
    except:
        print("❌ Prometheus not accessible")
        print("  Please run: kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring")
        checks_passed = False
    
    # Check Locust
    try:
        response = requests.get(f"{LOCUST_URL}/stats/requests", timeout=5)
        if response.status_code == 200:
            print("✓ Locust accessible")
        else:
            print("❌ Locust not responding correctly")
            checks_passed = False
    except:
        print("❌ Locust not accessible")
        print("  Please ensure Locust is running")
        checks_passed = False
    
    # Check HPA files
    scaling_dir = Path("/home/common/EECS6446_project/files/optimizations/scaling")
    hpa_files = [
        scaling_dir / "hpa_backup.yaml",
        scaling_dir / "cartservice-elascale-hpa.yaml",
        scaling_dir / "services-elascale-hpa.yaml"
    ]
    
    all_files_exist = True
    for hpa_file in hpa_files:
        if hpa_file.exists():
            print(f"✓ Found {hpa_file.name}")
        else:
            print(f"❌ Missing {hpa_file.name}")
            all_files_exist = False
            checks_passed = False
    
    print("\n" + "="*60 + "\n")
    
    if not checks_passed:
        print("❌ PREREQUISITES CHECK FAILED")
        print("Please fix the issues above before continuing.\n")
        return False
    
    print("✓ ALL PREREQUISITES PASSED\n")
    return True

# ============================================================
# Main Execution
# ============================================================
def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("EECS6446 PROJECT - INTERACTIVE UNIFIED EXPERIMENT")
    print("Notebook-Style Real-Time Visualization with Automated Comparison")
    print("="*70 + "\n")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    print("This experiment will:")
    print("  1. Run BASELINE configuration with live visualization (10 min)")
    print("  2. Wait 120s for stabilization")
    print("  3. Run ELASCALE configuration with live visualization (10 min)")
    print("  4. Progressive load pattern: 50 → 100 → 500 → 1000 → 500 → 100 users")
    print("  5. Real-time charts update during execution")
    print(f"  6. Total time: ~25 minutes\n")
    
    response = input("Continue with interactive experiment? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nExperiment cancelled.")
        sys.exit(0)
    
    all_results = {}
    
    # Run baseline experiment
    print("\n" + "="*70)
    print("EXPERIMENT 1/2: BASELINE HPA")
    print("="*70)
    baseline_data = run_progressive_load_experiment("baseline")
    if baseline_data:
        baseline_file = save_results(baseline_data, "baseline")
        all_results['baseline'] = baseline_data
    
    # Stabilization period
    print("\n" + "="*70)
    print("STABILIZATION PERIOD")
    print("="*70)
    print("\nWaiting 120 seconds for cluster to stabilize...")
    for i in range(120, 0, -10):
        print(f"  {i} seconds remaining...")
        time.sleep(10)
    
    # Run elascale experiment
    print("\n" + "="*70)
    print("EXPERIMENT 2/2: ELASCALE HPA")
    print("="*70)
    elascale_data = run_progressive_load_experiment("elascale")
    if elascale_data:
        elascale_file = save_results(elascale_data, "elascale")
        all_results['elascale'] = elascale_data
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70 + "\n")
    
    if all_results:
        print("Results saved:")
        for config_name in all_results.keys():
            print(f"  - {config_name}_results_*.csv")
            print(f"  - {config_name}_live_plots_*.png")
        
        print("\nNext steps:")
        print("  1. Review the generated plots")
        print("  2. Analyze CSV files for detailed metrics")
        print("  3. Compare baseline vs elascale performance")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Experiment interrupted by user (Ctrl+C)")
        stop_load_test()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        stop_load_test()
        sys.exit(1)
