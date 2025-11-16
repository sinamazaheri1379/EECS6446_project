#!/usr/bin/env python3
"""
EECS6446 Project - Unified Experiment Framework
Measures ALL baseline metrics for Baseline HPA vs Elascale-Optimized HPA

Metrics collected (matching baseline report):
- User load over time
- Throughput (requests/sec)
- Fault rate
- Average response time & 95th percentile
- CPU usage per service
- Pod counts (ordered vs ready) per service
- Memory, network metrics (additional)
"""

import time
import requests
import subprocess
import pandas as pd
from datetime import datetime
import json
import sys
from pathlib import Path
import numpy as np

# ============================================================
# Configuration
# ============================================================
PROMETHEUS_URL = "http://localhost:9090"
LOCUST_URL = "http://localhost:8089"
LOCUST_HOST = "http://localhost:8080"
NAMESPACE = "default"
OUTPUT_DIR = Path("/home/common/EECS6446_project/files/optimizations/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CRITICAL: Use exact same load pattern as baseline report
LOAD_SCENARIOS = [
    {"name": "baseline_50", "users": 50, "duration": 60, "spawn_rate": 10},
    {"name": "increase_100", "users": 100, "duration": 60, "spawn_rate": 20},
    {"name": "moderate_500", "users": 500, "duration": 60, "spawn_rate": 50},
    {"name": "peak_1000", "users": 1000, "duration": 210, "spawn_rate": 100},  # 3.5 minutes
    {"name": "decrease_500", "users": 500, "duration": 60, "spawn_rate": 50},
    {"name": "return_100", "users": 100, "duration": 60, "spawn_rate": 20},
]

# Services monitored (from baseline report)
SERVICES = [
    "frontend", 
    "cartservice", 
    "checkoutservice",
    "currencyservice", 
    "recommendationservice",
    "productcatalogservice"
]

# Elascale multi-factor weights
SERVICE_WEIGHTS = {
    "frontend": {"alpha": 0.4, "beta": 0.3, "gamma": 0.3, "lambda": 0.0},
    "cartservice": {"alpha": 0.3, "beta": 0.4, "gamma": 0.1, "lambda": 0.2},
    "checkoutservice": {"alpha": 0.5, "beta": 0.3, "gamma": 0.0, "lambda": 0.2},
    "currencyservice": {"alpha": 0.3, "beta": 0.5, "gamma": 0.0, "lambda": 0.2},
    "recommendationservice": {"alpha": 0.6, "beta": 0.4, "gamma": 0.0, "lambda": 0.0},
    "productcatalogservice": {"alpha": 0.3, "beta": 0.5, "gamma": 0.0, "lambda": 0.2},
}

# ============================================================
# Locust Automation Functions
# ============================================================

def start_load_test(users, spawn_rate):
    """Programmatically start Locust load test"""
    try:
        response = requests.post(
            f"{LOCUST_URL}/swarm",
            data={"user_count": users, "spawn_rate": spawn_rate}
        )
        if response.status_code == 200:
            print(f"   ✓ Locust started: {users} users @ {spawn_rate}/s spawn rate")
            return True
        else:
            print(f"   ❌ Locust start failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot start Locust: {e}")
        return False

def stop_load_test():
    """Stop Locust load test"""
    try:
        response = requests.get(f"{LOCUST_URL}/stop")
        if response.status_code == 200:
            print("   ✓ Locust stopped")
            return True
        else:
            print(f"   ⚠️  Locust stop returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ⚠️  Cannot stop Locust: {e}")
        return False

def get_locust_stats():
    """Get current Locust statistics"""
    try:
        response = requests.get(f"{LOCUST_URL}/stats/requests")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def reset_locust_stats():
    """Reset Locust statistics"""
    try:
        response = requests.get(f"{LOCUST_URL}/stats/reset")
        if response.status_code == 200:
            print("   ✓ Locust stats reset")
            return True
    except:
        print("   ⚠️  Cannot reset Locust stats")
    return False

# ============================================================
# Prometheus Query Functions
# ============================================================

def query_prometheus(query, timeout=10):
    """Query Prometheus and return results"""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=timeout
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return result.get("data", {}).get("result", [])
    except Exception as e:
        print(f"Prometheus query error: {e}")
    return []

def collect_service_metrics(service_name):
    """Collect all metrics for a specific service"""
    metrics = {
        "cpu_millicores": 0,
        "cpu_util_percent": 0,
        "memory_bytes": 0,
        "memory_util_percent": 0,
        "network_rx_bytes_per_sec": 0,
        "network_tx_bytes_per_sec": 0,
        "replicas_ordered": 0,
        "replicas_ready": 0,
    }
    
    # CPU usage (millicores)
    query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"{service_name}-.*"}}[1m])) * 1000'
    result = query_prometheus(query)
    metrics["cpu_millicores"] = float(result[0]["value"][1]) if result else 0
    
    # CPU utilization percentage
    query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"{service_name}-.*"}}[1m])) / sum(kube_pod_container_resource_requests{{namespace="{NAMESPACE}",pod=~"{service_name}-.*",resource="cpu"}}) * 100'
    result = query_prometheus(query)
    metrics["cpu_util_percent"] = float(result[0]["value"][1]) if result else 0
    
    # Memory usage (bytes)
    query = f'sum(container_memory_usage_bytes{{namespace="{NAMESPACE}",pod=~"{service_name}-.*"}})'
    result = query_prometheus(query)
    metrics["memory_bytes"] = float(result[0]["value"][1]) if result else 0
    
    # Memory utilization percentage
    query = f'sum(container_memory_usage_bytes{{namespace="{NAMESPACE}",pod=~"{service_name}-.*"}}) / sum(kube_pod_container_resource_requests{{namespace="{NAMESPACE}",pod=~"{service_name}-.*",resource="memory"}}) * 100'
    result = query_prometheus(query)
    metrics["memory_util_percent"] = float(result[0]["value"][1]) if result else 0
    
    # Network RX (bytes/sec)
    query = f'sum(rate(container_network_receive_bytes_total{{namespace="{NAMESPACE}",pod=~"{service_name}-.*"}}[1m]))'
    result = query_prometheus(query)
    metrics["network_rx_bytes_per_sec"] = float(result[0]["value"][1]) if result else 0
    
    # Network TX (bytes/sec)
    query = f'sum(rate(container_network_transmit_bytes_total{{namespace="{NAMESPACE}",pod=~"{service_name}-.*"}}[1m]))'
    result = query_prometheus(query)
    metrics["network_tx_bytes_per_sec"] = float(result[0]["value"][1]) if result else 0
    
    # Replicas (desired/ordered)
    query = f'kube_deployment_spec_replicas{{namespace="{NAMESPACE}",deployment="{service_name}"}}'
    result = query_prometheus(query)
    metrics["replicas_ordered"] = int(float(result[0]["value"][1])) if result else 0
    
    # Replicas (ready)
    query = f'kube_deployment_status_replicas_ready{{namespace="{NAMESPACE}",deployment="{service_name}"}}'
    result = query_prometheus(query)
    metrics["replicas_ready"] = int(float(result[0]["value"][1])) if result else 0
    
    return metrics

def collect_request_metrics():
    """Collect application-level request metrics"""
    metrics = {
        "throughput_rps": 0,
        "avg_response_time_ms": 0,
        "p95_response_time_ms": 0,
        "fault_rate_percent": 0,
        "total_requests": 0
    }
    
    # Throughput (requests per second)
    query = 'sum(rate(http_requests_total{job="frontend"}[1m]))'
    result = query_prometheus(query)
    metrics["throughput_rps"] = float(result[0]["value"][1]) if result else 0
    
    # Average response time (milliseconds)
    query = 'histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job="frontend"}[1m])) by (le)) * 1000'
    result = query_prometheus(query)
    metrics["avg_response_time_ms"] = float(result[0]["value"][1]) if result else 0
    
    # 95th percentile response time (milliseconds)
    query = 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="frontend"}[1m])) by (le)) * 1000'
    result = query_prometheus(query)
    metrics["p95_response_time_ms"] = float(result[0]["value"][1]) if result else 0
    
    # Fault rate (percentage)
    query = 'sum(rate(http_requests_total{job="frontend",status=~"5.."}[1m])) / sum(rate(http_requests_total{job="frontend"}[1m])) * 100'
    result = query_prometheus(query)
    if result and result[0]["value"][1] != "NaN":
        metrics["fault_rate_percent"] = float(result[0]["value"][1])
    else:
        metrics["fault_rate_percent"] = 0
    
    # Total request count
    query = 'sum(http_requests_total{job="frontend"})'
    result = query_prometheus(query)
    metrics["total_requests"] = float(result[0]["value"][1]) if result else 0
    
    return metrics

def collect_complete_snapshot():
    """Collect complete system snapshot"""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "requests": collect_request_metrics()
    }
    
    for service in SERVICES:
        snapshot["services"][service] = collect_service_metrics(service)
    
    return snapshot

# ============================================================
# HPA Configuration Management
# ============================================================

def apply_hpa_config(config_type):
    """
    Apply HPA configuration (baseline or elascale)
    
    Args:
        config_type: "baseline" or "elascale"
    """
    print(f"\n{'='*60}")
    print(f"Applying HPA Configuration: {config_type.upper()}")
    print(f"{'='*60}\n")
    
    config_dir = Path("/home/common/EECS6446_project/files/optimizations/hpa_configs")
    
    if config_type == "baseline":
        # Apply baseline CPU-only HPA (50% threshold)
        for service in SERVICES:
            hpa_file = config_dir / "baseline" / f"{service}-hpa.yaml"
            if hpa_file.exists():
                try:
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", str(hpa_file)],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if result.returncode == 0:
                        print(f"   ✓ Applied baseline HPA for {service}")
                    else:
                        print(f"   ❌ Failed to apply {service}: {result.stderr}")
                except Exception as e:
                    print(f"   ❌ Error applying {service}: {e}")
            else:
                print(f"   ⚠️  HPA file not found: {hpa_file}")
    
    elif config_type == "elascale":
        # Apply Elascale multi-factor HPA
        for service in SERVICES:
            hpa_file = config_dir / "elascale" / f"{service}-hpa.yaml"
            if hpa_file.exists():
                try:
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", str(hpa_file)],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if result.returncode == 0:
                        print(f"   ✓ Applied elascale HPA for {service}")
                    else:
                        print(f"   ❌ Failed to apply {service}: {result.stderr}")
                except Exception as e:
                    print(f"   ❌ Error applying {service}: {e}")
            else:
                print(f"   ⚠️  HPA file not found: {hpa_file}")
    
    print(f"\n{'='*60}\n")

# ============================================================
# Experiment Execution
# ============================================================

def run_single_experiment(config_name, load_scenarios):
    """
    Run complete experiment with specified HPA configuration
    
    Args:
        config_name: "baseline" or "elascale"
        load_scenarios: List of load scenarios
        
    Returns:
        List of metric snapshots
    """
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {config_name.upper()}")
    print(f"{'='*70}\n")
    
    # Apply HPA configuration
    apply_hpa_config(config_name)
    
    # CRITICAL: Wait for HPA to stabilize
    print("Waiting 60s for HPA controllers to stabilize...")
    time.sleep(60)
    
    # Reset Locust stats before starting
    reset_locust_stats()
    
    # Collect all metrics
    all_metrics = []
    experiment_start = time.time()
    scenario_number = 0
    
    for scenario in load_scenarios:
        scenario_number += 1
        print(f"\n{'-'*70}")
        print(f"SCENARIO {scenario_number}/{len(load_scenarios)}: {scenario['name']}")
        print(f"Users: {scenario['users']} | Duration: {scenario['duration']}s | Spawn Rate: {scenario['spawn_rate']}/s")
        print(f"{'-'*70}\n")
        
        # Start load test automatically
        if not start_load_test(scenario['users'], scenario['spawn_rate']):
            print(f"❌ Failed to start load test for scenario {scenario['name']}")
            continue
        
        # Wait for spawn to complete
        spawn_time = scenario['users'] / scenario['spawn_rate']
        print(f"Waiting {spawn_time:.1f}s for users to spawn...")
        time.sleep(spawn_time + 5)  # Extra 5s buffer
        
        # Collect metrics during scenario
        scenario_start = time.time()
        collection_interval = 10  # Collect every 10 seconds
        next_collection = scenario_start + collection_interval
        
        while time.time() < scenario_start + scenario['duration']:
            current_time = time.time()
            
            # Time to collect metrics?
            if current_time >= next_collection:
                snapshot = collect_complete_snapshot()
                snapshot['config'] = config_name
                snapshot['scenario'] = scenario['name']
                snapshot['scenario_users'] = scenario['users']
                snapshot['scenario_number'] = scenario_number
                snapshot['elapsed_total_seconds'] = current_time - experiment_start
                snapshot['elapsed_scenario_seconds'] = current_time - scenario_start
                
                all_metrics.append(snapshot)
                
                # Print status update
                frontend = snapshot['services']['frontend']
                cart = snapshot['services']['cartservice']
                req = snapshot['requests']
                
                elapsed = int(current_time - scenario_start)
                print(f"  [{elapsed:3d}s/{scenario['duration']}s] "
                      f"Users:{scenario['users']:4d} | "
                      f"Frontend:{frontend['replicas_ordered']:2d}({frontend['replicas_ready']:2d}) | "
                      f"Cart:{cart['replicas_ordered']:2d}({cart['replicas_ready']:2d}) | "
                      f"RPS:{req['throughput_rps']:6.1f} | "
                      f"P95:{req['p95_response_time_ms']:7.1f}ms | "
                      f"Faults:{req['fault_rate_percent']:4.2f}%")
                
                next_collection += collection_interval
            
            # Sleep briefly to avoid busy waiting
            time.sleep(1)
        
        # Stop load test
        stop_load_test()
        
        # Wait between scenarios (except after last one)
        if scenario_number < len(load_scenarios):
            print(f"\nWaiting 30s before next scenario...")
            time.sleep(30)
    
    print(f"\n✓ Experiment complete: {config_name}")
    print(f"  Total snapshots collected: {len(all_metrics)}")
    
    return all_metrics

def run_comparative_experiments():
    """
    Run BOTH baseline and elascale experiments sequentially
    """
    print(f"\n{'='*70}")
    print("EECS6446 PROJECT - COMPARATIVE EXPERIMENT")
    print("Baseline HPA vs Elascale-Optimized HPA")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    # Run BASELINE experiment
    print("\n" + "="*70)
    print("PHASE 1/2: BASELINE HPA (CPU-only, 50% threshold)")
    print("="*70)
    baseline_results = run_single_experiment("baseline", LOAD_SCENARIOS)
    all_results["baseline"] = baseline_results
    
    # CRITICAL: Stabilization period between configurations
    print(f"\n{'='*70}")
    print("STABILIZATION PERIOD")
    print(f"{'='*70}")
    print("Waiting 120s for cluster to return to steady state...")
    print("This ensures fair comparison between configurations.")
    time.sleep(120)
    
    # Run ELASCALE experiment
    print("\n" + "="*70)
    print("PHASE 2/2: ELASCALE HPA (Multi-factor optimized)")
    print("="*70)
    elascale_results = run_single_experiment("elascale", LOAD_SCENARIOS)
    all_results["elascale"] = elascale_results
    
    return all_results

# ============================================================
# Results Processing
# ============================================================
def save_results(all_results):
    """
    Save experimental results to CSV files
    
    Args:
        all_results: Dictionary with "baseline" and "elascale" keys
    """
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = []
    
    for config_name, snapshots in all_results.items():
        if not snapshots:
            print(f"⚠️  No data for {config_name}, skipping...")
            continue
        
        # Convert snapshots to DataFrame rows
        rows = []
        for snapshot in snapshots:
            req = snapshot["requests"]
            
            row = {
                "timestamp": snapshot["timestamp"],
                "config": snapshot["config"],
                "scenario": snapshot["scenario"],
                "scenario_users": snapshot["scenario_users"],
                "scenario_number": snapshot["scenario_number"],
                "elapsed_total_seconds": snapshot["elapsed_total_seconds"],
                "elapsed_scenario_seconds": snapshot["elapsed_scenario_seconds"],
                "throughput_rps": req["throughput_rps"],
                "avg_response_time_ms": req["avg_response_time_ms"],
                "p95_response_time_ms": req["p95_response_time_ms"],
                "fault_rate_percent": req["fault_rate_percent"],
                "total_requests": req["total_requests"],
            }
            
            # Service-specific metrics
            for service, metrics in snapshot["services"].items():
                row.update({
                    f"{service}_cpu_millicores": metrics["cpu_millicores"],
                    f"{service}_cpu_percent": metrics["cpu_util_percent"],
                    f"{service}_memory_bytes": metrics["memory_bytes"],
                    f"{service}_memory_percent": metrics["memory_util_percent"],
                    f"{service}_network_rx": metrics["network_rx_bytes_per_sec"],
                    f"{service}_network_tx": metrics["network_tx_bytes_per_sec"],
                    f"{service}_replicas_ordered": metrics["replicas_ordered"],
                    f"{service}_replicas_ready": metrics["replicas_ready"],
                })
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        filename = OUTPUT_DIR / f"{config_name}_complete_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✓ Saved {config_name}: {filename}")
        print(f"  - {len(rows)} data points")
        print(f"  - {df['scenario'].nunique()} scenarios")
        saved_files.append(filename)
    
    # Save combined file for easy comparison
    combined_rows = []
    for config_name, snapshots in all_results.items():
        for snapshot in snapshots:
            req = snapshot["requests"]
            row = {
                "timestamp": snapshot["timestamp"],
                "config": snapshot["config"],
                "scenario": snapshot["scenario"],
                "scenario_users": snapshot["scenario_users"],
                "elapsed_total_seconds": snapshot["elapsed_total_seconds"],
                "throughput_rps": req["throughput_rps"],
                "p95_response_time_ms": req["p95_response_time_ms"],
                "fault_rate_percent": req["fault_rate_percent"],
            }
            
            # Add key service metrics
            for service in ["frontend", "cartservice"]:
                metrics = snapshot["services"][service]
                row.update({
                    f"{service}_replicas_ordered": metrics["replicas_ordered"],
                    f"{service}_replicas_ready": metrics["replicas_ready"],
                    f"{service}_cpu_percent": metrics["cpu_util_percent"],
                })
            
            combined_rows.append(row)
    
    combined_df = pd.DataFrame(combined_rows)
    combined_filename = OUTPUT_DIR / f"comparative_summary_{timestamp}.csv"
    combined_df.to_csv(combined_filename, index=False)
    
    print(f"\n✓ Saved combined comparison: {combined_filename}")
    saved_files.append(combined_filename)
    
    print(f"\n{'='*70}\n")
    
    return saved_files
# ============================================================
# System Checks
# ============================================================

def check_prerequisites():
    """Check all prerequisites before running experiment"""
    print(f"\n{'='*60}")
    print("SYSTEM PREREQUISITES CHECK")
    print(f"{'='*60}\n")
    
    checks_passed = True
    
    # Check Prometheus
    print("1. Checking Prometheus connectivity...")
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "up"},
            timeout=5
        )
        if response.status_code == 200:
            print("   ✓ Prometheus connected")
        else:
            print("   ❌ Prometheus not responding")
            checks_passed = False
    except:
        print("   ❌ Cannot connect to Prometheus")
        print("   Run: kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring")
        checks_passed = False
    
    # Check Kubernetes
    print("\n2. Checking Kubernetes cluster...")
    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes"],
            capture_output=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0:
            print("   ✓ Cluster connected")
        else:
            print("   ❌ Cannot connect to cluster")
            checks_passed = False
    except:
        print("   ❌ kubectl not working")
        checks_passed = False
    
    # Check Locust
    print("\n3. Checking Locust availability...")
    try:
        response = requests.get(LOCUST_URL, timeout=5)
        print("   ✓ Locust web UI accessible")
    except:
        print("   ❌ Locust not accessible at http://localhost:8089")
        print("   Make sure Locust is running:")
        print("   1. kubectl port-forward svc/frontend 8080:8080 -n default")
        print("   2. locust -f locustfile.py --host=http://localhost:8080")
        checks_passed = False
    
    # Check Locust API
    print("\n4. Checking Locust API...")
    try:
        stats = get_locust_stats()
        if stats:
            print("   ✓ Locust API responding")
        else:
            print("   ⚠️  Locust API not responding (may need to start a test first)")
    except:
        print("   ⚠️  Cannot query Locust API")
    
    # Check services
    print("\n5. Checking services deployment...")
    try:
        result = subprocess.run(
            ["kubectl", "get", "deployments", "-n", NAMESPACE],
            capture_output=True,
            text=True,
            check=False
        )
        
        missing_services = []
        for service in SERVICES:
            if service not in result.stdout:
                missing_services.append(service)
        
        if not missing_services:
            print(f"   ✓ All {len(SERVICES)} services deployed")
        else:
            print(f"   ⚠️  Missing services: {missing_services}")
            checks_passed = False
    except:
        print("   ❌ Cannot check services")
        checks_passed = False
    
    # Check HPA config files
    print("\n6. Checking HPA configuration files...")
    config_dir = Path("/home/common/EECS6446_project/files/optimizations/hpa_configs")
    
    baseline_dir = config_dir / "baseline"
    elascale_dir = config_dir / "elascale"
    
    baseline_exists = baseline_dir.exists()
    elascale_exists = elascale_dir.exists()
    
    if baseline_exists and elascale_exists:
        print(f"   ✓ HPA config directories found")
        print(f"     - Baseline: {baseline_dir}")
        print(f"     - Elascale: {elascale_dir}")
    else:
        print(f"   ⚠️  HPA config directories missing:")
        if not baseline_exists:
            print(f"     - Missing: {baseline_dir}")
        if not elascale_exists:
            print(f"     - Missing: {elascale_dir}")
        print("   Experiments may fail without proper HPA configs")
    
    print(f"\n{'='*60}\n")
    
    if not checks_passed:
        print("❌ PREREQUISITES CHECK FAILED")
        print("Please fix the issues above before continuing.\n")
        return False
    
    print("✓ ALL PREREQUISITES PASSED")
    print("Ready to run experiments!\n")
    return True
# ============================================================
# Main Execution
# ============================================================

def main():
    """Main execution function"""
    print(f"\n{'='*70}")
    print("EECS6446 PROJECT - UNIFIED EXPERIMENT FRAMEWORK")
    print("Automated Baseline vs Elascale Comparison")
    print(f"{'='*70}\n")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Confirm with user before starting
    print("This experiment will:")
    print("  1. Run BASELINE HPA configuration (CPU-only, 50% threshold)")
    print("  2. Wait 120s for stabilization")
    print("  3. Run ELASCALE HPA configuration (multi-factor optimized)")
    print("  4. Each configuration tests 6 load scenarios")
    print(f"  5. Total estimated time: ~{(len(LOAD_SCENARIOS) * 2 * 90 + 120) / 60:.0f} minutes\n")
    
    response = input("Continue with automated experiments? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nExperiment cancelled by user.")
        sys.exit(0)
    
    # Run comparative experiments
    print("\nStarting automated experiments...")
    all_results = run_comparative_experiments()
    
    # Save results
    saved_files = save_results(all_results)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}\n")
    
    print("Results saved to:")
    for filepath in saved_files:
        print(f"  - {filepath}")
    
    print("\nNext steps:")
    print("1. Analyze results in Jupyter notebook:")
    print("   jupyter notebook /home/common/EECS6446_project/files/optimizations/analysis/elascale_analysis.ipynb")
    print("\n2. Generate comparison visualizations:")
    print("   python3 /home/common/EECS6446_project/files/optimizations/scripts/generate_comparison_plots.py")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Experiment interrupted by user (Ctrl+C)")
        print("Attempting to stop Locust...")
        stop_load_test()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to stop Locust...")
        stop_load_test()
        sys.exit(1)

