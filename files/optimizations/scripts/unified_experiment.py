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
NAMESPACE = "default"
OUTPUT_DIR = Path("/home/common/EECS6446_project/files/optimizations/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Locust automation settings
LOCUST_AUTOMATED = True  # Set to False for manual control
LOCUST_HOST = "http://localhost:8080"  # Frontend service URL
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

def start_locust_load(users, spawn_rate):
    """
    Start or update Locust load test via API
    
    Args:
        users: Number of users to simulate
        spawn_rate: Users spawned per second
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.post(
            f"{LOCUST_URL}/swarm",
            data={
                "user_count": users,
                "spawn_rate": spawn_rate,
                "host": LOCUST_HOST
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"  ✓ Locust started: {users} users @ {spawn_rate} users/sec spawn rate")
            return True
        else:
            print(f"  ❌ Locust API error: {response.status_code}")
            print(f"     Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Cannot connect to Locust API: {e}")
        print(f"     Make sure Locust is running at {LOCUST_URL}")
        return False
def stop_locust_load():
    """
    Stop Locust load test via API
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(
            f"{LOCUST_URL}/stop",
            timeout=10
        )
        
        if response.status_code == 200:
            print("  ✓ Locust stopped")
            return True
        else:
            print(f"  ⚠️  Locust stop returned: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  Cannot stop Locust: {e}")
        return False

def get_locust_stats():
    """
    Get current Locust statistics via API
    
    Returns:
        dict: Locust stats or None if failed
    """
    try:
        response = requests.get(
            f"{LOCUST_URL}/stats/requests",
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except requests.exceptions.RequestException:
        return None

def check_locust_ready():
    """
    Check if Locust is accessible and ready
    
    Returns:
        bool: True if Locust is ready, False otherwise
    """
    try:
        response = requests.get(LOCUST_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
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
            result = response.json()["data"]["result"]
            return result
        return []
    except Exception as e:
        print(f"⚠️  Prometheus query failed: {e}")
        return []

def query_prometheus_range(query, start, end, step="15s"):
    """Query Prometheus range and return results"""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": step
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()["data"]["result"]
            return result
        return []
    except Exception as e:
        print(f"⚠️  Prometheus range query failed: {e}")
        return []

# ============================================================
# Metric Collection (All Baseline Metrics)
# ============================================================

def collect_service_metrics(service):
    """
    Collect comprehensive metrics for a service
    Matches ALL metrics from baseline report
    """
    metrics = {
        "service": service,
        "timestamp": datetime.now().isoformat(),
    }
    
    # CPU utilization (millicores)
    query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*",namespace="{NAMESPACE}"}}[1m])) * 1000'
    result = query_prometheus(query)
    metrics["cpu_millicores"] = float(result[0]["value"][1]) if result else 0
    
    # CPU utilization percentage
    query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*",namespace="{NAMESPACE}"}}[1m])) * 100 / sum(kube_pod_container_resource_limits{{pod=~"{service}-.*",namespace="{NAMESPACE}",resource="cpu"}})'
    result = query_prometheus(query)
    metrics["cpu_util_percent"] = float(result[0]["value"][1]) if result else 0
    
    # Memory utilization (bytes)
    query = f'sum(container_memory_usage_bytes{{pod=~"{service}-.*",namespace="{NAMESPACE}"}})'
    result = query_prometheus(query)
    metrics["memory_bytes"] = float(result[0]["value"][1]) if result else 0
    
    # Memory utilization percentage
    query = f'sum(container_memory_usage_bytes{{pod=~"{service}-.*",namespace="{NAMESPACE}"}}) * 100 / sum(kube_pod_container_resource_limits{{pod=~"{service}-.*",namespace="{NAMESPACE}",resource="memory"}})'
    result = query_prometheus(query)
    metrics["memory_util_percent"] = float(result[0]["value"][1]) if result else 0
    
    # Network bytes received
    query = f'sum(rate(container_network_receive_bytes_total{{pod=~"{service}-.*",namespace="{NAMESPACE}"}}[1m]))'
    result = query_prometheus(query)
    metrics["network_rx_bytes_per_sec"] = float(result[0]["value"][1]) if result else 0
    
    # Network bytes transmitted
    query = f'sum(rate(container_network_transmit_bytes_total{{pod=~"{service}-.*",namespace="{NAMESPACE}"}}[1m]))'
    result = query_prometheus(query)
    metrics["network_tx_bytes_per_sec"] = float(result[0]["value"][1]) if result else 0
    
    # Replica count (ordered - what HPA wants)
    try:
        result = subprocess.run(
            ["kubectl", "get", "deployment", service, "-n", NAMESPACE, 
             "-o", "jsonpath={.status.replicas}"],
            capture_output=True, text=True, timeout=5, check=False
        )
        metrics["replicas_ordered"] = int(result.stdout) if result.stdout else 0
    except:
        metrics["replicas_ordered"] = 0
    
    # Ready replicas (what's actually ready)
    try:
        result = subprocess.run(
            ["kubectl", "get", "deployment", service, "-n", NAMESPACE,
             "-o", "jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True, timeout=5, check=False
        )
        metrics["replicas_ready"] = int(result.stdout) if result.stdout else 0
    except:
        metrics["replicas_ready"] = 0
    
    return metrics

def collect_request_metrics():
    """
    Collect request-level metrics from frontend
    Matches baseline report: throughput, response time, fault rate
    """
    metrics = {
        "timestamp": datetime.now().isoformat(),
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
    metrics["fault_rate_percent"] = float(result[0]["value"][1]) if result and result[0]["value"][1] != "NaN" else 0
    
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
# Experiment Execution
# ============================================================

def run_experiment(config_name, load_pattern):
    """
    Run complete experiment with specified HPA configuration
    
    Args:
        config_name: "baseline" or "elascale"
        load_pattern: List of load scenarios
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config_name.upper()}")
    print(f"{'='*60}\n")
    
    # Apply HPA configuration
    apply_hpa_config(config_name)
    
    print("Waiting 30s for HPA to stabilize...")
    time.sleep(30)
    
    # Collect all metrics
    all_metrics = []
    experiment_start = time.time()
    elapsed_minutes = 0
    
    for scenario in load_pattern:
        print(f"\n{'-'*60}")
        print(f"Scenario: {scenario['name']} - {scenario['users']} users for {scenario['duration']}s")
        print(f"{'-'*60}")
        
        # Start load - automated or manual
        if LOCUST_AUTOMATED:
            print(f"\n🤖 Automatically starting Locust load...")
            if not start_locust_load(scenario['users'], scenario['spawn_rate']):
                print("\n⚠️  Automated Locust control failed!")
                print("Falling back to manual control...")
                print(f"\n⚠️  MANUAL STEP: Set Locust to {scenario['users']} users")
                print(f"   1. Open Locust: {LOCUST_URL}")
                print(f"   2. Set users: {scenario['users']}")
                print(f"   3. Set spawn rate: {scenario['spawn_rate']}")
                print(f"   4. Press Enter when load is applied...")
                input()
        else:
            # Manual control
            print(f"\n⚠️  MANUAL STEP: Set Locust to {scenario['users']} users")
            print(f"   1. Open Locust: {LOCUST_URL}")
            print(f"   2. Set users: {scenario['users']}")
            print(f"   3. Set spawn rate: {scenario['spawn_rate']}")
            print(f"   4. Press Enter when load is applied...")
            input()
        
        # Wait a bit for load to ramp up
        if LOCUST_AUTOMATED:
            ramp_time = min(10, scenario['users'] / scenario['spawn_rate'])
            print(f"  Waiting {int(ramp_time)}s for load to ramp up...")
            time.sleep(ramp_time)
        
        # Collect metrics during scenario
        scenario_start = time.time()
        interval = 10  # Collect every 10 seconds
        
        while time.time() - scenario_start < scenario['duration']:
            snapshot = collect_complete_snapshot()
            snapshot['config'] = config_name
            snapshot['scenario'] = scenario['name']
            snapshot['scenario_users'] = scenario['users']
            snapshot['elapsed_total_seconds'] = time.time() - experiment_start
            snapshot['elapsed_minutes'] = elapsed_minutes + (time.time() - scenario_start) / 60
            
            all_metrics.append(snapshot)
            
            # Print status
            frontend = snapshot['services']['frontend']
            cart = snapshot['services']['cartservice']
            req = snapshot['requests']
            
            # Get Locust stats if available
            locust_info = ""
            if LOCUST_AUTOMATED:
                locust_stats = get_locust_stats()
                if locust_stats and 'user_count' in locust_stats:
                    locust_info = f"Locust:{locust_stats['user_count']:4d} | "
            
            print(f"  [{int(time.time() - scenario_start):3d}s] "
                  f"{locust_info}"
                  f"Frontend:{frontend['replicas_ordered']:2d}({frontend['replicas_ready']:2d}) | "
                  f"Cart:{cart['replicas_ordered']:2d}({cart['replicas_ready']:2d}) | "
                  f"RPS:{req['throughput_rps']:6.1f} | "
                  f"P95:{req['p95_response_time_ms']:7.1f}ms | "
                  f"Faults:{req['fault_rate_percent']:4.2f}%")
            
            time.sleep(interval)
        
        elapsed_minutes += scenario['duration'] / 60
    
    # Stop load at end of experiment
    if LOCUST_AUTOMATED:
        print(f"\n🛑 Stopping Locust load...")
        stop_locust_load()
    
    print(f"\n✓ Experiment complete: {config_name}")
    return all_metrics

def apply_hpa_config(config_type):
    """
    Apply HPA configuration
    config_type: 'baseline' or 'elascale'
    """
    print(f"\nApplying {config_type} HPA configuration...")
    
    # Remove existing HPAs
    print("  Removing existing HPAs...")
    subprocess.run(
        ["kubectl", "delete", "hpa", "--all", "-n", NAMESPACE],
        stdout=subprocess.PIPE, 
        stderr=subprocess.DEVNULL,
        check=False
    )
    
    time.sleep(5)
    
    if config_type == "baseline":
        # Apply baseline (50% CPU threshold from report)
        print("  Applying baseline HPA (50% CPU threshold)...")
        
        # You need to have baseline HPA YAML files ready
        # If using files from GitHub repo:
        baseline_path = "/home/common/EECS6446_project/files/optimizations/scaling/hpa_backup.yaml"
        
        if Path(baseline_path).exists():
            result = subprocess.run(
                ["kubectl", "apply", "-f", baseline_path],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print("  ✓ Baseline HPA applied")
            else:
                print(f"  ❌ Error: {result.stderr}")
        else:
            print(f"  ⚠️  Baseline HPA file not found: {baseline_path}")
            print("  Please ensure baseline HPA YAML exists")
            sys.exit(1)
            
    elif config_type == "elascale":
        # Apply Elascale optimized HPA
        print("  Applying Elascale HPA (multi-factor, optimized thresholds)...")
        
        elascale_paths = [
            "/home/common/EECS6446_project/files/optimizations/scaling/cartservice-elascale-hpa.yaml",
            "/home/common/EECS6446_project/files/optimizations/scaling/services-elascale-hpa.yaml"
        ]
        
        for path in elascale_paths:
            if Path(path).exists():
                result = subprocess.run(
                    ["kubectl", "apply", "-f", path],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    print(f"  ✓ Applied: {Path(path).name}")
                else:
                    print(f"  ❌ Error applying {Path(path).name}: {result.stderr}")
            else:
                print(f"  ⚠️  File not found: {path}")
    
    print("  Waiting 30s for HPA to initialize...")
    time.sleep(30)
# ============================================================
# Results Processing
# ============================================================

def flatten_and_save_results(metrics_list, config_name):
    """
    Flatten nested metrics structure and save to CSV
    Matches baseline report structure
    """
    rows = []
    
    for snapshot in metrics_list:
        # Base info
        row = {
            "timestamp": snapshot["timestamp"],
            "config": snapshot["config"],
            "scenario": snapshot["scenario"],
            "scenario_users": snapshot["scenario_users"],
            "elapsed_total_seconds": snapshot["elapsed_total_seconds"],
            "elapsed_minutes": snapshot["elapsed_minutes"],
        }
        
        # Request-level metrics
        req = snapshot["requests"]
        row.update({
            "throughput_rps": req["throughput_rps"],
            "avg_response_time_ms": req["avg_response_time_ms"],
            "p95_response_time_ms": req["p95_response_time_ms"],
            "fault_rate_percent": req["fault_rate_percent"],
            "total_requests": req["total_requests"],
        })
        
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
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = OUTPUT_DIR / f"{config_name}_complete_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✓ Results saved: {filename}")
    return df, filename
def check_prerequisites():
    """Check all prerequisites before running experiment"""
    print(f"\n{'='*60}")
    print("System Prerequisites Check")
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
    if LOCUST_AUTOMATED:
        if check_locust_ready():
            print(f"   ✓ Locust accessible at {LOCUST_URL}")
            print("   ✓ Automated load control enabled")
        else:
            print(f"   ❌ Locust not accessible at {LOCUST_URL}")
            print("   To use automated control, start Locust with:")
            print(f"     locust -f /path/to/locustfile.py --host={LOCUST_HOST} --web-port=8089")
            print("   OR set LOCUST_AUTOMATED=False in the script for manual control")
            checks_passed = False
    else:
        try:
            response = requests.get(LOCUST_URL, timeout=5)
            print(f"   ⚠️  Locust accessible (manual control mode)")
            print("   You will need to manually adjust load during experiment")
        except:
            print(f"   ⚠️  Locust not accessible")
            print("   Make sure to start Locust before running experiment")
            print(f"     locust -f /path/to/locustfile.py --host={LOCUST_HOST} --web-port=8089")
    
    # Check services
    print("\n4. Checking services deployment...")
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
    except:
        print("   ❌ Cannot check services")
        checks_passed = False
    
    print(f"\n{'='*60}\n")
    
    if not checks_passed:
        print("❌ Prerequisites check failed. Please fix issues before continuing.")
        sys.exit(1)
    
    return True
# ============================================================
# Main Execution
# ============================================================

def main():
    print(f"\n{'='*70}")
    print("EECS6446 Project - Unified Experiment Framework")
    print("Baseline HPA vs Elascale-Optimized HPA")
    print(f"{'='*70}\n")
    
    # Check prerequisites
    check_prerequisites()
    
    # Select configuration
    print("Select experiment configuration:")
    print("  1. Baseline (50% CPU threshold)")
    print("  2. Elascale (multi-factor, optimized)")
    print("  3. Both (run sequentially - takes ~40 minutes)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice not in ["1", "2", "3"]:
        print("Invalid choice")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("IMPORTANT: Load Pattern Information")
    print(f"{'='*60}")
    print("\nThis experiment uses the exact load pattern from baseline:")
    print("  - 50 users (1 min)")
    print("  - 100 users (1 min)")
    print("  - 500 users (1 min)")
    print("  - 1000 users (3.5 min)")
    print("  - 500 users (1 min)")
    print("  - 100 users (1 min)")
    print("\nTotal duration: ~10 minutes per configuration")
    print(f"{'='*60}\n")
    
    input("Press Enter when ready to begin...")
    
    # Run experiments
    all_results = {}
    
    if choice in ["1", "3"]:
        print("\n\n" + "="*70)
        print("RUNNING BASELINE CONFIGURATION")
        print("="*70)
        baseline_metrics = run_experiment("baseline", LOAD_SCENARIOS)
        baseline_df, baseline_file = flatten_and_save_results(baseline_metrics, "baseline")
        all_results["baseline"] = {"df": baseline_df, "file": baseline_file}
        
        if choice == "3":
            print("\nWaiting 120s before next configuration...")
            time.sleep(120)
    
    if choice in ["2", "3"]:
        print("\n\n" + "="*70)
        print("RUNNING ELASCALE CONFIGURATION")
        print("="*70)
        elascale_metrics = run_experiment("elascale", LOAD_SCENARIOS)
        elascale_df, elascale_file = flatten_and_save_results(elascale_metrics, "elascale")
        all_results["elascale"] = {"df": elascale_df, "file": elascale_file}
    
    # Summary
    print(f"\n\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}\n")
    
    print("Results saved to:")
    for config, data in all_results.items():
        print(f"  - {config}: {data['file']}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Generate comparison visualizations:")
    print(f"   python3 /home/claude/generate_unified_comparison.py")
    print("\n2. Analyze results in detail:")
    print("   - Check CSV files for raw data")
    print("   - Compare metrics against baseline report")
    print("\n3. Create final report with:")
    print("   - Performance improvements")
    print("   - Scaling behavior differences")
    print("   - Resource utilization comparison")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
