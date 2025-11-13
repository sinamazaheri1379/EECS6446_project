#!/usr/bin/env python3
"""
EECS6446 Elascale-Inspired MAPE-K Autoscaling Experiment

This script implements the Monitor-Analyze-Plan-Execute-Knowledge loop
to evaluate Elascale-optimized HPA vs baseline HPA.

Based on: Khazaei et al. (2017) "Elascale: Autoscaling and Monitoring as a Service"
"""

import time
import requests
import subprocess
import pandas as pd
from datetime import datetime
import json

# ============================================================
# Configuration
# ============================================================
PROMETHEUS_URL = "http://localhost:9090"
LOCUST_URL = "http://localhost:8089"
NAMESPACE = "default"

# Load test scenarios (users, duration in seconds)
LOAD_SCENARIOS = [
    {"users": 50, "duration": 300, "spawn_rate": 10},
    {"users": 100, "duration": 300, "spawn_rate": 10},
    {"users": 200, "duration": 300, "spawn_rate": 20},
    {"users": 500, "duration": 300, "spawn_rate": 50},
]

# Services to monitor
SERVICES = ["frontend", "cartservice", "checkoutservice", 
            "productcatalogservice", "recommendationservice"]

# Elascale multi-factor weights (from paper)
SERVICE_WEIGHTS = {
    "frontend": {"alpha": 0.4, "beta": 0.3, "gamma": 0.3, "lambda": 0.0},
    "cartservice": {"alpha": 0.3, "beta": 0.4, "gamma": 0.1, "lambda": 0.2},
    "checkoutservice": {"alpha": 0.5, "beta": 0.3, "gamma": 0.0, "lambda": 0.2},
    "productcatalogservice": {"alpha": 0.3, "beta": 0.5, "gamma": 0.0, "lambda": 0.2},
    "recommendationservice": {"alpha": 0.6, "beta": 0.4, "gamma": 0.0, "lambda": 0.0},
}

# ============================================================
# MAPE-K Loop Implementation
# ============================================================

# -------------------- MONITOR --------------------
def monitor_collect_metrics(service):
    """
    MONITOR Phase: Collect performance metrics from Prometheus
    Returns dict with CPU, memory, network utilization and replica count
    """
    metrics = {}
    
    try:
        # CPU utilization
        query = f'rate(container_cpu_usage_seconds_total{{pod=~"{service}.*"}}[1m]) * 100'
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
        if response.status_code == 200:
            result = response.json()["data"]["result"]
            if result:
                metrics["cpu_util"] = float(result[0]["value"][1])
            else:
                metrics["cpu_util"] = 0
        
        # Memory utilization
        query = f'container_memory_usage_bytes{{pod=~"{service}.*"}} / container_spec_memory_limit_bytes{{pod=~"{service}.*"}} * 100'
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
        if response.status_code == 200:
            result = response.json()["data"]["result"]
            if result:
                metrics["mem_util"] = float(result[0]["value"][1])
            else:
                metrics["mem_util"] = 0
        
        # Current replicas
        result = subprocess.run(
            ["kubectl", "get", "deployment", service, "-n", NAMESPACE, "-o", "json"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            deployment = json.loads(result.stdout)
            metrics["current_replicas"] = deployment["status"].get("replicas", 0)
            metrics["ready_replicas"] = deployment["status"].get("readyReplicas", 0)
        
    except Exception as e:
        print(f"Warning: Could not collect metrics for {service}: {e}")
        metrics = {
            "cpu_util": 0,
            "mem_util": 0,
            "current_replicas": 0,
            "ready_replicas": 0
        }
    
    return metrics

# -------------------- ANALYZE --------------------
def analyze_compute_scaling_score(metrics, service):
    """
    ANALYZE Phase: Compute scaling score using Elascale multi-factor formula
    f = Î±Â·CPU + Î²Â·MEM + Î³Â·NET + Î»Â·REP
    """
    weights = SERVICE_WEIGHTS[service]
    
    # Basic formula (network not implemented in this version)
    score = (
        weights["alpha"] * metrics["cpu_util"] +
        weights["beta"] * metrics["mem_util"]
    )
    
    # Add replication factor if applicable (Î» > 0)
    if weights["lambda"] > 0 and metrics["current_replicas"] > 0:
        # Simplified: assumes target ratio is met when score is balanced
        rep_factor = 100  # Placeholder for replication factor logic
        score += weights["lambda"] * (rep_factor / metrics["current_replicas"])
    
    return score

# -------------------- PLAN --------------------
def plan_determine_action(score, current_replicas, min_replicas, max_replicas):
    """
    PLAN Phase: Determine scaling action based on score
    Returns: ('scale_up'/'scale_down'/'maintain', target_replicas)
    """
    # Thresholds from Elascale configuration
    SCALE_UP_THRESHOLD = 70
    SCALE_DOWN_THRESHOLD = 40
    
    if score > SCALE_UP_THRESHOLD and current_replicas < max_replicas:
        # Aggressive scale-up (Elascale principle)
        target = min(current_replicas + 3, max_replicas)
        return ('scale_up', target)
    
    elif score < SCALE_DOWN_THRESHOLD and current_replicas > min_replicas:
        # Conservative scale-down
        target = max(current_replicas - 1, min_replicas)
        return ('scale_down', target)
    
    else:
        return ('maintain', current_replicas)

# -------------------- EXECUTE --------------------
def execute_apply_hpa_config(config_type):
    """
    EXECUTE Phase: Apply HPA configuration
    config_type: 'baseline' or 'elascale'
    """
    if config_type == 'baseline':
        print("Applying baseline HPA (CPU-only, 70% threshold)...")
        for service in SERVICES:
            subprocess.run([
                "kubectl", "autoscale", "deployment", service,
                "--cpu-percent=70", "--min=1", "--max=10",
                "-n", NAMESPACE
            ])
    
    elif config_type == 'elascale':
        print("Applying Elascale-optimized HPA (multi-factor)...")
        subprocess.run([
            "kubectl", "apply", "-f",
            "/home/EECS6446_project/files/optimizations/scaling/cartservice-elascale-hpa.yaml"
        ])
        subprocess.run([
            "kubectl", "apply", "-f",
            "/home/EECS6446_project/files/optimizations/scaling/services-elascale-hpa.yaml"
        ])
    
    # Wait for HPA to be ready
    time.sleep(30)

# -------------------- KNOWLEDGE --------------------
def knowledge_store_results(experiment_name, data):
    """
    KNOWLEDGE Phase: Store experimental results for learning
    """
    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/EECS6446_project/files/optimizations/results/{experiment_name}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")
    return filename

# ============================================================
# Load Testing Functions
# ============================================================
def start_load_test(users, spawn_rate):
    """Start Locust load test"""
    try:
        response = requests.post(
            f"{LOCUST_URL}/swarm",
            data={"user_count": users, "spawn_rate": spawn_rate}
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Could not start load test: {e}")
        return False

def stop_load_test():
    """Stop Locust load test"""
    try:
        requests.get(f"{LOCUST_URL}/stop")
        return True
    except:
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

# ============================================================
# Experiment Runner
# ============================================================
def run_experiment(hpa_config, scenario):
    """
    Run a single experiment with specified HPA config and load scenario
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {hpa_config.upper()}")
    print(f"Users: {scenario['users']}, Duration: {scenario['duration']}s")
    print(f"{'='*60}\n")
    
    # EXECUTE: Apply HPA configuration
    execute_apply_hpa_config(hpa_config)
    
    # Start load test
    if not start_load_test(scenario["users"], scenario["spawn_rate"]):
        print("Failed to start load test. Make sure Locust is running.")
        return None
    
    # MONITOR + ANALYZE loop
    results = []
    start_time = time.time()
    
    while time.time() - start_time < scenario["duration"]:
        timestamp = datetime.now()
        
        # Get Locust stats
        locust_stats = get_locust_stats()
        
        # Monitor each service
        for service in SERVICES:
            # MONITOR
            metrics = monitor_collect_metrics(service)
            
            # ANALYZE
            score = analyze_compute_scaling_score(metrics, service)
            
            # Record data
            result = {
                "timestamp": timestamp,
                "hpa_config": hpa_config,
                "scenario_users": scenario["users"],
                "service": service,
                "cpu_util": metrics["cpu_util"],
                "mem_util": metrics["mem_util"],
                "replicas": metrics["current_replicas"],
                "ready_replicas": metrics["ready_replicas"],
                "scaling_score": score,
            }
            
            # Add Locust metrics if available
            if locust_stats:
                result["response_time_p95"] = locust_stats.get("current_response_time_percentile_95", 0)
                result["fail_ratio"] = locust_stats.get("fail_ratio", 0)
                result["total_rps"] = locust_stats.get("total_rps", 0)
            
            results.append(result)
        
        # Print progress
        elapsed = int(time.time() - start_time)
        print(f"Progress: {elapsed}/{scenario['duration']}s", end='\r')
        
        # Wait before next monitoring cycle
        time.sleep(15)
    
    # Stop load test
    stop_load_test()
    
    # KNOWLEDGE: Store results
    filename = knowledge_store_results(f"{hpa_config}_{scenario['users']}users", results)
    
    print(f"\nâœ“ Experiment complete: {filename}\n")
    return results

# ============================================================
# Main Execution
# ============================================================
def main():
    print("\n" + "="*60)
    print("EECS6446 Elascale MAPE-K Autoscaling Experiment")
    print("="*60 + "\n")
    
    # Check prerequisites
    print("Checking prerequisites...")
    
    try:
        requests.get(PROMETHEUS_URL, timeout=5)
        print("âœ“ Prometheus accessible")
    except:
        print("âœ— Prometheus not accessible. Please run:")
        print("  kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring")
        return
    
    try:
        requests.get(LOCUST_URL, timeout=5)
        print("âœ“ Locust accessible")
    except:
        print("âœ— Locust not accessible. Please start Locust and run:")
        print("  kubectl port-forward svc/frontend 8080:8080 -n default")
        print("  locust -f locustfile.py --host=http://localhost:8080")
        return
    
    print("\n" + "="*60)
    print("Running Experiments")
    print("="*60 + "\n")
    
    all_results = []
    
    # Run experiments for each scenario with both HPA configs
    for scenario in LOAD_SCENARIOS:
        # Baseline HPA
        results_baseline = run_experiment("baseline", scenario)
        if results_baseline:
            all_results.extend(results_baseline)
        
        # Wait between experiments
        print("Waiting 60s for cluster to stabilize...")
        time.sleep(60)
        
        # Elascale HPA
        results_elascale = run_experiment("elascale", scenario)
        if results_elascale:
            all_results.extend(results_elascale)
        
        # Wait between scenarios
        print("Waiting 120s before next scenario...")
        time.sleep(120)
    
    # KNOWLEDGE: Store combined results
    if all_results:
        knowledge_store_results("complete_experiment", all_results)
    
    print("\n" + "="*60)
    print("All Experiments Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Analyze results in Jupyter notebook:")
    print("   jupyter notebook /home/EECS6446_project/files/optimizations/analysis/elascale_analysis.ipynb")
    print("\n2. Generate diagrams:")
    print("   python3 /home/EECS6446_project/files/optimizations/scripts/generate_diagrams.py")

if __name__ == "__main__":
    main()
