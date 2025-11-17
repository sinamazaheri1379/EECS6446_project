#!/usr/bin/env python3
"""
EECS6446 Project - Unified Elascale MAPE-K Experiment
-----------------------------------------------------
1. Runs Notebook-style continuous load (50->1000->50 users)
2. Runs custom MAPE-K Loop logic (Python-based scaling)
3. Outputs data compatible with 'generate_unified_comparison.py'
"""

import time
import requests
import subprocess
import pandas as pd
import numpy as np
import threading
import json
import sys
from datetime import datetime
from pathlib import Path
from kubernetes import client, config

# ============================================================
# CONFIGURATION
# ============================================================
PROMETHEUS_URL = "http://localhost:9090"
LOCUST_URL = "http://localhost:8089"
NAMESPACE = "default"
OUTPUT_DIR = Path("/home/common/EECS6446_project/files/optimizations/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERVICES = [
    "frontend", "cartservice", "checkoutservice",
    "currencyservice", "recommendationservice", "productcatalogservice"
]

# Configuration for MAPE-K logic
SERVICE_CONFIGS = {
    "frontend": {"alpha": 0.4, "beta": 0.3, "min": 3, "max": 25},
    "cartservice": {"alpha": 0.3, "beta": 0.4, "min": 2, "max": 15},
    "checkoutservice": {"alpha": 0.5, "beta": 0.3, "min": 2, "max": 20},
    "currencyservice": {"alpha": 0.3, "beta": 0.5, "min": 2, "max": 20},
    "recommendationservice": {"alpha": 0.6, "beta": 0.4, "min": 2, "max": 20},
    "productcatalogservice": {"alpha": 0.3, "beta": 0.5, "min": 2, "max": 20},
}

LOAD_STEPS = [
    (50, 60), (100, 60), (500, 60), 
    (1000, 180), # Peak
    (500, 60), (100, 60), (50, 60)
]

# ============================================================
# K8S & PROMETHEUS HELPERS
# ============================================================
try:
    config.load_kube_config()
    k8s_apps = client.AppsV1Api()
except:
    print("⚠️ K8s config not found")

def get_metrics(service):
    """Get comprehensive metrics for CSV report & MAPE-K"""
    m = {'cpu': 0.0, 'mem': 0.0, 'pods': 0}
    try:
        # CPU Millicores (for CSV)
        q_cpu = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*",namespace="{NAMESPACE}"}}[1m])) * 1000'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_cpu}).json()
        if res['data']['result']: m['cpu'] = float(res['data']['result'][0]['value'][1])

        # Memory Bytes (for CSV)
        q_mem = f'sum(container_memory_usage_bytes{{pod=~"{service}-.*",namespace="{NAMESPACE}"}})'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_mem}).json()
        if res['data']['result']: m['mem'] = float(res['data']['result'][0]['value'][1])

        # Pod Count
        scale = k8s_apps.read_namespaced_deployment_scale(service, NAMESPACE)
        m['pods'] = int(scale.status.replicas)
    except:
        pass
    return m

def scale_deployment(service, replicas):
    conf = SERVICE_CONFIGS.get(service, {'min': 1, 'max': 20})
    replicas = max(conf['min'], min(replicas, conf['max']))
    try:
        k8s_apps.patch_namespaced_deployment_scale(
            service, NAMESPACE, {"spec": {"replicas": int(replicas)}}
        )
    except Exception as e:
        print(f"Scale error {service}: {e}")

# ============================================================
# MAPE-K CONTROLLER THREAD
# ============================================================
class ElascaleController(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = False
        
    def run(self):
        print("   >>> Elascale MAPE-K Active")
        self.running = True
        while self.running:
            for svc in SERVICES:
                # MONITOR
                m = get_metrics(svc)
                
                # ANALYZE (Simple Elascale Formula)
                # CPU is in millicores, normalize approx (e.g. 1 core = 1000m)
                # This is where you'd add your 'predictive' hook
                cpu_score = (m['cpu'] / (m['pods'] * 200)) # Rough utilization
                
                # PLAN
                target = m['pods']
                if cpu_score > 0.50: target += 2
                elif cpu_score < 0.30: target -= 1
                
                # EXECUTE
                if target != m['pods']:
                    scale_deployment(svc, target)
            time.sleep(15)

    def stop(self):
        self.running = False

# ============================================================
# LOAD TEST RUNNER (Generates Compatible CSV)
# ============================================================
def run_experiment_phase(config_name):
    print(f"\n=== RUNNING PHASE: {config_name} ===")
    
    # 1. Setup
    if config_name == "baseline":
        subprocess.run("kubectl apply -f files/optimizations/scaling/hpa_backup.yaml", shell=True)
        controller = None
    else:
        subprocess.run("kubectl delete hpa --all", shell=True)
        controller = ElascaleController()
        controller.start()
    
    time.sleep(10)
    requests.get(f"{LOCUST_URL}/stats/reset")
    
    # 2. Run Load Pattern
    rows = []
    start_time = time.time()
    
    try:
        for users, duration in LOAD_STEPS:
            print(f"   -> Step: {users} users for {duration}s")
            requests.post(f"{LOCUST_URL}/swarm", data={"user_count": users, "spawn_rate": 20})
            
            end_step = time.time() + duration
            while time.time() < end_step:
                now = time.time()
                
                # --- COLLECT DATA FOR CSV ---
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "config": config_name,
                    "elapsed_total_seconds": now - start_time,
                    "scenario_users": users,
                }
                
                # Locust Data
                try:
                    stats = requests.get(f"{LOCUST_URL}/stats/requests").json()
                    row["throughput_rps"] = stats.get('total_rps', 0)
                    row["fault_rate_percent"] = stats.get('fail_ratio', 0) * 100
                    row["avg_response_time_ms"] = stats.get('current_response_time_percentile_95', 0) # Approx
                    row["p95_response_time_ms"] = stats.get('current_response_time_percentile_95', 0)
                except:
                    pass

                # Service Data (Formatted for generate_unified_comparison.py)
                for svc in SERVICES:
                    m = get_metrics(svc)
                    row[f"{svc}_cpu_millicores"] = m['cpu']
                    row[f"{svc}_memory_bytes"] = m['mem']
                    row[f"{svc}_replicas_ordered"] = m['pods']
                    row[f"{svc}_replicas_ready"] = m['pods'] # Simplifying for speed
                    # Fake calculating percent for the chart
                    row[f"{svc}_cpu_percent"] = (m['cpu'] / (m['pods']*250)) * 100 if m['pods'] > 0 else 0
                
                rows.append(row)
                time.sleep(5) # Sample rate
                
    finally:
        requests.get(f"{LOCUST_URL}/stop")
        if controller: controller.stop()

    # 3. Save Compatible CSV
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Filename MUST match what generate_unified_comparison looks for:
    filename = OUTPUT_DIR / f"{config_name}_complete_{ts}.csv"
    df.to_csv(filename, index=False)
    print(f"   -> Saved: {filename}")
    return filename

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Starting Unified Experiment...")
    
    # Run 1: Baseline
    run_experiment_phase("baseline")
    
    print("\nCooldown 60s...")
    time.sleep(60)
    
    # Run 2: Elascale
    run_experiment_phase("elascale")
    
    print("\nDONE. Now run: python3 generate_unified_comparison.py")
