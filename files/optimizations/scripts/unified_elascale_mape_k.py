#!/usr/bin/env python3
"""
EECS6446 Project - Unified Elascale MAPE-K Experiment (Robust & Validated)
--------------------------------------------------------------------------
1. Runs Notebook-style continuous load (50->1000->50 users)
2. Runs custom MAPE-K Loop logic with critical robustness fixes:
   - Weighted Average Strategy (Stable Scaling)
   - Robust Metric Collection (Handles missing data)
   - Ready-Replicas Logic (Prevents death spirals)
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
import joblib
import os
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

# Configuration for MAPE-K logic (UPDATED with YAML limits)
SERVICE_CONFIGS = {
    "frontend": {
        "alpha": 0.4, "beta": 0.3,
        "min": 3, "max": 25,
        "cpu_limit_millicores": 200,      # Matches YAML Limit
        "mem_limit_bytes": 128 * 1024 * 1024, # Matches YAML Limit (128Mi)
        "latency_target_seconds": 0.200,      # SLA Target
        "scale_up_threshold": 0.70,
        "scale_down_threshold": 0.40,
        "scale_up_increment": 2,
        "scale_down_increment": 1,
    },
    "cartservice": {
        "alpha": 0.3, "beta": 0.4,
        "min": 2, "max": 15,
        "cpu_limit_millicores": 300,      # Matches YAML Limit (300m)
        "mem_limit_bytes": 128 * 1024 * 1024, # Matches YAML Limit (128Mi)
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.60,
        "scale_down_threshold": 0.30,
        "scale_up_increment": 2,
        "scale_down_increment": 1,
    },
    "checkoutservice": {
        "alpha": 0.5, "beta": 0.3,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 128 * 1024 * 1024, # Matches YAML Limit (128Mi)
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65,
        "scale_down_threshold": 0.35,
        "scale_up_increment": 2,
        "scale_down_increment": 1,
    },
    "currencyservice": {
        "alpha": 0.3, "beta": 0.5,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 128 * 1024 * 1024, # Matches YAML Limit (128Mi)
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65,
        "scale_down_threshold": 0.35,
        "scale_up_increment": 1,
        "scale_down_increment": 1,
    },
    "recommendationservice": {
        "alpha": 0.6, "beta": 0.4,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 450 * 1024 * 1024, # Matches YAML Limit (450Mi)
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65,
        "scale_down_threshold": 0.35,
        "scale_up_increment": 1,
        "scale_down_increment": 1,
    },
    "productcatalogservice": {
        "alpha": 0.3, "beta": 0.5,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 128 * 1024 * 1024, # Matches YAML Limit (128Mi)
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65,
        "scale_down_threshold": 0.35,
        "scale_up_increment": 1,
        "scale_down_increment": 1,
    },
}

SERVICES = list(SERVICE_CONFIGS.keys())

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
    """Get metrics with proper error handling (Priority 2 Fix)"""
    m = {'cpu': 0.0, 'mem': 0.0, 'pods': 0, 'ready_pods': 0, 'latency': 0.0}
    try:
        # CPU Millicores
        q_cpu = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*",namespace="{NAMESPACE}"}}[1m])) * 1000'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_cpu}, timeout=5).json()
        if res.get('data', {}).get('result'):
            m['cpu'] = float(res['data']['result'][0]['value'][1])

        # Memory Bytes
        q_mem = f'sum(container_memory_usage_bytes{{pod=~"{service}-.*",namespace="{NAMESPACE}"}})'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_mem}, timeout=5).json()
        if res.get('data', {}).get('result'):
            m['mem'] = float(res['data']['result'][0]['value'][1])

        # Latency (P95) - Priority 2: Add check for empty result
        q_lat = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{app="{service}",namespace="{NAMESPACE}"}}[1m])) by (le))'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_lat}, timeout=5).json()
        if res.get('data', {}).get('result'):
            m['latency'] = float(res['data']['result'][0]['value'][1])
        else:
            m['latency'] = 0.0

        # Pod Count - Priority 3: Get Ready Replicas
        if k8s_apps:
            scale = k8s_apps.read_namespaced_deployment_scale(service, NAMESPACE)
            m['pods'] = int(scale.status.replicas)
            # Note: ready_replicas can be None if 0 are ready
            ready = getattr(scale.status, 'ready_replicas', 0)
            m['ready_pods'] = int(ready) if ready is not None else 0
            
    except Exception as e:
        print(f"Metrics error for {service}: {e}")
    
    return m

def scale_deployment(service, replicas):
    """Scale with proper bounds checking"""
    conf = SERVICE_CONFIGS.get(service, {'min': 1, 'max': 20})
    replicas = max(conf['min'], min(replicas, conf['max']))
    
    try:
        if k8s_apps:
            k8s_apps.patch_namespaced_deployment_scale(
                service, NAMESPACE, {"spec": {"replicas": int(replicas)}}
            )
            print(f"   ✓ Scaled {service} to {replicas} replicas")
        return True
    except Exception as e:
        print(f"   ✗ Scale error {service}: {e}")
        return False


def get_current_user_load():
    """Get actual current user load from Locust"""
    try:
        stats = requests.get("http://localhost:8089/stats/requests", timeout=2).json()
        return stats.get('user_count', 0)
    except:
        return 0

# ============================================================
# MAPE-K CONTROLLER THREAD
# ============================================================
class CAPAPlusController(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = False
        self.last_scale_time = {}
        self.last_scale_action = {}  # Track last action to prevent oscillation
        self.models = {}
        self.experiment_start_time = time.time()
        
        # Load Predictive Models
        for svc in ['frontend', 'cartservice', 'checkoutservice']:
            model_path = f"models/{svc}_cpu_predictor.pkl"
            if os.path.exists(model_path):
                try:
                    self.models[svc] = joblib.load(model_path)
                    print(f"   ✓ Loaded model for {svc}")
                except Exception as e:
                    print(f"   ✗ Model load error for {svc}: {e}")
        
    def run(self):
        print("\n>>> CAPA+ Controller Started (Robustness Improvements Applied) <<<")
        print("  - [Fix 1] Metric Validation (Handles empty/missing data)")
        print("  - [Fix 2] Provisioning Awareness (Uses Ready Pods)")
        print("  - [Logic] Weighted Average Strategy (Stability-Focused)\n")
        self.running = True
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.experiment_start_time
            
            # Get current load for predictive models
            current_users = get_current_user_load()
            for svc in SERVICES:
                try:
                    self._process_service(svc, current_time, elapsed, current_users)
                except Exception as e:
                    print(f"Error processing {svc}: {e}")
            time.sleep(15)

    def _process_service(self, svc, current_time, elapsed, current_users):
        """MAPE-K Loop for a single service"""
        
        # --- MONITOR ---
        m = get_metrics(svc)
        if m['pods'] == 0:
            return  # Skip if no pods
        
        conf = SERVICE_CONFIGS[svc]
        
        # --- ANALYZE (Level 4: Multi-Metric) ---
        
        # Priority 3 Fix: Use Ready Pods for capacity calculation to avoid death spirals
        active_capacity_count = max(1, m['ready_pods'])
        
        # 1. CPU Score
        cpu_limit = conf['cpu_limit_millicores']
        score_cpu = m['cpu'] / (active_capacity_count * cpu_limit)
        
        # 2. Memory Score
        mem_limit = conf.get('mem_limit_bytes', 500000000)
        score_mem = m['mem'] / (active_capacity_count * mem_limit)
        
        # 3. Latency Score
        lat_target = conf.get('latency_target_seconds', 0.5)
        score_lat = m['latency'] / lat_target if lat_target > 0 else 0
        
        # 4. Predictive Score
        predictive_util = 0
        if svc in self.models and current_users > 0:
            try:
                future_time = elapsed + 60
                input_df = pd.DataFrame(
                    [[current_users, future_time]], 
                    columns=['scenario_users', 'elapsed_total_seconds']
                )
                predicted_cpu = self.models[svc].predict(input_df)[0]
                # Predict against *Ordered* capacity (assuming they will be ready)
                total_capacity = m['pods'] * cpu_limit
                predictive_util = predicted_cpu / total_capacity if total_capacity > 0 else 0
            except Exception:
                pass 
        
        # Priority 1 Fix: Weighted Average Strategy
        # 0.5 CPU + 0.2 Mem + 0.2 Latency + 0.1 Prediction
        final_score = (0.5 * score_cpu) + (0.2 * score_mem) + (0.2 * score_lat) + (0.1 * predictive_util)

        # --- PLAN ---
        if svc not in self.last_scale_time:
            self.last_scale_time[svc] = 0
            self.last_scale_action[svc] = 'none'
        
        target = m['pods']
        action = 'none'
        
        # Scale UP (Aggressive)
        if final_score > conf['scale_up_threshold']:
            if self.last_scale_action[svc] == 'down' and (current_time - self.last_scale_time[svc]) < 60:
                print(f"[{svc}] Scale UP suppressed (recent scale-down)")
                return
            
            target += conf['scale_up_increment']
            action = 'up'
            reason = f"Score={final_score:.2f} (CPU:{score_cpu:.2f} Lat:{score_lat:.2f} Pred:{predictive_util:.2f})"
            print(f"[{svc}] Scale UP: {m['pods']} → {target} | {reason}")
            self.last_scale_time[svc] = current_time
            self.last_scale_action[svc] = 'up'
            
        # Scale DOWN (Conservative)
        elif final_score < conf['scale_down_threshold']:
            stabilization_window = 300
            if self.last_scale_action[svc] == 'up':
                stabilization_window = 600
            
            if current_time - self.last_scale_time[svc] > stabilization_window:
                target -= conf['scale_down_increment']
                action = 'down'
                print(f"[{svc}] Scale DOWN: {m['pods']} → {target} | Score={final_score:.2f} (Stable)")
                self.last_scale_time[svc] = current_time
                self.last_scale_action[svc] = 'down'
            else:
                pass

        # --- EXECUTE ---
        if action != 'none':
            target = max(conf['min'], min(target, conf['max']))
            
            if target != m['pods']:
                # Cost Awareness Logic
                node_tier = "Burstable (Cheap)"
                if target > 10:
                    node_tier = "On-Demand (Expensive)"
                print(f"   [COST] {svc} scaling to {target} on {node_tier} tier")
                scale_deployment(svc, target)
        
    def stop(self):
        self.running = False
        print("\n>>> CAPA+ Controller Stopped <<<")

# ============================================================
# LOAD TEST RUNNER
# ============================================================
def run_experiment_phase(config_name):
    print(f"\n=== RUNNING PHASE: {config_name} ===")
    
    # 1. Setup
    if config_name == "baseline":
        subprocess.run("kubectl apply -f ../scaling/hpa_backup.yaml", shell=True)
        controller = None
    else:
        subprocess.run("kubectl delete hpa --all", shell=True)
        controller = CAPAPlusController()
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
                
                # Locust Data (Robust Fix applied here)
                try:
                    stats = requests.get(f"{LOCUST_URL}/stats/requests").json()
                    row["throughput_rps"] = stats.get('total_rps', 0)
                    row["fault_rate_percent"] = stats.get('fail_ratio', 0) * 100
                    
                    # FIX: Robust extraction for Avg Response Time
                    # Try top-level key first (some versions)
                    avg_resp = stats.get('total_avg_response_time')
                    
                    # If not found, try finding "Total" in stats list
                    if avg_resp is None:
                        stats_list = stats.get('stats', [])
                        if isinstance(stats_list, list):
                            for entry in stats_list:
                                if isinstance(entry, dict) and entry.get('name') == 'Total':
                                    avg_resp = entry.get('avg_response_time', 0)
                                    break
                    
                    # Default to 0 if still not found
                    row["avg_response_time_ms"] = avg_resp if avg_resp is not None else 0
                    
                    row["p95_response_time_ms"] = stats.get('current_response_time_percentile_95', 0)
                except Exception as e:
                    print(f"Locust stats error: {e}")
                    # Provide defaults to prevent crash
                    row["throughput_rps"] = 0
                    row["fault_rate_percent"] = 0
                    row["avg_response_time_ms"] = 0
                    row["p95_response_time_ms"] = 0

                # Service Data (Formatted for generate_unified_comparison.py)
                for svc in SERVICES:
                    m = get_metrics(svc)
                    row[f"{svc}_cpu_millicores"] = m['cpu']
                    row[f"{svc}_memory_bytes"] = m['mem']
                    row[f"{svc}_replicas_ordered"] = m['pods']
                    row[f"{svc}_replicas_ready"] = m['ready_pods'] 
                    
                    # FIX: Use the actual configured limit instead of hardcoded 250
                    limit = SERVICE_CONFIGS[svc]['cpu_limit_millicores']
                    
                    # Calculate utilization based on READY pods for accurate visualization
                    ready_count = max(1, m['ready_pods'])
                    row[f"{svc}_cpu_percent"] = (m['cpu'] / (ready_count * limit)) * 100 if m['pods'] > 0 else 0
                
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

if __name__ == "__main__":
    print("Starting Unified Experiment...")
    run_experiment_phase("baseline")
    print("\nCooldown 60s...")
    time.sleep(60)
    run_experiment_phase("elascale")
    print("\nDONE. Now run: python3 generate_unified_comparison.py")
