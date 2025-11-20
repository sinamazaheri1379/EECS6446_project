#!/usr/bin/env python3
"""
EECS6446 Project - Unified Elascale MAPE-K Experiment (Level 5: RL-Enhanced)
----------------------------------------------------------------------------
1. Resets Cluster (Deletes HPA, scales to MIN replicas)
2. Runs Notebook-style continuous load (50->1000->50 users)
3. Runs custom MAPE-K Loop logic:
   - Multi-Metric Analysis (CPU, Memory, Latency)
   - Predictive Scaling (Machine Learning)
   - Shadow Q-Learning (Reinforcement Learning for Self-Optimization)
   - Weighted Average Execution Strategy
4. Outputs data compatible with 'generate_unified_comparison.py'
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
    "frontend": { "alpha": 0.4, "beta": 0.3, "min": 3, "max": 25, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.70, "scale_down_threshold": 0.40, "scale_up_increment": 2, "scale_down_increment": 1 },
    "cartservice": { "alpha": 0.3, "beta": 0.4, "min": 2, "max": 15, "cpu_limit_millicores": 300, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.60, "scale_down_threshold": 0.30, "scale_up_increment": 2, "scale_down_increment": 1 },
    "checkoutservice": { "alpha": 0.5, "beta": 0.3, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 2, "scale_down_increment": 1 },
    "currencyservice": { "alpha": 0.3, "beta": 0.5, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 1, "scale_down_increment": 1 },
    "recommendationservice": { "alpha": 0.6, "beta": 0.4, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 450 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 1, "scale_down_increment": 1 },
    "productcatalogservice": { "alpha": 0.3, "beta": 0.5, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 1, "scale_down_increment": 1 },
}

SERVICES = list(SERVICE_CONFIGS.keys())

# Load Pattern (Can be modified for Steady State)
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
    """Get metrics with container filtering and error handling"""
    m = {'cpu': 0.0, 'mem': 0.0, 'pods': 0, 'ready_pods': 0, 'latency': 0.0}
    try:
        # Fix 5: Filter by container="server" to exclude sidecars
        q_cpu = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}}[1m])) * 1000'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_cpu}, timeout=10).json()
        if res.get('data', {}).get('result'):
            m['cpu'] = float(res['data']['result'][0]['value'][1])

        q_mem = f'sum(container_memory_usage_bytes{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}})'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_mem}, timeout=10).json()
        if res.get('data', {}).get('result'):
            m['mem'] = float(res['data']['result'][0]['value'][1])

        # Latency (P95)
        q_lat = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{app="{service}",namespace="{NAMESPACE}"}}[1m])) by (le))'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_lat}, timeout=10).json()
        if res.get('data', {}).get('result'):
            m['latency'] = float(res['data']['result'][0]['value'][1])
        
        # Pod Count (Ready Replicas)
        if k8s_apps:
            scale = k8s_apps.read_namespaced_deployment_scale(service, NAMESPACE)
            m['pods'] = int(scale.status.replicas)
            ready = getattr(scale.status, 'ready_replicas', 0)
            m['ready_pods'] = int(ready) if ready is not None else 0
            
    except Exception as e:
        print(f"Metrics error for {service}: {e}")
    return m

def scale_deployment(service, replicas):
    conf = SERVICE_CONFIGS.get(service, {'min': 1, 'max': 20})
    replicas = max(conf['min'], min(replicas, conf['max']))
    try:
        if k8s_apps:
            k8s_apps.patch_namespaced_deployment_scale(service, NAMESPACE, {"spec": {"replicas": int(replicas)}})
            print(f"   ✓ Scaled {service} to {replicas} replicas")
        return True
    except Exception as e:
        print(f"   ✗ Scale error {service}: {e}")
        return False

def get_current_user_load():
    try:
        stats = requests.get(f"{LOCUST_URL}/stats/requests", timeout=2).json()
        return stats.get('user_count', 0)
    except:
        return 0

# ============================================================
# REINFORCEMENT LEARNING AGENT
# ============================================================
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {} 
        self.alpha = alpha     
        self.gamma = gamma     
        self.epsilon = epsilon 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01     

    def get_state(self, cpu_util, latency_score, pod_ratio):
        """3D State Space: (CPU, Latency, PodCount)"""
        cpu_state = 0 if cpu_util < 0.4 else (2 if cpu_util > 0.7 else 1)
        lat_state = 0 if latency_score < 0.8 else (2 if latency_score > 1.2 else 1)
        pod_state = 0 if pod_ratio < 0.3 else (2 if pod_ratio > 0.7 else 1)
        return (cpu_state, lat_state, pod_state)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [-0.1, 0.0, -0.1] # Bias for 'Stay'
        
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2]) 
        else:
            return np.argmax(self.q_table[state]) 
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table: self.q_table[state] = [-0.1, 0.0, -0.1]
        if next_state not in self.q_table: self.q_table[next_state] = [-0.1, 0.0, -0.1]
            
        old_q = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_q

# ============================================================
# MAPE-K CONTROLLER
# ============================================================
class CAPAPlusController(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = False
        self.last_scale_time = {}
        self.last_scale_action = {} 
        self.models = {}
        self.experiment_start_time = time.time()
        
        # Fix 2: Initialize RL Agents & State ONCE
        self.rl_agents = {svc: QLearningAgent() for svc in SERVICES}
        self.last_rl_state = {svc: None for svc in SERVICES}
        self.last_rl_action = {svc: None for svc in SERVICES}
        self.rl_metrics = {svc: [] for svc in SERVICES}
        
        # Fix 1: Safe Model Loading
        for svc in ['frontend', 'cartservice', 'checkoutservice']:
            model_path = f"models/{svc}_cpu_predictor.pkl"
            if os.path.exists(model_path):
                try:
                    self.models[svc] = joblib.load(model_path)
                    print(f"   ✓ Loaded model for {svc}")
                except Exception as e:
                    print(f"   ✗ Model load error for {svc}: {e}")

    def run(self):
        print("\n>>> CAPA+ Controller Started (Level 5: RL-Shadow Mode) <<<")
        self.running = True
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.experiment_start_time
            current_users = get_current_user_load()
            
            for svc in SERVICES:
                try:
                    self._process_service(svc, current_time, elapsed, current_users)
                except Exception as e:
                    print(f"Error processing {svc}: {e}")
            
            # Faster shutdown check
            for _ in range(15):
                if not self.running: break
                time.sleep(1)

    def _calculate_reward(self, svc, m, score_cpu, score_lat, action_taken):
        conf = SERVICE_CONFIGS[svc]
        
        # 1. Efficiency (Minimize Pods)
        pod_ratio = m['pods'] / conf['max']
        efficiency_reward = 1.0 - pod_ratio
        
        # 2. SLA Penalty (Latency)
        sla_penalty = 0
        if score_lat > 1.0:
            sla_penalty = -2.0 * (score_lat - 1.0)
        
        # 3. Stability (Reward consistency)
        stability_reward = 0.2 if action_taken == 1 else 0
        
        # 4. Readiness Penalty
        readiness_penalty = 0
        if m['pods'] > 0 and m['ready_pods'] < m['pods']:
            readiness_penalty = -0.5 * (1 - m['ready_pods'] / m['pods'])
        
        # 5. CPU Sweet Spot (40-70%)
        cpu_reward = 0
        if 0.4 <= score_cpu <= 0.7: cpu_reward = 0.5
        elif score_cpu < 0.2: cpu_reward = -0.3
        elif score_cpu > 0.9: cpu_reward = -0.5
        
        return (0.3 * efficiency_reward + 0.3 * sla_penalty + 0.2 * cpu_reward + 
                0.1 * stability_reward + 0.1 * readiness_penalty)

    def _process_service(self, svc, current_time, elapsed, current_users):
        m = get_metrics(svc)
        if m['pods'] == 0: return
        conf = SERVICE_CONFIGS[svc]
        
        # --- ANALYZE (Level 4 Metrics) ---
        active_capacity_count = max(1, m['ready_pods'])
        cpu_limit = conf['cpu_limit_millicores']
        score_cpu = m['cpu'] / (active_capacity_count * cpu_limit)
        
        mem_limit = conf.get('mem_limit_bytes', 500000000)
        score_mem = m['mem'] / (active_capacity_count * mem_limit)
        
        lat_target = conf.get('latency_target_seconds', 0.5)
        score_lat = m['latency'] / lat_target if lat_target > 0 else 0
        
        predictive_util = 0
        if svc in self.models and current_users > 0:
            try:
                input_df = pd.DataFrame([[current_users, elapsed + 60]], columns=['scenario_users', 'elapsed_total_seconds'])
                predicted_cpu = self.models[svc].predict(input_df)[0]
                # Fix 4: Consistent capacity calculation
                total_capacity = m['ready_pods'] * cpu_limit 
                predictive_util = predicted_cpu / total_capacity if total_capacity > 0 else 0
            except: pass

        # --- SHADOW RL LOGIC (Level 5) ---
        agent = self.rl_agents[svc]
        pod_ratio = m['pods'] / conf['max']
        current_state = agent.get_state(score_cpu, score_lat, pod_ratio)
        
        if self.last_rl_state[svc] is not None:
            reward = self._calculate_reward(svc, m, score_cpu, score_lat, self.last_rl_action[svc])
            agent.learn(self.last_rl_state[svc], self.last_rl_action[svc], reward, current_state)
            
            self.rl_metrics[svc].append({
                'timestamp': current_time, 'state': self.last_rl_state[svc],
                'action': self.last_rl_action[svc], 'reward': reward,
                'q_values': agent.q_table.get(self.last_rl_state[svc], [0,0,0]).copy()
            })
            agent.decay_epsilon()
            
        rl_action_idx = agent.choose_action(current_state)
        self.last_rl_state[svc] = current_state
        self.last_rl_action[svc] = rl_action_idx

        # --- ACTUAL EXECUTION (Weighted Average) ---
        final_score = (0.5 * score_cpu) + (0.2 * score_mem) + (0.2 * score_lat) + (0.1 * predictive_util)
        if max(score_cpu, score_mem, score_lat) > 0.95:
            final_score = max(score_cpu, score_mem, score_lat)

        if svc not in self.last_scale_time:
            self.last_scale_time[svc] = 0
            self.last_scale_action[svc] = 'none'
        
        target = m['pods']
        action = 'none'

        if final_score > conf['scale_up_threshold']:
            if self.last_scale_action[svc] == 'down' and (current_time - self.last_scale_time[svc]) < 60:
                return
            target += conf['scale_up_increment']
            action = 'up'
            self.last_scale_time[svc] = current_time
            self.last_scale_action[svc] = 'up'
            print(f"[{svc}] Scale UP: {m['pods']} → {target} | Score={final_score:.2f}")
            
        elif final_score < conf['scale_down_threshold']:
            stabilization_window = 300 if self.last_scale_action[svc] != 'up' else 600
            if current_time - self.last_scale_time[svc] > stabilization_window:
                target -= conf['scale_down_increment']
                action = 'down'
                self.last_scale_time[svc] = current_time
                self.last_scale_action[svc] = 'down'
                print(f"[{svc}] Scale DOWN: {m['pods']} → {target} | Score={final_score:.2f}")

        if action != 'none':
            target = max(conf['min'], min(target, conf['max']))
            if target != m['pods']:
                node_tier = "On-Demand (Expensive)" if target > 10 else "Burstable (Cheap)"
                print(f"   [COST] {svc} scaling to {target} on {node_tier} tier")
                scale_deployment(svc, target)

    def save_rl_metrics(self):
        for svc, metrics in self.rl_metrics.items():
            if metrics:
                df = pd.DataFrame(metrics)
                filename = OUTPUT_DIR / f"rl_metrics_{svc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                
                q_table_file = OUTPUT_DIR / f"q_table_{svc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                q_serial = {str(k): v for k, v in self.rl_agents[svc].q_table.items()}
                with open(q_table_file, 'w') as f:
                    json.dump(q_serial, f, indent=2)
    
    def stop(self):
        self.running = False

# ============================================================
# CLUSTER RESET LOGIC
# ============================================================
def reset_cluster():
    print("\n>>> Resetting Cluster State...")
    try:
        # Fix 6: Scope delete to namespace to be safe
        subprocess.run(f"kubectl delete hpa --all --namespace={NAMESPACE}", shell=True, check=False)
        
        # Fix 3: Reset to MIN replicas, not 1
        for svc, conf in SERVICE_CONFIGS.items():
            min_replicas = conf.get("min", 1)
            subprocess.run(f"kubectl scale deployment/{svc} --replicas={min_replicas} --namespace={NAMESPACE}", 
                         shell=True, check=False, stdout=subprocess.DEVNULL)
            
        print("   -> Waiting 30s for cluster to stabilize...")
        time.sleep(30)
        print("   ✓ Cluster Reset Complete.\n")
    except Exception as e:
        print(f"   Warning during reset: {e}")

# ============================================================
# LOAD TEST RUNNER
# ============================================================
def run_experiment_phase(config_name):
    reset_cluster()
    print(f"\n=== RUNNING PHASE: {config_name} ===")
    
    if config_name == "baseline":
        subprocess.run("kubectl apply -f ../scaling/hpa_backup.yaml", shell=True)
        controller = None
    else:
        controller = CAPAPlusController()
        controller.start()
    
    time.sleep(10)
    requests.get(f"{LOCUST_URL}/stats/reset")
    
    rows = []
    start_time = time.time()
    
    try:
        for users, duration in LOAD_STEPS:
            print(f"   -> Step: {users} users for {duration}s")
            requests.post(f"{LOCUST_URL}/swarm", data={"user_count": users, "spawn_rate": 20})
            end_step = time.time() + duration
            while time.time() < end_step:
                now = time.time()
                
                # Collect Data
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "config": config_name,
                    "elapsed_total_seconds": now - start_time,
                    "scenario_users": users,
                }
                
                # Locust Data (Robust)
                try:
                    stats = requests.get(f"{LOCUST_URL}/stats/requests", timeout=5).json()
                    row["throughput_rps"] = stats.get('total_rps', 0)
                    row["fault_rate_percent"] = stats.get('fail_ratio', 0) * 100
                    
                    avg_resp = 0
                    if 'total_avg_response_time' in stats:
                         avg_resp = stats['total_avg_response_time']
                    else:
                        stats_list = stats.get('stats', [])
                        if isinstance(stats_list, list):
                            for entry in stats_list:
                                if isinstance(entry, dict) and entry.get('name') == 'Total':
                                    avg_resp = entry.get('avg_response_time', 0)
                                    break
                    row["avg_response_time_ms"] = avg_resp
                    row["p95_response_time_ms"] = stats.get('current_response_time_percentile_95', 0)
                except Exception as e:
                    row["throughput_rps"] = 0
                    row["fault_rate_percent"] = 0
                    row["avg_response_time_ms"] = 0
                    row["p95_response_time_ms"] = 0

                for svc in SERVICES:
                    m = get_metrics(svc)
                    row[f"{svc}_cpu_millicores"] = m['cpu']
                    row[f"{svc}_memory_bytes"] = m['mem']
                    row[f"{svc}_replicas_ordered"] = m['pods']
                    row[f"{svc}_replicas_ready"] = m['ready_pods'] 
                    limit = SERVICE_CONFIGS[svc]['cpu_limit_millicores']
                    ready_count = max(1, m['ready_pods'])
                    row[f"{svc}_cpu_percent"] = (m['cpu'] / (ready_count * limit)) * 100 if m['pods'] > 0 else 0
                
                rows.append(row)
                time.sleep(5)
                
    finally:
        requests.get(f"{LOCUST_URL}/stop")
        if controller:
            controller.stop()
            controller.join() # Fix 4: Explicit join
            controller.save_rl_metrics() # Save Q-Learning data

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
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
