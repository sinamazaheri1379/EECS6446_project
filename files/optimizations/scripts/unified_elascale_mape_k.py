#!/usr/bin/env python3
"""
EECS6446 Project - Unified Elascale MAPE-K Experiment (Robust Level 4+)
-----------------------------------------------------------------------
Updates:
- Fix 3: Resets cluster to 'min' replicas defined in config (Fairness)
- Fix 4: Explicitly joins controller thread (Clean teardown)
- Fix 5: Safety division for prediction
- Fix 6: Filters Prometheus queries to main 'server' container (Accuracy)
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

# ... (Configuration remains the same) ...
PROMETHEUS_URL = "http://localhost:9090"
LOCUST_URL = "http://localhost:8089"
NAMESPACE = "default"
OUTPUT_DIR = Path("/home/common/EECS6446_project/files/optimizations/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ... (SERVICE_CONFIGS remains the same) ...
SERVICE_CONFIGS = {
    "frontend": { "alpha": 0.4, "beta": 0.3, "min": 3, "max": 25, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.70, "scale_down_threshold": 0.40, "scale_up_increment": 2, "scale_down_increment": 1 },
    "cartservice": { "alpha": 0.3, "beta": 0.4, "min": 2, "max": 15, "cpu_limit_millicores": 300, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.60, "scale_down_threshold": 0.30, "scale_up_increment": 2, "scale_down_increment": 1 },
    "checkoutservice": { "alpha": 0.5, "beta": 0.3, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 2, "scale_down_increment": 1 },
    "currencyservice": { "alpha": 0.3, "beta": 0.5, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 1, "scale_down_increment": 1 },
    "recommendationservice": { "alpha": 0.6, "beta": 0.4, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 450 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 1, "scale_down_increment": 1 },
    "productcatalogservice": { "alpha": 0.3, "beta": 0.5, "min": 2, "max": 20, "cpu_limit_millicores": 200, "mem_limit_bytes": 128 * 1024 * 1024, "latency_target_seconds": 0.200, "scale_up_threshold": 0.65, "scale_down_threshold": 0.35, "scale_up_increment": 1, "scale_down_increment": 1 },
}

SERVICES = list(SERVICE_CONFIGS.keys())
LOAD_STEPS = [(50, 60), (100, 60), (500, 60), (1000, 180), (500, 60), (100, 60), (50, 60)]
# LOAD_STEPS = [
#    (50, 60),       # Warm-up: 1 min to initialize pods/caches
#    (1000, 1800),   # STEADY STATE: 1000 users for 30 minutes (1800s)
#    (50, 60)        # Cool-down: 1 min to verify scale-down
# ]

try:
    config.load_kube_config()
    k8s_apps = client.AppsV1Api()
except:
    print("⚠️ K8s config not found")

def get_metrics(service):
    """Get metrics with proper error handling (Priority 2 Fix + Attribute Safety)"""
    m = {'cpu': 0.0, 'mem': 0.0, 'pods': 0, 'ready_pods': 0, 'latency': 0.0}
    try:
        # Fix 6: Filter by container="server" to exclude sidecars
        q_cpu = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}}[1m])) * 1000'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_cpu}, timeout=15).json()
        if res.get('data', {}).get('result'):
            m['cpu'] = float(res['data']['result'][0]['value'][1])

        q_mem = f'sum(container_memory_usage_bytes{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}})'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_mem}, timeout=15).json()
        if res.get('data', {}).get('result'):
            m['mem'] = float(res['data']['result'][0]['value'][1])

        # Latency (P95)
        q_lat = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{app="{service}",namespace="{NAMESPACE}"}}[1m])) by (le))'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_lat}, timeout=15).json()
        if res.get('data', {}).get('result'):
            m['latency'] = float(res['data']['result'][0]['value'][1])
        
        # Pod Count
        if k8s_apps:
            deploy = k8s_apps.read_namespaced_deployment(service, NAMESPACE)
            m['pods'] = deploy.status.replicas or 0
            m['ready_pods'] = deploy.status.ready_replicas or 0
            
    except Exception as e:
        print(f"Metrics error for {service}: {e}")
    return m

def scale_deployment(service, replicas):
    # ... (Same as before) ...
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


# Add this class BEFORE CAPAPlusController
class QLearningAgent:
    def __init__(self, actions=[0, 1, 2], alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {} 
        self.actions = actions # 0:Down, 1:Stay, 2:Up
        self.alpha = alpha     # Learning Rate
        self.gamma = gamma     # Discount Factor
        self.epsilon = epsilon # Start with more exploration (20%)
        self.epsilon_decay = 0.995  # Decay exploration over time
        self.epsilon_min = 0.01     # Minimum exploration rate

    def get_state(self, cpu_util, latency_score, pod_ratio):
        """
        Enhanced state discretization with pod count awareness
        """
        # CPU: 0=Low(<40%), 1=Target(40-70%), 2=High(>70%)
        cpu_state = 0 if cpu_util < 0.4 else (2 if cpu_util > 0.7 else 1)
        
        # Latency: 0=Good(<0.8x), 1=OK(0.8-1.2x), 2=Bad(>1.2x)
        lat_state = 0 if latency_score < 0.8 else (2 if latency_score > 1.2 else 1)
        
        # Pods: 0=Low(<0.3 of max), 1=Medium(0.3-0.7), 2=High(>0.7)
        pod_state = 0 if pod_ratio < 0.3 else (2 if pod_ratio > 0.7 else 1)
        
        return (cpu_state, lat_state, pod_state)

    def choose_action(self, state):
        if state not in self.q_table:
            # Initialize with slight bias toward "no change"
            self.q_table[state] = [-0.1, 0.0, -0.1]  # [down, stay, up]
        
        # Epsilon-Greedy with decay
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2])  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def decay_epsilon(self):
        """Reduce exploration over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [-0.1, 0.0, -0.1]
        if next_state not in self.q_table:
            self.q_table[next_state] = [-0.1, 0.0, -0.1]
            
        old_q = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Bellman Equation Update
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_q

class CAPAPlusController(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = False
        self.last_scale_time = {}
        self.last_scale_action = {} 
        self.models = {}
        self.experiment_start_time = time.time()
        
        # FIX 1: Create agents ONCE in __init__ for each service
        self.rl_agents = {svc: QLearningAgent() for svc in SERVICES}
        self.last_rl_state = {}
        self.last_rl_action = {}
        self.rl_metrics = {svc: [] for svc in SERVICES}  # Track performance
        
        # Load predictive models with Safety Check
        for svc in ['frontend', 'cartservice', 'checkoutservice']:
            model_path = f"models/{svc}_cpu_predictor.pkl"
            if os.path.exists(model_path):
                try:
                    self.models[svc] = joblib.load(model_path)
                    print(f"   ✓ Loaded model for {svc}")
                except Exception as e:
                    print(f"   ✗ Model load error for {svc}: {e}")

    def _calculate_reward(self, svc, m, score_cpu, score_lat, action_taken):
        """
        Sophisticated reward function considering multiple objectives
        """
        conf = SERVICE_CONFIGS[svc]
        
        # 1. Efficiency Reward (minimize resource usage)
        pod_ratio = m['pods'] / conf['max']
        efficiency_reward = 1.0 - pod_ratio  # Higher reward for fewer pods
        
        # 2. Performance Penalty (SLA violations)
        sla_penalty = 0
        if score_lat > 1.0:  # Latency exceeds target
            sla_penalty = -2.0 * (score_lat - 1.0)  # Proportional penalty
        
        # 3. Stability Reward (penalize frequent changes)
        stability_reward = 0
        if action_taken == 1:  # No change
            stability_reward = 0.2
        
        # 4. Ready Pod Penalty (penalize if pods aren't ready)
        readiness_penalty = 0
        if m['pods'] > 0 and m['ready_pods'] < m['pods']:
            readiness_penalty = -0.5 * (1 - m['ready_pods'] / m['pods'])
        
        # 5. CPU Utilization Reward (sweet spot between 40-70%)
        cpu_reward = 0
        if 0.4 <= score_cpu <= 0.7:
            cpu_reward = 0.5
        elif score_cpu < 0.2:  # Under-utilized
            cpu_reward = -0.3
        elif score_cpu > 0.9:  # Over-utilized
            cpu_reward = -0.5
        
        # Composite reward
        total_reward = (
            0.3 * efficiency_reward +
            0.3 * sla_penalty +
            0.2 * cpu_reward +
            0.1 * stability_reward +
            0.1 * readiness_penalty
        )
        
        return total_reward

    def _process_service(self, svc, current_time, elapsed, current_users):
        """Enhanced MAPE-K Loop with corrected Q-Learning"""
        
        # --- MONITOR ---
        m = get_metrics(svc)
        if m['pods'] == 0:
            return
        
        conf = SERVICE_CONFIGS[svc]
        active_capacity_count = max(1, m['ready_pods'])
        
        # --- ANALYZE ---
        cpu_limit = conf['cpu_limit_millicores']
        score_cpu = m['cpu'] / (active_capacity_count * cpu_limit)
        
        mem_limit = conf.get('mem_limit_bytes', 500000000)
        score_mem = m['mem'] / (active_capacity_count * mem_limit)
        
        lat_target = conf.get('latency_target_seconds', 0.5)
        score_lat = m['latency'] / lat_target if lat_target > 0 else 0
        
        # Predictive score (unchanged)
        predictive_util = 0
        if svc in self.models and current_users > 0:
            try:
                future_time = elapsed + 60
                input_df = pd.DataFrame(
                    [[current_users, future_time]], 
                    columns=['scenario_users', 'elapsed_total_seconds']
                )
                predicted_cpu = self.models[svc].predict(input_df)[0]
                total_capacity = m['pods'] * cpu_limit
                predictive_util = predicted_cpu / total_capacity if total_capacity > 0 else 0
            except Exception:
                pass
        
        # --- Q-LEARNING (Shadow Mode) ---
        agent = self.rl_agents[svc]
        pod_ratio = m['pods'] / conf['max']
        
        # Get current state
        current_state = agent.get_state(score_cpu, score_lat, pod_ratio)
        
        # Calculate reward for previous action (if exists)
        if svc in self.last_rl_state and svc in self.last_rl_action:
            # Map RL action to actual scaling action for reward calc
            # 0=Scale Down, 1=No Op, 2=Scale Up
            reward = self._calculate_reward(svc, m, score_cpu, score_lat, self.last_rl_action[svc])
            
            # Learn from experience
            agent.learn(self.last_rl_state[svc], self.last_rl_action[svc], reward, current_state)
            
            # Track metrics for analysis
            self.rl_metrics[svc].append({
                'timestamp': current_time,
                'state': self.last_rl_state[svc],
                'action': self.last_rl_action[svc],
                'reward': reward,
                'q_values': agent.q_table.get(self.last_rl_state[svc], [0,0,0]).copy()
            })
            
            # Decay exploration
            agent.decay_epsilon()
        
        # Choose next action (shadow - not executed)
        rl_action_idx = agent.choose_action(current_state)
        self.last_rl_state[svc] = current_state
        self.last_rl_action[svc] = rl_action_idx
        
        # Log Q-Learning decision
        actions = ["SCALE_DOWN", "NO_CHANGE", "SCALE_UP"]
        if svc in self.last_rl_state:
            print(f"   [RL-Shadow] {svc}: State{current_state} → {actions[rl_action_idx]} "
                  f"(ε={agent.epsilon:.3f}, Q={agent.q_table.get(current_state, [0,0,0])})")
        
        # --- ACTUAL EXECUTION (Original MAPE-K Logic) ---
        # Weighted Average Strategy (Stability-Focused)
        final_score = (0.5 * score_cpu) + (0.2 * score_mem) + (0.2 * score_lat) + (0.1 * predictive_util)
        
        # Safety Net (Max Strategy if critical)
        if max(score_cpu, score_mem, score_lat) > 0.95:
            final_score = max(score_cpu, score_mem, score_lat)

        # --- PLAN ---
        if svc not in self.last_scale_time:
            self.last_scale_time[svc] = 0
            self.last_scale_action[svc] = 'none'
        
        target = m['pods']
        action = 'none'
        
        # Scale UP (Aggressive)
        if final_score > conf['scale_up_threshold']:
            # Anti-oscillation
            if self.last_scale_action[svc] == 'down' and (current_time - self.last_scale_time[svc]) < 60:
                return
            
            target += conf['scale_up_increment']
            action = 'up'
            reason = f"Score={final_score:.2f} (CPU:{score_cpu:.2f} Lat:{score_lat:.2f})"
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

        # --- EXECUTE ---
        if action != 'none':
            # Enforce bounds
            target = max(conf['min'], min(target, conf['max']))
            
            if target != m['pods']:
                # Cost Awareness Logic
                node_tier = "Burstable (Cheap)"
                if target > 10:
                    node_tier = "On-Demand (Expensive)"
                print(f"   [COST] {svc} scaling to {target} on {node_tier} tier")
                
                scale_deployment(svc, target)
            
    def save_rl_metrics(self):
        """Save Q-Learning metrics for analysis"""
        for svc, metrics in self.rl_metrics.items():
            if metrics:
                df = pd.DataFrame(metrics)
                filename = OUTPUT_DIR / f"rl_metrics_{svc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved RL metrics: {filename}")
                
                # Also save Q-tables
                q_table_file = OUTPUT_DIR / f"q_table_{svc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                q_table_serializable = {
                    str(k): v for k, v in self.rl_agents[svc].q_table.items()
                }
                with open(q_table_file, 'w') as f:
                    json.dump(q_table_serializable, f, indent=2)
                print(f"Saved Q-table: {q_table_file}")

    def stop(self):
        self.running = False

# ============================================================
# CLUSTER RESET LOGIC
# ============================================================
def reset_cluster():
    print("\n>>> Resetting Cluster State...")
    try:
        subprocess.run("kubectl delete hpa --all --namespace=default", shell=True, check=False)
        
        # Fix 3: Reset to MIN replicas, not 1, for fairness
        for svc, conf in SERVICE_CONFIGS.items():
            min_replicas = conf.get("min", 1)
            subprocess.run(f"kubectl scale deployment/{svc} --replicas={min_replicas} --namespace=default", 
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
    # Always reset before starting a phase to prevent conflicts
    reset_cluster()
    print(f"\n=== RUNNING PHASE: {config_name} ===")
    
    # -----------------------------------
    # Phase setup: HPA vs CAPA+ controller
    # -----------------------------------
    controller = None
    if config_name == "baseline":
        subprocess.run("kubectl apply -f ../scaling/hpa_backup.yaml", shell=True)
    else:
        controller = CAPAPlusController()
        controller.start()
    
    # Give cluster + Locust a moment to settle
    time.sleep(10)

    # Try to reset Locust stats (non-fatal if it fails)
    try:
        requests.get(f"{LOCUST_URL}/stats/reset", timeout=3)
    except Exception as e:
        print(f"⚠️ Could not reset Locust stats: {e}")

    rows = []
    start_time = time.time()
    
    try:
        for users, duration in LOAD_STEPS:
            print(f"   -> Step: {users} users for {duration}s")
            
            # Start / update Locust swarm for this step
            try:
                requests.post(
                    f"{LOCUST_URL}/swarm",
                    data={"user_count": users, "spawn_rate": 20},
                    timeout=3,
                )
            except Exception as e:
                print(f"⚠️ Locust swarm error (users={users}): {e}")
            
            end_step = time.time() + duration

            while time.time() < end_step:
                now = time.time()
                
                # Base row fields
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "config": config_name,
                    "elapsed_total_seconds": now - start_time,
                    "scenario_users": users,
                }

                # -----------------------------------
                # Locust metrics (throughput, faults, RT)
                # -----------------------------------
                try:
                    stats = requests.get(
                        f"{LOCUST_URL}/stats/requests",
                        timeout=3
                    ).json()
                    
                    # Throughput and fault rate
                    row["throughput_rps"] = stats.get("total_rps", 0)
                    row["fault_rate_percent"] = stats.get("fail_ratio", 0) * 100

                    # Average response time (robust fallback)
                    avg_resp = 0

                    # Preferred: top-level "total_avg_response_time"
                    if "total_avg_response_time" in stats:
                        avg_resp = stats["total_avg_response_time"]
                    else:
                        stats_list = stats.get("stats", [])
                        if isinstance(stats_list, list) and stats_list:
                            # Look for an aggregate / total entry
                            found = False
                            for entry in stats_list:
                                if not isinstance(entry, dict):
                                    continue
                                name = entry.get("name", "")
                                # Common patterns: "Total", "", or contains "total"
                                if (
                                    name == "Total"
                                    or name == ""
                                    or "total" in name.lower()
                                ):
                                    avg_resp = entry.get("avg_response_time", 0)
                                    found = True
                                    break
                            # Final fallback: last entry if nothing matched
                            if not found:
                                last = stats_list[-1]
                                if isinstance(last, dict):
                                    avg_resp = last.get("avg_response_time", 0)

                    row["avg_response_time_ms"] = avg_resp
                    row["p95_response_time_ms"] = stats.get(
                        "current_response_time_percentile_95", 0
                    )

                except Exception as e:
                    print(f"⚠️ Locust stats error: {e}")
                    row["throughput_rps"] = 0
                    row["fault_rate_percent"] = 0
                    row["avg_response_time_ms"] = 0
                    row["p95_response_time_ms"] = 0

                # -----------------------------------
                # Service metrics from Prometheus + K8s
                # -----------------------------------
                for svc in SERVICES:
                    m = get_metrics(svc)
                    row[f"{svc}_cpu_millicores"] = m["cpu"]
                    row[f"{svc}_memory_bytes"] = m["mem"]
                    row[f"{svc}_replicas_ordered"] = m["pods"]
                    row[f"{svc}_replicas_ready"] = m["ready_pods"]

                    limit = SERVICE_CONFIGS[svc]["cpu_limit_millicores"]

                    # Option A (more semantic): 0% if no ready pods
                    if m["pods"] > 0 and m["ready_pods"] > 0:
                        row[f"{svc}_cpu_percent"] = (
                            m["cpu"] / (m["ready_pods"] * limit)
                        ) * 100
                    else:
                        row[f"{svc}_cpu_percent"] = 0

                    # If you prefer your original behavior, replace the block above with:
                    # ready_count = max(1, m["ready_pods"])
                    # row[f"{svc}_cpu_percent"] = (
                    #     (m["cpu"] / (ready_count * limit)) * 100 if m["pods"] > 0 else 0
                    # )

                rows.append(row)
                time.sleep(5)

    finally:
        # Stop Locust gracefully
        try:
            requests.get(f"{LOCUST_URL}/stop", timeout=3)
        except Exception as e:
            print(f"⚠️ Locust stop error: {e}")
        
        # Stop controller thread if running
        if controller:
            controller.stop()
            # Wait for thread to finish (avoids zombie threads between phases)
            try:
                controller.join()
            except RuntimeError:
                # In case thread was never started or already dead
                pass

    # -----------------------------------
    # Save collected data
    # -----------------------------------
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
