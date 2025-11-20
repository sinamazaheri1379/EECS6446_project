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
    (1000, 1200), # Peak
    (500, 60), (100, 60), (50, 60)
]

# ============================================================
# K8S & PROMETHEUS HELPERS
# ============================================================
k8s_apps = None
try:
    config.load_kube_config()
    k8s_apps = client.AppsV1Api()
except:
    print("⚠️ K8s config not found")

def get_metrics(service):
    """Get metrics with proper error handling"""
    m = {'cpu': 0.0, 'mem': 0.0, 'pods': 0, 'ready_pods': 0, 'latency': 0.0}
    try:
        # ... (Keep your existing Prometheus queries for CPU, Mem, Latency) ...
        # Filter by container="server" to exclude sidecars
        q_cpu = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}}[1m])) * 1000'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_cpu}, timeout=10).json()
        if res.get('data', {}).get('result'):
            m['cpu'] = float(res['data']['result'][0]['value'][1])

        q_mem = f'sum(container_memory_usage_bytes{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}})'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_mem}, timeout=10).json()
        if res.get('data', {}).get('result'):
            m['mem'] = float(res['data']['result'][0]['value'][1])

        q_lat = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{app="{service}",namespace="{NAMESPACE}"}}[1m])) by (le))'
        res = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': q_lat}, timeout=10).json()
        if res.get('data', {}).get('result'):
            m['latency'] = float(res['data']['result'][0]['value'][1])
        
        # --- FIX: Use read_namespaced_deployment for accurate Ready counts ---
        if k8s_apps:
            deploy = k8s_apps.read_namespaced_deployment(service, NAMESPACE)
            m['pods'] = deploy.status.replicas or 0
            m['ready_pods'] = deploy.status.ready_replicas or 0
            
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
# ============================================================
# REINFORCEMENT LEARNING AGENT (Enhanced Version)
# ============================================================
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Add statistics tracking
        self.total_steps = 0
        self.total_updates = 0
        
    def get_state(self, cpu_util, latency_score, pod_ratio):
        """3D State Space: (CPU, Latency, PodCount)"""
        cpu_state = 0 if cpu_util < 0.4 else (2 if cpu_util > 0.7 else 1)
        lat_state = 0 if latency_score < 0.8 else (2 if latency_score > 1.2 else 1)
        pod_state = 0 if pod_ratio < 0.3 else (2 if pod_ratio > 0.7 else 1)
        return (cpu_state, lat_state, pod_state)
    
    def _init_q_values(self, state):
        """Initialize Q-values based on state characteristics"""
        # Default: slight bias toward stability
        q_values = [-0.1, 0.0, -0.1]  # [down, stay, up]
        
        # Adjust based on state
        cpu_state, lat_state, pod_state = state
        
        # Critical states: bias toward appropriate action
        if cpu_state == 2 or lat_state == 2:  # High load/latency
            q_values[2] = 0.1  # Slight bias toward scaling up
        elif cpu_state == 0 and lat_state == 0 and pod_state == 2:  # Low load, many pods
            q_values[0] = 0.1  # Slight bias toward scaling down
            
        return q_values
        
    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = self._init_q_values(state)
        
        self.total_steps += 1
        
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2])
        else:
            # Break ties randomly (important for learning)
            q_values = self.q_table[state]
            max_q = max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if abs(q - max_q) < 1e-6]
            return np.random.choice(max_actions)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = self._init_q_values(state)
        if next_state not in self.q_table:
            self.q_table[next_state] = self._init_q_values(next_state)
            
        old_q = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_q
        
        self.total_updates += 1
    
    # Helper methods for analysis
    def get_policy_summary(self):
        """Get a summary of learned policy"""
        action_names = ["SCALE_DOWN", "NO_CHANGE", "SCALE_UP"]
        policy = {}
        
        for state, q_values in self.q_table.items():
            best_action = np.argmax(q_values)
            policy[state] = {
                'action': action_names[best_action],
                'confidence': max(q_values) - min(q_values),  # Spread indicates confidence
                'q_values': q_values
            }
        
        return policy
    
    def get_action_distribution(self):
        """Analyze what actions the policy prefers"""
        actions_count = {0: 0, 1: 0, 2: 0}
        
        for state, q_values in self.q_table.items():
            best_action = np.argmax(q_values)
            actions_count[best_action] += 1
            
        total = sum(actions_count.values())
        if total > 0:
            return {
                'scale_down': actions_count[0] / total,
                'no_change': actions_count[1] / total,
                'scale_up': actions_count[2] / total
            }
        return None
    
    def save_to_dict(self):
        """Save Q-table as dictionary for JSON serialization"""
        return {
            'q_table': {str(k): v for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'action_distribution': self.get_action_distribution()
        }
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
        
        # Initialize RL components per service
        self.rl_agents = {svc: QLearningAgent() for svc in SERVICES}
        self.last_rl_state = {svc: None for svc in SERVICES}
        self.last_rl_action = {svc: None for svc in SERVICES}
        self.last_metrics = {svc: None for svc in SERVICES}
        self.rl_metrics = {svc: [] for svc in SERVICES}
        
        # Load predictive models if available
        for svc in ['frontend', 'cartservice', 'checkoutservice']:
            model_path = f"models/{svc}_cpu_predictor.pkl"
            if os.path.exists(model_path):
                self.models[svc] = joblib.load(model_path)
                print(f"   ✓ Loaded predictive model for {svc}")
    def run(self):
        self.running = True
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.experiment_start_time
            current_users = get_current_user_load()
            
            for svc in SERVICES:
                self._process_service(svc, current_time, elapsed, current_users)
            
            time.sleep(5)  # or your chosen control interval
           
    def _calculate_reward(self, svc, prev_m, curr_m, prev_scores, curr_scores, action_taken):
        """
        Calculate reward based on state transition
        
        Rewards good outcomes:
        - Maintaining SLA with fewer pods
        - Improving latency
        - Stable CPU utilization (40-70%)
        
        Penalizes bad outcomes:
        - SLA violations
        - Over/under utilization
        - Unnecessary scaling actions
        """
        conf = SERVICE_CONFIGS[svc]
        
        # Component 1: SLA Compliance (most important)
        sla_reward = 0
        if curr_scores['latency'] <= 1.0:  # Meeting SLA
            sla_reward = 0.5
        else:  # Violating SLA
            sla_reward = -2.0 * (curr_scores['latency'] - 1.0)
        
        # Component 2: Resource Efficiency (minimize pods while meeting SLA)
        efficiency_reward = 0
        if curr_scores['latency'] <= 1.0:  # Only reward efficiency if SLA met
            pods_normalized = curr_m['pods'] / conf['max']
            efficiency_reward = (1.0 - pods_normalized) * 0.5
        
        # Component 3: CPU Utilization (sweet spot: 40-70%)
        cpu_reward = 0
        cpu_util = curr_scores['cpu']
        if 0.4 <= cpu_util <= 0.7:
            cpu_reward = 0.3
        elif cpu_util < 0.2:  # Severely under-utilized
            cpu_reward = -0.3
        elif cpu_util > 0.9:  # Severely over-utilized
            cpu_reward = -0.5
        
        # Component 4: Stability (penalize unnecessary changes)
        stability_penalty = 0
        if action_taken == 0:  # Scale down
            stability_penalty = -0.05
        elif action_taken == 2:  # Scale up
            stability_penalty = -0.1  # Slightly higher cost for scaling up
        # No penalty for staying (action_taken == 1)
        
        # Component 5: Improvement Bonus (reward if metrics improved)
        improvement_bonus = 0
        if prev_scores and curr_scores:
            lat_improved = prev_scores['latency'] - curr_scores['latency']
            if lat_improved > 0.1:  # Latency improved significantly
                improvement_bonus = 0.3
        
        # Composite reward
        total_reward = (
            sla_reward +           # Most important
            efficiency_reward +    # Save resources
            cpu_reward +          # Good utilization
            stability_penalty +   # Avoid thrashing
            improvement_bonus     # Reward improvements
        )
        
        return total_reward

    def _process_service(self, svc, current_time, elapsed, current_users):
        """
        MAPE-K Loop with Active Q-Learning
        
        Flow:
        1. Monitor current metrics
        2. Learn from previous action's outcome
        3. Choose new action using Q-learning
        4. Execute action with safety checks
        """
        
        # ============================================================
        # MONITOR: Collect Current Metrics
        # ============================================================
        m = get_metrics(svc)
        if m['pods'] == 0:
            return  # Skip if no pods running
        
        conf = SERVICE_CONFIGS[svc]
        
        # ============================================================
        # ANALYZE: Calculate Performance Scores
        # ============================================================
        
        # Use ready pods for accurate capacity (avoid death spirals)
        active_capacity = max(1, m['ready_pods'])
        
        # CPU Utilization (0-1+, where 1.0 = 100% of capacity)
        cpu_limit = conf['cpu_limit_millicores']
        score_cpu = m['cpu'] / (active_capacity * cpu_limit)
        
        # Memory Utilization
        mem_limit = conf.get('mem_limit_bytes', 128 * 1024 * 1024)
        score_mem = m['mem'] / (active_capacity * mem_limit)
        
        # Latency Score (0-2+, where 1.0 = meeting target)
        lat_target = conf.get('latency_target_seconds', 0.2)
        score_lat = m['latency'] / lat_target if lat_target > 0 else 0
        
        # Pod Ratio (0-1, current pods / max pods)
        pod_ratio = m['pods'] / conf['max']
        
        # Predictive Score (optional, if models available)
        predictive_util = 0
        if svc in self.models and current_users > 0:
            try:
                future_time = elapsed + 60  # Predict 60s ahead
                input_df = pd.DataFrame(
                    [[current_users, future_time]], 
                    columns=['scenario_users', 'elapsed_total_seconds']
                )
                predicted_cpu = self.models[svc].predict(input_df)[0]
                total_capacity = m['pods'] * cpu_limit
                predictive_util = predicted_cpu / total_capacity if total_capacity > 0 else 0
            except:
                pass  # Fail silently if prediction fails
        
        # Package current scores
        current_scores = {
            'cpu': score_cpu,
            'memory': score_mem,
            'latency': score_lat,
            'predictive': predictive_util
        }
        
        # ============================================================
        # Q-LEARNING: Learn from Previous & Choose Next Action
        # ============================================================
        
        agent = self.rl_agents[svc]
        
        # Get current state (discretized)
        current_state = agent.get_state(score_cpu, score_lat, pod_ratio)
        
        # Step 1: Learn from PREVIOUS action (if this isn't the first iteration)
        if (
             self.last_rl_state[svc] is not None and
             self.last_rl_action[svc] is not None and
             self.last_metrics[svc] is not None
            ):
            # Calculate reward for the transition
            prev_metrics = self.last_metrics[svc]['metrics']
            prev_scores = self.last_metrics[svc]['scores']
            
            reward = self._calculate_reward(
                svc, prev_metrics, m, prev_scores, current_scores, 
                self.last_rl_action[svc]
            )
            
            # Q-Learning update
            agent.learn(
                self.last_rl_state[svc], 
                self.last_rl_action[svc], 
                reward, 
                current_state
            )
            
            # Log for analysis
            self.rl_metrics[svc].append({
                'timestamp': current_time,
                'elapsed': elapsed,
                'users': current_users,
                'state': self.last_rl_state[svc],
                'action': self.last_rl_action[svc],
                'reward': reward,
                'cpu_util': score_cpu,
                'lat_score': score_lat,
                'pod_ratio': pod_ratio,
                'pods': m['pods'],
                'ready_pods': m['ready_pods'],
                'epsilon': agent.epsilon,
                'q_values': agent.q_table.get(self.last_rl_state[svc], [0,0,0]).copy()
            })
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Log learning
            action_names = ["DOWN", "STAY", "UP"]
            print(f"   [RL-Learn] {svc}: Reward={reward:+.2f} for {action_names[self.last_rl_action[svc]]}, "
                  f"ε={agent.epsilon:.3f}")
        
        # Step 2: Choose NEW action for current state
        rl_action = agent.choose_action(current_state)
        
        # Store for next iteration's learning
        self.last_rl_state[svc] = current_state
        self.last_rl_action[svc] = rl_action
        self.last_metrics[svc] = {
            'metrics': m.copy(),
            'scores': current_scores.copy()
        }
        
        # ============================================================
        # PLAN: Interpret RL Decision with Safety Checks
        # ============================================================
        
        # Initialize scaling history if needed
        if svc not in self.last_scale_time:
            self.last_scale_time[svc] = 0
            self.last_scale_action[svc] = 'none'
        
        # Safety Overrides (can override RL decision)
        safety_override = False
        original_action = rl_action
        
        # Critical Safety Check 1: Latency Emergency
        if score_lat > 2.0 and rl_action != 2:
            print(f"   ⚠️ SAFETY: Critical latency ({score_lat:.1f}x target), forcing SCALE_UP")
            rl_action = 2
            safety_override = True
        
        # Critical Safety Check 2: All pods not ready
        if m['ready_pods'] < m['pods'] * 0.5 and rl_action == 2:
            print(f"   ⚠️ SAFETY: Only {m['ready_pods']}/{m['pods']} pods ready, forcing NO_CHANGE")
            rl_action = 1
            safety_override = True
        
        # Critical Safety Check 3: At boundaries
        if m['pods'] >= conf['max'] and rl_action == 2:
            rl_action = 1  # Can't scale up beyond max
            safety_override = True
        elif m['pods'] <= conf['min'] and rl_action == 0:
            rl_action = 1  # Can't scale down below min
            safety_override = True
        
        # ============================================================
        # EXECUTE: Perform Scaling Action
        # ============================================================
        
        target = m['pods']
        action_taken = 'none'
        action_names = ["SCALE_DOWN", "NO_CHANGE", "SCALE_UP"]
        
        if rl_action == 2:  # SCALE UP
            # Anti-oscillation: Don't scale up immediately after scaling down
            if self.last_scale_action[svc] == 'down' and \
               (current_time - self.last_scale_time[svc]) < 60:
                print(f"   [RL] {svc}: UP suppressed (anti-oscillation)")
            else:
                target += conf['scale_up_increment']
                action_taken = 'up'
                self.last_scale_time[svc] = current_time
                self.last_scale_action[svc] = 'up'
                
        elif rl_action == 0:  # SCALE DOWN
            # Stabilization window (longer if we recently scaled up)
            stabilization = 300 if self.last_scale_action[svc] != 'up' else 600
            
            if current_time - self.last_scale_time[svc] > stabilization:
                target -= conf['scale_down_increment']
                action_taken = 'down'
                self.last_scale_time[svc] = current_time
                self.last_scale_action[svc] = 'down'
            else:
                print(f"   [RL] {svc}: DOWN suppressed (stabilization)")
        
        else:  # NO_CHANGE (rl_action == 1)
            action_taken = 'stay'
        
        # Apply bounds
        target = max(conf['min'], min(target, conf['max']))
        
        # Execute if there's a change
        if target != m['pods'] and action_taken != 'stay':
            # Log decision details
            q_vals = agent.q_table.get(current_state, [0, 0, 0])
            override_str = f" [SAFETY from {action_names[original_action]}]" if safety_override else ""
            
            print(f"   [RL-Active]{override_str} {svc}: {action_names[rl_action]} "
                  f"{m['pods']}→{target} pods")
            print(f"      State: CPU={score_cpu:.2f}, Lat={score_lat:.2f}x, Pods={pod_ratio:.2f}")
            print(f"      Q-values: [{q_vals[0]:.3f}, {q_vals[1]:.3f}, {q_vals[2]:.3f}]")
            
            if predictive_util > 0:
                print(f"      Predictive: {predictive_util:.2f} (60s forecast)")
            
            # Execute scaling
            scale_deployment(svc, target)
            
        elif action_taken == 'stay':
            # Log why we're not changing
            if elapsed % 60 < 15:  # Log every minute
                print(f"   [RL-Active] {svc}: NO_CHANGE (State: CPU={score_cpu:.2f}, "
                      f"Lat={score_lat:.2f}x)")

    def stop(self):
        """Stop controller and save RL data"""
        self.running = False
        
        # Save learned Q-tables and metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for svc in SERVICES:
            # Save metrics
            if self.rl_metrics[svc]:
                df = pd.DataFrame(self.rl_metrics[svc])
                filename = OUTPUT_DIR / f"rl_metrics_{svc}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"   Saved: {filename}")
            
            # Save Q-table
            agent = self.rl_agents[svc]
            q_data = agent.save_to_dict() if hasattr(agent, 'save_to_dict') else {
                'q_table': {str(k): v for k, v in agent.q_table.items()},
                'epsilon_final': agent.epsilon
            }
            
            q_file = OUTPUT_DIR / f"q_table_{svc}_{timestamp}.json"
            with open(q_file, 'w') as f:
                json.dump(q_data, f, indent=2)
            print(f"   Saved: {q_file}")
            
            # Print final policy summary
            print(f"\n   Final Policy for {svc}:")
            policy = agent.get_policy_summary() if hasattr(agent, 'get_policy_summary') else {}
            for state, info in list(policy.items())[:5]:  # Show top 5 states
                print(f"     State {state}: {info.get('action', 'N/A')}")
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
            controller.join()

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
