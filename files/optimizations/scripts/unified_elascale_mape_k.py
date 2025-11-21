#!/usr/bin/env python3
"""
EECS6446 Project - Unified Elascale MAPE-K Experiment (Level 5: RL-Enhanced)
----------------------------------------------------------------------------
1. Resets Cluster (Deletes HPA, scales to MIN replicas)
2. Runs notebook-style continuous load (50->1000->50 users)
3. Runs custom MAPE-K Loop logic:
   - Multi-Metric Analysis (CPU, Memory, Latency)
   - Predictive Scaling (Machine Learning)
   - Advanced Q-Learning (Reinforcement Learning for Self-Optimization)
   - Safety-checked execution strategy
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
from collections import defaultdict, deque
from kubernetes import client, config

# ============================================================
# CONFIGURATION
# ============================================================
PROMETHEUS_URL = "http://localhost:9090"
LOCUST_URL = "http://localhost:8089"
NAMESPACE = "default"
OUTPUT_DIR = Path("/home/common/EECS6446_project/files/optimizations/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Shared latency value for all services (in SECONDS) – warm start at target-ish
GLOBAL_P95_LATENCY = 0.20        # ~200 ms
LATENCY_EWMA_ALPHA = 0.2         # smoothing factor





# Configuration for MAPE-K logic (UPDATED with YAML limits)
SERVICE_CONFIGS = {
    "frontend": {
        "alpha": 0.4, "beta": 0.3,
        "min": 3, "max": 25,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 128 * 1024 * 1024,
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.70, "scale_down_threshold": 0.40,
        "scale_up_increment": 2, "scale_down_increment": 1
    },
    "cartservice": {
        "alpha": 0.3, "beta": 0.4,
        "min": 2, "max": 15,
        "cpu_limit_millicores": 300,
        "mem_limit_bytes": 128 * 1024 * 1024,
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.60, "scale_down_threshold": 0.30,
        "scale_up_increment": 2, "scale_down_increment": 1
    },
    "checkoutservice": {
        "alpha": 0.5, "beta": 0.3,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 128 * 1024 * 1024,
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65, "scale_down_threshold": 0.35,
        "scale_up_increment": 2, "scale_down_increment": 1
    },
    "currencyservice": {
        "alpha": 0.3, "beta": 0.5,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 128 * 1024 * 1024,
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65, "scale_down_threshold": 0.35,
        "scale_up_increment": 1, "scale_down_increment": 1
    },
    "recommendationservice": {
        "alpha": 0.6, "beta": 0.4,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 450 * 1024 * 1024,
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65, "scale_down_threshold": 0.35,
        "scale_up_increment": 1, "scale_down_increment": 1
    },
    "productcatalogservice": {
        "alpha": 0.3, "beta": 0.5,
        "min": 2, "max": 20,
        "cpu_limit_millicores": 200,
        "mem_limit_bytes": 128 * 1024 * 1024,
        "latency_target_seconds": 0.200,
        "scale_up_threshold": 0.65, "scale_down_threshold": 0.35,
        "scale_up_increment": 1, "scale_down_increment": 1
    },
}

SERVICES = list(SERVICE_CONFIGS.keys())

# Load Pattern (Can be modified for steady state)
LOAD_STEPS = [
    (50, 60), (100, 60), (500, 60),
    (1000, 1200),  # Peak
    (500, 60), (100, 60), (50, 60)
]

# ============================================================
# K8S & PROMETHEUS HELPERS
# ============================================================
try:
    config.load_kube_config()
    k8s_apps = client.AppsV1Api()
except Exception:
    print("⚠️ K8s config not found")
    k8s_apps = None

def update_global_latency_from_locust():
    """
    Update GLOBAL_P95_LATENCY using Locust stats with EMA smoothing.
    - Ignores periods with 0 users.
    - Uses current p95 if available, else ninetieth_response_time.
    - Applies exponential moving average to avoid jitter / zeros.
    """
    global GLOBAL_P95_LATENCY
    try:
        resp = requests.get(f"{LOCUST_URL}/stats/requests", timeout=5)
        data = resp.json()

        # No active load → keep previous latency estimate
        if data.get("state") != "running" or data.get("user_count", 0) == 0:
            return

        stats_list = data.get("stats", [])
        if not stats_list:
            return

        # Prefer the aggregated entry (name == "Aggregated"), else fallback
        agg = next((s for s in stats_list if s.get("name") == "Aggregated"),
                   stats_list[0])

        p95_ms = agg.get("current_response_time_percentile_95")
        if p95_ms is None:
            p95_ms = agg.get("ninetieth_response_time", 0.0)

        if not p95_ms or p95_ms <= 0:
            # nothing useful – keep last EMA value
            print("[Latency] p95 missing or non-positive, keeping previous EMA")
            return

        observed_sec = float(p95_ms) / 1000.0

        # EMA smoothing
        if GLOBAL_P95_LATENCY <= 0:
            GLOBAL_P95_LATENCY = observed_sec
        else:
            GLOBAL_P95_LATENCY = (
                LATENCY_EWMA_ALPHA * observed_sec +
                (1.0 - LATENCY_EWMA_ALPHA) * GLOBAL_P95_LATENCY
            )

        print(f"[Latency] p95≈{p95_ms:.1f} ms → EMA={GLOBAL_P95_LATENCY:.3f} s")

    except Exception as e:
        print(f"[Latency] Failed to fetch from Locust: {e}")
        # keep last good value


def get_metrics(service):
    """Get metrics with proper error handling."""
    m = {
        'cpu': 0.0,
        'mem': 0.0,
        'pods': 0,
        'ready_pods': 0,
        'latency': 0.0,   # seconds
    }

    try:
        # ---------- CPU (millicores) ----------
        q_cpu = (
            f'sum(rate(container_cpu_usage_seconds_total'
            f'{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}}[1m])) * 1000'
        )
        res = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={'query': q_cpu},
            timeout=10
        ).json()
        if res.get('data', {}).get('result'):
            m['cpu'] = float(res['data']['result'][0]['value'][1])

        # ---------- Memory (bytes) ----------
        q_mem = (
            f'sum(container_memory_usage_bytes'
            f'{{pod=~"{service}-.*",container="server",namespace="{NAMESPACE}"}})'
        )
        res = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={'query': q_mem},
            timeout=10
        ).json()
        if res.get('data', {}).get('result'):
            m['mem'] = float(res['data']['result'][0]['value'][1])

        # ---------- Latency (p95 from Locust, in seconds) ----------
        # RL uses this with: score_lat = m['latency'] / latency_target_seconds
        m['latency'] = GLOBAL_P95_LATENCY

        # ---------- Deployment status (pods / ready_pods) ----------
        if k8s_apps:
            deploy = k8s_apps.read_namespaced_deployment(service, NAMESPACE)
            m['pods'] = deploy.status.replicas or 0
            m['ready_pods'] = deploy.status.ready_replicas or 0

    except Exception as e:
        print(f"Metrics error for {service}: {e}")

    return m



def scale_deployment(service, replicas):
    """Scale deployment with min/max clamping."""
    conf = SERVICE_CONFIGS.get(service, {'min': 1, 'max': 20})
    replicas = max(conf['min'], min(replicas, conf['max']))
    try:
        if k8s_apps:
            k8s_apps.patch_namespaced_deployment_scale(
                service,
                NAMESPACE,
                {"spec": {"replicas": int(replicas)}}
            )
            print(f"   ✓ Scaled {service} to {replicas} replicas")
        return True
    except Exception as e:
        print(f"   ✗ Scale error {service}: {e}")
        return False


def get_current_user_load():
    try:
        stats = requests.get(f"{LOCUST_URL}/stats/requests", timeout=2).json()
        return stats.get('user_count', 0)
    except Exception:
        return 0

# ============================================================
# ADVANCED Q-LEARNING AGENT
# ============================================================


class QLearningAgent:
    """
    Advanced Q-Learning Agent for Kubernetes Autoscaling

    State features:
      - cpu_util (0..2 bucket)
      - cpu_trend (down / flat / up)
      - latency_score (0..2 bucket)
      - latency_trend (down / flat / up)
      - pod_ratio (0..2 bucket)
      - pod_trend (down / flat / up)
      - mem_util (0..2 bucket)

    Actions:
      0: SCALE_DOWN
      1: NO_CHANGE
      2: SCALE_UP
    """

    def __init__(
        self,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        use_replay=True,
        replay_size=200
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state (tuple) -> [Q(down), Q(stay), Q(up)]
        self.q_table = {}

        # Visit counts for exploration bonus
        self.state_visits = defaultdict(int)

        # Experience replay
        self.use_replay = use_replay
        self.replay_buffer = deque(maxlen=replay_size)

        self.total_steps = 0
        self.total_updates = 0

        self.actions = {0: "SCALE_DOWN", 1: "NO_CHANGE", 2: "SCALE_UP"}

        # Discretization thresholds
        self.cpu_thresholds = [0.3, 0.7]     # Low / Med / High
        self.lat_thresholds = [0.8, 1.2]     # <target / ~target / >target
        self.pod_thresholds = [0.3, 0.7]     # Few / Med / Many
        self.mem_thresholds = [0.4, 0.8]     # Low / Med / High

        # Trend cutoffs (per minute-ish, but we just use dimensionless)
        self.trend_neg = -0.05
        self.trend_pos = 0.05

    def discretize_value(self, value, thresholds):
        for i, thr in enumerate(thresholds):
            if value < thr:
                return i
        return len(thresholds)

    def discretize_trend(self, dv):
        """
        Map continuous trend to 3 buckets:
          0: decreasing
          1: flat
          2: increasing
        """
        if dv < self.trend_neg:
            return 0
        elif dv > self.trend_pos:
            return 2
        return 1

    def get_state(
       self,
       cpu_util,
       latency_score,
       pod_ratio,
       mem_util=None,
       cpu_trend=0.0,
       lat_trend=0.0,
       pod_trend=0.0
     ):
       cpu_state = self.discretize_value(cpu_util, self.cpu_thresholds)
       lat_state = self.discretize_value(latency_score, self.lat_thresholds)
       pod_state = self.discretize_value(pod_ratio, self.pod_thresholds)

       # Simplified state (no trends, no memory for now)
       return (cpu_state, lat_state, pod_state)

    def _init_q_values(self, state):
       # neutral initialization: mild preference for NO_CHANGE
       q_values = [0.0, 0.1, 0.0]

       cpu_state, lat_state, pod_state = state

       # mild hints only in extreme cases
       if lat_state >= 2 and cpu_state >= 1:
           q_values[2] = 0.2  # slight push toward scale up under true overload
       elif lat_state == 0 and cpu_state == 0 and pod_state == 2:
           q_values[0] = 0.2  # slight push toward scale down when very idle

       return q_values


    def choose_action(self, state, force_exploit=False):
       # Initialize new state
       if state not in self.q_table:
           self.q_table[state] = self._init_q_values(state)

       self.state_visits[state] += 1
       self.total_steps += 1

       # Pure exploitation (used by get_best_action, if needed)
       if force_exploit:
           q_values = self.q_table[state]
           max_q = max(q_values)
           best_actions = [i for i, q in enumerate(q_values) if abs(q - max_q) < 1e-6]
           return np.random.choice(best_actions)

       # Epsilon-greedy exploration
       if np.random.random() < self.epsilon:
           # slight bias to NO_CHANGE
           return np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])

       # UCB1-like bonus on top of exploitation
       q_values = self.q_table[state]
       visits = self.state_visits[state]
       # c controls exploration strength
       c = 0.5
       bonus = c * np.sqrt(np.log(self.total_steps + 1) / (visits + 1))
       adjusted = [q + bonus for q in q_values]
       max_q = max(adjusted)
       best_actions = [i for i, q in enumerate(adjusted) if abs(q - max_q) < 1e-6]
       return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = self._init_q_values(state)
        if next_state not in self.q_table:
            self.q_table[next_state] = self._init_q_values(next_state)

        if self.use_replay:
            self.replay_buffer.append((state, action, reward, next_state))

        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q
        self.total_updates += 1

        if self.use_replay and len(self.replay_buffer) >= 20:
            self._replay_learn()

    def _replay_learn(self, n_samples=8):
        samples = min(n_samples, len(self.replay_buffer))
        for _ in range(samples):
            s, a, r, ns = self.replay_buffer[np.random.randint(len(self.replay_buffer))]
            old_q = self.q_table[s][a]
            next_max = max(self.q_table[ns])
            new_q = old_q + self.alpha * (r + self.gamma * next_max - old_q)
            self.q_table[s][a] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_best_action(self, state):
        return self.choose_action(state, force_exploit=True)

    def get_policy(self):
        policy = {}
        for state, q_values in self.q_table.items():
            a = int(np.argmax(q_values))
            policy[state] = {
                "action": self.actions[a],
                "q_values": q_values,
                "visits": self.state_visits[state],
            }
        return policy

    def get_statistics(self):
        return {
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
            "states_discovered": len(self.q_table),
            "epsilon_current": self.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
        }

    def save(self, filepath):
        data = {
            "q_table": {str(k): v for k, v in self.q_table.items()},
            "state_visits": {str(k): v for k, v in self.state_visits.items()},
            "parameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
            },
            "statistics": self.get_statistics(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def visualize_policy(self):
        lines = ["=== Q-Learning Policy Summary ==="]
        for state, q_values in sorted(self.q_table.items(), key=lambda x: self.state_visits[x[0]], reverse=True)[:20]:
            a = int(np.argmax(q_values))
            lines.append(
                f"State={state}, best={self.actions[a]}, "
                f"Q=[{q_values[0]:.2f},{q_values[1]:.2f},{q_values[2]:.2f}], "
                f"visits={self.state_visits[state]}"
            )
        lines.append(str(self.get_statistics()))
        return "\n".join(lines)

# ============================================================
# MAPE-K CONTROLLER WITH RL
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
        self.rl_agents = {
            svc: QLearningAgent(
                alpha=0.1,
                gamma=0.9,
                epsilon=0.3,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                use_replay=True,
                replay_size=100
            )
            for svc in SERVICES
        }
        self.last_rl_state = {svc: None for svc in SERVICES}
        self.last_rl_action = {svc: None for svc in SERVICES}
        self.last_metrics = {svc: None for svc in SERVICES}
        self.rl_metrics = {svc: [] for svc in SERVICES}
        self.last_executed_action = {svc: None for svc in SERVICES}

        # Load predictive models if available
        for svc in ['frontend', 'cartservice', 'checkoutservice']:
            model_path = f"models/{svc}_cpu_predictor.pkl"
            if os.path.exists(model_path):
                self.models[svc] = joblib.load(model_path)
                print(f"   ✓ Loaded predictive model for {svc}")

    def run(self):
        print("\n>>> CAPA+ Controller Started (Level 5: Advanced RL) <<<")
        self.running = True
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.experiment_start_time
            update_global_latency_from_locust()
            current_users = get_current_user_load()

            for svc in SERVICES:
                try:
                    self._process_service(svc, current_time, elapsed, current_users)
                except Exception as e:
                    print(f"Error processing {svc}: {e}")

            # Faster shutdown checks with ~15s control period
            for _ in range(15):
                if not self.running:
                    break
                time.sleep(1)

    def _calculate_reward(self, svc, prev_m, curr_m, prev_scores, curr_scores, action_taken):
       """
       Improved, numerically-stable reward function for Kubernetes RL autoscaling.

       Goals:
         + Reward SLA adherence
         + Reward efficiency (fewer pods when SLA excellent)
         + Reward stable CPU in target zone
         + Reward readiness (only when SLA good)
         + Reward improvement relative to previous step

         - Penalize SLA violations
         - Penalize over/under CPU usage
         - Penalize memory pressure
         - Penalize scaling in unstable readiness conditions
         - Penalize unnecessary scaling
       """

       conf = SERVICE_CONFIGS[svc]
       lat = curr_scores["latency"]     # ratio vs target (0.0–5.0 typically)
       cpu = curr_scores["cpu"]         # 0.0–2.0 range
       mem = curr_scores["memory"]      # 0.0–2.0 range

       pods = curr_m["pods"]
       ready = curr_m["ready_pods"]
       ready_ratio = ready / max(1, pods)

       reward = 0.0

       # --------------------------------------------------------------------
       # (1) SLA / Latency reward (normalized)
       # --------------------------------------------------------------------
       # Map:
       #   lat <= 1.0 → positive reward up to +0.6
       #   lat > 1.0 → negative up to -1.0
       if lat <= 1.0:
           reward += 0.6 * (1.0 - lat)          # better when much below target
       else:
           reward -= min(1.0, (lat - 1.0))       # strong penalty for violation

       # --------------------------------------------------------------------
       # (2) CPU efficiency reward
       # Target zone = [40%, 70%]
       # --------------------------------------------------------------------
       if 0.4 <= cpu <= 0.7:
           reward += 0.4
       else:
           # Penalize under/over utilization smoothly
           reward -= 0.3 * abs(cpu - 0.55)

       # --------------------------------------------------------------------
       # (3) Memory pressure
       # --------------------------------------------------------------------
       if mem > 0.9:
           reward -= 0.6
       elif mem > 0.75:
           reward -= 0.3

       # --------------------------------------------------------------------
       # (4) Readiness reward (only helps when latency is good)
       # --------------------------------------------------------------------
       if lat < 1.0:
           reward += 0.4 * ready_ratio
       else:
           reward -= 0.2 * (1 - ready_ratio)

       # Scale-up during unready period = very bad
       if ready_ratio < 0.5 and action_taken == 2:
           reward -= 0.4

       # --------------------------------------------------------------------
       # (5) Penalize unnecessary scaling (larger penalty)
       # --------------------------------------------------------------------
       if action_taken == 2:     # SCALE UP
           reward -= 0.2
       elif action_taken == 0:   # SCALE DOWN
           reward -= 0.1
       
       if lat > 1.2 and action_taken == 2:
          reward += 0.4   
       # (6) Reward improvement from previous step
       # --------------------------------------------------------------------
       if prev_scores is not None:
           prev_lat = prev_scores["latency"]
           delta_lat = prev_lat - lat

           if delta_lat > 0.1:
               reward += 0.3
           elif delta_lat < -0.1:
               reward -= 0.3
       # --------------------------------------------------------------------
        # (7) Efficiency: fewer pods when SLA is excellent
        # --------------------------------------------------------------------
        # Define how "overprovisioned" we are:
        #   eff = 0 when pods == min
        #   eff → 1 as pods → max
       if conf["max"] > conf["min"]:
            eff = (pods - conf["min"]) / float(conf["max"] - conf["min"])
            eff = max(0.0, min(1.0, eff))
       else:
            eff = 0.0
       # --------------------------------------------------------------------
       # (7) Efficiency reward: using fewer pods when SLA is excellent
       # --------------------------------------------------------------------
       if lat < 0.8 and pods > conf["min"]:
           # Reward proportional to unused capacity
           reward += 0.3 * (1.0 - lat) * (1.0 - eff)

       return reward

    def _process_service(self, svc, current_time, elapsed, current_users):
       """
       MAPE-K Loop with active Q-Learning.

       1. Monitor metrics
       2. Learn from previous action (using EXECUTED action)
       3. Choose new action
       4. Apply safety checks and scale
       """
       # ----------------- MONITOR -----------------
       m = get_metrics(svc)
       if m['pods'] == 0:
           return

       conf = SERVICE_CONFIGS[svc]

       active_capacity = max(1, m['ready_pods'])

       # CPU util (0-1+)
       cpu_limit = conf['cpu_limit_millicores']
       score_cpu = m['cpu'] / (active_capacity * cpu_limit) if cpu_limit > 0 else 0.0

       # Memory util (0-1+)
       mem_limit = conf.get('mem_limit_bytes', 128 * 1024 * 1024)
       score_mem = m['mem'] / (active_capacity * mem_limit) if mem_limit > 0 else 0.0

       # Latency score (ratio vs target)
       lat_target = conf.get('latency_target_seconds', 0.2)
       score_lat = m['latency'] / lat_target if lat_target > 0 else 0.0

       # Pod ratio
       pod_ratio = m['pods'] / conf['max'] if conf['max'] > 0 else 0.0

       # Clamp to sane ranges
       score_cpu = float(max(0.0, min(score_cpu, 2.0)))
       score_lat = float(max(0.0, min(score_lat, 5.0)))
       score_mem = float(max(0.0, min(score_mem, 2.0)))
       pod_ratio = float(max(0.0, min(pod_ratio, 1.0)))

       # Predictive (unchanged, optional)
       predictive_util = 0.0
       if svc in self.models and current_users > 0:
           try:
               future_time = elapsed + 60
               input_df = pd.DataFrame(
                   [[current_users, future_time]],
                   columns=['scenario_users', 'elapsed_total_seconds']
               )
               predicted_cpu = self.models[svc].predict(input_df)[0]
               total_capacity = m['pods'] * cpu_limit
               predictive_util = predicted_cpu / total_capacity if total_capacity > 0 else 0.0
           except Exception:
               predictive_util = 0.0

       current_scores = {
           'cpu': score_cpu,
           'memory': score_mem,
           'latency': score_lat,
           'predictive': predictive_util,
       }

       agent = self.rl_agents[svc]

       # ----------------- BUILD CURRENT STATE -----------------
       # (Assuming get_state now ignores trends or takes them as kwargs)
       current_state = agent.get_state(
           cpu_util=score_cpu,
           latency_score=score_lat,
           pod_ratio=pod_ratio,
           mem_util=score_mem,
       )

       # ----------------- LEARN FROM PREVIOUS TRANSITION -----------------
       if (
           self.last_rl_state[svc] is not None and
           self.last_executed_action[svc] is not None and
           self.last_metrics[svc] is not None
       ):
           prev_metrics = self.last_metrics[svc]['metrics']
           prev_scores = self.last_metrics[svc]['scores']
           prev_action_idx = self.last_executed_action[svc]  # 0/1/2

           reward = self._calculate_reward(
               svc,
               prev_metrics,
               m,
               prev_scores,
               current_scores,
               prev_action_idx
           )

           agent.learn(
               self.last_rl_state[svc],
               prev_action_idx,
               reward,
               current_state
           )

           self.rl_metrics[svc].append({
               'timestamp': current_time,
               'elapsed': elapsed,
               'users': current_users,
               'state': self.last_rl_state[svc],
               'action': prev_action_idx,   # executed action
               'reward': reward,
               'cpu_util': score_cpu,
               'lat_score': score_lat,
               'pod_ratio': pod_ratio,
               'pods': m['pods'],
               'ready_pods': m['ready_pods'],
               'epsilon': agent.epsilon,
               'q_values': agent.q_table.get(
                   self.last_rl_state[svc],
                   [0.0, 0.0, 0.0]
               ).copy()
           })

           agent.decay_epsilon()

           act_names = ["DOWN", "STAY", "UP"]
           print(
               f"   [RL-Learn] {svc}: Reward={reward:+.2f} "
               f"for {act_names[prev_action_idx]}, "
               f"ε={agent.epsilon:.3f}"
           )

       # ----------------- CHOOSE NEW ACTION -----------------
       rl_action = agent.choose_action(current_state)  # 0/1/2
       action_names = ["SCALE_DOWN", "NO_CHANGE", "SCALE_UP"]

       # ----------------- PLAN (SAFETY) -----------------
       if svc not in self.last_scale_time:
           self.last_scale_time[svc] = 0
           self.last_scale_action[svc] = 'none'

       safety_override = False
       original_action = rl_action

       # Safety: critical latency - enforce SCALE_UP if not already
       if score_lat > 2.0 and rl_action != 2:
           print(
               f"   ⚠️ SAFETY: Critical latency ({score_lat:.1f}x target), "
               f"forcing SCALE_UP"
           )
           rl_action = 2
           safety_override = True

       # Safety: not enough ready pods to safely scale up (unless extreme latency)
       ready_ratio = m['ready_pods'] / max(1, m['pods'])
       if rl_action == 2 and ready_ratio < 0.4 and score_lat < 3.0:
           print(
               f"   ⚠️ SAFETY: Only {m['ready_pods']}/{m['pods']} pods ready "
               f"and latency not extreme, forcing NO_CHANGE"
           )
           rl_action = 1
           safety_override = True

       # Safety: boundaries
       if m['pods'] >= conf['max'] and rl_action == 2:
           rl_action = 1
           safety_override = True
       elif m['pods'] <= conf['min'] and rl_action == 0:
           rl_action = 1
           safety_override = True

       # ----------------- EXECUTE -----------------
       target = m['pods']
       action_taken = 'none'   # 'up' / 'down' / 'stay'

       if rl_action == 2:  # SCALE UP
           if (
               self.last_scale_action[svc] == 'down' and
               (current_time - self.last_scale_time[svc]) < 60
           ):
               print(f"   [RL] {svc}: UP suppressed (anti-oscillation)")
               action_taken = 'stay'
           else:
               target += conf['scale_up_increment']
               action_taken = 'up'
               self.last_scale_time[svc] = current_time
               self.last_scale_action[svc] = 'up'

       elif rl_action == 0:  # SCALE DOWN
           stabilization = 300 if self.last_scale_action[svc] != 'up' else 600
           if current_time - self.last_scale_time[svc] > stabilization:
               target -= conf['scale_down_increment']
               action_taken = 'down'
               self.last_scale_time[svc] = current_time
               self.last_scale_action[svc] = 'down'
           else:
               print(f"   [RL] {svc}: DOWN suppressed (stabilization)")
               action_taken = 'stay'
       else:
           action_taken = 'stay'

       # ---- Map textual action_taken to index 0/1/2 for the RL update ----
       if action_taken == 'up':
           executed_action_idx = 2       # SCALE_UP
       elif action_taken == 'down':
           executed_action_idx = 0       # SCALE_DOWN
       else:
           executed_action_idx = 1       # NO_CHANGE ('stay' or suppressed)

       self.last_executed_action[svc] = executed_action_idx

       target = max(conf['min'], min(target, conf['max']))

       if target != m['pods'] and action_taken != 'stay':
           q_vals = agent.q_table.get(current_state, [0.0, 0.0, 0.0])
           override_str = (
               f" [SAFETY from {action_names[original_action]}]"
               if safety_override else ""
           )

           print(
               f"   [RL-Active]{override_str} {svc}: {action_names[rl_action]} "
               f"{m['pods']}→{target} pods"
           )
           print(
               f"      State: CPU={score_cpu:.2f}, "
               f"Lat={score_lat:.2f}x, Pods={pod_ratio:.2f}"
           )
           print(
               "      Q-values: "
               f"[{q_vals[0]:.3f}, {q_vals[1]:.3f}, {q_vals[2]:.3f}]"
           )

           if predictive_util > 0:
               print(f"      Predictive: {predictive_util:.2f} (60s forecast)")

           scale_deployment(svc, target)

       elif action_taken == 'stay':
           if elapsed % 60 < 15:
               print(
                   f"   [RL-Active] {svc}: NO_CHANGE "
                   f"(State: CPU={score_cpu:.2f}, Lat={score_lat:.2f}x)"
               )

       # ----------------- SAVE STATE FOR NEXT ITERATION -----------------
       self.last_rl_state[svc] = current_state
       self.last_rl_action[svc] = rl_action
       self.last_metrics[svc] = {
           'metrics': m.copy(),
           'scores': current_scores.copy(),
           'elapsed': elapsed
       }


    def stop(self):
        """Stop controller and save RL data."""
        self.running = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for svc in SERVICES:
            # Metrics
            if self.rl_metrics[svc]:
                df = pd.DataFrame(self.rl_metrics[svc])
                filename = OUTPUT_DIR / f"rl_metrics_{svc}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"   Saved: {filename}")

            # Q-table & policy
            agent = self.rl_agents[svc]
            q_file = OUTPUT_DIR / f"q_table_{svc}_{timestamp}.json"
            agent.save(str(q_file))
            print(f"   Saved: {q_file}")

            print(f"\n   Final Policy for {svc}:")
            print(agent.visualize_policy())

# ============================================================
# CLUSTER RESET LOGIC
# ============================================================


def reset_cluster():
    print("\n>>> Resetting Cluster State...")
    try:
        subprocess.run(
            f"kubectl delete hpa --all --namespace={NAMESPACE}",
            shell=True,
            check=False
        )
        for svc, conf in SERVICE_CONFIGS.items():
            min_replicas = conf.get("min", 1)
            subprocess.run(
                f"kubectl scale deployment/{svc} "
                f"--replicas={min_replicas} --namespace={NAMESPACE}",
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL
            )

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
    try:
        requests.get(f"{LOCUST_URL}/stats/reset", timeout=5)
    except Exception:
        pass

    rows = []
    start_time = time.time()

    try:
        for users, duration in LOAD_STEPS:
            print(f"   -> Step: {users} users for {duration}s")
            try:
                requests.post(
                    f"{LOCUST_URL}/swarm",
                    data={"user_count": users, "spawn_rate": 20},
                    timeout=5
                )
            except Exception as e:
                print(f"Locust swarm error: {e}")

            end_step = time.time() + duration
            while time.time() < end_step:
                now = time.time()

                row = {
                    "timestamp": datetime.now().isoformat(),
                    "config": config_name,
                    "elapsed_total_seconds": now - start_time,
                    "scenario_users": users,
                }

                # Locust metrics
                try:
                    stats = requests.get(
                        f"{LOCUST_URL}/stats/requests",
                        timeout=5
                    ).json()
                    row["throughput_rps"] = stats.get('total_rps', 0)
                    row["fault_rate_percent"] = stats.get('fail_ratio', 0) * 100

                    avg_resp = 0
                    # 1) Use the correct top-level key
                    if 'total_average_response_time' in stats:
                       avg_resp = stats['total_average_response_time']
                    else:
                       # 2) Fallback to the "Aggregated" row
                       stats_list = stats.get('stats', [])
                       if isinstance(stats_list, list):
                          for entry in stats_list:
                              if isinstance(entry, dict) and entry.get('name') in ('Aggregated', 'Total'):
                                  avg_resp = entry.get('avg_response_time', 0)
                                  break
   
                    row["avg_response_time_ms"] = avg_resp
                    row["p95_response_time_ms"] = stats.get(
                        'current_response_time_percentile_95', 0
                    )
                except Exception:
                    row["throughput_rps"] = 0
                    row["fault_rate_percent"] = 0
                    row["avg_response_time_ms"] = 0
                    row["p95_response_time_ms"] = 0

                # Per-service metrics
                for svc in SERVICES:
                    m = get_metrics(svc)
                    row[f"{svc}_cpu_millicores"] = m['cpu']
                    row[f"{svc}_memory_bytes"] = m['mem']
                    row[f"{svc}_replicas_ordered"] = m['pods']
                    row[f"{svc}_replicas_ready"] = m['ready_pods']

                    limit = SERVICE_CONFIGS[svc]['cpu_limit_millicores']
                    ready_count = max(1, m['ready_pods'])
                    if m['pods'] > 0 and limit > 0:
                        row[f"{svc}_cpu_percent"] = (
                            m['cpu'] / (ready_count * limit)
                        ) * 100
                    else:
                        row[f"{svc}_cpu_percent"] = 0

                rows.append(row)
                time.sleep(5)

    finally:
        try:
            requests.get(f"{LOCUST_URL}/stop", timeout=5)
        except Exception:
            pass

        if controller:
            controller.stop()
            controller.join()

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = OUTPUT_DIR / f"{config_name}_complete_{ts}.csv"
    df.to_csv(filename, index=False)
    print(f"   -> Saved: {filename}")
    return filename

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Starting Unified Experiment...")
    run_experiment_phase("baseline")
    print("\nCooldown 60s...")
    time.sleep(60)
    run_experiment_phase("elascale")
    print("\nDONE. Now run: python3 generate_unified_comparison.py")
