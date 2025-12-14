#!/usr/bin/env python3
"""
unified_experiment_v3_clean.py
========================================================
CAPA+ Unified Experiment Framework v3 (Clean + Fixed)

Design goals:
- Runner produces raw, well-aligned data (NO statistical claims in runner).
- Deterministic paired workloads for baseline vs CAPA.
- Fixes critical issues discovered in v3_clean:
  1) Thrashing penalty works (prev_prev_action tracked properly)
  2) No EMA/cache leakage between runs (collector.reset() per run)
  3) Baseline cooldown/state independent of CAPA
  4) No Python hash() randomness for seeds (hashlib-based)
  5) RL updates use correct (prev_state, prev_action) -> next_state
  6) Windows signal safety for SIGTERM
  7) Epsilon decays once per GLOBAL tick (not per service)
  8) Locust aggregate-latency limitation documented in logs and CSV

Outputs:
- results/<exp_name>/baseline_run_<ts>.csv
- results/<exp_name>/capa_run_<ts>.csv
- results/<exp_name>/run_meta_<ts>.json
- Optional checkpoints/<...> for RL tables

Usage:
  python unified_experiment_v3_clean.py diagnose
  python unified_experiment_v3_clean.py run --mode baseline --patterns warmup,step --duration 1800
  python unified_experiment_v3_clean.py run --mode capa --patterns warmup,step --duration 1800
  python unified_experiment_v3_clean.py paired --patterns warmup,step --duration 1800 --reset-replicas

Notes:
- Locust stats endpoint provides aggregate latency by default. If you need per-service latency,
  integrate Prometheus/OTel/Envoy histograms and set Config.latency_source="prometheus".
"""

import os
import sys
import time
import json
import math
import random
import signal
import argparse
import logging
import subprocess
import hashlib
import ast
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("capa_unified_v3.log")]
)
logger = logging.getLogger("CAPA+v3_clean")


try:
    import requests
except ImportError:
    logger.error(
        "Missing required dependency: requests\n"
        "Install with: pip install requests\n"
        "Or use: pip install -r requirements.txt"
    )
    sys.exit(1)

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ServiceConfig:
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 50
    target_latency_ms: float = 200.0
    capacity_per_pod: float = 100.0

@dataclass
class Config:
    prometheus_url: str = "http://localhost:9090"
    locust_url: str = "http://localhost:8089"
    namespace: str = "default"

    # If locust only: latency is global aggregate (same for all services)
    latency_source: str = "locust"  # "locust" | "prometheus" (stubbed)

    services: Dict[str, ServiceConfig] = field(default_factory=lambda: {
        "frontend": ServiceConfig(1, 10, 50, 200, 100),
        "recommendationservice": ServiceConfig(1, 5, 50, 100, 150),
        "productcatalogservice": ServiceConfig(1, 5, 50, 100, 200),
        "cartservice": ServiceConfig(1, 5, 50, 100, 100),
        "checkoutservice": ServiceConfig(1, 5, 50, 150, 80),
    })

    # Control loop
    control_interval_sec: float = 15.0
    cooldown_sec: float = 60.0

    # EMA smoothing
    smoothing_alpha: float = 0.3

    # Robustness: delayed reward settling window
    scaling_settle_timeout_sec: float = 180.0
    readiness_tolerance: int = 0

    # Trends thresholds (for discretization)
    cpu_trend_threshold: float = 0.05
    latency_trend_threshold: float = 0.10

    # RL
    learning_rate: float = 0.10
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.10
    epsilon_decay_per_tick: float = 0.9995  # GLOBAL decay per tick (fixed)

    # Workload randomization (deterministic)
    workload_scale_min: float = 0.8
    workload_scale_max: float = 1.2
    workload_noise_sigma: float = 15.0

    # Results
    results_dir: str = "./results"
    exp_name: str = "capa_experiment"


CONFIG = Config()

LOAD_PATTERNS = {
    "warmup": [50, 100, 150, 100, 50],
    "step":   [50, 100, 300, 500, 600, 500, 300, 100],
    "gradual":[50, 150, 250, 350, 450, 550, 450, 250],
    "sine":   [300, 477, 550, 477, 300, 123, 50, 123],
    "spike":  [50, 50, 800, 100, 50, 700, 50, 50],
    "stress": [100, 300, 600, 800, 600, 300, 100, 50],
}


# =============================================================================
# ENUMS / DATA
# =============================================================================

class ControlMode(str, Enum):
    BASELINE = "baseline"
    CAPA = "capa"

class ScalingAction(int, Enum):
    SCALE_DOWN = 0
    STAY = 1
    SCALE_UP = 2

@dataclass
class ServiceMetrics:
    timestamp: float
    service: str
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0
    arrival_rate_rps: float = 0.0
    failure_rate: float = 0.0
    cpu_utilization: float = 0.0
    ready_replicas: int = 1
    current_replicas: int = 1
    desired_replicas: int = 1
    latency_scope: str = "unknown"  # "global" or "per_service"
    latency_source: str = "unknown" # "locust" or "prometheus"
    resource_source: str = "unknown" # "prometheus" or "kubectl"

@dataclass
class PendingScale:
    start_time: float
    prev_state: Tuple[int, int, int, int, int]
    prev_action: ScalingAction
    last_action_before: Optional[ScalingAction]  # Was: prev_prev_action (a_{t-1} for thrash check)
    target_replicas: int

@dataclass
class TickRecord:
    # minimal record for robust analysis alignment
    run_id: str
    mode: str
    seed: int
    pattern: str
    users: int
    step_idx: int
    tick: int
    timestamp: float
    service: str
    # metrics
    latency_p95_ms: float
    latency_avg_ms: float
    arrival_rate_rps: float
    cpu_utilization: float
    failure_rate: float
    ready_replicas: int
    desired_replicas: int
    # decisions
    action: str
    baseline_action: str
    rl_action: str
    decision_source: str
    reward: float
    # state
    state: str
    lat_ratio: float
    pod_ratio: float
    cpu_trend: float
    lat_trend: float
    executed: bool = False  # Add this field
    notes: str = ""


# =============================================================================
# METRICS COLLECTORS
# =============================================================================

class LocustClient:
    def __init__(self, url: str):
        self.url = url
        self._cache = {}
        self._cache_ts = 0.0
        self._ttl = 2.0
        self.log = logging.getLogger("Locust")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stats/requests", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> Optional[Dict[str, Any]]:
        if (time.time() - self._cache_ts) < self._ttl:
            return self._cache
        try:
            r = requests.get(f"{self.url}/stats/requests", timeout=5)
            if r.status_code == 200:
                self._cache = r.json()
                self._cache_ts = time.time()
                return self._cache
        except Exception as e:
            self.log.warning(f"Locust stats failed: {e}")
        return None

    def set_users(self, count: int, spawn_rate: Optional[int] = None) -> bool:
        if spawn_rate is None:
            spawn_rate = max(1, count // 10)
        try:
            r = requests.post(
                f"{self.url}/swarm",
                data={"user_count": count, "spawn_rate": spawn_rate},
                timeout=5
            )
            return r.status_code == 200
        except Exception:
            return False

    def stop(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stop", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
    
    def reset_cache(self):
        """Clear cached stats to prevent leak between experiment runs."""
        self._cache = {}
        self._cache_ts = 0.0


class PrometheusClient:
    def __init__(self, url: str, namespace: str):
        self.url = url
        self.ns = namespace
        self.log = logging.getLogger("Prometheus")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/v1/status/config", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def query(self, promql: str) -> Optional[float]:
        try:
            r = requests.get(f"{self.url}/api/v1/query", params={"query": promql}, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            if data.get("status") != "success":
                return None
            result = data.get("data", {}).get("result", [])
            if not result:
                return None
            vals = []
            for item in result:
                try:
                    vals.append(float(item["value"][1]))
                except Exception:
                    continue
            return float(np.mean(vals)) if vals else None
        except Exception:
            return None

    def get_cpu_utilization(self, deployment: str) -> Optional[float]:
        # CPU usage / CPU requests
        q = (
            f'sum(rate(container_cpu_usage_seconds_total{{namespace="{self.ns}", pod=~"{deployment}.*", container!=""}}[2m]))'
            f' / sum(kube_pod_container_resource_requests{{namespace="{self.ns}", pod=~"{deployment}.*", resource="cpu"}})'
        )
        v = self.query(q)
        if v is None:
            return None
        return min(max(v, 0.0), 2.0)

    def get_pod_counts(self, deployment: str) -> Tuple[int, int, int]:
        ready = self.query(f'kube_deployment_status_replicas_ready{{namespace="{self.ns}",deployment="{deployment}"}}')
        cur = self.query(f'kube_deployment_status_replicas{{namespace="{self.ns}",deployment="{deployment}"}}')
        des = self.query(f'kube_deployment_spec_replicas{{namespace="{self.ns}",deployment="{deployment}"}}')
        r = int(ready) if ready is not None else 1
        c = int(cur) if cur is not None else r
        d = int(des) if des is not None else c
        return r, c, d


class KubectlClient:
    def __init__(self, namespace: str):
        self.ns = namespace

    def get_pod_counts(self, deployment: str) -> Tuple[int, int, int]:
        try:
            p = subprocess.run(
                ["kubectl", "get", "deployment", deployment, "-n", self.ns,
                 "-o", "jsonpath={.status.readyReplicas},{.status.replicas},{.spec.replicas}"],
                capture_output=True, text=True, timeout=10
            )
            parts = p.stdout.strip().split(",")
            ready = int(parts[0]) if parts and parts[0] else 1
            cur = int(parts[1]) if len(parts) > 1 and parts[1] else ready
            des = int(parts[2]) if len(parts) > 2 and parts[2] else cur
            return ready, cur, des
        except Exception:
            return 1, 1, 1

    def scale_deployment(self, deployment: str, replicas: int) -> bool:
        try:
            p = subprocess.run(
                ["kubectl", "scale", "deployment", deployment, "-n", self.ns, f"--replicas={replicas}"],
                capture_output=True, text=True, timeout=30
            )
            return p.returncode == 0
        except Exception:
            return False


class UnifiedMetricsCollector:
    """
    Collects:
    - latency (usually global aggregate from Locust)
    - arrival rate/fail rate from Locust
    - CPU + replicas from Prometheus (fallback to kubectl)
    Applies EMA smoothing per-service (must be reset per run).
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.locust = LocustClient(cfg.locust_url)
        self.prom = PrometheusClient(cfg.prometheus_url, cfg.namespace)
        self.kubectl = KubectlClient(cfg.namespace)
        self.locust_ok = self.locust.is_available()
        self.prom_ok = self.prom.is_available()
        self._smoothed: Dict[str, ServiceMetrics] = {}
        self.log = logging.getLogger("Metrics")

        if self.locust_ok:
            self.log.info("Locust available: latency_scope=global (unless per-service metrics implemented).")
        else:
            self.log.warning("Locust not available: latency fields may be 0.")

        self.log.info(f"Prometheus available={self.prom_ok}")

    def reset(self):
        """Reset all cached state for a fresh run."""
        self._smoothed.clear()
        self.locust.reset_cache()  # ADD THIS LINE

    def _ema(self, service: str, m: ServiceMetrics) -> ServiceMetrics:
        if service not in self._smoothed:
            self._smoothed[service] = m
            return m

        alpha = self.cfg.smoothing_alpha
        prev = self._smoothed[service]

        def blend(field: str):
            old = float(getattr(prev, field, 0.0) or 0.0)
            new = float(getattr(m, field, 0.0) or 0.0)
            # only blend if new is present
            if new > 0:
                setattr(m, field, alpha * new + (1 - alpha) * old)

        for f in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_avg_ms",
                  "arrival_rate_rps", "failure_rate", "cpu_utilization"]:
            blend(f)

        self._smoothed[service] = m
        return m

    def collect(self, service: str) -> ServiceMetrics:
        now = time.time()
        m = ServiceMetrics(timestamp=now, service=service)

        # 1) Locust (aggregate)
        if self.locust_ok:
            st = self.locust.get_stats()
            if st:
                # aggregate latency percentiles
                m.latency_p50_ms = float(st.get("current_response_time_percentile_50", 0) or 0)
                m.latency_p95_ms = float(st.get("current_response_time_percentile_95", 0) or 0)
                # crude p99 estimate if not available
                m.latency_p99_ms = float(st.get("current_response_time_percentile_99", 0) or 0)
                if m.latency_p99_ms <= 0 and m.latency_p95_ms > 0:
                    m.latency_p99_ms = 1.30 * m.latency_p95_ms

                # avg_response_time often available per endpoint; try aggregated/total
                avg_ms = 0.0
                stats_list = st.get("stats", [])
                if isinstance(stats_list, list):
                    for e in stats_list:
                        if e.get("name") in ("Aggregated", "Total"):
                            avg_ms = float(e.get("avg_response_time", 0) or 0)
                            break
                m.latency_avg_ms = avg_ms
                m.arrival_rate_rps = float(st.get("total_rps", 0) or 0)
                m.failure_rate = float(st.get("fail_ratio", 0) or 0)
                m.latency_scope = "global"
                m.latency_source = "locust"

        # 2) Prometheus or kubectl
        if self.prom_ok:
            cpu = self.prom.get_cpu_utilization(service)
            if cpu is not None:
                m.cpu_utilization = float(cpu)
            r, c, d = self.prom.get_pod_counts(service)
            m.ready_replicas, m.current_replicas, m.desired_replicas = r, c, d
            m.resource_source = "prometheus"
        else:
            r, c, d = self.kubectl.get_pod_counts(service)
            m.ready_replicas, m.current_replicas, m.desired_replicas = r, c, d
            m.resource_source = "kubectl"

        return self._ema(service, m)


# =============================================================================
# BASELINE CONTROLLER
# =============================================================================

class BaselineHPA:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def decide(self, service: str, cpu_util: float, ready: int) -> ScalingAction:
        sc = self.cfg.services.get(service)
        if not sc:
            return ScalingAction.STAY
        target = sc.target_cpu_percent / 100.0
        up_th = target * 1.4
        down_th = target * 0.6

        if cpu_util > up_th and ready < sc.max_replicas:
            return ScalingAction.SCALE_UP
        if cpu_util < down_th and ready > sc.min_replicas:
            return ScalingAction.SCALE_DOWN
        return ScalingAction.STAY


# =============================================================================
# RL AGENT (Double Q-learning, 243 states)
# =============================================================================

class DoubleQAgent:
    def __init__(self, service: str, cfg: Config):
        self.service = service
        self.cfg = cfg
        self.n_actions = 3
        self.qA = defaultdict(lambda: [0.0] * self.n_actions)
        self.qB = defaultdict(lambda: [0.0] * self.n_actions)
        self.useA = True
        self.alpha = cfg.learning_rate
        self.gamma = cfg.discount_factor
        self.eps = cfg.epsilon_start

        self.state_visits = defaultdict(int)
        self.action_counts = defaultdict(int)

    def discretize(self, cpu: float, lat_ratio: float, pod_ratio: float,
                   cpu_trend: float, lat_trend: float) -> Tuple[int, int, int, int, int]:
        # CPU
        if cpu < 0.4: cpu_l = 0
        elif cpu < 0.7: cpu_l = 1
        else: cpu_l = 2

        # Latency ratio
        if lat_ratio <= 0.8: lat_l = 0
        elif lat_ratio <= 1.1: lat_l = 1
        else: lat_l = 2

        # Pods ratio
        if pod_ratio < 0.3: pod_l = 0
        elif pod_ratio < 0.7: pod_l = 1
        else: pod_l = 2

        # Trends
        ct = self.cfg.cpu_trend_threshold
        if cpu_trend < -ct: cpu_t = 0
        elif cpu_trend > ct: cpu_t = 2
        else: cpu_t = 1

        lt = self.cfg.latency_trend_threshold
        if lat_trend < -lt: lat_t = 0
        elif lat_trend > lt: lat_t = 2
        else: lat_t = 1

        return (cpu_l, lat_l, pod_l, cpu_t, lat_t)

    def choose(self, state: Tuple[int, int, int, int, int], training: bool) -> ScalingAction:
        self.state_visits[state] += 1
        if training and random.random() < self.eps:
            # smart exploration: if SLA violated, bias toward UP
            if state[1] == 2 and random.random() < 0.5:
                a = ScalingAction.SCALE_UP
            else:
                a = ScalingAction(random.randint(0, self.n_actions - 1))
        else:
            q = [(a + b) / 2 for a, b in zip(self.qA[state], self.qB[state])]
            a = ScalingAction(int(np.argmax(q)))
        self.action_counts[int(a)] += 1
        return a

    def update(self, prev_state: Tuple[int, int, int, int, int], prev_action: ScalingAction,
               reward: float, next_state: Tuple[int, int, int, int, int]) -> None:
        # Double Q update
        if self.useA:
            best_next = int(np.argmax(self.qA[next_state]))
            target = reward + self.gamma * self.qB[next_state][best_next]
            old = self.qA[prev_state][int(prev_action)]
            self.qA[prev_state][int(prev_action)] = old + self.alpha * (target - old)
        else:
            best_next = int(np.argmax(self.qB[next_state]))
            target = reward + self.gamma * self.qA[next_state][best_next]
            old = self.qB[prev_state][int(prev_action)]
            self.qB[prev_state][int(prev_action)] = old + self.alpha * (target - old)

        self.useA = not self.useA

    def set_epsilon(self, v: float) -> None:
        self.eps = max(self.cfg.epsilon_min, min(1.0, float(v)))

    def decay_epsilon_once(self) -> None:
        if self.eps > self.cfg.epsilon_min:
            self.eps *= self.cfg.epsilon_decay_per_tick
            if self.eps < self.cfg.epsilon_min:
                self.eps = self.cfg.epsilon_min

    def save(self, path: str) -> None:
        data = {
            "service": self.service,
            "eps": self.eps,
            "qA": {str(k): v for k, v in self.qA.items()},
            "qB": {str(k): v for k, v in self.qB.items()},
            "state_visits": {str(k): int(v) for k, v in self.state_visits.items()},
            "action_counts": {str(k): int(v) for k, v in self.action_counts.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.eps = float(data.get("eps", self.eps))
        for k, v in data.get("qA", {}).items():
            self.qA[ast.literal_eval(k)] = v  # Safe!
        for k, v in data.get("qB", {}).items():
            self.qB[ast.literal_eval(k)] = v  # Safe!


# =============================================================================
# REWARD (Thrashing fixed)
# =============================================================================

class Reward:
    """
    Reward components:
    - SLA (latency ratio)
    - efficiency (cpu)
    - stability (readiness ratio)
    - thrashing (compare current action vs immediately previous action)
    """
    def __init__(self, weights: Tuple[float, float, float, float] = (0.45, 0.25, 0.15, 0.15)):
        self.w_sla, self.w_eff, self.w_stab, self.w_thr = weights

    def calc(self,
             lat_ratio: float,
             cpu: float,
             action_being_rewarded: ScalingAction,
             pods_ready_ratio: float,
             last_action: Optional[ScalingAction],  # Was: prev_prev_action
             latency_valid: bool = True) -> float:  # Added for issue #3

        # 1) SLA - handle missing latency (Issue #3)
        if not latency_valid:
            r_sla = 0.0  # Neutral when data missing
        elif lat_ratio <= 0.8: r_sla = 1.0
        elif lat_ratio <= 1.0: r_sla = 0.5
        elif lat_ratio <= 1.5: r_sla = -0.5
        elif lat_ratio <= 2.0: r_sla = -1.0
        else: r_sla = -2.0

        # 2) Efficiency
        if 0.5 <= cpu <= 0.7: r_eff = 1.0
        elif 0.3 <= cpu <= 0.85: r_eff = 0.3
        elif cpu < 0.1: r_eff = -0.5
        else: r_eff = -0.3

        # 3) Stability
        if pods_ready_ratio < 1.0 and action_being_rewarded != ScalingAction.STAY:
            r_stab = -1.0
        elif action_being_rewarded == ScalingAction.SCALE_UP:
            r_stab = -0.2
        elif action_being_rewarded == ScalingAction.SCALE_DOWN:
            r_stab = -0.1
        else:
            r_stab = 0.1

        # 4) Thrashing: consecutive flip-flop (t vs t-1)
        r_thr = 0.0
        if last_action is not None:
            # UP immediately after DOWN, or DOWN immediately after UP
            if ((action_being_rewarded == ScalingAction.SCALE_UP and last_action == ScalingAction.SCALE_DOWN) or
                (action_being_rewarded == ScalingAction.SCALE_DOWN and last_action == ScalingAction.SCALE_UP)):
                r_thr = -1.0
            # Any scaling after different scaling (less severe)
            elif (action_being_rewarded != ScalingAction.STAY and 
                  last_action != ScalingAction.STAY and 
                  action_being_rewarded != last_action):
                r_thr = -0.3

        r = self.w_sla * r_sla + self.w_eff * r_eff + self.w_stab * r_stab + self.w_thr * r_thr
        return float(max(-2.0, min(2.0, r)))


# =============================================================================
# CAPA CONTROLLER (fixed update logic + pending settle)
# =============================================================================

class CAPAController:
    def __init__(self, cfg: Config, metrics: UnifiedMetricsCollector, dry_run: bool = False):
        self.cfg = cfg
        self.metrics = metrics
        self.dry_run = dry_run
        self.kubectl = KubectlClient(cfg.namespace)
        self.baseline = BaselineHPA(cfg)

        self.agents = {svc: DoubleQAgent(svc, cfg) for svc in cfg.services.keys()}
        self.reward = Reward()

        # State histories for correct RL updates
        self.prev_state: Dict[str, Tuple[int, int, int, int, int]] = {}
        self.prev_action: Dict[str, ScalingAction] = {}
        self.last_action: Dict[str, Optional[ScalingAction]] = {}  # Was: prev_prev_action

        # Trends
        self.prev_metrics: Dict[str, ServiceMetrics] = {}
        self.prev_lat_ratio: Dict[str, float] = {}

        # Cooldown
        self.last_scale_time: Dict[str, float] = {}

        # Delayed reward queue
        self.pending: Dict[str, PendingScale] = {}

        # Training control
        self.learning_enabled: bool = True

    def reset_state_for_new_run(self):
        self.prev_state.clear()
        self.prev_action.clear()
        self.last_action.clear()  # Was: prev_prev_action
        self.prev_metrics.clear()
        self.prev_lat_ratio.clear()
        self.last_scale_time.clear()
        self.pending.clear()

    def set_learning(self, enabled: bool):
        self.learning_enabled = bool(enabled)

    def decay_eps_once_per_tick(self):
        # FIX: decay is done once per global tick (outside service loop)
        for a in self.agents.values():
            a.decay_epsilon_once()

    def _can_scale(self, service: str) -> bool:
        last = self.last_scale_time.get(service, 0.0)
        return (time.time() - last) >= self.cfg.cooldown_sec

    def _compute_new_reps(self, action: ScalingAction, cur: int, sc: ServiceConfig) -> int:
        if action == ScalingAction.SCALE_UP:
            return min(cur + 1, sc.max_replicas)
        if action == ScalingAction.SCALE_DOWN:
            return max(cur - 1, sc.min_replicas)
        return cur

    def _execute(self, service: str, action: ScalingAction, cur_ready: int, sc: ServiceConfig) -> Tuple[bool, int]:
        new_reps = self._compute_new_reps(action, cur_ready, sc)
        if new_reps == cur_ready:
            return False, new_reps
        if self.dry_run:
            self.last_scale_time[service] = time.time()
            return True, new_reps
        ok = self.kubectl.scale_deployment(service, new_reps)
        if ok:
            self.last_scale_time[service] = time.time()
        return ok, new_reps

    def step(self, service: str) -> Dict[str, Any]:
        """
        Returns a dict with:
        - metrics, state, actions, reward
        """
        sc = self.cfg.services[service]
        m = self.metrics.collect(service)
        # Track whether latency data is valid
        latency_valid = (m.latency_avg_ms > 0 or m.latency_p95_ms > 0)
        # Use p95 if available, fallback to avg (Issue #8)
        latency_for_ratio = m.latency_p95_ms if m.latency_p95_ms > 0 else m.latency_avg_ms
        # Neutral ratio (1.0) when data missing instead of 0.0
        if latency_for_ratio > 0 and sc.target_latency_ms > 0:
            lat_ratio = latency_for_ratio / sc.target_latency_ms
        else:
            lat_ratio = 1.0  # Neutral
            latency_valid = False
        pod_ratio = (m.ready_replicas / sc.max_replicas) if sc.max_replicas > 0 else 0.0

        # trends
        cpu_tr = 0.0
        lat_tr = 0.0
        if service in self.prev_metrics:
            pm = self.prev_metrics[service]
            cpu_tr = float(m.cpu_utilization - pm.cpu_utilization)
            prev_lr = float(self.prev_lat_ratio.get(service, lat_ratio))
            lat_tr = float(lat_ratio - prev_lr)
        self.prev_lat_ratio[service] = lat_ratio

        # discretize
        cur_state = self.agents[service].discretize(m.cpu_utilization, lat_ratio, pod_ratio, cpu_tr, lat_tr)

        # actions
        baseline_action = self.baseline.decide(service, m.cpu_utilization, m.ready_replicas)
        rl_action = self.agents[service].choose(cur_state, training=self.learning_enabled)

        # select action (CAPA uses RL)
        chosen = rl_action
        source = "rl"

        # cooldown
        if chosen != ScalingAction.STAY and not self._can_scale(service):
            chosen = ScalingAction.STAY
            source = "cooldown_block"

        executed = False
        target_reps = m.desired_replicas  # Changed from m.ready_replicas
        if chosen != ScalingAction.STAY:
            executed, target_reps = self._execute(service, chosen, m.desired_replicas, sc)

        # learning update:
        # 1) if there is pending settle, check settle/timeout and update for that previous scaling action
        reward_value = self._maybe_update_learning(service, m, cur_state, lat_ratio, latency_valid)
        # 2) if we executed scaling now, register pending settle with the action being rewarded later
        if executed and self.learning_enabled:
            self.pending[service] = PendingScale(
                start_time=time.time(),
                prev_state=cur_state,                    # state at time of action selection
                prev_action=chosen,                      # action to be rewarded later
                last_action_before=self.prev_action.get(service),  # Action at t-1
                target_replicas=target_reps
            )

        # store histories AFTER update registration
        self.prev_metrics[service] = m
        # track action history for thrashing:
        # prev_prev_action <= prev_action, prev_action <= chosen
        self.last_action[service] = self.prev_action.get(service)
        self.prev_action[service] = chosen
        self.prev_state[service] = cur_state

        return {
            "metrics": m,
            "cur_state": cur_state,
            "cpu_trend": cpu_tr,
            "lat_trend": lat_tr,
            "lat_ratio": lat_ratio,
            "pod_ratio": pod_ratio,
            "baseline_action": baseline_action,
            "rl_action": rl_action,
            "action": chosen,
            "decision_source": source,
            "reward": reward_value,
            "executed": executed,
            "latency_valid": latency_valid,  # ADD THIS
        }

    # Replace _is_settled with action-aware version:
    def _is_settled(self, m: ServiceMetrics, target: int, action: ScalingAction) -> bool:
        """
        Check if scaling action has settled.
    
        For SCALE_UP: ready >= target (pods are running)
        For SCALE_DOWN: ready <= target AND desired == target (pods terminated)
        """
        tol = self.cfg.readiness_tolerance
    
        if action == ScalingAction.SCALE_UP:
            # Pods should be ready
            return m.ready_replicas >= (target - tol)
    
        elif action == ScalingAction.SCALE_DOWN:
            # Pods should be terminated AND desired should match
            return (m.ready_replicas <= (target + tol) and 
                    m.desired_replicas == target)
    
        else:
            # STAY is always "settled"
            return True

    def _maybe_update_learning(self, service: str, m: ServiceMetrics, cur_state: Tuple[int,int,int,int,int], lat_ratio: float, latency_valid: bool = True) -> float:
        if not self.learning_enabled:
            return 0.0

        # Case A: settle pending scaling action
        if service in self.pending:
            pend = self.pending[service]
            elapsed = time.time() - pend.start_time
            settled = self._is_settled(m, pend.target_replicas, pend.prev_action)
            timed_out = elapsed >= self.cfg.scaling_settle_timeout_sec

            if settled or timed_out:
                pods_ready_ratio = (m.ready_replicas / max(1, m.current_replicas))
                r = self.reward.calc(
                    lat_ratio=float(lat_ratio),
                    cpu=float(m.cpu_utilization),
                    action_being_rewarded=pend.prev_action,
                    pods_ready_ratio=float(pods_ready_ratio),
                    last_action=pend.last_action_before,
                    latency_valid=latency_valid  # ADD THIS
                )
                self.agents[service].update(pend.prev_state, pend.prev_action, r, cur_state)
                del self.pending[service]
                return r
            return 0.0
        # Case B: no pending action
        if service in self.prev_state and service in self.prev_action:
            prev_state = self.prev_state[service]
            prev_action = self.prev_action[service]
            last_act = self.last_action.get(service)
            pods_ready_ratio = (m.ready_replicas / max(1, m.current_replicas))
            r = self.reward.calc(
              lat_ratio=float(lat_ratio),
              cpu=float(m.cpu_utilization),
              action_being_rewarded=prev_action,
              pods_ready_ratio=float(pods_ready_ratio),
              last_action=last_act,
              latency_valid=latency_valid  # ADD THIS
            )
            self.agents[service].update(prev_state, prev_action, r, cur_state)
            return r

        return 0.0

    def save_agents(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for svc, agent in self.agents.items():
            agent.save(os.path.join(directory, f"{svc}_agent.json"))

    def load_agents(self, directory: str):
        for svc, agent in self.agents.items():
            p = os.path.join(directory, f"{svc}_agent.json")
            if os.path.exists(p):
                agent.load(p)


# =============================================================================
# BASELINE RUNNER (independent cooldown/state)
# =============================================================================

class BaselineController:
    def __init__(self, cfg: Config, metrics: UnifiedMetricsCollector, dry_run: bool = False):
        self.cfg = cfg
        self.metrics = metrics
        self.dry_run = dry_run
        self.kubectl = KubectlClient(cfg.namespace)
        self.hpa = BaselineHPA(cfg)
        self.last_scale_time: Dict[str, float] = {}

    def reset_state_for_new_run(self):
        self.last_scale_time.clear()

    def _can_scale(self, service: str) -> bool:
        last = self.last_scale_time.get(service, 0.0)
        return (time.time() - last) >= self.cfg.cooldown_sec

    def _compute_new_reps(self, action: ScalingAction, cur: int, sc: ServiceConfig) -> int:
        if action == ScalingAction.SCALE_UP:
            return min(cur + 1, sc.max_replicas)
        if action == ScalingAction.SCALE_DOWN:
            return max(cur - 1, sc.min_replicas)
        return cur

    def _execute(self, service: str, action: ScalingAction, cur_ready: int, sc: ServiceConfig) -> Tuple[bool, int]:
        new_reps = self._compute_new_reps(action, cur_ready, sc)
        if new_reps == cur_ready:
            return False, new_reps
        if self.dry_run:
            self.last_scale_time[service] = time.time()
            return True, new_reps
        ok = self.kubectl.scale_deployment(service, new_reps)
        if ok:
            self.last_scale_time[service] = time.time()
        return ok, new_reps

    def step(self, service: str) -> Dict[str, Any]:
        sc = self.cfg.services[service]
        m = self.metrics.collect(service)
        latency_valid = (m.latency_avg_ms > 0 or m.latency_p95_ms > 0)
        lat_ratio = (m.latency_avg_ms / sc.target_latency_ms) if (m.latency_avg_ms > 0 and sc.target_latency_ms > 0) else 1.0  # Changed from 0.0 to 1.0
        pod_ratio = (m.ready_replicas / sc.max_replicas) if sc.max_replicas > 0 else 0.0
        action = self.hpa.decide(service, m.cpu_utilization, m.ready_replicas)
        source = "baseline_hpa"
        executed = False  # Track execution status
        
        if action != ScalingAction.STAY and not self._can_scale(service):
            action = ScalingAction.STAY
            source = "cooldown_block"
    
        if action != ScalingAction.STAY:
            executed, _ = self._execute(service, action, m.desired_replicas, sc)
            if not executed:
                source = "execute_failed"  # Optional: track failures
    
        return {
            "metrics": m,
            "cur_state": None,
            "cpu_trend": 0.0,
            "lat_trend": 0.0,
            "lat_ratio": lat_ratio,
            "pod_ratio": pod_ratio,
            "baseline_action": action,
            "rl_action": ScalingAction.STAY,
            "action": action,
            "decision_source": source,
            "reward": 0.0,
            "executed": executed,  # Add this to output
            "latency_valid": latency_valid,  # ADD THIS
        }


# =============================================================================
# RUNNER (deterministic patterns + aligned ticks + reset between runs)
# =============================================================================

class ExperimentRunner:
    def __init__(self, cfg: Config, seed: int, dry_run: bool = False):
        self.cfg = cfg
        self.seed = int(seed) & 0xFFFFFFFF
        self.dry_run = dry_run

        self.metrics = UnifiedMetricsCollector(cfg)
        self.locust = self.metrics.locust
        self.kubectl = KubectlClient(cfg.namespace)

        self.baseline_ctrl = BaselineController(cfg, self.metrics, dry_run=dry_run)
        self.capa_ctrl = CAPAController(cfg, self.metrics, dry_run=dry_run)

        self._running = False
        self._stop_reason = ""

        self._install_signal_handlers()

    def _install_signal_handlers(self):
        def _stop(sig=None, frame=None):
            self._running = False
            self._stop_reason = f"signal={sig}"
        signal.signal(signal.SIGINT, _stop)
        if hasattr(signal, "SIGTERM"):
            try:
                signal.signal(signal.SIGTERM, _stop)
            except Exception:
                pass

    @staticmethod
    def _stable_seed(pattern_name: str, base_seed: int) -> int:
        # FIX: deterministic seed (no Python hash())
        h = hashlib.sha256(pattern_name.encode("utf-8")).digest()
        pseed = int.from_bytes(h[:4], "big")
        return (pseed ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF

    def _randomized_pattern(self, pattern_name: str) -> List[int]:
        base = LOAD_PATTERNS.get(pattern_name, [100])
        rng = random.Random(self._stable_seed(pattern_name, self.seed))

        scale = rng.uniform(self.cfg.workload_scale_min, self.cfg.workload_scale_max)

        out = []
        for u in base:
            val = int(u * scale + rng.gauss(0, self.cfg.workload_noise_sigma))
            out.append(max(10, val))
        return out

    def reset_cluster_replicas(self):
        # Best-effort: set all services to min_replicas for comparability
        for svc, sc in self.cfg.services.items():
            if self.dry_run:
                continue
            self.kubectl.scale_deployment(svc, sc.min_replicas)
        # Give time to settle
        time.sleep(min(60.0, max(10.0, self.cfg.control_interval_sec * 2)))

    def _prepare_run(self):
        # FIX: reset EMA/collector caches between runs
        self.metrics.reset()
        self.baseline_ctrl.reset_state_for_new_run()
        self.capa_ctrl.reset_state_for_new_run()

    def run(self,
            mode: ControlMode,
            patterns: List[str],
            duration_sec: int,
            results_dir: str,
            reset_replicas: bool = False,
            capa_learning: bool = True,
            capa_checkpoint_in: Optional[str] = None,
            capa_checkpoint_out: Optional[str] = None) -> str:

        os.makedirs(results_dir, exist_ok=True)
        self._prepare_run()

        if reset_replicas:
            logger.info("Resetting replicas to min_replicas for comparability...")
            self.reset_cluster_replicas()

        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{mode.value}_{run_ts}"
        csv_path = os.path.join(results_dir, f"{mode.value}_run_{run_ts}.csv")
        meta_path = os.path.join(results_dir, f"run_meta_{run_ts}_{mode.value}.json")

        # CAPA load checkpoint (optional)
        if mode == ControlMode.CAPA and capa_checkpoint_in:
            logger.info(f"Loading CAPA agents from: {capa_checkpoint_in}")
            self.capa_ctrl.load_agents(capa_checkpoint_in)

        if mode == ControlMode.CAPA:
            self.capa_ctrl.set_learning(capa_learning)

        # If locust is not running, continue but note.
        if not self.locust.is_available():
            logger.warning("Locust is not reachable; load changes may not apply; latency may be 0.")

        # duration split among patterns equally
        if not patterns:
            patterns = ["warmup"]
        per_pattern = max(1.0, float(duration_sec) / float(len(patterns)))

        # CSV header
        header = [
            "run_id","mode","seed","pattern","users","step_idx","tick","timestamp","service", "executed",
            "latency_p95_ms","latency_avg_ms","arrival_rate_rps","cpu_utilization","failure_rate",
            "ready_replicas","desired_replicas",
            "action","baseline_action","rl_action","decision_source","reward",
            "state","lat_ratio","pod_ratio","cpu_trend","lat_trend",
            "latency_valid",
            "latency_scope","latency_source","resource_source","notes"
        ]

        meta = {
            "run_id": run_id,
            "mode": mode.value,
            "seed": self.seed,
            "patterns": patterns,
            "duration_sec": duration_sec,
            "control_interval_sec": self.cfg.control_interval_sec,
            "cooldown_sec": self.cfg.cooldown_sec,
            "latency_source": self.cfg.latency_source,
            "latency_scope_expected": "global" if self.cfg.latency_source == "locust" else "per_service",
            "services": {k: asdict(v) for k, v in self.cfg.services.items()},
            "notes": [
                "If locust is used, latency_scope is global aggregate; per-service latency requires Prometheus/OTel integration."
            ]
        }

        logger.info(f"Starting run: {run_id} -> {csv_path}")
        self._running = True
        self._stop_reason = ""

        tick = 0
        csv_file = None
        try:
            # Open file ONCE at the start
            csv_file = open(csv_path, "w", buffering=1)  # Line buffering
            csv_file.write(",".join(header) + "\n")
            for pat in patterns:
                if not self._running:
                    break
                schedule = self._randomized_pattern(pat)
                step_dur = per_pattern / max(1, len(schedule))

                logger.info(f"Pattern={pat} steps={len(schedule)} step_dur~{step_dur:.1f}s")

                for step_idx, users in enumerate(schedule):
                    if not self._running:
                        break

                    # apply load
                    if self.locust.is_available():
                        self.locust.set_users(users)

                    step_start = time.time()
                    while self._running and (time.time() - step_start) < step_dur:
                        # GLOBAL epsilon decay once per tick (FIX)
                        if mode == ControlMode.CAPA and self.capa_ctrl.learning_enabled:
                            self.capa_ctrl.decay_eps_once_per_tick()

                        for svc in self.cfg.services.keys():
                            if mode == ControlMode.BASELINE:
                                out = self.baseline_ctrl.step(svc)
                            else:
                                out = self.capa_ctrl.step(svc)

                            m: ServiceMetrics = out["metrics"]
                            rec = TickRecord(
                                run_id=run_id,
                                mode=mode.value,
                                seed=self.seed,
                                pattern=pat,
                                users=int(users),
                                step_idx=int(step_idx),
                                tick=int(tick),
                                timestamp=float(time.time()),
                                service=svc,
                                latency_p95_ms=float(m.latency_p95_ms),
                                latency_avg_ms=float(m.latency_avg_ms),
                                arrival_rate_rps=float(m.arrival_rate_rps),
                                cpu_utilization=float(m.cpu_utilization),
                                failure_rate=float(m.failure_rate),
                                ready_replicas=int(m.ready_replicas),
                                desired_replicas=int(m.desired_replicas),
                                action=str(out["action"].name),
                                baseline_action=str(out["baseline_action"].name) if hasattr(out["baseline_action"], "name") else str(out["baseline_action"]),
                                rl_action=str(out["rl_action"].name) if hasattr(out["rl_action"], "name") else str(out["rl_action"]),
                                decision_source=str(out["decision_source"]),
                                reward=float(out.get("reward", 0.0) or 0.0),
                                state=str(out["cur_state"]) if out["cur_state"] is not None else "",
                                lat_ratio=float(out["lat_ratio"]),
                                pod_ratio=float(out["pod_ratio"]),
                                cpu_trend=float(out["cpu_trend"]),
                                lat_trend=float(out["lat_trend"]),
                                executed=bool(out.get("executed", False)),
                                notes=""
                            )

                            row = [
                                rec.run_id, rec.mode, str(rec.seed), rec.pattern, str(rec.users), str(rec.step_idx),
                                str(rec.tick), f"{rec.timestamp:.6f}", rec.service,
                                str(int(rec.executed)),  # Move here to match header position!
                                f"{rec.latency_p95_ms:.6f}", f"{rec.latency_avg_ms:.6f}", f"{rec.arrival_rate_rps:.6f}",
                                f"{rec.cpu_utilization:.6f}", f"{rec.failure_rate:.6f}",
                                str(rec.ready_replicas), str(rec.desired_replicas),
                                rec.action, rec.baseline_action, rec.rl_action, rec.decision_source, f"{rec.reward:.6f}",
                                rec.state, f"{rec.lat_ratio:.6f}", f"{rec.pod_ratio:.6f}", f"{rec.cpu_trend:.6f}", f"{rec.lat_trend:.6f}",
                                str(int(out.get("latency_valid", True))),  # Added for Issue #3
                                m.latency_scope, m.latency_source, m.resource_source,
                                rec.notes.replace(",", ";")
                            ]
                            # Write directly to open file handle
                            csv_file.write(",".join(row) + "\n")

                        tick += 1
                        time.sleep(self.cfg.control_interval_sec)

        finally:
            self._running = False
            # Close CSV file
            if csv_file:
                csv_file.close()
            if self.locust.is_available():
                self.locust.stop()

            meta["stop_reason"] = self._stop_reason or "completed"
            meta["ticks"] = tick
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # CAPA save checkpoint (optional)
            if mode == ControlMode.CAPA and capa_checkpoint_out:
                logger.info(f"Saving CAPA agents to: {capa_checkpoint_out}")
                self.capa_ctrl.save_agents(capa_checkpoint_out)

            logger.info(f"Run complete. meta={meta_path}")
            return csv_path


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def run_diagnostics(cfg: Config) -> int:
    ok = True
    m = UnifiedMetricsCollector(cfg)

    logger.info("=== DIAGNOSE ===")
    logger.info(f"Locust available: {m.locust_ok} ({cfg.locust_url})")
    logger.info(f"Prometheus available: {m.prom_ok} ({cfg.prometheus_url})")

    if not m.locust_ok:
        logger.warning("Locust not reachable: aggregate latency/load control disabled.")
        ok = False

    if not m.prom_ok:
        logger.warning("Prometheus not reachable: CPU utilization may be missing; falling back to kubectl for replicas.")
        # not necessarily fatal

    # kubectl sanity
    kc = KubectlClient(cfg.namespace)
    svc0 = next(iter(cfg.services.keys()))
    r, c, d = kc.get_pod_counts(svc0)
    logger.info(f"kubectl sample: {svc0} ready={r} current={c} desired={d}")
    logger.info("=== END DIAGNOSE ===")
    return 0 if ok else 2


# =============================================================================
# CLI
# =============================================================================

def parse_patterns(s: str) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["diagnose", "run", "paired"], help="Action")
    ap.add_argument("--mode", choices=["baseline", "capa"], default="baseline", help="run: mode")
    ap.add_argument("--patterns", default="warmup,step", help="Comma-separated patterns")
    ap.add_argument("--duration", type=int, default=1800, help="Total duration seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset-replicas", action="store_true", help="Reset replicas to min before each run")

    ap.add_argument("--results-dir", default=CONFIG.results_dir)
    ap.add_argument("--exp-name", default=CONFIG.exp_name)

    # CAPA checkpointing
    ap.add_argument("--capa-load", default="", help="Directory containing *_agent.json to load")
    ap.add_argument("--capa-save", default="", help="Directory to save agents after run")
    ap.add_argument("--no-learning", action="store_true", help="CAPA: disable learning (evaluation-like)")

    # URLs
    ap.add_argument("--prometheus", default=CONFIG.prometheus_url)
    ap.add_argument("--locust", default=CONFIG.locust_url)
    ap.add_argument("--namespace", default=CONFIG.namespace)

    args = ap.parse_args()

    # apply overrides
    CONFIG.prometheus_url = args.prometheus
    CONFIG.locust_url = args.locust
    CONFIG.namespace = args.namespace
    CONFIG.results_dir = args.results_dir
    CONFIG.exp_name = args.exp_name

    results_dir = os.path.join(CONFIG.results_dir, CONFIG.exp_name)
    patterns = parse_patterns(args.patterns)

    if args.command == "diagnose":
        code = run_diagnostics(CONFIG)
        raise SystemExit(code)

    runner = ExperimentRunner(CONFIG, seed=args.seed, dry_run=args.dry_run)

    if args.command == "run":
        mode = ControlMode(args.mode)
        capa_learning = (not args.no_learning)
        runner.run(
            mode=mode,
            patterns=patterns,
            duration_sec=args.duration,
            results_dir=results_dir,
            reset_replicas=args.reset_replicas,
            capa_learning=capa_learning,
            capa_checkpoint_in=(args.capa_load or None),
            capa_checkpoint_out=(args.capa_save or None),
        )

    if args.command == "paired":
        # Paired run: baseline then CAPA with identical schedule seed/patterns
        logger.info("=== PAIRED: baseline then capa ===")
        runner.run(
            mode=ControlMode.BASELINE,
            patterns=patterns,
            duration_sec=args.duration,
            results_dir=results_dir,
            reset_replicas=args.reset_replicas
        )
        # small cooldown gap
        time.sleep(min(15.0, CONFIG.control_interval_sec))
        runner.run(
            mode=ControlMode.CAPA,
            patterns=patterns,
            duration_sec=args.duration,
            results_dir=results_dir,
            reset_replicas=args.reset_replicas,
            capa_learning=(not args.no_learning),
            capa_checkpoint_in=(args.capa_load or None),
            capa_checkpoint_out=(args.capa_save or None),
        )

if __name__ == "__main__":
    main()
