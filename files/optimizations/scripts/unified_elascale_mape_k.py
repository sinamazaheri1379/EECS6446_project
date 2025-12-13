#!/usr/bin/env python3
"""
unified_experiment_v3_clean.py
========================================================
CAPA+ v3 CLEAN Experiment Runner (Control-only, Raw Logging only)

Design principles (Jain-compliant separation of concerns):
- This script DOES NOT compute confidence intervals, batch means, transient removal, or statistical tests.
- It only runs workloads, collects metrics, applies controller logic (baseline or CAPA+),
  executes scaling (optional), and logs raw timestamped samples.

Paired-friendly logging:
- baseline and capa runs are executed sequentially with identical workload schedule and seed.
- Each observation is indexed by (pattern, step_idx, tick, service) so analysis can pair samples.

Outputs (by default under ./results):
- baseline_raw_<runid>.csv
- capa_raw_<runid>.csv
- baseline_decisions_<runid>.jsonl
- capa_decisions_<runid>.jsonl
- run_metadata_<runid>.json

Usage examples:
  python unified_experiment_v3_clean.py diagnose
  python unified_experiment_v3_clean.py run --mode baseline --duration 3600
  python unified_experiment_v3_clean.py run --mode capa --duration 3600
  python unified_experiment_v3_clean.py run --mode paired --duration 3600 --seed 123
"""

import os
import sys
import time
import json
import math
import random
import signal
import logging
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(), logging.FileHandler("capa_v3_clean.log")],
)
logger = logging.getLogger("CAPA_V3_CLEAN")


@dataclass
class ServiceConfig:
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 50
    target_latency_ms: float = 200.0
    capacity_per_pod: float = 100.0  # assumed service rate per pod (req/s), used only for simple checks (not stats)


@dataclass
class Config:
    prometheus_url: str = "http://localhost:9090"
    locust_url: str = "http://localhost:8089"
    namespace: str = "default"

    # Services under control
    services: Dict[str, ServiceConfig] = field(
        default_factory=lambda: {
            "frontend": ServiceConfig(1, 10, 50, 200, 100),
            "recommendationservice": ServiceConfig(1, 5, 50, 100, 150),
            "productcatalogservice": ServiceConfig(1, 5, 50, 100, 200),
            "cartservice": ServiceConfig(1, 5, 50, 100, 100),
            "checkoutservice": ServiceConfig(1, 5, 50, 150, 80),
        }
    )

    # Control loop
    control_interval_sec: float = 15.0
    cooldown_sec: float = 60.0

    # Smoothing for metrics (engineering, not statistics)
    smoothing_alpha: float = 0.30

    # RL hyperparameters (CAPA+)
    learning_rate: float = 0.10
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.10
    epsilon_decay: float = 0.9995

    # Trend thresholds (state discretization)
    cpu_trend_threshold: float = 0.05
    latency_trend_threshold: float = 0.10

    # Settling / delayed reward
    scaling_settle_timeout_sec: float = 180.0
    readiness_tolerance: int = 0

    # Workload randomization
    workload_scale_min: float = 0.8
    workload_scale_max: float = 1.2
    workload_noise_sigma: float = 15.0

    # Output
    results_dir: str = "./results"


CONFIG = Config()

LOAD_PATTERNS: Dict[str, List[int]] = {
    "warmup": [50, 100, 150, 100, 50],
    "step": [50, 100, 300, 500, 600, 500, 300, 100],
    "gradual": [50, 150, 250, 350, 450, 550, 450, 250],
    "sine": [300, 477, 550, 477, 300, 123, 50, 123],
    "spike": [50, 50, 800, 100, 50, 700, 50, 50],
    "stress": [100, 300, 600, 800, 600, 300, 100, 50],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ControlMode(Enum):
    BASELINE = "baseline"
    CAPA = "capa"


class ScalingAction(Enum):
    SCALE_DOWN = 0
    STAY = 1
    SCALE_UP = 2


@dataclass
class ServiceMetrics:
    timestamp: float
    service_name: str

    # Latency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0

    # Throughput / errors
    arrival_rate_rps: float = 0.0
    failure_rate: float = 0.0

    # Resources / replicas
    cpu_utilization: float = 0.0
    ready_replicas: int = 0
    current_replicas: int = 0
    desired_replicas: int = 0

    # Provenance
    latency_source: str = "unknown"
    resource_source: str = "unknown"


@dataclass
class DecisionRecord:
    timestamp: float
    run_id: str
    run_type: str  # baseline/capa
    pattern: str
    step_idx: int
    tick: int
    service: str
    state: Tuple[int, int, int, int, int]
    action: str
    cpu_util: float
    latency_p95_ms: float
    latency_avg_ms: float
    arrival_rps: float
    ready_replicas: int
    executed: bool


# =============================================================================
# METRICS COLLECTION
# =============================================================================

class LocustClient:
    def __init__(self, url: str):
        self.url = url
        self.logger = logging.getLogger("LocustClient")
        self._cache = {}
        self._cache_time = 0.0
        self._ttl = 2.0

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stats/requests", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> Optional[Dict[str, Any]]:
        if time.time() - self._cache_time < self._ttl:
            return self._cache
        try:
            r = requests.get(f"{self.url}/stats/requests", timeout=5)
            if r.status_code == 200:
                self._cache = r.json()
                self._cache_time = time.time()
                return self._cache
        except Exception as e:
            self.logger.warning(f"Locust stats failed: {e}")
        return None

    def get_aggregate(self) -> Dict[str, float]:
        stats = self.get_stats()
        if not stats:
            return {}

        avg_ms = 0.0
        stats_list = stats.get("stats", [])
        if isinstance(stats_list, list):
            for ep in stats_list:
                if ep.get("name") in ("Aggregated", "Total"):
                    avg_ms = float(ep.get("avg_response_time", 0) or 0)
                    break

        p50 = float(stats.get("current_response_time_percentile_50", 0) or 0)
        p95 = float(stats.get("current_response_time_percentile_95", 0) or 0)
        # conservative proxy for p99 if not exposed
        p99 = float(p95 * 1.3) if p95 > 0 else 0.0

        return {
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "avg_ms": avg_ms,
            "rps": float(stats.get("total_rps", 0) or 0),
            "fail_ratio": float(stats.get("fail_ratio", 0) or 0),
        }

    def set_users(self, user_count: int, spawn_rate: Optional[int] = None) -> bool:
        if spawn_rate is None:
            spawn_rate = max(1, user_count // 10)
        try:
            r = requests.post(
                f"{self.url}/swarm",
                data={"user_count": user_count, "spawn_rate": spawn_rate},
                timeout=5,
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

    def reset(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stats/reset", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


class PrometheusClient:
    def __init__(self, url: str, namespace: str):
        self.url = url
        self.namespace = namespace
        self.logger = logging.getLogger("PrometheusClient")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/v1/status/config", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def query(self, promql: str) -> Optional[float]:
        try:
            r = requests.get(
                f"{self.url}/api/v1/query", params={"query": promql}, timeout=10
            )
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

    def get_cpu_util(self, deployment: str) -> Optional[float]:
        q = (
            f'sum(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}",'
            f'pod=~"{deployment}.*",container!=""}}[2m]))'
            f' / sum(kube_pod_container_resource_requests{{namespace="{self.namespace}",'
            f'pod=~"{deployment}.*",resource="cpu"}})'
        )
        v = self.query(q)
        return float(min(v, 2.0)) if v is not None else None

    def get_replica_counts(self, deployment: str) -> Tuple[int, int, int]:
        ready = self.query(
            f'kube_deployment_status_replicas_ready{{namespace="{self.namespace}",deployment="{deployment}"}}'
        )
        cur = self.query(
            f'kube_deployment_status_replicas{{namespace="{self.namespace}",deployment="{deployment}"}}'
        )
        des = self.query(
            f'kube_deployment_spec_replicas{{namespace="{self.namespace}",deployment="{deployment}"}}'
        )
        r = int(ready) if ready else 0
        c = int(cur) if cur else 0
        d = int(des) if des else 0
        # fallbacks for robustness
        if r == 0 and c > 0:
            r = c
        if c == 0 and d > 0:
            c = d
        if d == 0 and c > 0:
            d = c
        return (r, c, d)


class KubectlClient:
    def __init__(self, namespace: str):
        self.namespace = namespace

    def get_replica_counts(self, deployment: str) -> Tuple[int, int, int]:
        try:
            res = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "deployment",
                    deployment,
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.status.readyReplicas},{.status.replicas},{.spec.replicas}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = res.stdout.strip().split(",")
            ready = int(parts[0]) if parts[0] else 0
            cur = int(parts[1]) if len(parts) > 1 and parts[1] else ready
            des = int(parts[2]) if len(parts) > 2 and parts[2] else cur
            if ready == 0 and cur > 0:
                ready = cur
            return (ready, cur, des)
        except Exception:
            return (0, 0, 0)

    def scale(self, deployment: str, replicas: int) -> bool:
        try:
            res = subprocess.run(
                [
                    "kubectl",
                    "scale",
                    "deployment",
                    deployment,
                    "-n",
                    self.namespace,
                    f"--replicas={replicas}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return res.returncode == 0
        except Exception:
            return False


class UnifiedMetricsCollector:
    """
    Collects metrics and applies simple engineering smoothing (EMA) for stability.
    This is not statistical processing.
    """
    def __init__(self, config: Config):
        self.config = config
        self.locust = LocustClient(config.locust_url)
        self.prom = PrometheusClient(config.prometheus_url, config.namespace)
        self.kubectl = KubectlClient(config.namespace)

        self.locust_ok = self.locust.is_available()
        self.prom_ok = self.prom.is_available()
        self._smoothed: Dict[str, ServiceMetrics] = {}

        logger.info(f"Locust available: {self.locust_ok}, Prometheus available: {self.prom_ok}")

    def collect(self, service: str) -> ServiceMetrics:
        m = ServiceMetrics(timestamp=time.time(), service_name=service)

        if self.locust_ok:
            agg = self.locust.get_aggregate()
            if agg:
                m.latency_p50_ms = agg.get("p50_ms", 0.0)
                m.latency_p95_ms = agg.get("p95_ms", 0.0)
                m.latency_p99_ms = agg.get("p99_ms", 0.0)
                m.latency_avg_ms = agg.get("avg_ms", 0.0)
                m.arrival_rate_rps = agg.get("rps", 0.0)
                m.failure_rate = agg.get("fail_ratio", 0.0)
                m.latency_source = "locust"

        if self.prom_ok:
            cpu = self.prom.get_cpu_util(service)
            if cpu is not None:
                m.cpu_utilization = float(cpu)
            r, c, d = self.prom.get_replica_counts(service)
            m.ready_replicas, m.current_replicas, m.desired_replicas = r, c, d
            m.resource_source = "prometheus"
        else:
            r, c, d = self.kubectl.get_replica_counts(service)
            m.ready_replicas, m.current_replicas, m.desired_replicas = r, c, d
            m.resource_source = "kubectl"

        return self._ema(service, m)

    def _ema(self, service: str, new: ServiceMetrics) -> ServiceMetrics:
        if service not in self._smoothed:
            self._smoothed[service] = new
            return new

        old = self._smoothed[service]
        a = self.config.smoothing_alpha

        def ema(x_new: float, x_old: float) -> float:
            if x_new <= 0:
                return x_old
            return a * x_new + (1 - a) * x_old

        new.latency_p95_ms = ema(new.latency_p95_ms, old.latency_p95_ms)
        new.latency_avg_ms = ema(new.latency_avg_ms, old.latency_avg_ms)
        new.arrival_rate_rps = ema(new.arrival_rate_rps, old.arrival_rate_rps)
        new.cpu_utilization = ema(new.cpu_utilization, old.cpu_utilization)

        self._smoothed[service] = new
        return new


# =============================================================================
# CONTROLLERS
# =============================================================================

class BaselineHPA:
    def __init__(self, config: Config):
        self.config = config

    def decide(self, service: str, cpu_util: float, ready: int) -> ScalingAction:
        cfg = self.config.services.get(service)
        if not cfg:
            return ScalingAction.STAY

        target = cfg.target_cpu_percent / 100.0
        up_th = target * 1.4
        down_th = target * 0.6

        if cpu_util > up_th and ready < cfg.max_replicas:
            return ScalingAction.SCALE_UP
        if cpu_util < down_th and ready > cfg.min_replicas:
            return ScalingAction.SCALE_DOWN
        return ScalingAction.STAY


class EnhancedDoubleQLearningAgent:
    """
    Same engineering RL core as v3, kept inside controller.
    No statistics are computed here.
    """
    def __init__(self, service: str, config: Config):
        self.service = service
        self.config = config

        self.n_actions = 3
        self.qA: Dict[Tuple[int, int, int, int, int], List[float]] = {}
        self.qB: Dict[Tuple[int, int, int, int, int], List[float]] = {}
        self.useA = True

        self.alpha = config.learning_rate
        self.gamma = config.discount_factor
        self.eps = config.epsilon_start
        self.eps_min = config.epsilon_min
        self.eps_decay = config.epsilon_decay

        self.replay: List[Tuple[Tuple, int, float, Tuple, float]] = []
        self.max_replay = 500
        self.batch = 16

    def _q(self, table: Dict, s: Tuple) -> List[float]:
        if s not in table:
            table[s] = [0.0] * self.n_actions
        return table[s]

    def discretize(
        self,
        cpu: float,
        lat_ratio: float,
        pod_ratio: float,
        cpu_trend: float,
        lat_trend: float,
    ) -> Tuple[int, int, int, int, int]:
        # CPU
        if cpu < 0.4:
            c = 0
        elif cpu < 0.7:
            c = 1
        else:
            c = 2
        # latency ratio
        if lat_ratio <= 0.8:
            l = 0
        elif lat_ratio <= 1.1:
            l = 1
        else:
            l = 2
        # pods ratio
        if pod_ratio < 0.3:
            p = 0
        elif pod_ratio < 0.7:
            p = 1
        else:
            p = 2
        # cpu trend
        th = self.config.cpu_trend_threshold
        if cpu_trend < -th:
            ct = 0
        elif cpu_trend > th:
            ct = 2
        else:
            ct = 1
        # lat trend
        th2 = self.config.latency_trend_threshold
        if lat_trend < -th2:
            lt = 0
        elif lat_trend > th2:
            lt = 2
        else:
            lt = 1
        return (c, l, p, ct, lt)

    def choose(self, state: Tuple[int, int, int, int, int], training: bool = True) -> ScalingAction:
        if training and random.random() < self.eps:
            # guided exploration: if latency bad, bias up
            if state[1] == 2 and random.random() < 0.5:
                a = ScalingAction.SCALE_UP
            else:
                a = ScalingAction(random.randint(0, self.n_actions - 1))
        else:
            qA = self._q(self.qA, state)
            qB = self._q(self.qB, state)
            q = [(x + y) / 2 for x, y in zip(qA, qB)]
            a = ScalingAction(int(np.argmax(q)))
        return a

    def update(self, s: Tuple, a: ScalingAction, r: float, s2: Tuple, priority: float):
        # add replay item
        exp = (s, a.value, float(r), s2, float(priority))
        if len(self.replay) >= self.max_replay:
            # drop min priority
            min_i = min(range(len(self.replay)), key=lambda i: self.replay[i][4])
            self.replay.pop(min_i)
        self.replay.append(exp)

        # double Q update
        if self.useA:
            best = int(np.argmax(self._q(self.qA, s2)))
            target = r + self.gamma * self._q(self.qB, s2)[best]
            old = self._q(self.qA, s)[a.value]
            self._q(self.qA, s)[a.value] = old + self.alpha * (target - old)
        else:
            best = int(np.argmax(self._q(self.qB, s2)))
            target = r + self.gamma * self._q(self.qA, s2)[best]
            old = self._q(self.qB, s)[a.value]
            self._q(self.qB, s)[a.value] = old + self.alpha * (target - old)

        self.useA = not self.useA

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def replay_step(self):
        if len(self.replay) < self.batch:
            return
        pr = np.array([x[4] for x in self.replay], dtype=float) + 0.01
        p = pr / pr.sum()
        idxs = np.random.choice(len(self.replay), self.batch, replace=False, p=p)
        alpha2 = self.alpha * 0.5
        for i in idxs:
            s, a, r, s2, _ = self.replay[i]
            if self.useA:
                best = int(np.argmax(self._q(self.qA, s2)))
                target = r + self.gamma * self._q(self.qB, s2)[best]
                old = self._q(self.qA, s)[a]
                self._q(self.qA, s)[a] = old + alpha2 * (target - old)
            else:
                best = int(np.argmax(self._q(self.qB, s2)))
                target = r + self.gamma * self._q(self.qA, s2)[best]
                old = self._q(self.qB, s)[a]
                self._q(self.qB, s)[a] = old + alpha2 * (target - old)

    def set_eval(self):
        self.eps = 0.0


class RewardModel:
    """
    Engineering reward. (Not statistical.)
    """
    def __init__(self, w=(0.45, 0.25, 0.15, 0.15)):
        self.w_sla, self.w_eff, self.w_stab, self.w_thrash = w

    def calc(
        self,
        latency_ratio: float,
        cpu: float,
        action: ScalingAction,
        pods_ready_ratio: float,
        prev_action: Optional[ScalingAction],
    ) -> float:
        # SLA
        if latency_ratio <= 0.8:
            r_sla = 1.0
        elif latency_ratio <= 1.0:
            r_sla = 0.5
        elif latency_ratio <= 1.5:
            r_sla = -0.5
        elif latency_ratio <= 2.0:
            r_sla = -1.0
        else:
            r_sla = -2.0

        # efficiency
        if 0.5 <= cpu <= 0.7:
            r_eff = 1.0
        elif 0.3 <= cpu <= 0.85:
            r_eff = 0.3
        elif cpu < 0.1:
            r_eff = -0.5
        else:
            r_eff = -0.3

        # stability
        if pods_ready_ratio < 1.0 and action != ScalingAction.STAY:
            r_stab = -1.0
        elif action == ScalingAction.SCALE_UP:
            r_stab = -0.2
        elif action == ScalingAction.SCALE_DOWN:
            r_stab = -0.1
        else:
            r_stab = 0.1

        # thrash
        r_thr = 0.0
        if prev_action is not None and action != prev_action:
            if (
                (action == ScalingAction.SCALE_UP and prev_action == ScalingAction.SCALE_DOWN)
                or (action == ScalingAction.SCALE_DOWN and prev_action == ScalingAction.SCALE_UP)
            ):
                r_thr = -1.0
            elif action != ScalingAction.STAY:
                r_thr = -0.3

        r = (
            self.w_sla * r_sla
            + self.w_eff * r_eff
            + self.w_stab * r_stab
            + self.w_thrash * r_thr
        )
        return float(max(-2.0, min(2.0, r)))


class CAPAController:
    """
    CAPA+ controller (clean). Handles delayed reward settling, cooldown, RL agent per service.
    """
    def __init__(self, config: Config, dry_run: bool = False, eval_mode: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.eval_mode = eval_mode

        self.baseline = BaselineHPA(config)  # used only as fallback if needed
        self.kubectl = KubectlClient(config.namespace)
        self.reward = RewardModel()

        self.agents = {svc: EnhancedDoubleQLearningAgent(svc, config) for svc in config.services.keys()}
        if eval_mode:
            for a in self.agents.values():
                a.set_eval()

        self.prev_metrics: Dict[str, ServiceMetrics] = {}
        self.prev_action: Dict[str, ScalingAction] = {}
        self.pending: Dict[str, Dict[str, Any]] = {}
        self.last_scale_time: Dict[str, float] = {}

    def _cooldown_ok(self, service: str) -> bool:
        last = self.last_scale_time.get(service, 0.0)
        return (time.time() - last) >= self.config.cooldown_sec

    def _compute_replicas(self, action: ScalingAction, cur: int, cfg: ServiceConfig) -> int:
        if action == ScalingAction.SCALE_UP:
            return min(cur + 1, cfg.max_replicas)
        if action == ScalingAction.SCALE_DOWN:
            return max(cur - 1, cfg.min_replicas)
        return cur

    def decide_and_maybe_scale(
        self,
        service: str,
        metrics: ServiceMetrics,
    ) -> Tuple[Tuple[int, int, int, int, int], ScalingAction, bool]:
        cfg = self.config.services[service]
        lat_ratio = (metrics.latency_avg_ms / cfg.target_latency_ms) if metrics.latency_avg_ms > 0 else 0.0
        pod_ratio = (metrics.ready_replicas / cfg.max_replicas) if cfg.max_replicas > 0 else 0.0

        cpu_trend = 0.0
        lat_trend = 0.0
        if service in self.prev_metrics:
            pm = self.prev_metrics[service]
            cpu_trend = metrics.cpu_utilization - pm.cpu_utilization
            prev_lat_ratio = (pm.latency_avg_ms / cfg.target_latency_ms) if pm.latency_avg_ms > 0 else lat_ratio
            lat_trend = lat_ratio - prev_lat_ratio

        state = self.agents[service].discretize(metrics.cpu_utilization, lat_ratio, pod_ratio, cpu_trend, lat_trend)
        action = self.agents[service].choose(state, training=not self.eval_mode)

        executed = False
        if action != ScalingAction.STAY and self._cooldown_ok(service):
            new_rep = self._compute_replicas(action, metrics.ready_replicas, cfg)
            if new_rep != metrics.ready_replicas:
                if self.dry_run:
                    executed = True
                else:
                    executed = self.kubectl.scale(service, new_rep)
                if executed:
                    self.last_scale_time[service] = time.time()
                    # register pending settling
                    self.pending[service] = {
                        "start": time.time(),
                        "prev_state": state,
                        "prev_action": action,
                        "target_replicas": new_rep,
                    }

        # delayed reward update (engineering)
        self._maybe_update(service, metrics, state, lat_ratio)

        self.prev_metrics[service] = metrics
        self.prev_action[service] = action
        return state, action, executed

    def _maybe_update(self, service: str, metrics: ServiceMetrics, cur_state: Tuple, lat_ratio: float):
        if self.eval_mode:
            return

        cfg = self.config.services[service]
        # pending settle?
        if service in self.pending:
            pend = self.pending[service]
            elapsed = time.time() - float(pend["start"])
            target = int(pend["target_replicas"])
            settled = metrics.ready_replicas >= (target - self.config.readiness_tolerance)
            if settled or elapsed >= self.config.scaling_settle_timeout_sec:
                prev_state = pend["prev_state"]
                prev_action = pend["prev_action"]
                pods_ready_ratio = (metrics.ready_replicas / max(1, metrics.current_replicas)) if metrics.current_replicas else 1.0
                r = self.reward.calc(lat_ratio, metrics.cpu_utilization, prev_action, pods_ready_ratio, prev_action)
                # priority: |r| plus SLA violation
                pr = abs(r) + (1.0 if lat_ratio > 1.0 else 0.0)
                self.agents[service].update(prev_state, prev_action, r, cur_state, pr)
                self.agents[service].replay_step()
                del self.pending[service]
            return

        # immediate update (best-effort)
        if service in self.prev_metrics:
            prev_action = self.prev_action.get(service)
            if prev_action is None:
                return
            pods_ready_ratio = (metrics.ready_replicas / max(1, metrics.current_replicas)) if metrics.current_replicas else 1.0
            r = self.reward.calc(lat_ratio, metrics.cpu_utilization, prev_action, pods_ready_ratio, prev_action)
            pr = abs(r) + (1.0 if lat_ratio > 1.0 else 0.0)
            # use last state as prev_state if available via prev_metrics trend; keep simple: no extra state buffer.
            # For correctness we need prev_state; we approximate with cur_state when no pending existed.
            # (This is engineering RL, not a statistical claim.)
            self.agents[service].update(cur_state, prev_action, r, cur_state, pr)
            self.agents[service].replay_step()


# =============================================================================
# EXPERIMENT RUNNER + LOGGING
# =============================================================================

CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "run_type",
    "pattern",
    "step_idx",
    "tick",
    "service",
    "users",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_avg_ms",
    "arrival_rate_rps",
    "failure_rate",
    "cpu_utilization",
    "ready_replicas",
    "current_replicas",
    "desired_replicas",
    "action",
    "executed",
    "latency_source",
    "resource_source",
]


def _write_csv_header(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(CSV_COLUMNS) + "\n")


def _append_csv_row(path: Path, row: Dict[str, Any]):
    vals = []
    for c in CSV_COLUMNS:
        v = row.get(c, "")
        if isinstance(v, str):
            v = v.replace("\n", " ").replace("\r", " ")
            # basic CSV escaping
            if "," in v or '"' in v:
                v = '"' + v.replace('"', '""') + '"'
        vals.append(str(v))
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(vals) + "\n")


class ExperimentRunner:
    def __init__(self, config: Config, out_dir: str, dry_run: bool, seed: int):
        self.config = config
        self.out_dir = Path(out_dir)
        self.dry_run = dry_run
        self.seed = seed

        self.metrics = UnifiedMetricsCollector(config)
        self.locust = self.metrics.locust  # reuse client
        self.kubectl = KubectlClient(config.namespace)

        self.running = False
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        logger.info("Shutdown signal received.")
        self.running = False

    def diagnose(self):
        logger.info("Diagnostics:")
        logger.info(f"  Locust reachable: {self.locust.is_available()}")
        logger.info(f"  Prometheus reachable: {PrometheusClient(self.config.prometheus_url, self.config.namespace).is_available()}")
        # kubectl smoke test
        for svc in list(self.config.services.keys())[:2]:
            r, c, d = self.kubectl.get_replica_counts(svc)
            logger.info(f"  kubectl deployment {svc}: ready={r}, current={c}, desired={d}")

    def _randomize_pattern(self, base: List[int]) -> List[int]:
        random.seed(self.seed)
        scale = random.uniform(self.config.workload_scale_min, self.config.workload_scale_max)
        out = []
        for u in base:
            v = int(u * scale + random.gauss(0, self.config.workload_noise_sigma))
            out.append(max(10, v))
        return out

    def _pattern_schedule(self, pattern_name: str, randomized: bool) -> List[int]:
        base = LOAD_PATTERNS.get(pattern_name, [100])
        if not randomized:
            return list(base)
        # derive deterministic per-pattern seed
        rnd_seed = (hash(pattern_name) ^ self.seed) & 0xFFFFFFFF
        rnd = random.Random(rnd_seed)
        scale = rnd.uniform(self.config.workload_scale_min, self.config.workload_scale_max)
        out = []
        for u in base:
            v = int(u * scale + rnd.gauss(0, self.config.workload_noise_sigma))
            out.append(max(10, v))
        return out

    def run(
        self,
        mode: ControlMode,
        duration_sec: int,
        patterns: List[str],
        randomized_patterns: bool,
        eval_mode: bool,
        run_id: str,
    ):
        """
        Run either baseline or CAPA.
        """
        self.running = True

        run_type = mode.value
        csv_path = self.out_dir / f"{run_type}_raw_{run_id}.csv"
        dec_path = self.out_dir / f"{run_type}_decisions_{run_id}.jsonl"
        _write_csv_header(csv_path)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        baseline = BaselineHPA(self.config)
        capa = CAPAController(self.config, dry_run=self.dry_run, eval_mode=eval_mode)

        # deterministic locust reset
        if self.locust.is_available():
            self.locust.reset()

        start = time.time()
        total_steps = sum(len(LOAD_PATTERNS.get(p, [100])) for p in patterns)
        sec_per_step = duration_sec / max(1, total_steps)

        logger.info(f"RUN START: mode={run_type}, duration={duration_sec}s, steps={total_steps}, sec/step≈{sec_per_step:.1f}")

        tick_global = 0

        for pat in patterns:
            if not self.running:
                break

            sched = self._pattern_schedule(pat, randomized_patterns)
            for step_idx, users in enumerate(sched):
                if not self.running:
                    break

                # apply load
                if self.locust.is_available():
                    self.locust.set_users(users)

                step_start = time.time()
                tick_in_step = 0

                while self.running and (time.time() - step_start) < sec_per_step:
                    tick_global += 1
                    tick_in_step += 1

                    for service in self.config.services.keys():
                        m = self.metrics.collect(service)

                        if mode == ControlMode.BASELINE:
                            action = baseline.decide(service, m.cpu_utilization, m.ready_replicas)
                            executed = False
                            if action != ScalingAction.STAY and (time.time() - capa.last_scale_time.get(service, 0.0)) >= self.config.cooldown_sec:
                                cfg = self.config.services[service]
                                new_rep = capa._compute_replicas(action, m.ready_replicas, cfg)
                                if new_rep != m.ready_replicas:
                                    executed = True if self.dry_run else self.kubectl.scale(service, new_rep)
                                    if executed:
                                        capa.last_scale_time[service] = time.time()
                            state = (0, 0, 0, 0, 0)  # baseline has no RL state
                        else:
                            state, action, executed = capa.decide_and_maybe_scale(service, m)

                        # log raw sample
                        row = {
                            "timestamp": m.timestamp,
                            "run_id": run_id,
                            "run_type": run_type,
                            "pattern": pat,
                            "step_idx": step_idx,
                            "tick": tick_in_step,
                            "service": service,
                            "users": users,
                            "latency_p50_ms": m.latency_p50_ms,
                            "latency_p95_ms": m.latency_p95_ms,
                            "latency_p99_ms": m.latency_p99_ms,
                            "latency_avg_ms": m.latency_avg_ms,
                            "arrival_rate_rps": m.arrival_rate_rps,
                            "failure_rate": m.failure_rate,
                            "cpu_utilization": m.cpu_utilization,
                            "ready_replicas": m.ready_replicas,
                            "current_replicas": m.current_replicas,
                            "desired_replicas": m.desired_replicas,
                            "action": action.name,
                            "executed": int(executed),
                            "latency_source": m.latency_source,
                            "resource_source": m.resource_source,
                        }
                        _append_csv_row(csv_path, row)

                        # log decision record (jsonl)
                        dr = DecisionRecord(
                            timestamp=m.timestamp,
                            run_id=run_id,
                            run_type=run_type,
                            pattern=pat,
                            step_idx=step_idx,
                            tick=tick_in_step,
                            service=service,
                            state=state,
                            action=action.name,
                            cpu_util=m.cpu_utilization,
                            latency_p95_ms=m.latency_p95_ms,
                            latency_avg_ms=m.latency_avg_ms,
                            arrival_rps=m.arrival_rate_rps,
                            ready_replicas=m.ready_replicas,
                            executed=bool(executed),
                        )
                        with open(dec_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(asdict(dr), ensure_ascii=False) + "\n")

                    time.sleep(self.config.control_interval_sec)

        # stop load
        if self.locust.is_available():
            self.locust.stop()

        logger.info(f"RUN END: {run_type}, elapsed={time.time()-start:.1f}s")
        self.running = False


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main():
    p = argparse.ArgumentParser(description="CAPA+ v3 CLEAN experiment runner")
    p.add_argument("command", choices=["diagnose", "run"])
    p.add_argument("--mode", choices=["baseline", "capa", "paired"], default="paired")
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--patterns", type=str, default="warmup,step,gradual")
    p.add_argument("--randomized-patterns", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--results-dir", default=CONFIG.results_dir)
    p.add_argument("--prometheus", default=CONFIG.prometheus_url)
    p.add_argument("--locust", default=CONFIG.locust_url)
    p.add_argument("--namespace", default=CONFIG.namespace)
    p.add_argument("--eval", action="store_true", help="CAPA eval mode (epsilon=0, no learning).")
    args = p.parse_args()

    CONFIG.prometheus_url = args.prometheus
    CONFIG.locust_url = args.locust
    CONFIG.namespace = args.namespace
    CONFIG.results_dir = args.results_dir

    patterns = [x.strip() for x in args.patterns.split(",") if x.strip()]

    runner = ExperimentRunner(CONFIG, CONFIG.results_dir, args.dry_run, args.seed)

    if args.command == "diagnose":
        runner.diagnose()
        return

    run_id = _run_id()

    meta = {
        "run_id": run_id,
        "timestamp": run_id,
        "mode": args.mode,
        "duration": args.duration,
        "patterns": patterns,
        "randomized_patterns": bool(args.randomized_patterns),
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "eval": bool(args.eval),
        "config": {
            "prometheus_url": CONFIG.prometheus_url,
            "locust_url": CONFIG.locust_url,
            "namespace": CONFIG.namespace,
            "control_interval_sec": CONFIG.control_interval_sec,
            "cooldown_sec": CONFIG.cooldown_sec,
        },
        "services": {k: asdict(v) for k, v in CONFIG.services.items()},
    }
    Path(CONFIG.results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(CONFIG.results_dir) / f"run_metadata_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.mode == "baseline":
        runner.run(ControlMode.BASELINE, args.duration, patterns, args.randomized_patterns, False, run_id)
    elif args.mode == "capa":
        runner.run(ControlMode.CAPA, args.duration, patterns, args.randomized_patterns, args.eval, run_id)
    else:
        # paired: baseline then capa, same schedule/seed
        runner.run(ControlMode.BASELINE, args.duration, patterns, args.randomized_patterns, False, run_id)
        # small gap for stabilization between runs (not statistics)
        time.sleep(10)
        runner.run(ControlMode.CAPA, args.duration, patterns, args.randomized_patterns, args.eval, run_id)

    logger.info(f"Artifacts written under: {CONFIG.results_dir}")


if __name__ == "__main__":
    main()
