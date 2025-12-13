#!/usr/bin/env python3
"""
Unified CAPA+ Experiment Framework for Kubernetes Autoscaling
==============================================================

Patched version (Dec 2025):
- SHADOW mode: learning disabled to avoid off-policy / causal mismatch
- Reward settling window after scaling: update only after pods settle (ready ~= desired) or timeout
- CLI: hybrid mode maps to TrainingPhase.HYBRID_50 (previously incorrect)
- Safe state serialization (no eval)

Author: EECS6446 Cloud Computing Project
Date: December 2025
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import subprocess
import math
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


# =============================================================================
# SECTION 1: CONFIGURATION AND CONSTANTS
# =============================================================================

@dataclass
class ServiceConfig:
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 50
    target_latency_ms: float = 200.0
    capacity_per_pod: float = 100.0  # μ: requests/second one pod can handle


@dataclass
class Config:
    prometheus_url: str = "http://localhost:9090"
    locust_url: str = "http://localhost:8089"
    namespace: str = "default"

    services: Dict[str, ServiceConfig] = field(default_factory=lambda: {
        'frontend': ServiceConfig(1, 10, 50, 200, 100),
        'recommendationservice': ServiceConfig(1, 5, 50, 100, 150),
        'productcatalogservice': ServiceConfig(1, 5, 50, 100, 200),
        'cartservice': ServiceConfig(1, 5, 50, 100, 100),
        'checkoutservice': ServiceConfig(1, 5, 50, 150, 80),
    })

    train_patterns: List[str] = field(default_factory=lambda: ['step', 'gradual', 'sine'])
    test_patterns: List[str] = field(default_factory=lambda: ['spike', 'random'])

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.10
    epsilon_decay: float = 0.9995

    stability_threshold: float = 0.9
    target_utilization: float = 0.7
    little_law_error_threshold: float = 0.2

    smoothing_alpha: float = 0.3

    control_interval_sec: float = 15.0
    cooldown_sec: float = 60.0

    # NEW: settling window after scaling (seconds)
    scaling_settle_timeout_sec: float = 180.0
    # NEW: readiness tolerance (ready >= desired - tol)
    readiness_tolerance: int = 0


CONFIG = Config()

LOAD_PATTERNS = {
    'warmup': [50, 100, 150, 100, 50],
    'step': [50, 100, 300, 500, 600, 500, 300, 100],
    'gradual': [50, 150, 250, 350, 450, 550, 450, 250],
    'sine': [300, 477, 550, 477, 300, 123, 50, 123],
    'spike': [50, 50, 800, 100, 50, 700, 50, 50],
    'stress': [100, 300, 600, 800, 600, 300, 100, 50],
    'random': None,
}

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(), logging.FileHandler('capa_experiment.log')]
)
logger = logging.getLogger('CAPA+')


# =============================================================================
# SECTION 2: ENUMS AND DATA CLASSES
# =============================================================================

class ControlMode(Enum):
    SHADOW = "shadow"
    HYBRID = "hybrid"
    ACTIVE = "active"
    EVALUATION = "eval"


class ScalingAction(Enum):
    SCALE_DOWN = 0
    STAY = 1
    SCALE_UP = 2


class TrainingPhase(Enum):
    NOT_STARTED = "not_started"
    SHADOW = "shadow"
    HYBRID_25 = "hybrid_25"
    HYBRID_50 = "hybrid_50"
    HYBRID_75 = "hybrid_75"
    ACTIVE = "active"
    EVALUATION = "evaluation"
    COMPLETED = "completed"


@dataclass
class ServiceMetrics:
    timestamp: float = 0.0
    service_name: str = ""

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_max_ms: float = 0.0

    arrival_rate: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    failure_rate: float = 0.0

    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    memory_bytes: int = 0

    current_replicas: int = 0
    ready_replicas: int = 0
    desired_replicas: int = 0

    latency_source: str = "unknown"
    resource_source: str = "unknown"


@dataclass
class SystemState:
    timestamp: float = 0.0
    arrival_rate: float = 0.0
    response_time_avg_sec: float = 0.0
    predicted_queue_length: float = 0.0  # projection: lambda * T

    num_servers: int = 0
    service_rate_per_server: float = 0.0
    utilization: float = 0.0
    is_stable: bool = False

    # If you later add measured N, compute error; otherwise keep 0.
    little_law_error: float = 0.0
    in_steady_state: bool = False


@dataclass
class ScalingDecision:
    timestamp: float
    service: str
    state: Tuple[int, int, int]

    rl_action: ScalingAction
    baseline_action: ScalingAction
    actual_action: ScalingAction
    decision_source: str

    metrics_snapshot: Dict
    system_state: Dict

    reward: float = 0.0
    latency_after_ms: float = 0.0


@dataclass
class TrainingState:
    phase: TrainingPhase = TrainingPhase.NOT_STARTED
    epoch: int = 0
    total_iterations: int = 0
    current_pattern: str = ""
    rl_authority: float = 0.0
    start_time: float = 0.0
    last_checkpoint_time: float = 0.0


# =============================================================================
# SECTION 3: METRICS COLLECTION
# =============================================================================

class LocustMetricsCollector:
    def __init__(self, locust_url: str = "http://localhost:8089"):
        self.url = locust_url
        self.logger = logging.getLogger('Locust')
        self._cache = {}
        self._cache_time = 0
        self._cache_ttl = 2.0

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stats/requests", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> Optional[Dict]:
        if time.time() - self._cache_time < self._cache_ttl:
            return self._cache

        try:
            r = requests.get(f"{self.url}/stats/requests", timeout=5)
            if r.status_code == 200:
                self._cache = r.json()
                self._cache_time = time.time()
                return self._cache
        except Exception as e:
            self.logger.warning(f"Failed to get Locust stats: {e}")
        return None

    def _get_avg_from_stats(self, stats: Dict) -> float:
        for endpoint in stats.get('stats', []):
            if endpoint.get('name') == 'Aggregated':
                return endpoint.get('avg_response_time', 0) or 0
        return 0

    def _get_max_from_stats(self, stats: Dict) -> float:
        for endpoint in stats.get('stats', []):
            if endpoint.get('name') == 'Aggregated':
                return endpoint.get('max_response_time', 0) or 0
        return 0

    def get_aggregate_latency(self) -> Dict:
        stats = self.get_stats()
        if not stats:
            return {}

        p95 = stats.get('current_response_time_percentile_95', 0) or 0
        return {
            'p50_ms': stats.get('current_response_time_percentile_50', 0) or 0,
            'p95_ms': p95,
            'p99_ms': (p95 * 1.3) if p95 else 0.0,
            'avg_ms': self._get_avg_from_stats(stats),
            'max_ms': self._get_max_from_stats(stats),
            'rps': stats.get('total_rps', 0) or 0,
            'fail_ratio': stats.get('fail_ratio', 0) or 0
        }

    def set_user_count(self, count: int) -> bool:
        try:
            r = requests.post(
                f"{self.url}/swarm",
                data={'user_count': count, 'spawn_rate': max(1, count // 10)},
                timeout=5
            )
            return r.status_code == 200
        except Exception:
            return False


class PrometheusMetricsCollector:
    def __init__(self, prometheus_url: str = "http://localhost:9090", namespace: str = "default"):
        self.url = prometheus_url
        self.namespace = namespace
        self.logger = logging.getLogger('Prometheus')

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/v1/status/config", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def query(self, promql: str) -> Optional[float]:
        try:
            r = requests.get(
                f"{self.url}/api/v1/query",
                params={'query': promql},
                timeout=10
            )
            if r.status_code != 200:
                return None

            data = r.json()
            if data.get('status') != 'success':
                return None

            result = data.get('data', {}).get('result', [])
            if not result:
                return None

            # PATCH: if multiple series return, take mean of values
            vals = []
            for item in result:
                try:
                    vals.append(float(item['value'][1]))
                except Exception:
                    continue
            return float(np.mean(vals)) if vals else None

        except Exception as e:
            self.logger.debug(f"Query failed: {promql[:80]}... - {e}")
            return None

    def get_cpu_utilization(self, deployment: str) -> Optional[float]:
        query = f'''
        sum(rate(container_cpu_usage_seconds_total{{
            namespace="{self.namespace}",
            pod=~"{deployment}.*",
            container!=""
        }}[2m]))
        /
        sum(kube_pod_container_resource_requests{{
            namespace="{self.namespace}",
            pod=~"{deployment}.*",
            resource="cpu"
        }})
        '''
        result = self.query(query)
        if result is not None:
            return min(result, 2.0)
        return None

    def get_memory_utilization(self, deployment: str) -> Optional[float]:
        query = f'''
        sum(container_memory_working_set_bytes{{
            namespace="{self.namespace}",
            pod=~"{deployment}.*",
            container!=""
        }})
        /
        sum(kube_pod_container_resource_requests{{
            namespace="{self.namespace}",
            pod=~"{deployment}.*",
            resource="memory"
        }})
        '''
        return self.query(query)

    def get_pod_counts(self, deployment: str) -> Tuple[int, int, int]:
        ready = self.query(f'kube_deployment_status_replicas_ready{{namespace="{self.namespace}", deployment="{deployment}"}}')
        current = self.query(f'kube_deployment_status_replicas{{namespace="{self.namespace}", deployment="{deployment}"}}')
        desired = self.query(f'kube_deployment_spec_replicas{{namespace="{self.namespace}", deployment="{deployment}"}}')

        return (int(ready) if ready else 1, int(current) if current else 1, int(desired) if desired else 1)

    def has_latency_metrics(self) -> Tuple[bool, str]:
        patterns = [
            ('istio_request_duration_milliseconds_bucket', 'Istio'),
            ('nginx_ingress_controller_request_duration_seconds_bucket', 'NGINX Ingress'),
            ('http_request_duration_seconds_bucket', 'Generic HTTP'),
        ]
        for metric, source in patterns:
            result = self.query(f'count({metric})')
            if result and result > 0:
                return True, source
        return False, "None (use Locust for latency)"


class KubectlMetricsCollector:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.logger = logging.getLogger('Kubectl')

    def get_pod_counts(self, deployment: str) -> Tuple[int, int, int]:
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'deployment', deployment, '-n', self.namespace,
                 '-o', 'jsonpath={.status.readyReplicas},{.status.replicas},{.spec.replicas}'],
                capture_output=True, text=True, timeout=10
            )
            parts = result.stdout.strip().split(',')
            ready = int(parts[0]) if parts[0] else 1
            current = int(parts[1]) if len(parts) > 1 and parts[1] else ready
            desired = int(parts[2]) if len(parts) > 2 and parts[2] else current
            return (ready, current, desired)
        except Exception:
            return (1, 1, 1)

    def scale_deployment(self, deployment: str, replicas: int) -> bool:
        try:
            result = subprocess.run(
                ['kubectl', 'scale', 'deployment', deployment,
                 '-n', self.namespace, f'--replicas={replicas}'],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Failed to scale {deployment}: {e}")
            return False


class UnifiedMetricsCollector:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.logger = logging.getLogger('Metrics')

        self.locust = LocustMetricsCollector(config.locust_url)
        self.prometheus = PrometheusMetricsCollector(config.prometheus_url, config.namespace)
        self.kubectl = KubectlMetricsCollector(config.namespace)

        self.locust_available = self.locust.is_available()
        self.prometheus_available = self.prometheus.is_available()

        self._smoothed: Dict[str, ServiceMetrics] = {}

        self.logger.info(f"Locust available: {self.locust_available}")
        self.logger.info(f"Prometheus available: {self.prometheus_available}")

    def collect(self, service: str) -> ServiceMetrics:
        metrics = ServiceMetrics(timestamp=time.time(), service_name=service)

        if self.locust_available:
            latency = self.locust.get_aggregate_latency()
            if latency:
                metrics.latency_p50_ms = latency.get('p50_ms', 0)
                metrics.latency_p95_ms = latency.get('p95_ms', 0)
                metrics.latency_p99_ms = latency.get('p99_ms', 0)
                metrics.latency_avg_ms = latency.get('avg_ms', 0)
                metrics.latency_max_ms = latency.get('max_ms', 0)
                metrics.arrival_rate = latency.get('rps', 0)
                metrics.failure_rate = latency.get('fail_ratio', 0)
                metrics.latency_source = 'locust'

        if self.prometheus_available:
            cpu = self.prometheus.get_cpu_utilization(service)
            if cpu is not None:
                metrics.cpu_utilization = cpu

            mem = self.prometheus.get_memory_utilization(service)
            if mem is not None:
                metrics.memory_utilization = mem

            ready, current, desired = self.prometheus.get_pod_counts(service)
            metrics.ready_replicas = ready
            metrics.current_replicas = current
            metrics.desired_replicas = desired
            metrics.resource_source = 'prometheus'
        else:
            ready, current, desired = self.kubectl.get_pod_counts(service)
            metrics.ready_replicas = ready
            metrics.current_replicas = current
            metrics.desired_replicas = desired
            metrics.resource_source = 'kubectl'

        return self._apply_smoothing(service, metrics)

    def _apply_smoothing(self, service: str, new: ServiceMetrics) -> ServiceMetrics:
        if service not in self._smoothed:
            self._smoothed[service] = new
            return new

        old = self._smoothed[service]
        alpha = self.config.smoothing_alpha

        for field_name in ['latency_p50_ms', 'latency_p95_ms', 'latency_avg_ms',
                           'arrival_rate', 'cpu_utilization']:
            old_val = getattr(old, field_name, 0) or 0
            new_val = getattr(new, field_name, 0) or 0
            if new_val > 0:
                setattr(new, field_name, alpha * new_val + (1 - alpha) * old_val)

        self._smoothed[service] = new
        return new

    def collect_all(self) -> Dict[str, ServiceMetrics]:
        return {svc: self.collect(svc) for svc in self.config.services.keys()}


# =============================================================================
# SECTION 4: LITTLE'S LAW (Projection + Stability)
# =============================================================================

class LittleLawValidator:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.logger = logging.getLogger('LittleLaw')
        self.history: Dict[str, deque] = {}

    def validate(self, service: str, metrics: ServiceMetrics, service_capacity: float = None) -> SystemState:
        if service_capacity is None:
            svc_config = self.config.services.get(service)
            service_capacity = svc_config.capacity_per_pod if svc_config else 100.0

        lambda_rate = metrics.arrival_rate
        response_time_sec = metrics.latency_avg_ms / 1000.0
        num_pods = max(1, metrics.ready_replicas)

        predicted_queue = lambda_rate * response_time_sec  # projection
        total_capacity = num_pods * service_capacity
        utilization = lambda_rate / total_capacity if total_capacity > 0 else 1.0
        is_stable = utilization < self.config.stability_threshold

        in_steady_state, _ = self._check_steady_state(service)

        state = SystemState(
            timestamp=time.time(),
            arrival_rate=lambda_rate,
            response_time_avg_sec=response_time_sec,
            predicted_queue_length=predicted_queue,
            num_servers=num_pods,
            service_rate_per_server=service_capacity,
            utilization=utilization,
            is_stable=is_stable,
            little_law_error=0.0,   # no measured N in this implementation
            in_steady_state=in_steady_state
        )

        if service not in self.history:
            self.history[service] = deque(maxlen=30)
        self.history[service].append(state)

        return state

    def _check_steady_state(self, service: str) -> Tuple[bool, str]:
        if service not in self.history or len(self.history[service]) < 5:
            return False, "Insufficient history"

        history = list(self.history[service])[-10:]

        rates = [s.arrival_rate for s in history if s.arrival_rate > 0]
        if rates:
            mean = np.mean(rates)
            if mean > 0:
                cv = np.std(rates) / mean
                if cv > 0.3:
                    return False, f"Arrival rate unstable (CV={cv:.2f})"

        times = [s.response_time_avg_sec for s in history if s.response_time_avg_sec > 0]
        if times:
            mean = np.mean(times)
            if mean > 0:
                cv = np.std(times) / mean
                if cv > 0.4:
                    return False, f"Response time unstable (CV={cv:.2f})"

        utils = [s.utilization for s in history]
        if len(utils) >= 5:
            x = np.arange(len(utils))
            slope = np.polyfit(x, utils, 1)[0]
            if slope > 0.01:
                return False, f"Utilization trending up (slope={slope:.3f})"

        return True, "System in steady-state"

    def calculate_optimal_pods(self, arrival_rate: float, service_capacity: float, target_utilization: float = None) -> Tuple[int, Dict]:
        if target_utilization is None:
            target_utilization = self.config.target_utilization
        if service_capacity <= 0:
            return 1, {}

        min_stable = math.ceil(arrival_rate / service_capacity) if arrival_rate > 0 else 1
        min_stable = max(1, min_stable)

        optimal = math.ceil(arrival_rate / (target_utilization * service_capacity)) if arrival_rate > 0 else 1
        optimal = max(1, optimal, min_stable)

        max_rps = optimal * service_capacity * self.config.stability_threshold
        return optimal, {
            'min_stable_pods': min_stable,
            'optimal_pods': optimal,
            'max_sustainable_rps': max_rps,
            'at_utilization': target_utilization
        }


# =============================================================================
# SECTION 5: DOUBLE Q-LEARNING
# =============================================================================

def _state_to_key(state: Tuple[int, int, int]) -> str:
    return ",".join(str(x) for x in state)

def _key_to_state(key: str) -> Tuple[int, int, int]:
    parts = key.split(",")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


class DoubleQLearningAgent:
    def __init__(self, service: str, config: Config = CONFIG):
        self.service = service
        self.config = config
        self.logger = logging.getLogger(f'RL:{service}')

        self.state_dims = (2, 2, 2)
        self.n_actions = 3

        self.q_table_A: Dict[Tuple[int, int, int], List[float]] = defaultdict(lambda: [0.0] * self.n_actions)
        self.q_table_B: Dict[Tuple[int, int, int], List[float]] = defaultdict(lambda: [0.0] * self.n_actions)
        self.use_table_A = True

        self.alpha = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay

        self.replay_buffer: deque = deque(maxlen=500)
        self.batch_size = 16

        self.total_updates = 0
        self.state_visits: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.action_counts: Dict[int, int] = defaultdict(int)

    def discretize_state(self, cpu_util: float, latency_ratio: float, pod_ratio: float) -> Tuple[int, int, int]:
        cpu_level = 1 if cpu_util > 0.5 else 0
        latency_level = 1 if latency_ratio > 1.0 else 0
        pod_level = 1 if pod_ratio > 0.5 else 0
        return (cpu_level, latency_level, pod_level)

    def choose_action(self, state: Tuple[int, int, int], training: bool = True) -> ScalingAction:
        self.state_visits[state] += 1

        if training and random.random() < self.epsilon:
            action = ScalingAction(random.randint(0, self.n_actions - 1))
        else:
            q_A = self.q_table_A[state]
            q_B = self.q_table_B[state]
            q_avg = [(a + b) / 2 for a, b in zip(q_A, q_B)]
            action = ScalingAction(int(np.argmax(q_avg)))

        self.action_counts[action.value] += 1
        return action

    def update(self, state: Tuple[int, int, int], action: ScalingAction, reward: float, next_state: Tuple[int, int, int]):
        noisy_reward = reward + float(np.random.normal(0, 0.1))
        self.replay_buffer.append((state, action.value, noisy_reward, next_state))

        if self.use_table_A:
            best_action = int(np.argmax(self.q_table_A[next_state]))
            target = noisy_reward + self.gamma * self.q_table_B[next_state][best_action]
            old_q = self.q_table_A[state][action.value]
            self.q_table_A[state][action.value] = old_q + self.alpha * (target - old_q)
        else:
            best_action = int(np.argmax(self.q_table_B[next_state]))
            target = noisy_reward + self.gamma * self.q_table_A[next_state][best_action]
            old_q = self.q_table_B[state][action.value]
            self.q_table_B[state][action.value] = old_q + self.alpha * (target - old_q)

        self.use_table_A = not self.use_table_A
        self.total_updates += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        indices = random.sample(range(len(self.replay_buffer)), self.batch_size)
        replay_alpha = self.alpha * 0.5

        for idx in indices:
            state, action, reward, next_state = self.replay_buffer[idx]
            if self.use_table_A:
                best_action = int(np.argmax(self.q_table_A[next_state]))
                target = reward + self.gamma * self.q_table_B[next_state][best_action]
                old_q = self.q_table_A[state][action]
                self.q_table_A[state][action] = old_q + replay_alpha * (target - old_q)
            else:
                best_action = int(np.argmax(self.q_table_B[next_state]))
                target = reward + self.gamma * self.q_table_A[next_state][best_action]
                old_q = self.q_table_B[state][action]
                self.q_table_B[state][action] = old_q + replay_alpha * (target - old_q)

    def get_stats(self) -> Dict:
        return {
            'service': self.service,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'states_discovered': len(self.state_visits),
            'state_visits': { _state_to_key(k): v for k, v in self.state_visits.items() },
            'action_counts': dict(self.action_counts),
            'replay_buffer_size': len(self.replay_buffer),
        }

    def save(self, path: str):
        data = {
            'service': self.service,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'state_visits': { _state_to_key(k): v for k, v in self.state_visits.items() },
            'action_counts': dict(self.action_counts),
            'q_table_A': { _state_to_key(k): v for k, v in self.q_table_A.items() },
            'q_table_B': { _state_to_key(k): v for k, v in self.q_table_B.items() },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        self.epsilon = data.get('epsilon', self.epsilon)
        self.total_updates = data.get('total_updates', 0)

        for k, v in data.get('q_table_A', {}).items():
            self.q_table_A[_key_to_state(k)] = v
        for k, v in data.get('q_table_B', {}).items():
            self.q_table_B[_key_to_state(k)] = v
        for k, v in data.get('state_visits', {}).items():
            self.state_visits[_key_to_state(k)] = v


# =============================================================================
# SECTION 6: BASELINE
# =============================================================================

class BaselineHPAController:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.logger = logging.getLogger('BaselineHPA')

    def decide(self, service: str, cpu_util: float, ready_pods: int) -> ScalingAction:
        svc_config = self.config.services.get(service)
        if not svc_config:
            return ScalingAction.STAY

        target_cpu = svc_config.target_cpu_percent / 100.0
        scale_up_threshold = target_cpu * 1.4
        scale_down_threshold = target_cpu * 0.6

        if cpu_util > scale_up_threshold and ready_pods < svc_config.max_replicas:
            return ScalingAction.SCALE_UP
        if cpu_util < scale_down_threshold and ready_pods > svc_config.min_replicas:
            return ScalingAction.SCALE_DOWN
        return ScalingAction.STAY


# =============================================================================
# SECTION 7: REWARD
# =============================================================================

class RewardCalculator:
    def __init__(self, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)):
        self.w_sla, self.w_eff, self.w_stab = weights

    def calculate(self, latency_ratio: float, cpu_util: float, action: ScalingAction, pods_ready_ratio: float) -> float:
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

        if 0.5 <= cpu_util <= 0.7:
            r_eff = 1.0
        elif 0.3 <= cpu_util <= 0.85:
            r_eff = 0.3
        elif cpu_util < 0.1:
            r_eff = -0.5
        else:
            r_eff = -0.3

        if pods_ready_ratio < 1.0 and action != ScalingAction.STAY:
            r_stab = -1.0
        elif action == ScalingAction.SCALE_UP:
            r_stab = -0.2
        elif action == ScalingAction.SCALE_DOWN:
            r_stab = -0.1
        else:
            r_stab = 0.1

        reward = (self.w_sla * r_sla + self.w_eff * r_eff + self.w_stab * r_stab)
        return float(max(-2.0, min(2.0, reward)))


# =============================================================================
# SECTION 8: MAPE-K CONTROLLER
# =============================================================================

@dataclass
class PendingScale:
    start_time: float
    prev_state: Tuple[int, int, int]
    prev_action: ScalingAction
    target_replicas: int


class MAPEKController:
    def __init__(self, config: Config = CONFIG, mode: ControlMode = ControlMode.SHADOW, dry_run: bool = False):
        self.config = config
        self.mode = mode
        self.dry_run = dry_run

        self.logger = logging.getLogger('MAPE-K')

        self.metrics = UnifiedMetricsCollector(config)
        self.little_law = LittleLawValidator(config)
        self.baseline = BaselineHPAController(config)
        self.reward_calc = RewardCalculator()
        self.kubectl = KubectlMetricsCollector(config.namespace)

        self.agents: Dict[str, DoubleQLearningAgent] = {
            svc: DoubleQLearningAgent(svc, config) for svc in config.services.keys()
        }

        # PATCH: learning policy by mode (SHADOW: disabled)
        self.learning_enabled = False
        self.rl_authority = 0.0
        self.set_mode(mode)

        self.previous_states: Dict[str, Tuple[int, int, int]] = {}
        self.previous_actions: Dict[str, ScalingAction] = {}
        self.previous_metrics: Dict[str, ServiceMetrics] = {}

        self.pending_scale: Dict[str, PendingScale] = {}

        self.last_scale_time: Dict[str, float] = {}
        self.cooldown_sec = config.cooldown_sec

        self.decision_history: List[ScalingDecision] = []
        self.metrics_history: List[Dict] = []

        self.shadow_stats = {'agreements': 0, 'disagreements': 0}

    def set_mode(self, mode: ControlMode, rl_authority: float = None):
        self.mode = mode
        if mode == ControlMode.SHADOW:
            self.rl_authority = 0.0
            self.learning_enabled = False  # PATCH: avoid off-policy learning
        elif mode == ControlMode.ACTIVE:
            self.rl_authority = 1.0
            self.learning_enabled = True
        elif mode == ControlMode.EVALUATION:
            self.rl_authority = 1.0
            self.learning_enabled = False
        elif mode == ControlMode.HYBRID:
            self.rl_authority = float(rl_authority if rl_authority is not None else 0.5)
            self.learning_enabled = True

        self.logger.info(f"Mode: {mode.value}, RL authority: {self.rl_authority:.0%}, Learning: {self.learning_enabled}")

    def control_loop_iteration(self) -> Dict[str, ScalingDecision]:
        decisions: Dict[str, ScalingDecision] = {}
        for service, svc_config in self.config.services.items():
            try:
                d = self._process_service(service, svc_config)
                if d:
                    decisions[service] = d
            except Exception as e:
                self.logger.error(f"Error processing {service}: {e}")
        return decisions

    def _process_service(self, service: str, svc_config: ServiceConfig) -> Optional[ScalingDecision]:
        # MONITOR
        metrics = self.metrics.collect(service)
        self.metrics_history.append({'timestamp': time.time(), 'service': service, **asdict(metrics)})

        # ANALYZE
        system_state = self.little_law.validate(service, metrics, svc_config.capacity_per_pod)
        latency_ratio = (metrics.latency_avg_ms / svc_config.target_latency_ms) if metrics.latency_avg_ms > 0 else 0.0
        pod_ratio = (metrics.ready_replicas / svc_config.max_replicas) if svc_config.max_replicas > 0 else 0.0

        state = self.agents[service].discretize_state(metrics.cpu_utilization, latency_ratio, pod_ratio)

        # PLAN
        rl_action = self.agents[service].choose_action(state, training=self.learning_enabled)
        baseline_action = self.baseline.decide(service, metrics.cpu_utilization, metrics.ready_replicas)
        actual_action, decision_source = self._select_action(rl_action, baseline_action)

        if rl_action == baseline_action:
            self.shadow_stats['agreements'] += 1
        else:
            self.shadow_stats['disagreements'] += 1

        if not self._can_scale(service, actual_action):
            actual_action = ScalingAction.STAY

        # EXECUTE (and register pending settle)
        if actual_action != ScalingAction.STAY:
            new_replicas = self._compute_new_replicas(actual_action, metrics.ready_replicas, svc_config)
            executed = self._execute_scaling(service, metrics.ready_replicas, new_replicas)
            if executed:
                # PATCH: register pending scaling for delayed reward
                self.pending_scale[service] = PendingScale(
                    start_time=time.time(),
                    prev_state=self.previous_states.get(service, state),
                    prev_action=actual_action,
                    target_replicas=new_replicas
                )

        # KNOWLEDGE: delayed update logic
        self._maybe_update_learning(service, svc_config, metrics, state, latency_ratio)

        # Store for next iteration (state and action actually applied)
        self.previous_states[service] = state
        self.previous_actions[service] = actual_action
        self.previous_metrics[service] = metrics

        decision = ScalingDecision(
            timestamp=time.time(),
            service=service,
            state=state,
            rl_action=rl_action,
            baseline_action=baseline_action,
            actual_action=actual_action,
            decision_source=decision_source,
            metrics_snapshot=asdict(metrics),
            system_state=asdict(system_state)
        )
        self.decision_history.append(decision)
        self._log_decision(service, state, rl_action, baseline_action, actual_action, system_state, metrics)
        return decision

    def _select_action(self, rl_action: ScalingAction, baseline_action: ScalingAction) -> Tuple[ScalingAction, str]:
        if self.mode == ControlMode.SHADOW:
            return baseline_action, 'baseline'
        if self.mode in (ControlMode.ACTIVE, ControlMode.EVALUATION):
            return rl_action, 'rl'
        if self.mode == ControlMode.HYBRID:
            return (rl_action, 'rl') if random.random() < self.rl_authority else (baseline_action, 'baseline')
        return baseline_action, 'baseline'

    def _can_scale(self, service: str, action: ScalingAction) -> bool:
        if action == ScalingAction.STAY:
            return True
        last = self.last_scale_time.get(service, 0.0)
        return (time.time() - last) >= self.cooldown_sec

    def _compute_new_replicas(self, action: ScalingAction, current_pods: int, cfg: ServiceConfig) -> int:
        if action == ScalingAction.SCALE_UP:
            return min(current_pods + 1, cfg.max_replicas)
        if action == ScalingAction.SCALE_DOWN:
            return max(current_pods - 1, cfg.min_replicas)
        return current_pods

    def _execute_scaling(self, service: str, current_pods: int, new_replicas: int) -> bool:
        if new_replicas == current_pods:
            return False

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would scale {service}: {current_pods} -> {new_replicas}")
            self.last_scale_time[service] = time.time()
            return True

        ok = self.kubectl.scale_deployment(service, new_replicas)
        if ok:
            self.logger.info(f"Scaled {service}: {current_pods} -> {new_replicas}")
            self.last_scale_time[service] = time.time()
            return True

        self.logger.error(f"Failed to scale {service}")
        return False

    def _is_settled(self, metrics: ServiceMetrics, target_replicas: int) -> bool:
        tol = int(self.config.readiness_tolerance)
        return metrics.ready_replicas >= (target_replicas - tol)

    def _maybe_update_learning(self, service: str, svc_config: ServiceConfig, metrics: ServiceMetrics,
                              current_state: Tuple[int, int, int], latency_ratio: float):
        if not self.learning_enabled:
            return

        # If we have a pending scale, wait until settled or timeout
        if service in self.pending_scale:
            pend = self.pending_scale[service]
            elapsed = time.time() - pend.start_time
            settled = self._is_settled(metrics, pend.target_replicas)
            timed_out = elapsed >= self.config.scaling_settle_timeout_sec

            if not (settled or timed_out):
                return  # skip update until we can attribute outcome

            # Now compute reward and update using pend.prev_state/action -> current_state
            reward = self.reward_calc.calculate(
                latency_ratio=latency_ratio,
                cpu_util=metrics.cpu_utilization,
                action=pend.prev_action,
                pods_ready_ratio=(metrics.ready_replicas / max(1, metrics.current_replicas))
            )
            self.agents[service].update(pend.prev_state, pend.prev_action, reward, current_state)
            self.agents[service].replay()

            del self.pending_scale[service]
            return

        # No pending scale; safe to do one-step update if we have previous state
        if service in self.previous_states and service in self.previous_actions:
            prev_state = self.previous_states[service]
            prev_action = self.previous_actions[service]
            reward = self.reward_calc.calculate(
                latency_ratio=latency_ratio,
                cpu_util=metrics.cpu_utilization,
                action=prev_action,
                pods_ready_ratio=(metrics.ready_replicas / max(1, metrics.current_replicas))
            )
            self.agents[service].update(prev_state, prev_action, reward, current_state)
            self.agents[service].replay()

    def _log_decision(self, service: str, state: Tuple[int, int, int], rl: ScalingAction, baseline: ScalingAction,
                      actual: ScalingAction, sys_state: SystemState, metrics: ServiceMetrics):
        stable_mark = "OK" if sys_state.is_stable else "UNSTABLE"
        self.logger.info(
            f"[{service}] S={state} Act={actual.name} "
            f"(RL={rl.name} Base={baseline.name}) "
            f"rho={sys_state.utilization:.1%} {stable_mark} "
            f"P95={metrics.latency_p95_ms:.0f}ms CPU={metrics.cpu_utilization:.1%} Pods={metrics.ready_replicas}"
        )

    def get_shadow_analysis(self) -> Dict:
        total = self.shadow_stats['agreements'] + self.shadow_stats['disagreements']
        if total == 0:
            return {'agreement_rate': 0.0, 'total_decisions': 0}
        return {
            'agreement_rate': self.shadow_stats['agreements'] / total,
            'total_decisions': total,
            'agreements': self.shadow_stats['agreements'],
            'disagreements': self.shadow_stats['disagreements']
        }

    def save_state(self, directory: str):
        os.makedirs(directory, exist_ok=True)

        for service, agent in self.agents.items():
            agent.save(os.path.join(directory, f"{service}_agent.json"))

        with open(os.path.join(directory, "shadow_stats.json"), 'w') as f:
            json.dump(self.shadow_stats, f, indent=2)

        decisions_data = []
        for d in self.decision_history[-1000:]:
            dd = asdict(d)
            dd['state'] = _state_to_key(d.state)
            dd['rl_action'] = d.rl_action.name
            dd['baseline_action'] = d.baseline_action.name
            dd['actual_action'] = d.actual_action.name
            decisions_data.append(dd)

        with open(os.path.join(directory, "decision_history.json"), 'w') as f:
            json.dump(decisions_data, f, indent=2)

        with open(os.path.join(directory, "metrics_history.json"), 'w') as f:
            json.dump(self.metrics_history[-5000:], f, indent=2)

        self.logger.info(f"State saved to {directory}")

    def load_state(self, directory: str):
        for service, agent in self.agents.items():
            path = os.path.join(directory, f"{service}_agent.json")
            if os.path.exists(path):
                agent.load(path)

        shadow_path = os.path.join(directory, "shadow_stats.json")
        if os.path.exists(shadow_path):
            with open(shadow_path, 'r') as f:
                self.shadow_stats = json.load(f)

        self.logger.info(f"State loaded from {directory}")


# =============================================================================
# SECTION 9: LOCUST MANAGER
# =============================================================================

class LocustManager:
    def __init__(self, locust_url: str = "http://localhost:8089"):
        self.url = locust_url
        self.logger = logging.getLogger('LocustMgr')

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stats/requests", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def set_users(self, count: int, spawn_rate: int = None) -> bool:
        if spawn_rate is None:
            spawn_rate = max(1, count // 10)
        try:
            r = requests.post(
                f"{self.url}/swarm",
                data={'user_count': count, 'spawn_rate': spawn_rate},
                timeout=5
            )
            return r.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to set users: {e}")
            return False

    def stop(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stop", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def reset_stats(self) -> bool:
        try:
            r = requests.get(f"{self.url}/stats/reset", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# =============================================================================
# SECTION 10: TRAINING ORCHESTRATOR
# =============================================================================

class TrainingOrchestrator:
    def __init__(self, config: Config = CONFIG, checkpoint_dir: str = "./checkpoints", dry_run: bool = False):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.dry_run = dry_run

        self.controller = MAPEKController(config, ControlMode.SHADOW, dry_run)
        self.locust = LocustManager(config.locust_url)
        self.logger = logging.getLogger('Training')

        self.running = False
        self.training_state = TrainingState()
        self.results: List[Dict] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        self.logger.info("Shutdown signal received, saving checkpoint...")
        self.running = False
        self.save_checkpoint()

    def save_checkpoint(self, label: str = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if label:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_{label}_{timestamp}")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{self.training_state.epoch}_{timestamp}")

        os.makedirs(path, exist_ok=True)
        self.controller.save_state(path)

        state_data = {
            'phase': self.training_state.phase.value,
            'epoch': self.training_state.epoch,
            'total_iterations': self.training_state.total_iterations,
            'rl_authority': self.training_state.rl_authority,
            'results': self.results[-100:]
        }
        with open(os.path.join(path, "training_state.json"), 'w') as f:
            json.dump(state_data, f, indent=2)

        self.training_state.last_checkpoint_time = time.time()
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        self.controller.load_state(path)
        state_file = os.path.join(path, "training_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                data = json.load(f)
            self.training_state.phase = TrainingPhase(data.get('phase', 'shadow'))
            self.training_state.epoch = int(data.get('epoch', 0))
            self.training_state.total_iterations = int(data.get('total_iterations', 0))
            self.training_state.rl_authority = float(data.get('rl_authority', 0.0))

        self.logger.info(f"Loaded checkpoint: epoch={self.training_state.epoch}, phase={self.training_state.phase.value}")

    def run_pattern(self, pattern_name: str, duration_sec: float) -> Dict:
        if pattern_name == 'random':
            pattern = [50 + random.randint(-30, 400) for _ in range(8)]
        else:
            pattern = LOAD_PATTERNS.get(pattern_name, [100, 200, 300, 200, 100])

        step_duration = duration_sec / len(pattern)
        pattern_metrics = []

        self.logger.info(f"Running pattern: {pattern_name} ({len(pattern)} steps, {step_duration:.0f}s each)")

        for step_idx, users in enumerate(pattern):
            if not self.running:
                break

            if self.locust.is_running():
                self.locust.set_users(users)

            step_start = time.time()
            while self.running and (time.time() - step_start) < step_duration:
                decisions = self.controller.control_loop_iteration()
                self.training_state.total_iterations += 1

                for svc, d in decisions.items():
                    pattern_metrics.append({
                        'timestamp': time.time(),
                        'pattern': pattern_name,
                        'step': step_idx,
                        'users': users,
                        'service': svc,
                        'latency_p95': d.metrics_snapshot.get('latency_p95_ms', 0),
                        'cpu': d.metrics_snapshot.get('cpu_utilization', 0),
                        'action': d.actual_action.name
                    })

                time.sleep(self.config.control_interval_sec)

            self.training_state.epoch += 1

        if pattern_metrics:
            latencies = [m['latency_p95'] for m in pattern_metrics if m['latency_p95'] > 0]
            return {
                'pattern': pattern_name,
                'phase': self.training_state.phase.value,
                'samples': len(pattern_metrics),
                'latency_p95_mean': float(np.mean(latencies)) if latencies else 0.0,
                'latency_p95_max': float(np.max(latencies)) if latencies else 0.0,
                'latency_p95_std': float(np.std(latencies)) if latencies else 0.0
            }

        return {'pattern': pattern_name, 'samples': 0}

    def run_phase(self, phase: TrainingPhase, duration_sec: float, patterns: List[str],
                  rl_authority: float = 0.0, learning: bool = True) -> List[Dict]:
        self.training_state.phase = phase
        self.training_state.rl_authority = rl_authority

        if phase == TrainingPhase.SHADOW:
            self.controller.set_mode(ControlMode.SHADOW)
        elif phase in (TrainingPhase.HYBRID_25, TrainingPhase.HYBRID_50, TrainingPhase.HYBRID_75):
            self.controller.set_mode(ControlMode.HYBRID, rl_authority)
        elif phase == TrainingPhase.ACTIVE:
            self.controller.set_mode(ControlMode.ACTIVE)
        elif phase == TrainingPhase.EVALUATION:
            self.controller.set_mode(ControlMode.EVALUATION)

        # Respect explicit learning flag (though SHADOW will override to False)
        if self.controller.mode != ControlMode.SHADOW:
            self.controller.learning_enabled = bool(learning)

        phase_results = []
        pattern_duration = duration_sec / max(1, len(patterns))

        self.logger.info(f"PHASE: {phase.value.upper()} Duration: {duration_sec/3600:.2f}h Patterns: {patterns}")
        self.logger.info(f"RL Authority: {rl_authority:.0%} Learning: {self.controller.learning_enabled}")

        for pattern_name in patterns:
            if not self.running:
                break
            result = self.run_pattern(pattern_name, pattern_duration)
            phase_results.append(result)
            self.results.append(result)
            self.logger.info(f"Pattern {pattern_name}: P95={result.get('latency_p95_mean', 0):.1f}ms")

        return phase_results

    def run_full_training(self, total_duration_hours: float = 8.0,
                          shadow_fraction: float = 0.30, hybrid_fraction: float = 0.25,
                          active_fraction: float = 0.25, eval_fraction: float = 0.20):
        self.running = True
        self.training_state.start_time = time.time()
        total_sec = total_duration_hours * 3600.0

        try:
            self.run_phase(TrainingPhase.SHADOW, total_sec * shadow_fraction,
                           ['warmup'] + self.config.train_patterns * 3,
                           rl_authority=0.0, learning=False)
            self.save_checkpoint("shadow_complete")
            if not self.running:
                return

            for authority, phase in [(0.25, TrainingPhase.HYBRID_25),
                                     (0.50, TrainingPhase.HYBRID_50),
                                     (0.75, TrainingPhase.HYBRID_75)]:
                if not self.running:
                    break
                self.run_phase(phase, total_sec * hybrid_fraction / 3.0,
                               self.config.train_patterns, rl_authority=authority, learning=True)
            self.save_checkpoint("hybrid_complete")
            if not self.running:
                return

            self.run_phase(TrainingPhase.ACTIVE, total_sec * active_fraction,
                           self.config.train_patterns * 3, rl_authority=1.0, learning=True)
            self.save_checkpoint("active_complete")
            if not self.running:
                return

            self.run_phase(TrainingPhase.EVALUATION, total_sec * eval_fraction,
                           self.config.test_patterns * 3, rl_authority=1.0, learning=False)

            self.training_state.phase = TrainingPhase.COMPLETED
            self.save_checkpoint("final")

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.save_checkpoint("error")
            raise
        finally:
            self.running = False


# =============================================================================
# SECTION 11: DIAGNOSTICS
# =============================================================================

def run_diagnostics(config: Config):
    print("\n" + "="*60)
    print("CAPA+ SYSTEM DIAGNOSTICS")
    print("="*60)

    print("\n--- Prometheus ---")
    prom = PrometheusMetricsCollector(config.prometheus_url, config.namespace)
    if prom.is_available():
        print(f"Connected to {config.prometheus_url}")
        has_latency, source = prom.has_latency_metrics()
        print(f"Latency metrics: {'YES' if has_latency else 'NO'} ({source})")
    else:
        print(f"Cannot connect to {config.prometheus_url}")

    print("\n--- Locust ---")
    locust = LocustMetricsCollector(config.locust_url)
    if locust.is_available():
        print(f"Connected to {config.locust_url}")
        stats = locust.get_aggregate_latency()
        print(f"RPS: {stats.get('rps', 0):.1f} P50: {stats.get('p50_ms', 0):.1f}ms P95: {stats.get('p95_ms', 0):.1f}ms")
    else:
        print(f"Cannot connect to {config.locust_url}")

    print("\n--- Kubernetes ---")
    try:
        result = subprocess.run(['kubectl', 'get', 'nodes', '-o', 'wide'],
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("kubectl connected")
            print(result.stdout)
        else:
            print("kubectl error")
            print(result.stderr)
    except Exception as e:
        print(f"Kubernetes error: {e}")

    print(f"\n--- Deployments in '{config.namespace}' ---")
    try:
        result = subprocess.run(['kubectl', 'get', 'deployments', '-n', config.namespace],
                               capture_output=True, text=True, timeout=10)
        print(result.stdout if result.returncode == 0 else result.stderr)
    except Exception as e:
        print(f"Error: {e}")


# =============================================================================
# SECTION 12: CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CAPA+ Unified Experiment Framework (Patched)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--prometheus', default='http://localhost:9090', help='Prometheus URL')
    common.add_argument('--locust', default='http://localhost:8089', help='Locust URL')
    common.add_argument('--namespace', '-n', default='default', help='Kubernetes namespace')

    subparsers.add_parser('diagnose', parents=[common], help='Run diagnostics')

    val_parser = subparsers.add_parser('validate', parents=[common], help='Validate Little\'s Law (projection)')
    val_parser.add_argument('--continuous', '-c', action='store_true', help='Run continuously')
    val_parser.add_argument('--service', '-s', default='frontend', help='Service to validate')
    val_parser.add_argument('--interval', type=int, default=5, help='Interval in seconds')

    train_parser = subparsers.add_parser('train', parents=[common], help='Run training')
    train_parser.add_argument('--mode', choices=['shadow', 'hybrid', 'active', 'full'],
                              default='shadow', help='Training mode')
    train_parser.add_argument('--duration', type=int, default=3600, help='Duration in seconds')
    train_parser.add_argument('--dry-run', action='store_true', help='Simulate scaling')
    train_parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint path')

    analyze_parser = subparsers.add_parser('analyze', parents=[common], help='Analyze results')
    analyze_parser.add_argument('--data-dir', default='./checkpoints', help='Data directory')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    CONFIG.prometheus_url = args.prometheus
    CONFIG.locust_url = args.locust
    CONFIG.namespace = args.namespace

    if args.command == 'diagnose':
        run_diagnostics(CONFIG)
        return

    if args.command == 'validate':
        validator = LittleLawValidator(CONFIG)
        collector = UnifiedMetricsCollector(CONFIG)

        if args.continuous:
            print("Press Ctrl+C to stop\n")
            try:
                while True:
                    metrics = collector.collect(args.service)
                    svc_config = CONFIG.services.get(args.service)
                    capacity = svc_config.capacity_per_pod if svc_config else 100
                    state = validator.validate(args.service, metrics, capacity)
                    steady, reason = validator._check_steady_state(args.service)
                    status = "STABLE" if state.is_stable else "UNSTABLE"

                    print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
                    print(f"Status: {status}")
                    print(f"lambda (arrival): {state.arrival_rate:.1f} req/s")
                    print(f"T (response):     {state.response_time_avg_sec*1000:.1f} ms")
                    print(f"N = lambda*T:     {state.predicted_queue_length:.1f}")
                    print(f"rho (util):       {state.utilization:.1%}")
                    print(f"Pods (c):         {state.num_servers}")
                    print(f"Steady-state:     {reason}")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
        else:
            metrics = collector.collect(args.service)
            state = validator.validate(args.service, metrics, 100)
            print(json.dumps(asdict(state), indent=2, default=str))
        return

    if args.command == 'train':
        orchestrator = TrainingOrchestrator(CONFIG, checkpoint_dir=args.checkpoint_dir, dry_run=args.dry_run)
        if args.resume:
            orchestrator.load_checkpoint(args.resume)

        if args.mode == 'full':
            orchestrator.run_full_training(args.duration / 3600.0)
            return

        # PATCH: correct mapping for non-full modes
        mode_to_phase = {
            'shadow': TrainingPhase.SHADOW,
            'hybrid': TrainingPhase.HYBRID_50,
            'active': TrainingPhase.ACTIVE,
        }
        mode_to_ctrl = {
            'shadow': (ControlMode.SHADOW, 0.0, False),
            'hybrid': (ControlMode.HYBRID, 0.5, True),
            'active': (ControlMode.ACTIVE, 1.0, True),
        }

        phase = mode_to_phase[args.mode]
        ctrl_mode, authority, learning = mode_to_ctrl[args.mode]

        orchestrator.running = True
        # set controller mode explicitly
        orchestrator.controller.set_mode(ctrl_mode, authority)

        patterns = (CONFIG.train_patterns * 3) if args.mode != 'shadow' else (['warmup'] + CONFIG.train_patterns * 3)
        orchestrator.run_phase(phase, float(args.duration), patterns, rl_authority=authority, learning=learning)
        orchestrator.running = False
        return

    if args.command == 'analyze':
        print(f"Analyzing results from: {args.data_dir}")
        checkpoints = sorted(Path(args.data_dir).glob("checkpoint_*"))
        if not checkpoints:
            print("No checkpoints found")
            return
        latest = checkpoints[-1]
        print(f"Latest checkpoint: {latest}")

        state_file = latest / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            print(f"\nPhase: {data.get('phase')}")
            print(f"Epoch: {data.get('epoch')}")
            print(f"Iterations: {data.get('total_iterations')}")
            results = data.get('results', [])
            if results:
                print(f"\nResults ({len(results)} patterns):")
                for r in results[-10:]:
                    print(f"  {r.get('pattern')}: P95={r.get('latency_p95_mean', 0):.1f}ms")


if __name__ == '__main__':
    main()
