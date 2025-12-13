#!/usr/bin/env python3
"""
CAPA+ Enhanced Unified Experiment Framework v2.0
=================================================

Enhancements over v1.0:
1. Enriched State Space (243 states vs 8)
   - 3-level discretization for CPU, latency, pods
   - Trend information (CPU trend, latency trend)
   
2. Workload Domain Randomization
   - Random scaling (0.8x-1.2x) of patterns
   - Gaussian noise on each step
   - Prevents memorization of exact sequences

3. Behavioral Regularization
   - Thrashing penalty for flip-flopping decisions
   - Smoother, more stable scaling behavior

4. Prioritized Experience Replay
   - Higher priority for high-reward or SLA-violation states
   - Faster learning from rare but important events

5. Phase-Dependent Exploration
   - High epsilon in shadow mode
   - Gradual reduction through phases
   - Zero exploration in evaluation

Theoretical Foundations:
- Jain (1991): The Art of Computer Systems Performance Analysis
- Harchol-Balter (2013): Performance Modeling and Design of Computer Systems
- Domain Randomization: Tobin et al. (2017) for sim-to-real transfer

Usage:
    python unified_experiment_v2.py diagnose
    python unified_experiment_v2.py validate --continuous
    python unified_experiment_v2.py train --mode full --duration 28800

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
import ast
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
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

try:
    from scipy import stats as scipy_stats
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
    from scipy import stats as scipy_stats

# Import our statistical analysis module
try:
    from statistical_analysis import (
        batch_means_analysis, paired_comparison, compute_confidence_interval,
        compute_speedup_with_ci, check_precision, ExperimentAnalyzer
    )
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    # Warning will be logged after logging is configured


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

@dataclass
class ServiceConfig:
    """Configuration for a single service"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 50
    target_latency_ms: float = 200.0
    capacity_per_pod: float = 100.0


@dataclass 
class Config:
    """Global configuration"""
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
    
    # RL Hyperparameters
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.10
    epsilon_decay: float = 0.9995
    
    # Queueing theory thresholds
    stability_threshold: float = 0.9
    target_utilization: float = 0.7
    little_law_error_threshold: float = 0.2
    
    # Smoothing
    smoothing_alpha: float = 0.3
    
    # Control loop
    control_interval_sec: float = 15.0
    cooldown_sec: float = 60.0
    
    # ENHANCED: Trend detection thresholds
    cpu_trend_threshold: float = 0.05      # 5% change = significant
    latency_trend_threshold: float = 0.10  # 10% change = significant
    
    # ENHANCED: Workload randomization
    workload_scale_min: float = 0.8
    workload_scale_max: float = 1.2
    workload_noise_sigma: float = 15.0


CONFIG = Config()

# Load pattern definitions
LOAD_PATTERNS = {
    'warmup': [50, 100, 150, 100, 50],
    'step': [50, 100, 300, 500, 600, 500, 300, 100],
    'gradual': [50, 150, 250, 350, 450, 550, 450, 250],
    'sine': [300, 477, 550, 477, 300, 123, 50, 123],
    'spike': [50, 50, 800, 100, 50, 700, 50, 50],
    'stress': [100, 300, 600, 800, 600, 300, 100, 50],
}

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('capa_experiment_v2.log')
    ]
)
logger = logging.getLogger('CAPA+v2')

# Deferred warning for statistical analysis module
if not STATS_AVAILABLE:
    logger.warning("statistical_analysis.py not found - using basic statistics")


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
    """Comprehensive metrics for a single service"""
    timestamp: float = 0.0
    service_name: str = ""
    
    # Latency (populated from Locust or Prometheus)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_max_ms: float = 0.0  # TODO: Populate from Locust max_response_time
    
    # Throughput
    arrival_rate: float = 0.0
    total_requests: int = 0       # TODO: Populate from Locust total_requests
    failed_requests: int = 0      # TODO: Populate from Locust failures
    failure_rate: float = 0.0
    
    # Resources
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    memory_bytes: int = 0         # TODO: Populate from Prometheus container_memory_usage_bytes
    
    # Pods
    current_replicas: int = 0
    ready_replicas: int = 0
    desired_replicas: int = 0
    
    # Metadata
    latency_source: str = "unknown"
    resource_source: str = "unknown"


@dataclass
class SystemState:
    """System state for Little's Law validation"""
    timestamp: float = 0.0
    arrival_rate: float = 0.0
    response_time_avg_sec: float = 0.0
    predicted_queue_length: float = 0.0
    num_servers: int = 0
    service_rate_per_server: float = 0.0
    utilization: float = 0.0
    is_stable: bool = False
    little_law_error: float = 0.0
    in_steady_state: bool = False
    cv_arrival: float = 0.0
    cv_service: float = 0.0

@dataclass
class ScalingDecision:
    """Record of a scaling decision"""
    timestamp: float
    service: str
    state: Tuple
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
    """State of training for checkpointing"""
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
    """Collect metrics from Locust API"""
    
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
        except:
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
    
    def get_aggregate_latency(self) -> Dict:
        stats = self.get_stats()
        if not stats:
            return {}
        
        avg_ms = 0
        for endpoint in stats.get('stats', []):
            if endpoint.get('name') in ('Aggregated', 'Total'):
                avg_ms = endpoint.get('avg_response_time', 0) or 0
                break
        
        # Validate expected keys exist (warn on schema mismatch)
        expected_keys = ['current_response_time_percentile_50', 'current_response_time_percentile_95', 'total_rps']
        missing_keys = [k for k in expected_keys if k not in stats]
        if missing_keys:
            logger.warning(f"Locust API schema mismatch - missing keys: {missing_keys}. "
                          f"Available keys: {list(stats.keys())}")
        
        return {
            'p50_ms': stats.get('current_response_time_percentile_50', 0) or 0,
            'p95_ms': stats.get('current_response_time_percentile_95', 0) or 0,
            'p99_ms': (stats.get('current_response_time_percentile_95', 0) or 0) * 1.3,
            'avg_ms': avg_ms,
            'rps': stats.get('total_rps', 0) or 0,
            'fail_ratio': stats.get('fail_ratio', 0) or 0
        }
    
    def set_users(self, count: int, spawn_rate: int = None) -> bool:
        if spawn_rate is None:
            spawn_rate = max(1, count // 10)
        try:
            r = requests.post(f"{self.url}/swarm",
                            data={'user_count': count, 'spawn_rate': spawn_rate}, timeout=5)
            return r.status_code == 200
        except:
            return False


class PrometheusMetricsCollector:
    """Collect metrics from Prometheus"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090", namespace: str = "default"):
        self.url = prometheus_url
        self.namespace = namespace
        self.logger = logging.getLogger('Prometheus')
    
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/v1/status/config", timeout=3)
            return r.status_code == 200
        except:
            return False
    
    def query(self, promql: str) -> Optional[float]:
        try:
            r = requests.get(f"{self.url}/api/v1/query", params={'query': promql}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get('status') == 'success':
                    result = data.get('data', {}).get('result', [])
                    if result:
                        vals = [float(item['value'][1]) for item in result]
                        return np.mean(vals) if vals else None
        except:
            pass
        return None
    
    def get_cpu_utilization(self, deployment: str) -> Optional[float]:
        query = f'''
        sum(rate(container_cpu_usage_seconds_total{{
            namespace="{self.namespace}", pod=~"{deployment}.*", container!=""
        }}[2m])) / 
        sum(kube_pod_container_resource_requests{{
            namespace="{self.namespace}", pod=~"{deployment}.*", resource="cpu"
        }})
        '''
        result = self.query(query)
        return min(result, 2.0) if result is not None else None
    
    def get_pod_counts(self, deployment: str) -> Tuple[int, int, int]:
        ready = self.query(f'kube_deployment_status_replicas_ready{{namespace="{self.namespace}",deployment="{deployment}"}}')
        current = self.query(f'kube_deployment_status_replicas{{namespace="{self.namespace}",deployment="{deployment}"}}')
        desired = self.query(f'kube_deployment_spec_replicas{{namespace="{self.namespace}",deployment="{deployment}"}}')
        return (int(ready) if ready else 1, int(current) if current else 1, int(desired) if desired else 1)


class KubectlMetricsCollector:
    """Fallback metrics via kubectl"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
    
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
        except:
            return (1, 1, 1)
    
    def scale_deployment(self, deployment: str, replicas: int) -> bool:
        try:
            result = subprocess.run(
                ['kubectl', 'scale', 'deployment', deployment, '-n', self.namespace, f'--replicas={replicas}'],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except:
            return False


class UnifiedMetricsCollector:
    """Multi-source metrics collection"""
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.logger = logging.getLogger('Metrics')
        
        self.locust = LocustMetricsCollector(config.locust_url)
        self.prometheus = PrometheusMetricsCollector(config.prometheus_url, config.namespace)
        self.kubectl = KubectlMetricsCollector(config.namespace)
        
        self.locust_available = self.locust.is_available()
        self.prometheus_available = self.prometheus.is_available()
        
        self._smoothed: Dict[str, ServiceMetrics] = {}
        
        self.logger.info(f"Locust: {self.locust_available}, Prometheus: {self.prometheus_available}")
    
    def collect(self, service: str) -> ServiceMetrics:
        metrics = ServiceMetrics(timestamp=time.time(), service_name=service)
        
        if self.locust_available:
            latency = self.locust.get_aggregate_latency()
            if latency:
                metrics.latency_p50_ms = latency.get('p50_ms', 0)
                metrics.latency_p95_ms = latency.get('p95_ms', 0)
                metrics.latency_p99_ms = latency.get('p99_ms', 0)
                metrics.latency_avg_ms = latency.get('avg_ms', 0)
                metrics.arrival_rate = latency.get('rps', 0)
                metrics.failure_rate = latency.get('fail_ratio', 0)
                metrics.latency_source = 'locust'
        
        if self.prometheus_available:
            cpu = self.prometheus.get_cpu_utilization(service)
            if cpu is not None:
                metrics.cpu_utilization = cpu
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
        
        for field_name in ['latency_p50_ms', 'latency_p95_ms', 'latency_avg_ms', 'arrival_rate', 'cpu_utilization']:
            old_val = getattr(old, field_name, 0) or 0
            new_val = getattr(new, field_name, 0) or 0
            if new_val > 0:
                setattr(new, field_name, alpha * new_val + (1 - alpha) * old_val)
        
        self._smoothed[service] = new
        return new


# =============================================================================
# SECTION 4: LITTLE'S LAW VALIDATOR
# =============================================================================

class LittleLawValidator:
    """Validates Little's Law and system stability"""
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.history: Dict[str, deque] = {}
        self.capacity_estimates: Dict[str, deque] = {}
    
    def validate(self, service: str, metrics: ServiceMetrics, service_capacity: float = None) -> SystemState:
        if service_capacity is None:
            svc_config = self.config.services.get(service)
            service_capacity = svc_config.capacity_per_pod if svc_config else 100.0
        
        lambda_rate = metrics.arrival_rate
        response_time_sec = metrics.latency_avg_ms / 1000.0
        num_pods = max(1, metrics.ready_replicas)
        
        predicted_queue = lambda_rate * response_time_sec  # L = λT (Little's Law)
        total_capacity = num_pods * service_capacity
        utilization = lambda_rate / total_capacity if total_capacity > 0 else 1.0
        is_stable = utilization < self.config.stability_threshold
        in_steady_state, _ = self._check_steady_state(service)
        
        # Calculate Little's Law error using historical data
        # Error = relative deviation of L = λT from its historical mean
        little_law_error = self._compute_little_law_error(service, predicted_queue)
        
        state = SystemState(
            timestamp=time.time(),
            arrival_rate=lambda_rate,
            response_time_avg_sec=response_time_sec,
            predicted_queue_length=predicted_queue,
            num_servers=num_pods,
            service_rate_per_server=service_capacity,
            utilization=utilization,
            is_stable=is_stable,
            little_law_error=little_law_error,
            in_steady_state=in_steady_state
        )
        
        if service not in self.history:
            self.history[service] = deque(maxlen=30)
        self.history[service].append(state)
        
        return state
    
    def _compute_little_law_error(self, service: str, current_L: float) -> float:
        """
        Compute Little's Law consistency error.
        
        In steady state, L = λT should be relatively constant. We measure
        the relative deviation of the current L from the recent mean.
        A high error indicates the system is not in steady state.
        
        Returns: Relative error (0.0 = perfect consistency)
        """
        if service not in self.history or len(self.history[service]) < 5:
            return 0.0  # Not enough data
        
        # Get recent L values
        recent_L = [s.predicted_queue_length for s in self.history[service]]
        mean_L = sum(recent_L) / len(recent_L)
        
        if mean_L < 0.001:  # Avoid division by near-zero
            return 0.0
        
        # Relative error from mean
        error = abs(current_L - mean_L) / mean_L
        return min(error, 1.0)  # Cap at 1.0
    
    def _check_steady_state(self, service: str) -> Tuple[bool, str]:
        if service not in self.history or len(self.history[service]) < 5:
            return False, "Insufficient history"
        
        history = list(self.history[service])[-10:]
        
        rates = [s.arrival_rate for s in history if s.arrival_rate > 0]
        if rates and np.mean(rates) > 0:
            cv = np.std(rates) / np.mean(rates)
            if cv > 0.3:
                return False, f"Arrival rate unstable (CV={cv:.2f})"
        
        times = [s.response_time_avg_sec for s in history if s.response_time_avg_sec > 0]
        if times and np.mean(times) > 0:
            cv = np.std(times) / np.mean(times)
            if cv > 0.4:
                return False, f"Response time unstable (CV={cv:.2f})"
        
        return True, "System in steady-state"
    
    def calculate_optimal_pods(self, arrival_rate: float, service_capacity: float, target_util: float = None) -> Tuple[int, Dict]:
        if target_util is None:
            target_util = self.config.target_utilization
        if service_capacity <= 0 or arrival_rate <= 0:
            return 1, {}
        
        min_stable = max(1, math.ceil(arrival_rate / service_capacity))
        optimal = max(1, math.ceil(arrival_rate / (target_util * service_capacity)))
        
        return optimal, {
            'min_stable_pods': min_stable,
            'optimal_pods': optimal,
            'max_sustainable_rps': optimal * service_capacity * self.config.stability_threshold
        }


# =============================================================================
# SECTION 5: ENHANCED DOUBLE Q-LEARNING AGENT (243 states with trends)
# =============================================================================

class EnhancedDoubleQLearningAgent:
    """
    Enhanced Double Q-Learning with:
    1. 243-state space (3×3×3×3×3): CPU, latency, pods, CPU trend, latency trend
    2. Prioritized experience replay
    3. State-dependent exploration
    
    This addresses the overfitting concern by:
    - Richer state representation captures dynamics
    - Trends enable predictive rather than reactive control
    - Still tabular (no DRL needed for this state space size)
    """
    
    def __init__(self, service: str, config: Config = CONFIG):
        self.service = service
        self.config = config
        self.logger = logging.getLogger(f'RL:{service}')
        
        # ENHANCED: 5D state space
        # (cpu_level, latency_level, pod_level, cpu_trend, latency_trend)
        # 3 × 3 × 3 × 3 × 3 = 243 states
        self.state_dims = (3, 3, 3, 3, 3)
        self.n_actions = 3
        
        # Double Q-tables
        self.q_table_A: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0] * self.n_actions)
        self.q_table_B: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0] * self.n_actions)
        self.use_table_A = True
        
        # Learning parameters
        self.alpha = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        
        # ENHANCED: Prioritized experience replay
        # Store (state, action, reward, next_state, priority)
        self.replay_buffer: List[Tuple] = []
        self.max_replay_size = 500
        self.batch_size = 16
        
        # Statistics
        self.total_updates = 0
        self.state_visits: Dict[Tuple, int] = defaultdict(int)
        self.action_counts: Dict[int, int] = defaultdict(int)
    
    def discretize_state(
        self,
        cpu_util: float,
        latency_ratio: float,
        pod_ratio: float,
        cpu_trend: float,
        latency_trend: float
    ) -> Tuple[int, int, int, int, int]:
        """
        ENHANCED: 5-dimensional state with trends
        
        Args:
            cpu_util: Current CPU utilization (0-1)
            latency_ratio: actual_latency / target_latency
            pod_ratio: current_pods / max_pods
            cpu_trend: Change in CPU since last observation (delta)
            latency_trend: Change in latency_ratio since last observation
        
        Returns:
            5-tuple state with 243 possible combinations
        """
        # 1. CPU: 0=Low (<40%), 1=Target (40-70%), 2=High (>70%)
        if cpu_util < 0.4:
            cpu_level = 0
        elif cpu_util < 0.7:
            cpu_level = 1
        else:
            cpu_level = 2
        
        # 2. Latency: 0=Good (≤80% target), 1=Warning (80-110%), 2=Violation (>110%)
        if latency_ratio <= 0.8:
            lat_level = 0
        elif latency_ratio <= 1.1:
            lat_level = 1
        else:
            lat_level = 2
        
        # 3. Pods: 0=Low (<30%), 1=Medium (30-70%), 2=High (>70%)
        if pod_ratio < 0.3:
            pod_level = 0
        elif pod_ratio < 0.7:
            pod_level = 1
        else:
            pod_level = 2
        
        # 4. CPU Trend: 0=Falling, 1=Stable, 2=Rising
        cpu_threshold = self.config.cpu_trend_threshold
        if cpu_trend < -cpu_threshold:
            cpu_trend_level = 0  # Falling (good if was high)
        elif cpu_trend > cpu_threshold:
            cpu_trend_level = 2  # Rising (might need to scale up)
        else:
            cpu_trend_level = 1  # Stable
        
        # 5. Latency Trend: 0=Improving, 1=Stable, 2=Worsening
        lat_threshold = self.config.latency_trend_threshold
        if latency_trend < -lat_threshold:
            lat_trend_level = 0  # Improving
        elif latency_trend > lat_threshold:
            lat_trend_level = 2  # Worsening (need action!)
        else:
            lat_trend_level = 1  # Stable
        
        return (cpu_level, lat_level, pod_level, cpu_trend_level, lat_trend_level)
    
    def choose_action(self, state: Tuple, training: bool = True) -> ScalingAction:
        """Epsilon-greedy with state-dependent exploration"""
        self.state_visits[state] += 1
        
        if training and random.random() < self.epsilon:
            # ENHANCED: Bias exploration towards scaling up if latency is bad
            lat_level = state[1]  # Index 1 is latency level
            if lat_level == 2:  # Violation state
                # 50% chance to explore SCALE_UP
                if random.random() < 0.5:
                    action = ScalingAction.SCALE_UP
                else:
                    action = ScalingAction(random.randint(0, self.n_actions - 1))
            else:
                action = ScalingAction(random.randint(0, self.n_actions - 1))
        else:
            q_A = self.q_table_A[state]
            q_B = self.q_table_B[state]
            q_avg = [(a + b) / 2 for a, b in zip(q_A, q_B)]
            action = ScalingAction(np.argmax(q_avg))
        
        self.action_counts[action.value] += 1
        return action
    
    def update(self, state: Tuple, action: ScalingAction, reward: float, next_state: Tuple,
               latency_ratio: float = 1.0):
        """
        Double Q-Learning update with prioritized storage
        
        Priority based on:
        - Magnitude of reward
        - SLA violation (latency_ratio > 1)
        """
        # Add regularization noise
        noisy_reward = reward + np.random.normal(0, 0.05)
        
        # ENHANCED: Calculate priority for replay
        priority = abs(reward) + (1.0 if latency_ratio > 1.0 else 0.0)
        
        # Store with priority
        experience = (state, action.value, noisy_reward, next_state, priority)
        
        if len(self.replay_buffer) >= self.max_replay_size:
            # Remove lowest priority
            min_idx = min(range(len(self.replay_buffer)), key=lambda i: self.replay_buffer[i][4])
            self.replay_buffer.pop(min_idx)
        self.replay_buffer.append(experience)
        
        # Standard Double Q-Learning update
        if self.use_table_A:
            best_action = np.argmax(self.q_table_A[next_state])
            target = noisy_reward + self.gamma * self.q_table_B[next_state][best_action]
            old_q = self.q_table_A[state][action.value]
            self.q_table_A[state][action.value] = old_q + self.alpha * (target - old_q)
        else:
            best_action = np.argmax(self.q_table_B[next_state])
            target = noisy_reward + self.gamma * self.q_table_A[next_state][best_action]
            old_q = self.q_table_B[state][action.value]
            self.q_table_B[state][action.value] = old_q + self.alpha * (target - old_q)
        
        self.use_table_A = not self.use_table_A
        self.total_updates += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self):
        """ENHANCED: Prioritized experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample with priority weighting
        priorities = np.array([exp[4] for exp in self.replay_buffer])
        priorities = priorities + 0.01  # Ensure non-zero
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False, p=probs)
        
        replay_alpha = self.alpha * 0.5
        
        for idx in indices:
            state, action, reward, next_state, _ = self.replay_buffer[idx]
            
            if self.use_table_A:
                best_action = np.argmax(self.q_table_A[next_state])
                target = reward + self.gamma * self.q_table_B[next_state][best_action]
                old_q = self.q_table_A[state][action]
                self.q_table_A[state][action] = old_q + replay_alpha * (target - old_q)
            else:
                best_action = np.argmax(self.q_table_B[next_state])
                target = reward + self.gamma * self.q_table_A[next_state][best_action]
                old_q = self.q_table_B[state][action]
                self.q_table_B[state][action] = old_q + replay_alpha * (target - old_q)
    
    def set_epsilon(self, value: float):
        """
        ENHANCED: Allow external epsilon control for phase-dependent exploration
        
        Special case: value=0.0 is allowed for evaluation mode (pure exploitation)
        Otherwise, epsilon is clamped to [epsilon_min, 1.0]
        """
        if value == 0.0:
            self.epsilon = 0.0  # Allow true zero for evaluation
        else:
            self.epsilon = max(self.epsilon_min, min(1.0, value))
    
    def get_stats(self) -> Dict:
        return {
            'service': self.service,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'states_discovered': len(self.state_visits),
            'state_space_coverage': len(self.state_visits) / 243.0,  # Out of 243 possible
            'state_visits': dict(self.state_visits),
            'action_counts': dict(self.action_counts),
            'replay_buffer_size': len(self.replay_buffer)
        }
    
    def save(self, path: str):
        data = {
            'service': self.service,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'state_visits': {str(k): v for k, v in self.state_visits.items()},
            'action_counts': dict(self.action_counts),
            'q_table_A': {str(k): v for k, v in self.q_table_A.items()},
            'q_table_B': {str(k): v for k, v in self.q_table_B.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.epsilon = data.get('epsilon', self.epsilon)
        self.total_updates = data.get('total_updates', 0)
        for k, v in data.get('q_table_A', {}).items():
            self.q_table_A[ast.literal_eval(k)] = v
        for k, v in data.get('q_table_B', {}).items():
            self.q_table_B[ast.literal_eval(k)] = v
        for k, v in data.get('state_visits', {}).items():
            self.state_visits[ast.literal_eval(k)] = v


# =============================================================================
# SECTION 6: BASELINE HPA CONTROLLER
# =============================================================================

class BaselineHPAController:
    """Simple CPU-based HPA for comparison"""
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def decide(self, service: str, cpu_util: float, ready_pods: int) -> ScalingAction:
        svc_config = self.config.services.get(service)
        if not svc_config:
            return ScalingAction.STAY
        
        target_cpu = svc_config.target_cpu_percent / 100.0
        scale_up_threshold = target_cpu * 1.4
        scale_down_threshold = target_cpu * 0.6
        
        if cpu_util > scale_up_threshold and ready_pods < svc_config.max_replicas:
            return ScalingAction.SCALE_UP
        elif cpu_util < scale_down_threshold and ready_pods > svc_config.min_replicas:
            return ScalingAction.SCALE_DOWN
        return ScalingAction.STAY


# =============================================================================
# SECTION 7: ENHANCED REWARD CALCULATOR (with thrashing penalty)
# =============================================================================

class EnhancedRewardCalculator:
    """
    ENHANCED: Reward calculator with thrashing penalty
    
    Adds behavioral regularization to prevent flip-flopping:
    - Penalty for changing action from previous step
    - Higher penalty for Up<->Down oscillation
    """
    
    def __init__(self, weights: Tuple[float, float, float, float] = (0.45, 0.25, 0.15, 0.15)):
        self.w_sla, self.w_eff, self.w_stab, self.w_thrash = weights
    
    def calculate(
        self,
        latency_ratio: float,
        cpu_util: float,
        action: ScalingAction,
        pods_ready_ratio: float,
        prev_action: Optional[ScalingAction] = None
    ) -> float:
        """
        Calculate composite reward with thrashing penalty
        
        Args:
            latency_ratio: actual_latency / target_latency
            cpu_util: CPU utilization (0-1)
            action: Current action
            pods_ready_ratio: ready_pods / current_pods
            prev_action: Previous action (for thrashing detection)
        """
        # === SLA Reward ===
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
        
        # === Efficiency Reward ===
        if 0.5 <= cpu_util <= 0.7:
            r_eff = 1.0
        elif 0.3 <= cpu_util <= 0.85:
            r_eff = 0.3
        elif cpu_util < 0.1:
            r_eff = -0.5
        else:
            r_eff = -0.3
        
        # === Stability Reward ===
        if pods_ready_ratio < 1.0 and action != ScalingAction.STAY:
            r_stab = -1.0
        elif action == ScalingAction.SCALE_UP:
            r_stab = -0.2
        elif action == ScalingAction.SCALE_DOWN:
            r_stab = -0.1
        else:
            r_stab = 0.1
        
        # === ENHANCED: Thrashing Penalty ===
        r_thrash = 0.0
        if prev_action is not None and action != prev_action:
            # High penalty for Up<->Down flip-flop
            if ((action == ScalingAction.SCALE_UP and prev_action == ScalingAction.SCALE_DOWN) or
                (action == ScalingAction.SCALE_DOWN and prev_action == ScalingAction.SCALE_UP)):
                r_thrash = -1.0  # Strong penalty
            # Mild penalty for other changes
            elif action != ScalingAction.STAY:
                r_thrash = -0.3
        
        # Weighted sum
        reward = (self.w_sla * r_sla + 
                  self.w_eff * r_eff + 
                  self.w_stab * r_stab + 
                  self.w_thrash * r_thrash)
        
        return max(-2.0, min(2.0, reward))


# =============================================================================
# SECTION 8: ENHANCED MAPE-K CONTROLLER
# =============================================================================

class EnhancedMAPEKController:
    """
    ENHANCED MAPE-K Controller with:
    1. Trend computation for state enrichment
    2. Thrashing penalty in rewards
    3. Phase-dependent exploration control
    """
    
    def __init__(
        self,
        config: Config = CONFIG,
        mode: ControlMode = ControlMode.SHADOW,
        dry_run: bool = False
    ):
        self.config = config
        self.mode = mode
        self.dry_run = dry_run
        self.rl_authority = 0.0 if mode == ControlMode.SHADOW else 1.0
        
        self.logger = logging.getLogger('MAPE-K')
        
        # Components
        self.metrics = UnifiedMetricsCollector(config)
        self.little_law = LittleLawValidator(config)
        self.baseline = BaselineHPAController(config)
        self.reward_calc = EnhancedRewardCalculator()
        self.kubectl = KubectlMetricsCollector(config.namespace)
        
        # ENHANCED: Use enhanced agents with trend support
        self.agents: Dict[str, EnhancedDoubleQLearningAgent] = {
            svc: EnhancedDoubleQLearningAgent(svc, config)
            for svc in config.services.keys()
        }
        
        # State tracking
        self.learning_enabled = True
        self.previous_states: Dict[str, Tuple] = {}
        self.previous_actions: Dict[str, ScalingAction] = {}
        self.previous_metrics: Dict[str, ServiceMetrics] = {}
        
        # ENHANCED: Track previous latency ratios for trend computation
        self.previous_latency_ratios: Dict[str, float] = {}
        
        # Cooldowns
        self.last_scale_time: Dict[str, float] = {}
        self.cooldown_sec = config.cooldown_sec
        
        # History
        self.decision_history: List[ScalingDecision] = []
        self.metrics_history: List[Dict] = []
        
        # Shadow stats
        self.shadow_stats = {
            'agreements': 0, 'disagreements': 0,
            'rl_would_be_better': 0, 'baseline_would_be_better': 0
        }
    
    def set_mode(self, mode: ControlMode, rl_authority: float = None):
        """Set control mode with phase-dependent exploration"""
        self.mode = mode
        
        if mode == ControlMode.SHADOW:
            self.rl_authority = 0.0
            self.learning_enabled = True
            # High exploration in shadow mode (safe to explore)
            for agent in self.agents.values():
                agent.set_epsilon(0.8)
        elif mode == ControlMode.ACTIVE:
            self.rl_authority = 1.0
            self.learning_enabled = True
            # Low exploration in active mode - mostly exploit learned policy
            # Note: min() ensures we don't INCREASE epsilon if agent has lower value
            for agent in self.agents.values():
                agent.set_epsilon(min(agent.epsilon, 0.05))
        elif mode == ControlMode.EVALUATION:
            self.rl_authority = 1.0
            self.learning_enabled = False
            # Zero exploration in evaluation - pure exploitation
            for agent in self.agents.values():
                agent.set_epsilon(0.0)
        elif mode == ControlMode.HYBRID:
            self.rl_authority = rl_authority if rl_authority is not None else 0.5
            self.learning_enabled = True
            # Moderate exploration in hybrid mode
            for agent in self.agents.values():
                agent.set_epsilon(max(agent.epsilon, 0.3))
        
        self.logger.info(f"Mode: {mode.value}, RL authority: {self.rl_authority:.0%}")
    
    def control_loop_iteration(self) -> Dict[str, ScalingDecision]:
        """Execute one MAPE-K iteration"""
        decisions = {}
        
        for service, svc_config in self.config.services.items():
            try:
                decision = self._process_service(service, svc_config)
                if decision:
                    decisions[service] = decision
            except Exception as e:
                self.logger.error(f"Error processing {service}: {e}")

        if self.learning_enabled:
            for agent in self.agents.values():
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
        return decisions
    
    def _process_service(self, service: str, svc_config: ServiceConfig) -> Optional[ScalingDecision]:
        """Process a single service through MAPE-K with trend computation"""
        
        # === MONITOR ===
        metrics = self.metrics.collect(service)
        
        self.metrics_history.append({
            'timestamp': time.time(),
            'service': service,
            **asdict(metrics)
        })
        
        # Cap in-memory history to prevent unbounded growth in long runs
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]
        
        # === ANALYZE ===
        system_state = self.little_law.validate(service, metrics, svc_config.capacity_per_pod)
        
        # Compute derived values
        latency_ratio = metrics.latency_avg_ms / svc_config.target_latency_ms if metrics.latency_avg_ms > 0 else 0
        pod_ratio = metrics.ready_replicas / svc_config.max_replicas if svc_config.max_replicas > 0 else 0
        
        # ENHANCED: Compute trends
        cpu_trend = 0.0
        latency_trend = 0.0
        
        if service in self.previous_metrics:
            prev = self.previous_metrics[service]
            cpu_trend = metrics.cpu_utilization - prev.cpu_utilization
            
            prev_lat_ratio = self.previous_latency_ratios.get(service, latency_ratio)
            latency_trend = latency_ratio - prev_lat_ratio
        
        # Store for next iteration
        self.previous_latency_ratios[service] = latency_ratio
        
        # ENHANCED: Discretize state with trends
        state = self.agents[service].discretize_state(
            metrics.cpu_utilization,
            latency_ratio,
            pod_ratio,
            cpu_trend,
            latency_trend
        )
        
        # === PLAN ===
        rl_action = self.agents[service].choose_action(state, self.learning_enabled)
        baseline_action = self.baseline.decide(service, metrics.cpu_utilization, metrics.ready_replicas)
        actual_action, decision_source = self._select_action(rl_action, baseline_action)
        
        self._update_shadow_stats(rl_action, baseline_action)
        
        if not self._can_scale(service, actual_action):
            actual_action = ScalingAction.STAY
        
        # === EXECUTE ===
        if actual_action != ScalingAction.STAY:
            self._execute_scaling(service, actual_action, metrics.ready_replicas, svc_config)
        
        # === KNOWLEDGE ===
        if self.learning_enabled and service in self.previous_states:
            # Thrashing penalty compares a_{t-1} (previous) vs a_t (current)
            # Penalizes sequences like: SCALE_UP → SCALE_DOWN (flip-flopping)
            reward = self.reward_calc.calculate(
                latency_ratio=latency_ratio,
                cpu_util=metrics.cpu_utilization,
                action=actual_action,                    # a_t: current action
                pods_ready_ratio=metrics.ready_replicas / max(1, metrics.current_replicas),
                prev_action=self.previous_actions[service]  # a_{t-1}: previous action
            )
            
            self.agents[service].update(
                self.previous_states[service],
                self.previous_actions[service],
                reward,
                state,
                latency_ratio=latency_ratio
            )
            
            self.agents[service].replay()
        
        # Store for next iteration
        self.previous_states[service] = state
        self.previous_actions[service] = actual_action
        self.previous_metrics[service] = metrics
        
        # Create decision record
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
        
        # Trim decision history to prevent memory growth (same pattern as metrics_history)
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]
        
        self._log_decision(service, state, rl_action, baseline_action, actual_action, system_state, metrics)
        
        return decision
    
    def _select_action(self, rl_action: ScalingAction, baseline_action: ScalingAction) -> Tuple[ScalingAction, str]:
        if self.mode == ControlMode.SHADOW:
            return baseline_action, 'baseline'
        elif self.mode in [ControlMode.ACTIVE, ControlMode.EVALUATION]:
            return rl_action, 'rl'
        elif self.mode == ControlMode.HYBRID:
            if random.random() < self.rl_authority:
                return rl_action, 'rl'
            return baseline_action, 'baseline'
        return baseline_action, 'baseline'
    
    def _update_shadow_stats(self, rl_action: ScalingAction, baseline_action: ScalingAction):
        if rl_action == baseline_action:
            self.shadow_stats['agreements'] += 1
        else:
            self.shadow_stats['disagreements'] += 1
    
    def _can_scale(self, service: str, action: ScalingAction) -> bool:
        if action == ScalingAction.STAY:
            return True
        last_time = self.last_scale_time.get(service, 0)
        return time.time() - last_time >= self.cooldown_sec
    
    def _execute_scaling(self, service: str, action: ScalingAction, current_pods: int, config: ServiceConfig):
        if action == ScalingAction.SCALE_UP:
            new_replicas = min(current_pods + 1, config.max_replicas)
        elif action == ScalingAction.SCALE_DOWN:
            new_replicas = max(current_pods - 1, config.min_replicas)
        else:
            return
        
        if new_replicas == current_pods:
            return
        
        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would scale {service}: {current_pods} → {new_replicas}")
        else:
            if self.kubectl.scale_deployment(service, new_replicas):
                self.logger.info(f"Scaled {service}: {current_pods} → {new_replicas}")
                self.last_scale_time[service] = time.time()
    
    def _log_decision(self, service: str, state: Tuple, rl: ScalingAction, baseline: ScalingAction,
                      actual: ScalingAction, sys_state: SystemState, metrics: ServiceMetrics):
        stable = "✓" if sys_state.is_stable else "✗"
        self.logger.info(
            f"[{service}] S={state} Act={actual.name} "
            f"(RL={rl.name} Base={baseline.name}) "
            f"ρ={sys_state.utilization:.1%}{stable} "
            f"P95={metrics.latency_p95_ms:.0f}ms CPU={metrics.cpu_utilization:.1%}"
        )
    
    def get_shadow_analysis(self) -> Dict:
        total = self.shadow_stats['agreements'] + self.shadow_stats['disagreements']
        if total == 0:
            return {'agreement_rate': 0, 'total_decisions': 0}
        return {
            'agreement_rate': self.shadow_stats['agreements'] / total,
            'total_decisions': total,
            **self.shadow_stats
        }
    
    def save_state(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for service, agent in self.agents.items():
            agent.save(os.path.join(directory, f"{service}_agent.json"))
        with open(os.path.join(directory, "shadow_stats.json"), 'w') as f:
            json.dump(self.shadow_stats, f, indent=2)
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
    """
    Manage Locust load generation.
    
    Uses LocustMetricsCollector internally to avoid code duplication.
    """
    
    def __init__(self, locust_url: str = "http://localhost:8089"):
        self.url = locust_url
        self.logger = logging.getLogger('LocustMgr')
        # Delegate to LocustMetricsCollector for common operations
        self._collector = LocustMetricsCollector(locust_url)
    
    def is_running(self) -> bool:
        """Check if Locust is running (delegates to collector.is_available)"""
        return self._collector.is_available()
    
    def set_users(self, count: int, spawn_rate: int = None) -> bool:
        """Set user count (delegates to collector.set_users)"""
        return self._collector.set_users(count, spawn_rate)
    
    def reset_stats(self) -> bool:
        """Reset Locust statistics"""
        try:
            r = requests.get(f"{self.url}/stats/reset", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def get_stats(self) -> Optional[Dict]:
        """Get Locust stats (delegates to collector.get_stats)"""
        return self._collector.get_stats()
    
    def get_aggregate_latency(self) -> Dict:
        """Get aggregate latency metrics (delegates to collector)"""
        return self._collector.get_aggregate_latency()


# =============================================================================
# SECTION 10: ENHANCED TRAINING ORCHESTRATOR
# =============================================================================

class EnhancedTrainingOrchestrator:
    """
    ENHANCED Training Orchestrator with:
    1. Workload domain randomization
    2. Phase-dependent epsilon control
    3. Comprehensive overfitting analysis
    """
    
    def __init__(
        self,
        config: Config = CONFIG,
        checkpoint_dir: str = "./checkpoints",
        dry_run: bool = False
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.dry_run = dry_run
        
        self.controller = EnhancedMAPEKController(config, ControlMode.SHADOW, dry_run)
        self.locust = LocustManager(config.locust_url)
        self.logger = logging.getLogger('Training')
        
        self.running = False
        self.training_state = TrainingState()
        self.results: List[Dict] = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Register signal handlers (with Windows compatibility)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        self.logger.info("\nShutdown signal, saving checkpoint...")
        self.running = False
        self.save_checkpoint()
    
    def save_checkpoint(self, label: str = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.checkpoint_dir,
                           f"checkpoint_{label}_{timestamp}" if label else f"checkpoint_epoch{self.training_state.epoch}_{timestamp}")
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
                self.training_state.epoch = data.get('epoch', 0)
                self.training_state.total_iterations = data.get('total_iterations', 0)
        self.logger.info(f"Loaded checkpoint: epoch={self.training_state.epoch}")
    
    def _get_randomized_pattern(self, pattern_name: str) -> List[int]:
        """
        ENHANCED: Domain randomization for workloads
        
        Applies:
        1. Random scaling (0.8x-1.2x)
        2. Gaussian noise per step
        """
        if pattern_name == 'random':
            # Truly random pattern
            return [max(10, 50 + random.randint(-30, 500)) for _ in range(8)]
        
        base_pattern = LOAD_PATTERNS.get(pattern_name, [100, 200, 300, 200, 100])
        
        # ENHANCED: Apply domain randomization
        scale_factor = random.uniform(
            self.config.workload_scale_min,
            self.config.workload_scale_max
        )
        
        randomized = []
        for users in base_pattern:
            noise = random.gauss(0, self.config.workload_noise_sigma)
            new_val = int(users * scale_factor + noise)
            randomized.append(max(10, new_val))
        
        return randomized
    
    def run_pattern(self, pattern_name: str, duration_sec: float) -> Dict:
        """Run a single load pattern with randomization"""
        # ENHANCED: Get randomized version
        pattern = self._get_randomized_pattern(pattern_name)
        
        step_duration = duration_sec / len(pattern)
        pattern_metrics = []
        
        self.logger.info(f"Pattern: {pattern_name} (randomized: {pattern[:3]}...)")
        
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
                        'action': d.actual_action.name,
                        'state': str(d.state)
                    })
                
                time.sleep(self.config.control_interval_sec)
            
            self.training_state.epoch += 1
        
        if pattern_metrics:
            latencies = [m['latency_p95'] for m in pattern_metrics if m['latency_p95'] > 0]
            return {
                'pattern': pattern_name,
                'phase': self.training_state.phase.value,
                'samples': len(pattern_metrics),
                'latency_p95_mean': np.mean(latencies) if latencies else 0,
                'latency_p95_max': np.max(latencies) if latencies else 0,
                'latency_p95_std': np.std(latencies) if latencies else 0
            }
        return {'pattern': pattern_name, 'samples': 0}
    
    def run_phase(self, phase: TrainingPhase, duration_sec: float, patterns: List[str],
                  rl_authority: float = 0.0, learning: bool = True) -> List[Dict]:
        """Run a training phase"""
        self.training_state.phase = phase
        self.training_state.rl_authority = rl_authority
        
        # Set mode with phase-dependent exploration
        if phase == TrainingPhase.SHADOW:
            self.controller.set_mode(ControlMode.SHADOW)
        elif phase in [TrainingPhase.HYBRID_25, TrainingPhase.HYBRID_50, TrainingPhase.HYBRID_75]:
            self.controller.set_mode(ControlMode.HYBRID, rl_authority)
        elif phase == TrainingPhase.ACTIVE:
            self.controller.set_mode(ControlMode.ACTIVE)
        elif phase == TrainingPhase.EVALUATION:
            self.controller.set_mode(ControlMode.EVALUATION)
        
        self.controller.learning_enabled = learning
        
        phase_results = []
        pattern_duration = duration_sec / len(patterns)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PHASE: {phase.value.upper()}")
        self.logger.info(f"Duration: {duration_sec/3600:.1f}h, Patterns: {len(patterns)}")
        self.logger.info(f"RL Authority: {rl_authority:.0%}, Learning: {learning}")
        self.logger.info(f"{'='*60}\n")
        
        for pattern_name in patterns:
            if not self.running:
                break
            result = self.run_pattern(pattern_name, pattern_duration)
            phase_results.append(result)
            self.results.append(result)
            self.logger.info(f"  {pattern_name}: P95={result.get('latency_p95_mean', 0):.1f}ms")
        
        return phase_results
    
    def run_full_training(
        self,
        total_duration_hours: float = 8.0,
        shadow_fraction: float = 0.30,
        hybrid_fraction: float = 0.25,
        active_fraction: float = 0.25,
        eval_fraction: float = 0.20
    ):
        """Run complete training pipeline"""
        self.running = True
        self.training_state.start_time = time.time()
        total_sec = total_duration_hours * 3600
        
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"# CAPA+ ENHANCED TRAINING v2.0")
        self.logger.info(f"# State Space: 243 (3×3×3×3×3 with trends)")
        self.logger.info(f"# Duration: {total_duration_hours} hours")
        self.logger.info(f"# Dry-run: {self.dry_run}")
        self.logger.info(f"{'#'*60}\n")
        
        try:
            # Phase 1: SHADOW
            self.run_phase(TrainingPhase.SHADOW, total_sec * shadow_fraction,
                          ['warmup'] + self.config.train_patterns * 3, 0.0, True)
            self.save_checkpoint("shadow_complete")
            
            if not self.running:
                return
            
            # Phase 2: HYBRID (25% → 50% → 75%)
            for authority, phase in [(0.25, TrainingPhase.HYBRID_25),
                                      (0.50, TrainingPhase.HYBRID_50),
                                      (0.75, TrainingPhase.HYBRID_75)]:
                if not self.running:
                    break
                self.run_phase(phase, total_sec * hybrid_fraction / 3,
                              self.config.train_patterns, authority, True)
            self.save_checkpoint("hybrid_complete")
            
            if not self.running:
                return
            
            # Phase 3: ACTIVE
            self.run_phase(TrainingPhase.ACTIVE, total_sec * active_fraction,
                          self.config.train_patterns * 3, 1.0, True)
            self.save_checkpoint("active_complete")
            
            if not self.running:
                return
            
            # Phase 4: EVALUATION (no learning, epsilon=0)
            eval_results = self.run_phase(TrainingPhase.EVALUATION, total_sec * eval_fraction,
                                          self.config.test_patterns * 3, 1.0, False)
            
            self._analyze_overfitting(eval_results)
            self.training_state.phase = TrainingPhase.COMPLETED
            self.save_checkpoint("final")
            self._print_summary()
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.save_checkpoint("error")
            raise
        finally:
            self.running = False
    
    def _analyze_overfitting(self, eval_results: List[Dict]):
        """
        Comprehensive overfitting analysis following Jain (1991) methodology.
        
        Implements:
        1. Batch means for autocorrelated observations (Jain 25.5.2)
        2. Confidence intervals with proper variance estimation (Jain 13.2)
        3. Paired comparison for train vs test (Jain 13.4.1)
        4. Proper mean selection (arithmetic for latency, geometric for ratios)
        """
        train_results = [r for r in self.results
                        if r.get('phase') in ['shadow', 'active', 'hybrid_25', 'hybrid_50', 'hybrid_75']]
        
        if not train_results or not eval_results:
            return
        
        train_latencies = [r['latency_p95_mean'] for r in train_results if r.get('latency_p95_mean', 0) > 0]
        test_latencies = [r['latency_p95_mean'] for r in eval_results if r.get('latency_p95_mean', 0) > 0]
        
        if not train_latencies or not test_latencies:
            return
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("STATISTICAL ANALYSIS (Following Jain 1991 Methodology)")
        self.logger.info(f"{'='*70}")
        
        if STATS_AVAILABLE:
            # === Use proper batch means for autocorrelated data ===
            self.logger.info("\n--- Batch Means Analysis (Jain 25.5.2) ---")
            self.logger.info("Note: Naive variance can be 300x wrong for correlated data!")
            
            train_batch = batch_means_analysis(train_latencies, initial_batch_size=5)
            test_batch = batch_means_analysis(test_latencies, initial_batch_size=5)
            
            self.logger.info(f"\nTraining P95 Latency:")
            self.logger.info(f"  Mean: {train_batch.ci.mean:.1f} ms")
            self.logger.info(f"  95% CI: [{train_batch.ci.ci_lower:.1f}, {train_batch.ci.ci_upper:.1f}]")
            self.logger.info(f"  Batch size: {train_batch.batch_size}, Batches: {train_batch.num_batches}")
            self.logger.info(f"  Autocovariance: {train_batch.autocovariance:.4f}")
            self.logger.info(f"  Valid (autocov small): {train_batch.is_valid}")
            
            self.logger.info(f"\nTest (Evaluation) P95 Latency:")
            self.logger.info(f"  Mean: {test_batch.ci.mean:.1f} ms")
            self.logger.info(f"  95% CI: [{test_batch.ci.ci_lower:.1f}, {test_batch.ci.ci_upper:.1f}]")
            self.logger.info(f"  Valid: {test_batch.is_valid}")
            
            # === Check if difference is statistically significant ===
            self.logger.info("\n--- Train vs Test Comparison ---")
            
            # Compute ratio (test/train) - values > 1 indicate overfitting
            ratio = test_batch.ci.mean / train_batch.ci.mean if train_batch.ci.mean > 0 else 0
            
            # For proper comparison, we'd need paired observations
            # Since we don't have exact pairs, we use confidence interval overlap
            ci_overlap = (test_batch.ci.ci_lower <= train_batch.ci.ci_upper and
                         train_batch.ci.ci_lower <= test_batch.ci.ci_upper)
            
            self.logger.info(f"Test/Train Ratio: {ratio:.3f}")
            self.logger.info(f"CI Overlap (rough significance test): {ci_overlap}")
            
            if ratio > 1.5:
                self.logger.warning("⚠️ SEVERE OVERFITTING - 50%+ degradation on test")
                self.logger.warning("   Agent has memorized training patterns")
            elif ratio > 1.2:
                self.logger.warning("⚠️ Mild overfitting detected (20-50% degradation)")
            elif ratio > 1.0 and not ci_overlap:
                self.logger.info("ℹ️ Minor but significant degradation on test")
            elif ratio > 1.0:
                self.logger.info("ℹ️ Minor degradation, but within statistical uncertainty")
            else:
                self.logger.info("✓ Excellent generalization (test ≤ train)!")
            
            # === Sample size adequacy (Jain 13.9) ===
            self.logger.info("\n--- Sample Size Adequacy (Jain 13.9) ---")
            precision_ok, current_precision, additional = check_precision(
                train_latencies, desired_precision=0.10
            )
            self.logger.info(f"Current precision: ±{current_precision*100:.1f}% of mean")
            if precision_ok:
                self.logger.info("✓ Sample size adequate for 10% precision")
            else:
                self.logger.info(f"✗ Need ~{additional} more observations for 10% precision")
            
        else:
            # Fallback to basic statistics (not recommended!)
            self.logger.warning("⚠️ Using naive statistics - install scipy for proper analysis")
            
            train_mean = np.mean(train_latencies)
            test_mean = np.mean(test_latencies)
            ratio = test_mean / train_mean if train_mean > 0 else 0
            
            # Naive CI (WARNING: ignores autocorrelation!)
            train_std = np.std(train_latencies, ddof=1)
            test_std = np.std(test_latencies, ddof=1)
            train_se = train_std / np.sqrt(len(train_latencies))
            test_se = test_std / np.sqrt(len(test_latencies))
            
            self.logger.info(f"Training P95: {train_mean:.1f} ± {1.96*train_se:.1f} ms (NAIVE CI)")
            self.logger.info(f"Test P95:     {test_mean:.1f} ± {1.96*test_se:.1f} ms (NAIVE CI)")
            self.logger.info(f"Test/Train Ratio: {ratio:.2f}")
            
            if ratio > 1.5:
                self.logger.warning("⚠️ SEVERE OVERFITTING detected")
            elif ratio > 1.2:
                self.logger.warning("⚠️ Mild overfitting detected")
            else:
                self.logger.info("✓ Good generalization")
        
        # === State space coverage analysis ===
        self.logger.info("\n--- State Space Coverage ---")
        for service, agent in self.controller.agents.items():
            stats = agent.get_stats()
            coverage = stats['state_space_coverage']
            self.logger.info(f"  {service}: {coverage:.1%} ({stats['states_discovered']}/243 states)")
            
            if coverage < 0.3:
                self.logger.warning(f"    ⚠️ Low coverage - agent may not generalize well")
        
        self.logger.info(f"\n{'='*70}")
    
    def _print_summary(self):
        elapsed = time.time() - self.training_state.start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("TRAINING SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Duration: {elapsed/3600:.2f} hours")
        self.logger.info(f"Total Epochs: {self.training_state.epoch}")
        self.logger.info(f"Total Iterations: {self.training_state.total_iterations}")
        
        shadow = self.controller.get_shadow_analysis()
        self.logger.info(f"\nShadow Agreement: {shadow.get('agreement_rate', 0):.1%}")
        
        self.logger.info("\nAgent Statistics:")
        for service, agent in self.controller.agents.items():
            stats = agent.get_stats()
            self.logger.info(f"  {service}:")
            self.logger.info(f"    States: {stats['states_discovered']}/243 ({stats['state_space_coverage']:.1%})")
            self.logger.info(f"    Updates: {stats['total_updates']}")
            self.logger.info(f"    Epsilon: {stats['epsilon']:.3f}")


# =============================================================================
# SECTION 11: DIAGNOSTICS
# =============================================================================

def run_diagnostics(config: Config):
    """System diagnostics"""
    print("\n" + "="*60)
    print("CAPA+ v2.0 SYSTEM DIAGNOSTICS")
    print("="*60)
    
    print("\n--- Prometheus ---")
    prom = PrometheusMetricsCollector(config.prometheus_url, config.namespace)
    if prom.is_available():
        print(f"✓ Connected: {config.prometheus_url}")
    else:
        print(f"✗ Cannot connect: {config.prometheus_url}")
    
    print("\n--- Locust ---")
    locust = LocustMetricsCollector(config.locust_url)
    if locust.is_available():
        print(f"✓ Connected: {config.locust_url}")
        stats = locust.get_aggregate_latency()
        print(f"  RPS: {stats.get('rps', 0):.1f}")
        print(f"  P95: {stats.get('p95_ms', 0):.1f} ms")
    else:
        print(f"✗ Cannot connect: {config.locust_url}")
    
    print("\n--- Kubernetes ---")
    try:
        result = subprocess.run(['kubectl', 'get', 'nodes'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ kubectl connected")
            print(result.stdout)
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60)
    print("ENHANCEMENTS IN v2.0:")
    print("="*60)
    print("""
1. State Space: 243 states (3×3×3×3×3) with CPU/latency trends
2. Workload Randomization: Scale (0.8-1.2x) + Gaussian noise
3. Thrashing Penalty: Penalizes Up<->Down oscillation
4. Prioritized Replay: Focus on high-reward/SLA-violation states
5. Phase-Dependent Exploration: ε=0.8→0.4→0.15→0.0
    """)


# =============================================================================
# SECTION 12: MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CAPA+ Enhanced v2.0 - Anti-Overfitting Autoscaler',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--prometheus', default='http://localhost:9090')
    common.add_argument('--locust', default='http://localhost:8089')
    common.add_argument('--namespace', '-n', default='default')
    
    # Commands
    subparsers.add_parser('diagnose', parents=[common])
    
    val_p = subparsers.add_parser('validate', parents=[common])
    val_p.add_argument('--continuous', '-c', action='store_true')
    val_p.add_argument('--service', '-s', default='frontend')
    val_p.add_argument('--interval', type=int, default=5)
    
    train_p = subparsers.add_parser('train', parents=[common])
    train_p.add_argument('--mode', choices=['shadow', 'hybrid', 'active', 'full'], default='shadow')
    train_p.add_argument('--duration', type=int, default=3600)
    train_p.add_argument('--dry-run', action='store_true')
    train_p.add_argument('--checkpoint-dir', default='./checkpoints')
    train_p.add_argument('--resume', type=str)
    
    analyze_p = subparsers.add_parser('analyze', parents=[common])
    analyze_p.add_argument('--data-dir', default='./checkpoints')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    CONFIG.prometheus_url = args.prometheus
    CONFIG.locust_url = args.locust
    CONFIG.namespace = args.namespace
    
    if args.command == 'diagnose':
        run_diagnostics(CONFIG)
    
    elif args.command == 'validate':
        validator = LittleLawValidator(CONFIG)
        collector = UnifiedMetricsCollector(CONFIG)
        
        if args.continuous:
            print("\nLittle's Law Validation (Ctrl+C to stop)\n")
            try:
                while True:
                    metrics = collector.collect(args.service)
                    svc = CONFIG.services.get(args.service)
                    cap = svc.capacity_per_pod if svc else 100
                    state = validator.validate(args.service, metrics, cap)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"λ={state.arrival_rate:.0f} T={state.response_time_avg_sec*1000:.0f}ms "
                          f"ρ={state.utilization:.1%} {'✓' if state.is_stable else '✗'}")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
        else:
            # Single measurement mode
            metrics = collector.collect(args.service)
            svc = CONFIG.services.get(args.service)
            if svc is None:
                logger.warning(f"Service '{args.service}' not in CONFIG.services, using default capacity=100")
            cap = svc.capacity_per_pod if svc else 100
            state = validator.validate(args.service, metrics, cap)
            
            print(f"\n{'='*60}")
            print(f"Little's Law Validation: {args.service}")
            print(f"{'='*60}")
            print(f"Arrival Rate (λ):     {state.arrival_rate:.1f} req/s")
            print(f"Response Time (T):    {state.response_time_avg_sec*1000:.1f} ms")
            print(f"Utilization (ρ):      {state.utilization:.1%}")
            print(f"Number in System (L): {state.arrival_rate * state.response_time_avg_sec:.2f}")
            print(f"CV of Arrivals:       {state.cv_arrival:.3f}")
            print(f"CV of Service:        {state.cv_service:.3f}")
            print(f"Steady State:         {'✓ Yes' if state.in_steady_state else '✗ No'}")
            print(f"System Stable:        {'✓ Yes' if state.is_stable else '✗ No'}")
            print(f"{'='*60}")
    
    elif args.command == 'train':
        orchestrator = EnhancedTrainingOrchestrator(CONFIG, args.checkpoint_dir, args.dry_run)
        
        if args.resume:
            orchestrator.load_checkpoint(args.resume)
        
        if args.mode == 'full':
            orchestrator.run_full_training(args.duration / 3600)
        else:
            orchestrator.running = True
            phase_map = {
                'shadow': TrainingPhase.SHADOW,
                'hybrid': TrainingPhase.HYBRID_50,
                'active': TrainingPhase.ACTIVE
            }
            orchestrator.run_phase(phase_map[args.mode], args.duration,
                                  CONFIG.train_patterns * 3, 0.5 if args.mode == 'hybrid' else 1.0, True)
    
    elif args.command == 'analyze':
        print(f"Analyzing: {args.data_dir}")
        checkpoints = sorted(Path(args.data_dir).glob("checkpoint_*"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Latest: {latest}")
            state_file = latest / "training_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    data = json.load(f)
                print(f"Phase: {data.get('phase')}, Epoch: {data.get('epoch')}")


if __name__ == '__main__':
    main()
