#!/usr/bin/env python3
"""
Unified CAPA+ (Elascale) MAPE-K Controller with Anti-Overfitting Measures

This implementation incorporates academic best practices from:
- Jain (1991): The Art of Computer Systems Performance Analysis
- Harchol-Balter (2013): Performance Modeling and Design of Computer Systems
- INTROD_1: Introduction to Computer System Performance Evaluation

Key Features:
- Shadow Mode / Hybrid Mode / Active Mode for safe RL training
- Reduced state space (8 states) to prevent overfitting
- Double Q-Learning to reduce overestimation bias
- Prioritized Experience Replay for diverse learning
- Exponential averaging for metric smoothing (INTROD_1 Eq. 5.3)
- Multiple load patterns with train/test split
- Proper reward function following Jain's Rule 1

Author: EECS6446 Cloud Computing Project
Date: November 2025
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import threading

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Service configurations for Online Boutique
SERVICE_CONFIGS = {
    'frontend': {
        'min': 1, 'max': 10,
        'target_cpu': 50, 'target_latency_ms': 200,
        'scale_up_threshold': 70, 'scale_down_threshold': 30
    },
    'recommendationservice': {
        'min': 1, 'max': 5,
        'target_cpu': 50, 'target_latency_ms': 100,
        'scale_up_threshold': 70, 'scale_down_threshold': 30
    },
    'productcatalogservice': {
        'min': 1, 'max': 5,
        'target_cpu': 50, 'target_latency_ms': 100,
        'scale_up_threshold': 70, 'scale_down_threshold': 30
    },
    'cartservice': {
        'min': 1, 'max': 5,
        'target_cpu': 50, 'target_latency_ms': 100,
        'scale_up_threshold': 70, 'scale_down_threshold': 30
    },
    'checkoutservice': {
        'min': 1, 'max': 5,
        'target_cpu': 50, 'target_latency_ms': 150,
        'scale_up_threshold': 70, 'scale_down_threshold': 30
    },
}

# Load patterns for training and testing (Jain Ch. 11 - avoid ratio games)
LOAD_PATTERNS = {
    # Training patterns (75%)
    'step': [50, 100, 300, 500, 600, 500, 300, 100],      # Original pattern
    'gradual': [50, 150, 250, 350, 450, 550, 450, 250],   # Linear ramp
    'sine': [300, 477, 550, 477, 300, 123, 50, 123],      # Sinusoidal
    
    # Testing patterns (25%) - NEVER train on these
    'spike': [50, 50, 800, 100, 50, 700, 50, 50],         # Sudden spikes
    'random': None,  # Generated at runtime
}

TRAIN_PATTERNS = ['step', 'gradual', 'sine']
TEST_PATTERNS = ['spike', 'random']

# Prometheus endpoint
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:9090')

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger('CAPA+')


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ControlMode(Enum):
    """Controller operating modes"""
    SHADOW = "shadow"       # RL observes, baseline controls
    HYBRID = "hybrid"       # Probabilistic mix
    ACTIVE = "active"       # RL controls


class ScalingAction(Enum):
    """Possible scaling actions"""
    SCALE_DOWN = 0
    STAY = 1
    SCALE_UP = 2


@dataclass
class ServiceMetrics:
    """Metrics for a single service at a point in time"""
    timestamp: float
    service: str
    cpu_util: float
    memory_util: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    request_rate: float
    error_rate: float
    pods_desired: int
    pods_ready: int
    pods_available: int


@dataclass
class ScalingDecision:
    """Record of a scaling decision"""
    timestamp: float
    service: str
    state: Tuple
    rl_action: int
    baseline_action: int
    actual_action: int
    decision_source: str  # 'rl', 'baseline', 'hybrid'
    metrics_before: Dict
    reward: Optional[float] = None


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    pattern_name: str
    mode: str
    start_time: float
    end_time: float
    metrics_history: List[Dict] = field(default_factory=list)
    decisions_history: List[Dict] = field(default_factory=list)
    
    # Summary statistics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_cpu_util: float = 0.0
    avg_pods: float = 0.0
    total_scaling_actions: int = 0
    sla_violations: int = 0
    fault_rate: float = 0.0


# =============================================================================
# METRIC SMOOTHER (INTROD_1 Section 2.5.1)
# =============================================================================

class MetricSmoother:
    """
    Exponential averaging for metric smoothing
    
    From INTROD_1 Section 2.5.1, Equation 5.3:
    PI(k) = α × PI(k-1) + (1-α) × R(k)
    
    Higher α = more weight on past (smoother, slower response)
    Typical α = 0.3 for responsive smoothing
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.values: Dict[str, float] = {}
        self.trends: Dict[str, float] = {}
    
    def update(self, key: str, raw_value: float) -> Tuple[float, float]:
        """
        Update smoothed value and compute trend
        
        Returns:
            (smoothed_value, trend)
        """
        if key not in self.values:
            self.values[key] = raw_value
            self.trends[key] = 0.0
        else:
            old_val = self.values[key]
            # Exponential moving average
            self.values[key] = self.alpha * old_val + (1 - self.alpha) * raw_value
            # Trend = rate of change
            self.trends[key] = raw_value - old_val
        
        return self.values[key], self.trends[key]
    
    def get(self, key: str) -> Tuple[float, float]:
        """Get current smoothed value and trend"""
        return self.values.get(key, 0.0), self.trends.get(key, 0.0)
    
    def reset(self):
        """Reset all smoothed values"""
        self.values.clear()
        self.trends.clear()


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Samples experiences based on TD-error priority, ensuring diverse
    learning and preventing memorization of recent sequences.
    
    From Schaul et al. (2015) and adapted for tabular Q-learning.
    """
    
    def __init__(self, capacity: int = 500, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent (0 = uniform, 1 = full prioritization)
        self.beta = beta    # Importance sampling exponent
        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.position = 0
    
    def add(self, state: Tuple, action: int, reward: float, 
            next_state: Tuple, td_error: float):
        """Add experience with priority based on TD-error"""
        priority = (abs(td_error) + 0.01) ** self.alpha
        
        experience = (state, action, reward, next_state)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int = 16) -> List[Tuple]:
        """Sample batch based on priorities"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Convert priorities to probabilities
        priorities_array = np.array(self.priorities)
        probs = priorities_array / priorities_array.sum()
        
        # Sample without replacement
        indices = np.random.choice(
            len(self.buffer), 
            size=min(batch_size, len(self.buffer)),
            p=probs,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# DOUBLE Q-LEARNING AGENT (Harchol-Balter variance discussion)
# =============================================================================

class DoubleQLearningAgent:
    """
    Double Q-Learning Agent with Anti-Overfitting Measures
    
    Key features:
    - Double Q-Learning to reduce overestimation bias
    - Reduced state space (8 states) for better generalization
    - Reward noise injection
    - Q-value regularization (weight decay)
    - Slower epsilon decay with higher floor
    - Prioritized experience replay
    
    Based on:
    - Harchol-Balter Ch. 14: Variance reduction
    - Jain Ch. 6: Model simplicity for generalization
    - INTROD_1 Ch. 3: Statistical estimation
    """
    
    def __init__(
        self,
        service_name: str,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.10,
        weight_decay: float = 0.001,
        reward_noise_std: float = 0.1
    ):
        self.service_name = service_name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.weight_decay = weight_decay
        self.reward_noise_std = reward_noise_std
        
        # Double Q-Learning: Two Q-tables
        self.q_table_A: Dict[Tuple, List[float]] = {}
        self.q_table_B: Dict[Tuple, List[float]] = {}
        self.use_A = True  # Alternates which table to update
        
        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=500)
        
        # Statistics
        self.total_updates = 0
        self.state_visits: Dict[Tuple, int] = defaultdict(int)
        
        # REDUCED state thresholds (2 buckets per dimension = 8 total states)
        # From Jain Ch. 6: Simpler models generalize better
        self.cpu_threshold = 0.5      # Low / High
        self.lat_threshold = 1.0      # Good / Bad (relative to SLA)
        self.pod_threshold = 0.5      # Few / Many
        
        logger.info(f"[{service_name}] DoubleQLearningAgent initialized")
        logger.info(f"  State space: 2×2×2 = 8 states (reduced for generalization)")
        logger.info(f"  ε={epsilon}, decay={epsilon_decay}, min={epsilon_min}")
    
    def get_state(
        self,
        cpu_util: float,
        latency_score: float,
        pod_ratio: float,
        **kwargs
    ) -> Tuple[int, int, int]:
        """
        Convert continuous metrics to discrete state
        
        REDUCED state space (8 states) to prevent overfitting
        """
        cpu_state = 0 if cpu_util < self.cpu_threshold else 1
        lat_state = 0 if latency_score < self.lat_threshold else 1
        pod_state = 0 if pod_ratio < self.pod_threshold else 1
        
        return (cpu_state, lat_state, pod_state)
    
    def _init_q_values(self, state: Tuple) -> List[float]:
        """Initialize Q-values for a new state with slight preference for STAY"""
        return [0.0, 0.1, 0.0]  # [SCALE_DOWN, STAY, SCALE_UP]
    
    def _ensure_state_exists(self, state: Tuple):
        """Ensure state exists in both Q-tables"""
        if state not in self.q_table_A:
            self.q_table_A[state] = self._init_q_values(state)
        if state not in self.q_table_B:
            self.q_table_B[state] = self._init_q_values(state)
    
    def get_q_values(self, state: Tuple) -> List[float]:
        """Get averaged Q-values from both tables"""
        self._ensure_state_exists(state)
        q_a = self.q_table_A[state]
        q_b = self.q_table_B[state]
        return [(a + b) / 2 for a, b in zip(q_a, q_b)]
    
    def choose_action(self, state: Tuple, training: bool = True) -> int:
        """
        Choose action using ε-greedy policy
        
        Args:
            state: Current state tuple
            training: If False, use pure exploitation (ε=0)
        
        Returns:
            Action index (0=SCALE_DOWN, 1=STAY, 2=SCALE_UP)
        """
        self._ensure_state_exists(state)
        self.state_visits[state] += 1
        
        effective_epsilon = self.epsilon if training else 0.0
        
        if np.random.random() < effective_epsilon:
            return np.random.randint(0, 3)
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))
    
    def learn(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple
    ) -> float:
        """
        Double Q-Learning update with regularization
        
        Returns:
            TD-error for prioritized replay
        """
        # Add noise to reward to prevent memorization
        noisy_reward = reward + np.random.normal(0, self.reward_noise_std)
        noisy_reward = np.clip(noisy_reward, -2.0, 2.0)
        
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        
        # Double Q-Learning: Use one table to select action, other to evaluate
        if self.use_A:
            # Use A to select best action, B to evaluate it
            best_action = int(np.argmax(self.q_table_A[next_state]))
            target = noisy_reward + self.gamma * self.q_table_B[next_state][best_action]
            
            old_q = self.q_table_A[state][action]
            td_error = target - old_q
            new_q = old_q + self.alpha * td_error
            
            # Apply weight decay (regularization)
            new_q *= (1 - self.weight_decay)
            self.q_table_A[state][action] = new_q
        else:
            # Use B to select best action, A to evaluate it
            best_action = int(np.argmax(self.q_table_B[next_state]))
            target = noisy_reward + self.gamma * self.q_table_A[next_state][best_action]
            
            old_q = self.q_table_B[state][action]
            td_error = target - old_q
            new_q = old_q + self.alpha * td_error
            
            # Apply weight decay
            new_q *= (1 - self.weight_decay)
            self.q_table_B[state][action] = new_q
        
        # Alternate tables
        self.use_A = not self.use_A
        
        # Store in replay buffer
        self.replay_buffer.add(state, action, noisy_reward, next_state, td_error)
        
        # Replay from buffer
        if len(self.replay_buffer) >= 16:
            self._replay_batch(batch_size=8)
        
        self.total_updates += 1
        
        # Periodic global shrinkage
        if self.total_updates % 100 == 0:
            self._apply_global_shrinkage()
        
        return td_error
    
    def _replay_batch(self, batch_size: int = 8):
        """Replay experiences from buffer"""
        batch = self.replay_buffer.sample(batch_size)
        
        for state, action, reward, next_state in batch:
            self._ensure_state_exists(state)
            self._ensure_state_exists(next_state)
            
            # Use averaged Q-values for replay
            q_values = self.get_q_values(state)
            next_q_values = self.get_q_values(next_state)
            
            old_q = q_values[action]
            target = reward + self.gamma * max(next_q_values)
            
            # Smaller learning rate for replay
            replay_alpha = 0.5 * self.alpha
            new_q = old_q + replay_alpha * (target - old_q)
            
            # Update both tables slightly
            self.q_table_A[state][action] = 0.5 * (self.q_table_A[state][action] + new_q)
            self.q_table_B[state][action] = 0.5 * (self.q_table_B[state][action] + new_q)
    
    def _apply_global_shrinkage(self, factor: float = 0.99):
        """Shrink all Q-values toward zero to prevent extreme values"""
        for state in self.q_table_A:
            self.q_table_A[state] = [q * factor for q in self.q_table_A[state]]
        for state in self.q_table_B:
            self.q_table_B[state] = [q * factor for q in self.q_table_B[state]]
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            'service': self.service_name,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'states_discovered': len(self.q_table_A),
            'replay_buffer_size': len(self.replay_buffer),
            'q_table_A': {str(k): v for k, v in self.q_table_A.items()},
            'q_table_B': {str(k): v for k, v in self.q_table_B.items()},
            'state_visits': {str(k): v for k, v in self.state_visits.items()}
        }
    
    def save(self, filepath: str):
        """Save agent state to file"""
        state = {
            'service_name': self.service_name,
            'q_table_A': {str(k): v for k, v in self.q_table_A.items()},
            'q_table_B': {str(k): v for k, v in self.q_table_B.items()},
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'state_visits': {str(k): v for k, v in self.state_visits.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"[{self.service_name}] Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Convert string keys back to tuples
        self.q_table_A = {eval(k): v for k, v in state['q_table_A'].items()}
        self.q_table_B = {eval(k): v for k, v in state['q_table_B'].items()}
        self.epsilon = state['epsilon']
        self.total_updates = state['total_updates']
        self.state_visits = defaultdict(int, {eval(k): v for k, v in state['state_visits'].items()})
        
        logger.info(f"[{self.service_name}] Agent loaded from {filepath}")


# =============================================================================
# BASELINE HPA CONTROLLER
# =============================================================================

class BaselineHPAController:
    """
    Simple CPU-based Horizontal Pod Autoscaler
    
    Mimics Kubernetes HPA behavior for comparison baseline.
    """
    
    def __init__(self):
        self.cooldown_until: Dict[str, float] = {}
        self.cooldown_period = 30.0  # seconds
    
    def decide(self, service: str, metrics: Dict) -> int:
        """
        Make scaling decision based on CPU utilization
        
        Returns:
            Action index (0=SCALE_DOWN, 1=STAY, 2=SCALE_UP)
        """
        config = SERVICE_CONFIGS[service]
        cpu_util = metrics.get('cpu_util', 0.5)
        current_pods = metrics.get('pods_ready', 1)
        
        # Check cooldown
        if service in self.cooldown_until:
            if time.time() < self.cooldown_until[service]:
                return ScalingAction.STAY.value
        
        # Simple threshold-based decision
        if cpu_util > config['scale_up_threshold'] / 100.0:
            if current_pods < config['max']:
                self.cooldown_until[service] = time.time() + self.cooldown_period
                return ScalingAction.SCALE_UP.value
        elif cpu_util < config['scale_down_threshold'] / 100.0:
            if current_pods > config['min']:
                self.cooldown_until[service] = time.time() + self.cooldown_period
                return ScalingAction.SCALE_DOWN.value
        
        return ScalingAction.STAY.value


# =============================================================================
# REWARD CALCULATOR (Jain Ch. 12 - Rule 1)
# =============================================================================

class RewardCalculator:
    """
    Reward function following Jain's principles
    
    From Jain Ch. 12, Rule 1: "The sum must have physical meaning"
    
    Three weighted objectives:
    - SLA compliance (50%): Primary objective
    - Resource efficiency (30%): Secondary objective
    - Stability (20%): Anti-oscillation
    """
    
    def __init__(self):
        # Weights must sum to 1.0
        self.w_sla = 0.50
        self.w_efficiency = 0.30
        self.w_stability = 0.20
        
        # Noise for anti-overfitting
        self.noise_std = 0.1
    
    def calculate(
        self,
        service: str,
        latency_score: float,  # ratio to target (1.0 = at target)
        cpu_util: float,       # 0-1
        pods: int,
        ready_pods: int,
        action_taken: int
    ) -> float:
        """
        Calculate reward for a scaling decision
        
        Args:
            service: Service name
            latency_score: Current latency / target latency
            cpu_util: CPU utilization (0-1)
            pods: Total pods
            ready_pods: Ready pods
            action_taken: Action that was taken (0, 1, 2)
        
        Returns:
            Reward value (clipped to [-2, 2])
        """
        config = SERVICE_CONFIGS[service]
        ready_ratio = ready_pods / max(1, pods)
        
        # (1) SLA REWARD (50% weight)
        if latency_score <= 0.8:
            sla_reward = 1.0      # Excellent - well below SLA
        elif latency_score <= 1.0:
            sla_reward = 0.5      # Good - meeting SLA
        elif latency_score <= 1.5:
            sla_reward = -0.5     # Degraded - slightly over SLA
        else:
            sla_reward = -1.0     # SLA violation
        
        # (2) EFFICIENCY REWARD (30% weight)
        # Target: 50-70% CPU utilization (optimal operating point)
        if 0.5 <= cpu_util <= 0.7:
            eff_reward = 1.0      # Optimal range
        elif 0.3 <= cpu_util < 0.5 or 0.7 < cpu_util <= 0.85:
            eff_reward = 0.3      # Acceptable
        else:
            eff_reward = -0.5     # Inefficient (too low or too high)
        
        # Penalize over-provisioning when SLA is excellent
        if latency_score < 0.8 and pods > config['min'] + 2:
            eff_reward -= 0.3
        
        # (3) STABILITY REWARD (20% weight)
        stability_reward = 0.0
        
        # Penalize scaling when pods aren't ready
        if ready_ratio < 0.8 and action_taken != ScalingAction.STAY.value:
            stability_reward = -1.0
        # Slight penalty for any scaling action (prefer stability)
        elif action_taken == ScalingAction.SCALE_UP.value:
            stability_reward = -0.2
        elif action_taken == ScalingAction.SCALE_DOWN.value:
            stability_reward = -0.1
        
        # WEIGHTED COMBINATION
        base_reward = (
            self.w_sla * sla_reward +
            self.w_efficiency * eff_reward +
            self.w_stability * stability_reward
        )
        
        # Add noise for anti-overfitting
        noise = np.random.normal(0, self.noise_std)
        reward = base_reward + noise
        
        # Clip to prevent extreme values
        return float(np.clip(reward, -2.0, 2.0))


# =============================================================================
# LITTLE'S LAW PREDICTOR (Harchol-Balter Ch. 6)
# =============================================================================

class LittleLawPredictor:
    """
    Proactive scaling using Little's Law
    
    From Harchol-Balter Chapter 6:
    N = λ × T
    
    Where:
    - N = number in system (expected queue length)
    - λ = arrival rate (requests/sec)
    - T = mean response time (seconds)
    """
    
    def __init__(self, concurrent_per_pod: int = 10):
        self.concurrent_per_pod = concurrent_per_pod
    
    def estimate_optimal_pods(
        self,
        service: str,
        arrival_rate: float,
        target_latency_sec: float
    ) -> int:
        """
        Estimate optimal pod count using Little's Law
        
        Args:
            service: Service name
            arrival_rate: Requests per second
            target_latency_sec: Target response time in seconds
        
        Returns:
            Estimated optimal number of pods
        """
        config = SERVICE_CONFIGS[service]
        
        # Expected number in system at target latency
        expected_queue = arrival_rate * target_latency_sec
        
        # Required pods to handle this queue
        optimal_pods = int(np.ceil(expected_queue / self.concurrent_per_pod))
        
        # Clamp to service limits
        return max(config['min'], min(optimal_pods, config['max']))


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Collects metrics from Prometheus and Kubernetes
    """
    
    def __init__(self, prometheus_url: str = PROMETHEUS_URL):
        self.prometheus_url = prometheus_url
        self.smoother = MetricSmoother(alpha=0.3)
    
    def query_prometheus(self, query: str) -> Optional[float]:
        """Execute PromQL query and return result"""
        try:
            import requests
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=5
            )
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                return float(data['data']['result'][0]['value'][1])
            return None
        except Exception as e:
            logger.warning(f"Prometheus query failed: {e}")
            return None
    
    def get_pod_count(self, service: str, namespace: str = 'default') -> Tuple[int, int, int]:
        """Get pod counts (desired, ready, available) from kubectl"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'deployment', service, '-n', namespace,
                 '-o', 'jsonpath={.spec.replicas},{.status.readyReplicas},{.status.availableReplicas}'],
                capture_output=True, text=True, timeout=10
            )
            parts = result.stdout.strip().split(',')
            desired = int(parts[0]) if parts[0] else 1
            ready = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            available = int(parts[2]) if len(parts) > 2 and parts[2] else 0
            return desired, ready, available
        except Exception as e:
            logger.warning(f"Failed to get pod count for {service}: {e}")
            return 1, 1, 1
    
    def collect_service_metrics(self, service: str) -> ServiceMetrics:
        """Collect all metrics for a service"""
        config = SERVICE_CONFIGS[service]
        
        # Query Prometheus for various metrics
        cpu_query = f'avg(rate(container_cpu_usage_seconds_total{{pod=~"{service}.*"}}[1m]))'
        mem_query = f'avg(container_memory_usage_bytes{{pod=~"{service}.*"}})'
        lat_p50_query = f'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{{service="{service}"}}[1m]))'
        lat_p95_query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service}"}}[1m]))'
        lat_p99_query = f'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{service="{service}"}}[1m]))'
        rate_query = f'sum(rate(http_requests_total{{service="{service}"}}[1m]))'
        error_query = f'sum(rate(http_requests_total{{service="{service}",status=~"5.."}}[1m]))'
        
        # Collect raw values
        raw_cpu = self.query_prometheus(cpu_query) or 0.0
        raw_mem = self.query_prometheus(mem_query) or 0.0
        raw_lat_p50 = (self.query_prometheus(lat_p50_query) or 0.0) * 1000  # to ms
        raw_lat_p95 = (self.query_prometheus(lat_p95_query) or 0.0) * 1000
        raw_lat_p99 = (self.query_prometheus(lat_p99_query) or 0.0) * 1000
        raw_rate = self.query_prometheus(rate_query) or 0.0
        raw_error = self.query_prometheus(error_query) or 0.0
        
        # Get pod counts
        pods_desired, pods_ready, pods_available = self.get_pod_count(service)
        
        # Apply exponential smoothing
        cpu_util, _ = self.smoother.update(f'{service}_cpu', raw_cpu)
        lat_p95, lat_trend = self.smoother.update(f'{service}_lat', raw_lat_p95)
        
        return ServiceMetrics(
            timestamp=time.time(),
            service=service,
            cpu_util=cpu_util,
            memory_util=raw_mem / (1024 * 1024 * 1024),  # to GB
            latency_p50_ms=raw_lat_p50,
            latency_p95_ms=lat_p95,
            latency_p99_ms=raw_lat_p99,
            request_rate=raw_rate,
            error_rate=raw_error / max(raw_rate, 0.001),
            pods_desired=pods_desired,
            pods_ready=pods_ready,
            pods_available=pods_available
        )
    
    def collect_all_metrics(self) -> Dict[str, ServiceMetrics]:
        """Collect metrics for all services"""
        metrics = {}
        for service in SERVICE_CONFIGS:
            metrics[service] = self.collect_service_metrics(service)
        return metrics


# =============================================================================
# SCALING EXECUTOR
# =============================================================================

class ScalingExecutor:
    """
    Executes scaling decisions via kubectl
    """
    
    def __init__(self, namespace: str = 'default', dry_run: bool = False):
        self.namespace = namespace
        self.dry_run = dry_run
        self.scaling_history: List[Dict] = []
    
    def scale(self, service: str, action: int, current_pods: int) -> bool:
        """
        Execute scaling action
        
        Args:
            service: Service name
            action: Action (0=down, 1=stay, 2=up)
            current_pods: Current pod count
        
        Returns:
            True if scaling was executed
        """
        config = SERVICE_CONFIGS[service]
        
        if action == ScalingAction.STAY.value:
            return False
        
        if action == ScalingAction.SCALE_UP.value:
            new_pods = min(current_pods + 1, config['max'])
        else:  # SCALE_DOWN
            new_pods = max(current_pods - 1, config['min'])
        
        if new_pods == current_pods:
            return False
        
        self.scaling_history.append({
            'timestamp': time.time(),
            'service': service,
            'action': 'scale_up' if action == ScalingAction.SCALE_UP.value else 'scale_down',
            'from_pods': current_pods,
            'to_pods': new_pods
        })
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would scale {service}: {current_pods} → {new_pods}")
            return True
        
        try:
            cmd = [
                'kubectl', 'scale', 'deployment', service,
                f'--replicas={new_pods}',
                '-n', self.namespace
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Scaled {service}: {current_pods} → {new_pods}")
                return True
            else:
                logger.error(f"Scaling failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Scaling error: {e}")
            return False


# =============================================================================
# SHADOW MODE ANALYZER
# =============================================================================

class ShadowModeAnalyzer:
    """
    Analyzes Shadow Mode logs to detect overfitting and readiness
    """
    
    def __init__(self, decisions: List[ScalingDecision]):
        self.decisions = decisions
    
    def agreement_rate(self) -> float:
        """Calculate how often RL agrees with baseline"""
        if not self.decisions:
            return 0.0
        
        agreed = sum(
            1 for d in self.decisions 
            if d.rl_action == d.baseline_action
        )
        return agreed / len(self.decisions)
    
    def analyze_disagreements(self) -> Dict:
        """Analyze cases where RL and baseline disagree"""
        disagreements = [d for d in self.decisions if d.rl_action != d.baseline_action]
        
        if not disagreements:
            return {'total': 0, 'rl_better': 0, 'baseline_better': 0}
        
        rl_better = 0
        baseline_better = 0
        
        # Simple heuristic: check subsequent metrics
        for i, d in enumerate(disagreements):
            # Find next decision for same service
            for j in range(i + 1, len(self.decisions)):
                if self.decisions[j].service == d.service:
                    next_d = self.decisions[j]
                    # Compare latency outcomes
                    if next_d.metrics_before.get('latency_score', 1.0) > 1.2:
                        # High latency after - who would have helped?
                        if d.rl_action == ScalingAction.SCALE_UP.value:
                            rl_better += 1
                        else:
                            baseline_better += 1
                    break
        
        return {
            'total': len(disagreements),
            'rl_better': rl_better,
            'baseline_better': baseline_better,
            'rl_win_rate': rl_better / max(1, rl_better + baseline_better)
        }
    
    def ready_for_active_mode(self, min_observations: int = 500) -> Tuple[bool, Dict]:
        """
        Check if RL is ready to transition from Shadow to Active mode
        
        Criteria from INTROD_1: Validate before deployment
        """
        agreement = self.agreement_rate()
        disagreement_analysis = self.analyze_disagreements()
        
        criteria = {
            'sufficient_observations': len(self.decisions) >= min_observations,
            'reasonable_agreement': agreement >= 0.6,
            'rl_adds_value': disagreement_analysis['rl_win_rate'] >= 0.4,
        }
        
        ready = all(criteria.values())
        
        return ready, {
            'ready': ready,
            'observations': len(self.decisions),
            'agreement_rate': agreement,
            'disagreement_analysis': disagreement_analysis,
            'criteria': criteria
        }


# =============================================================================
# MAPE-K CONTROLLER
# =============================================================================

class MAPEKController:
    """
    MAPE-K (Monitor-Analyze-Plan-Execute-Knowledge) Controller
    
    Implements the CAPA+ autoscaling strategy with:
    - Shadow/Hybrid/Active modes for safe RL training
    - Double Q-Learning with anti-overfitting measures
    - Proper reward function following Jain's principles
    - Little's Law for proactive scaling hints
    """
    
    def __init__(
        self,
        mode: ControlMode = ControlMode.SHADOW,
        namespace: str = 'default',
        dry_run: bool = False,
        rl_authority: float = 0.0
    ):
        self.mode = mode
        self.rl_authority = rl_authority  # For hybrid mode (0-1)
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.scaling_executor = ScalingExecutor(namespace, dry_run)
        self.baseline_hpa = BaselineHPAController()
        self.reward_calculator = RewardCalculator()
        self.littles_law = LittleLawPredictor()
        
        # RL Agents (one per service)
        self.rl_agents: Dict[str, DoubleQLearningAgent] = {}
        for service in SERVICE_CONFIGS:
            self.rl_agents[service] = DoubleQLearningAgent(service_name=service)
        
        # State tracking
        self.previous_states: Dict[str, Tuple] = {}
        self.previous_actions: Dict[str, int] = {}
        self.previous_metrics: Dict[str, ServiceMetrics] = {}
        
        # History
        self.decisions_history: List[ScalingDecision] = []
        self.metrics_history: List[Dict] = []
        
        # Control flags
        self.running = False
        self.learning_enabled = True
        
        logger.info(f"MAPEKController initialized in {mode.value} mode")
    
    def set_mode(self, mode: ControlMode, rl_authority: float = 0.5):
        """Change controller mode"""
        self.mode = mode
        self.rl_authority = rl_authority
        logger.info(f"Mode changed to {mode.value} (RL authority: {rl_authority:.0%})")
    
    def set_learning(self, enabled: bool):
        """Enable/disable learning"""
        self.learning_enabled = enabled
        logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
    
    # -------------------------------------------------------------------------
    # MAPE-K Phases
    # -------------------------------------------------------------------------
    
    def monitor(self) -> Dict[str, ServiceMetrics]:
        """MONITOR phase: Collect metrics from all services"""
        return self.metrics_collector.collect_all_metrics()
    
    def analyze(self, metrics: Dict[str, ServiceMetrics]) -> Dict[str, Dict]:
        """ANALYZE phase: Convert metrics to states and scores"""
        analysis = {}
        
        for service, m in metrics.items():
            config = SERVICE_CONFIGS[service]
            
            # Calculate scores (normalized to target)
            latency_score = m.latency_p95_ms / config['target_latency_ms']
            cpu_score = m.cpu_util / (config['target_cpu'] / 100.0)
            pod_ratio = m.pods_ready / config['max']
            ready_ratio = m.pods_ready / max(1, m.pods_desired)
            
            # Get RL state
            state = self.rl_agents[service].get_state(
                cpu_util=m.cpu_util,
                latency_score=latency_score,
                pod_ratio=pod_ratio
            )
            
            # Little's Law prediction
            littles_optimal = self.littles_law.estimate_optimal_pods(
                service,
                m.request_rate,
                config['target_latency_ms'] / 1000.0
            )
            
            analysis[service] = {
                'state': state,
                'latency_score': latency_score,
                'cpu_score': cpu_score,
                'pod_ratio': pod_ratio,
                'ready_ratio': ready_ratio,
                'littles_optimal': littles_optimal,
                'metrics': m
            }
        
        return analysis
    
    def plan(self, analysis: Dict[str, Dict]) -> Dict[str, ScalingDecision]:
        """PLAN phase: Decide scaling actions"""
        decisions = {}
        
        for service, a in analysis.items():
            m = a['metrics']
            state = a['state']
            
            # Get RL agent's decision
            rl_action = self.rl_agents[service].choose_action(
                state, 
                training=self.learning_enabled
            )
            
            # Get baseline HPA decision
            baseline_metrics = {
                'cpu_util': m.cpu_util,
                'pods_ready': m.pods_ready
            }
            baseline_action = self.baseline_hpa.decide(service, baseline_metrics)
            
            # Determine actual action based on mode
            if self.mode == ControlMode.SHADOW:
                actual_action = baseline_action
                decision_source = 'baseline'
            elif self.mode == ControlMode.ACTIVE:
                actual_action = rl_action
                decision_source = 'rl'
            else:  # HYBRID
                if np.random.random() < self.rl_authority:
                    actual_action = rl_action
                    decision_source = 'rl'
                else:
                    actual_action = baseline_action
                    decision_source = 'baseline'
            
            decisions[service] = ScalingDecision(
                timestamp=time.time(),
                service=service,
                state=state,
                rl_action=rl_action,
                baseline_action=baseline_action,
                actual_action=actual_action,
                decision_source=decision_source,
                metrics_before={
                    'cpu_util': m.cpu_util,
                    'latency_p95_ms': m.latency_p95_ms,
                    'latency_score': a['latency_score'],
                    'pods_ready': m.pods_ready,
                    'pods_desired': m.pods_desired,
                    'request_rate': m.request_rate
                }
            )
        
        return decisions
    
    def execute(self, decisions: Dict[str, ScalingDecision]) -> Dict[str, bool]:
        """EXECUTE phase: Apply scaling decisions"""
        results = {}
        
        for service, decision in decisions.items():
            m = decision.metrics_before
            executed = self.scaling_executor.scale(
                service,
                decision.actual_action,
                m['pods_ready']
            )
            results[service] = executed
            
            # Store for learning
            self.previous_states[service] = decision.state
            self.previous_actions[service] = decision.actual_action
        
        return results
    
    def knowledge(self, decisions: Dict[str, ScalingDecision], 
                  current_metrics: Dict[str, ServiceMetrics]):
        """KNOWLEDGE phase: Learn from outcomes"""
        if not self.learning_enabled:
            return
        
        for service, decision in decisions.items():
            if service not in self.previous_states:
                continue
            
            prev_state = self.previous_states[service]
            action = self.previous_actions[service]
            m = current_metrics[service]
            config = SERVICE_CONFIGS[service]
            
            # Calculate reward
            latency_score = m.latency_p95_ms / config['target_latency_ms']
            reward = self.reward_calculator.calculate(
                service=service,
                latency_score=latency_score,
                cpu_util=m.cpu_util,
                pods=m.pods_desired,
                ready_pods=m.pods_ready,
                action_taken=action
            )
            
            # Get current state
            curr_state = self.rl_agents[service].get_state(
                cpu_util=m.cpu_util,
                latency_score=latency_score,
                pod_ratio=m.pods_ready / config['max']
            )
            
            # Learn
            self.rl_agents[service].learn(prev_state, action, reward, curr_state)
            
            # Update decision with reward
            decision.reward = reward
        
        # Decay epsilon for all agents
        for agent in self.rl_agents.values():
            agent.decay_epsilon()
        
        # Store history
        for decision in decisions.values():
            self.decisions_history.append(decision)
    
    # -------------------------------------------------------------------------
    # Main Control Loop
    # -------------------------------------------------------------------------
    
    def control_loop_iteration(self) -> Dict[str, ScalingDecision]:
        """Single iteration of the MAPE-K control loop"""
        
        # MONITOR
        current_metrics = self.monitor()
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': {svc: asdict(m) for svc, m in current_metrics.items()}
        })
        
        # KNOWLEDGE (learn from previous iteration)
        if self.previous_metrics:
            # Use current metrics to evaluate previous decisions
            pass  # Learning happens after execution
        
        # ANALYZE
        analysis = self.analyze(current_metrics)
        
        # PLAN
        decisions = self.plan(analysis)
        
        # EXECUTE
        self.execute(decisions)
        
        # KNOWLEDGE (learn from this iteration's outcome)
        if self.previous_metrics:
            self.knowledge(
                {svc: self.decisions_history[-len(decisions) + i] 
                 for i, svc in enumerate(decisions.keys()) 
                 if len(self.decisions_history) > i},
                current_metrics
            )
        
        # Store for next iteration
        self.previous_metrics = current_metrics
        
        return decisions
    
    def run(self, duration_sec: int = 300, interval_sec: int = 10):
        """Run control loop for specified duration"""
        self.running = True
        start_time = time.time()
        iteration = 0
        
        logger.info(f"Starting control loop for {duration_sec}s with {interval_sec}s interval")
        
        while self.running and (time.time() - start_time) < duration_sec:
            iteration += 1
            iter_start = time.time()
            
            try:
                decisions = self.control_loop_iteration()
                
                # Log summary
                for svc, d in decisions.items():
                    action_name = ScalingAction(d.actual_action).name
                    logger.debug(
                        f"[{svc}] State={d.state} Action={action_name} "
                        f"(RL={ScalingAction(d.rl_action).name}, "
                        f"Baseline={ScalingAction(d.baseline_action).name})"
                    )
            except Exception as e:
                logger.error(f"Control loop error: {e}")
            
            # Wait for next iteration
            elapsed = time.time() - iter_start
            sleep_time = max(0, interval_sec - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.running = False
        logger.info(f"Control loop completed. {iteration} iterations.")
    
    def stop(self):
        """Stop the control loop"""
        self.running = False
    
    # -------------------------------------------------------------------------
    # Analysis and Reporting
    # -------------------------------------------------------------------------
    
    def get_shadow_analysis(self) -> Dict:
        """Get Shadow Mode analysis"""
        analyzer = ShadowModeAnalyzer(self.decisions_history)
        ready, analysis = analyzer.ready_for_active_mode()
        return analysis
    
    def get_agent_stats(self) -> Dict[str, Dict]:
        """Get statistics for all RL agents"""
        return {svc: agent.get_stats() for svc, agent in self.rl_agents.items()}
    
    def save_state(self, directory: str):
        """Save controller state to directory"""
        os.makedirs(directory, exist_ok=True)
        
        # Save agents
        for service, agent in self.rl_agents.items():
            agent.save(os.path.join(directory, f'{service}_agent.json'))
        
        # Save history
        with open(os.path.join(directory, 'decisions_history.json'), 'w') as f:
            json.dump(
                [asdict(d) if hasattr(d, '__dataclass_fields__') else d.__dict__ 
                 for d in self.decisions_history],
                f, indent=2, default=str
            )
        
        with open(os.path.join(directory, 'metrics_history.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        logger.info(f"State saved to {directory}")
    
    def load_state(self, directory: str):
        """Load controller state from directory"""
        for service, agent in self.rl_agents.items():
            agent_file = os.path.join(directory, f'{service}_agent.json')
            if os.path.exists(agent_file):
                agent.load(agent_file)
        
        logger.info(f"State loaded from {directory}")


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Runs experiments with proper train/test split
    
    Following Jain Ch. 16 experimental design principles
    """
    
    def __init__(self, controller: MAPEKController):
        self.controller = controller
        self.train_results: List[ExperimentResult] = []
        self.test_results: List[ExperimentResult] = []
    
    def get_load_pattern(self, pattern_name: str) -> List[int]:
        """Get load pattern by name"""
        if pattern_name == 'random':
            # Generate random pattern
            pattern = [50]
            for _ in range(7):
                delta = np.random.randint(-200, 300)
                next_val = max(50, min(1000, pattern[-1] + delta))
                pattern.append(next_val)
            return pattern
        return LOAD_PATTERNS[pattern_name]
    
    def run_experiment(
        self,
        pattern_name: str,
        phase_duration_sec: int = 60,
        is_training: bool = True
    ) -> ExperimentResult:
        """
        Run single experiment with a load pattern
        
        Args:
            pattern_name: Name of load pattern
            phase_duration_sec: Duration of each load phase
            is_training: If True, enable learning
        """
        pattern = self.get_load_pattern(pattern_name)
        
        # Configure controller
        self.controller.set_learning(is_training)
        if not is_training:
            # Disable exploration during testing
            for agent in self.controller.rl_agents.values():
                agent.epsilon = 0.0
        
        result = ExperimentResult(
            pattern_name=pattern_name,
            mode=self.controller.mode.value,
            start_time=time.time(),
            end_time=0.0
        )
        
        logger.info(f"Starting experiment: {pattern_name} ({'train' if is_training else 'test'})")
        logger.info(f"Load pattern: {pattern}")
        
        # Run through load phases
        for phase_idx, load_level in enumerate(pattern):
            logger.info(f"Phase {phase_idx + 1}/{len(pattern)}: {load_level} users")
            
            # In real deployment, this would adjust Locust load
            # Here we simulate by running the control loop
            phase_start = time.time()
            
            while (time.time() - phase_start) < phase_duration_sec:
                try:
                    decisions = self.controller.control_loop_iteration()
                    
                    # Record metrics
                    for svc, d in decisions.items():
                        result.metrics_history.append({
                            'timestamp': time.time(),
                            'phase': phase_idx,
                            'load_level': load_level,
                            'service': svc,
                            **d.metrics_before
                        })
                        result.decisions_history.append(asdict(d))
                except Exception as e:
                    logger.error(f"Experiment error: {e}")
                
                time.sleep(10)  # Control loop interval
        
        result.end_time = time.time()
        
        # Calculate summary statistics
        if result.metrics_history:
            latencies = [m.get('latency_p95_ms', 0) for m in result.metrics_history]
            cpus = [m.get('cpu_util', 0) for m in result.metrics_history]
            pods = [m.get('pods_ready', 1) for m in result.metrics_history]
            
            result.avg_latency_ms = np.mean(latencies) if latencies else 0
            result.p95_latency_ms = np.percentile(latencies, 95) if latencies else 0
            result.p99_latency_ms = np.percentile(latencies, 99) if latencies else 0
            result.avg_cpu_util = np.mean(cpus) if cpus else 0
            result.avg_pods = np.mean(pods) if pods else 0
            result.total_scaling_actions = sum(
                1 for d in result.decisions_history 
                if d.get('actual_action') != ScalingAction.STAY.value
            )
        
        # Store result
        if is_training:
            self.train_results.append(result)
        else:
            self.test_results.append(result)
        
        logger.info(f"Experiment complete: P95={result.p95_latency_ms:.1f}ms, "
                   f"Avg pods={result.avg_pods:.1f}")
        
        return result
    
    def run_training_pipeline(
        self,
        epochs: int = 3,
        phase_duration_sec: int = 60
    ):
        """
        Run full training pipeline with Shadow → Hybrid → Active progression
        """
        logger.info("="*60)
        logger.info("PHASE 1: SHADOW MODE TRAINING")
        logger.info("="*60)
        
        self.controller.set_mode(ControlMode.SHADOW)
        
        for epoch in range(epochs):
            pattern = TRAIN_PATTERNS[epoch % len(TRAIN_PATTERNS)]
            logger.info(f"\nEpoch {epoch + 1}/{epochs}: {pattern} pattern")
            self.run_experiment(pattern, phase_duration_sec, is_training=True)
        
        # Check readiness
        analysis = self.controller.get_shadow_analysis()
        logger.info(f"\nShadow Mode Analysis:")
        logger.info(f"  Observations: {analysis['observations']}")
        logger.info(f"  Agreement rate: {analysis['agreement_rate']:.1%}")
        
        if not analysis['ready']:
            logger.warning("Not ready for active mode. Continue shadow training.")
            return
        
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: HYBRID MODE")
        logger.info("="*60)
        
        for authority in [0.25, 0.50, 0.75]:
            self.controller.set_mode(ControlMode.HYBRID, rl_authority=authority)
            pattern = TRAIN_PATTERNS[0]
            logger.info(f"\nRL authority: {authority:.0%}")
            result = self.run_experiment(pattern, phase_duration_sec, is_training=True)
            
            # Safety check
            if result.p95_latency_ms > 1000:  # 1 second threshold
                logger.warning(f"High latency detected. Reducing authority.")
                break
        
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: ACTIVE MODE TRAINING")
        logger.info("="*60)
        
        self.controller.set_mode(ControlMode.ACTIVE)
        
        for pattern in TRAIN_PATTERNS:
            logger.info(f"\nActive training: {pattern}")
            self.run_experiment(pattern, phase_duration_sec, is_training=True)
        
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: EVALUATION (held-out patterns)")
        logger.info("="*60)
        
        for pattern in TEST_PATTERNS:
            logger.info(f"\nTesting on UNSEEN pattern: {pattern}")
            self.run_experiment(pattern, phase_duration_sec, is_training=False)
        
        # Final analysis
        self.check_overfitting()
    
    def check_overfitting(self) -> float:
        """
        Check for overfitting by comparing train vs test performance
        
        From Jain: If test >> train, model is overfitting
        """
        if not self.train_results or not self.test_results:
            logger.warning("Insufficient data for overfitting check")
            return 0.0
        
        train_p95 = np.mean([r.p95_latency_ms for r in self.train_results])
        test_p95 = np.mean([r.p95_latency_ms for r in self.test_results])
        
        ratio = test_p95 / max(train_p95, 0.001)
        
        logger.info("\n" + "="*60)
        logger.info("OVERFITTING ANALYSIS")
        logger.info("="*60)
        logger.info(f"Training P95 (seen patterns):   {train_p95:.1f} ms")
        logger.info(f"Testing P95 (unseen patterns):  {test_p95:.1f} ms")
        logger.info(f"Generalization ratio:           {ratio:.2f}")
        
        if ratio > 1.5:
            logger.warning("❌ OVERFITTING DETECTED: Test 50%+ worse than training")
        elif ratio > 1.2:
            logger.warning("⚠️ MILD OVERFITTING: Test 20%+ worse than training")
        else:
            logger.info("✓ GOOD GENERALIZATION: Test similar to training")
        
        return ratio
    
    def save_results(self, directory: str):
        """Save experiment results"""
        os.makedirs(directory, exist_ok=True)
        
        with open(os.path.join(directory, 'train_results.json'), 'w') as f:
            json.dump([asdict(r) for r in self.train_results], f, indent=2, default=str)
        
        with open(os.path.join(directory, 'test_results.json'), 'w') as f:
            json.dump([asdict(r) for r in self.test_results], f, indent=2, default=str)
        
        logger.info(f"Results saved to {directory}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CAPA+ MAPE-K Controller')
    parser.add_argument('--mode', choices=['shadow', 'hybrid', 'active'], 
                       default='shadow', help='Control mode')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Dry run (no actual scaling)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration in seconds')
    parser.add_argument('--interval', type=int, default=10,
                       help='Control loop interval in seconds')
    parser.add_argument('--train', action='store_true',
                       help='Run full training pipeline')
    parser.add_argument('--load-state', type=str,
                       help='Load state from directory')
    parser.add_argument('--save-state', type=str,
                       help='Save state to directory')
    parser.add_argument('--namespace', type=str, default='default',
                       help='Kubernetes namespace')
    
    args = parser.parse_args()
    
    # Create controller
    mode_map = {
        'shadow': ControlMode.SHADOW,
        'hybrid': ControlMode.HYBRID,
        'active': ControlMode.ACTIVE
    }
    
    controller = MAPEKController(
        mode=mode_map[args.mode],
        namespace=args.namespace,
        dry_run=args.dry_run
    )
    
    # Load previous state if specified
    if args.load_state:
        controller.load_state(args.load_state)
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        controller.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.train:
            # Run full training pipeline
            runner = ExperimentRunner(controller)
            runner.run_training_pipeline(epochs=3, phase_duration_sec=60)
            
            if args.save_state:
                runner.save_results(args.save_state)
                controller.save_state(args.save_state)
        else:
            # Run single control loop
            controller.run(duration_sec=args.duration, interval_sec=args.interval)
            
            if args.save_state:
                controller.save_state(args.save_state)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Print final statistics
        stats = controller.get_agent_stats()
        for svc, s in stats.items():
            logger.info(f"[{svc}] Final ε={s['epsilon']:.3f}, "
                       f"updates={s['total_updates']}, "
                       f"states={s['states_discovered']}")


if __name__ == '__main__':
    main()
