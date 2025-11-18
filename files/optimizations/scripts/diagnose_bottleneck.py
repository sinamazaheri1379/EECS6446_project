#!/usr/bin/env python3
"""
EECS6446 Project - Bottleneck Analysis
--------------------------------------
Analyzes where response time is actually being spent
"""

import requests
import pandas as pd
from datetime import datetime

PROMETHEUS_URL = "http://localhost:9090"

def analyze_response_time_components():
    """
    Breakdown: Response Time = Pod Provisioning Time + Contribution Time
    We need to measure:
    1. Pod startup latency (Pending -> Running -> Ready)
    2. Redis latency (CartService dependency)
    3. Network latency (Service mesh overhead)
    4. Actual compute time
    """
    
    results = {}
    
    # 1. Pod Readiness Lag
    query = """
    avg(
        (kube_pod_status_ready_time - kube_pod_created) > 0
    ) by (pod)
    """
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
    data = r.json()
    if data['data']['result']:
        results['avg_pod_ready_time_sec'] = sum(
            float(x['value'][1]) for x in data['data']['result']
        ) / len(data['data']['result'])
    
    # 2. Redis Latency (if available)
    query = 'redis_command_duration_seconds_sum / redis_command_duration_seconds_count'
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
    data = r.json()
    if data['data']['result']:
        results['redis_latency_ms'] = float(data['data']['result'][0]['value'][1]) * 1000
    
    # 3. Istio Network Latency (if using service mesh)
    query = 'histogram_quantile(0.95, istio_request_duration_milliseconds_bucket)'
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
    data = r.json()
    if data['data']['result']:
        results['istio_p95_latency_ms'] = float(data['data']['result'][0]['value'][1])
    
    # 4. CPU Throttling
    query = '''
    sum(rate(container_cpu_cfs_throttled_seconds_total[1m])) by (pod) /
    sum(rate(container_cpu_usage_seconds_total[1m])) by (pod)
    '''
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
    data = r.json()
    if data['data']['result']:
        results['throttling_ratio'] = {
            x['metric']['pod']: float(x['value'][1]) 
            for x in data['data']['result']
        }
    
    return results

def check_redis_connection_pool():
    """CartService depends on Redis - check if connection pooling is a bottleneck"""
    query = 'redis_connected_clients'
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
    data = r.json()
    if data['data']['result']:
        return int(data['data']['result'][0]['value'][1])
    return None

if __name__ == "__main__":
    print("=== Bottleneck Analysis ===\n")
    
    components = analyze_response_time_components()
    
    print("1. Pod Provisioning:")
    print(f"   Avg Time to Ready: {components.get('avg_pod_ready_time_sec', 'N/A')}s")
    
    print("\n2. Redis Performance:")
    print(f"   Latency: {components.get('redis_latency_ms', 'N/A')}ms")
    print(f"   Connected Clients: {check_redis_connection_pool()}")
    
    print("\n3. Network Overhead:")
    print(f"   P95 Request Duration: {components.get('istio_p95_latency_ms', 'N/A')}ms")
    
    print("\n4. CPU Throttling:")
    if 'throttling_ratio' in components:
        for pod, ratio in components['throttling_ratio'].items():
            if ratio > 0.1:  # >10% throttled
                print(f"   ⚠️  {pod}: {ratio*100:.1f}% throttled")
    
    # Hypothesis Testing
    print("\n=== Hypothesis ===")
    if components.get('avg_pod_ready_time_sec', 0) > 30:
        print("❌ Pod startup time is the bottleneck (>30s to ready)")
        print("   → Solution: Use Pod Prewarming or faster container images")
    
    elif components.get('redis_latency_ms', 0) > 50:
        print("❌ Redis latency is high")
        print("   → Solution: Scale Redis or optimize queries")
    
    elif components.get('istio_p95_latency_ms', 0) > 100:
        print("❌ Service mesh overhead is significant")
        print("   → Solution: Bypass mesh for internal calls")
    
    else:
        print("✅ No obvious bottleneck - CPU scaling should help")
        print("   → But it's not helping, so check:")
        print("   - Request queueing in application")
        print("   - Lock contention")
        print("   - Database connection limits")
