#!/usr/bin/env python3
"""
run_replications.py - Run R independent paired experiments
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

def run_replications(
    num_replications: int = 5,
    base_seed: int = 1000,
    patterns: str = "warmup,step",
    duration: int = 1800,
    stabilization_sec: int = 300,  # 5 min between replications
    results_base: str = "./results",
    exp_name: str = "meta_experiment"
):
    """
    Run R independent paired experiments with different seeds.
    
    Key for independence:
    - Each replication uses seed = base_seed + i
    - Replicas reset between runs
    - Stabilization period between replications
    """
    
    results_dir = os.path.join(results_base, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    replication_log = []
    
    for i in range(num_replications):
        seed = base_seed + i
        rep_id = f"rep_{i:02d}_seed_{seed}"
        rep_dir = os.path.join(results_dir, rep_id)
        os.makedirs(rep_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"REPLICATION {i+1}/{num_replications} (seed={seed})")
        print(f"{'='*60}")
        
        # Run paired experiment with unique seed
        cmd = [
            sys.executable, "unified_experiment_v3_clean.py", "paired",
            "--seed", str(seed),
            "--patterns", patterns,
            "--duration", str(duration),
            "--reset-replicas",  # Critical for independence!
            "--results-dir", results_base,
            "--exp-name", os.path.join(exp_name, rep_id),
        ]
        
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=False)
        end_time = datetime.now()
        
        replication_log.append({
            "replication": i,
            "seed": seed,
            "rep_dir": rep_dir,
            "return_code": result.returncode,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        })
        
        if result.returncode != 0:
            print(f"WARNING: Replication {i} failed with code {result.returncode}")
        
        # Run per-replication analysis
        analysis_cmd = [
            sys.executable, "generate_unified_comparison_academic.py",
            "--data-dir", rep_dir,
            "--output-dir", rep_dir,
        ]
        subprocess.run(analysis_cmd)
        
        # Rename output for meta-analysis collection
        src = os.path.join(rep_dir, "academic_report.json")
        dst = os.path.join(results_dir, f"academic_summary_{rep_id}.json")
        if os.path.exists(src):
            os.rename(src, dst)
        
        # Stabilization period (skip after last replication)
        if i < num_replications - 1:
            print(f"Stabilization period: {stabilization_sec}s...")
            time.sleep(stabilization_sec)
    
    # Save replication log
    log_path = os.path.join(results_dir, "replication_log.json")
    with open(log_path, "w") as f:
        json.dump(replication_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETED {num_replications} REPLICATIONS")
    print(f"Results: {results_dir}")
    print(f"Run meta_analysis_replications.py --path '{results_dir}/academic_summary_*.json'")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-R", "--replications", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--patterns", default="warmup,step")
    ap.add_argument("--duration", type=int, default=1800)
    ap.add_argument("--stabilization", type=int, default=300)
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--exp-name", default="meta_experiment")
    args = ap.parse_args()
    
    run_replications(
        num_replications=args.replications,
        base_seed=args.base_seed,
        patterns=args.patterns,
        duration=args.duration,
        stabilization_sec=args.stabilization,
        results_base=args.results_dir,
        exp_name=args.exp_name,
    )
