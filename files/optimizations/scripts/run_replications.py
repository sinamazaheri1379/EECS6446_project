#!/usr/bin/env python3
"""
run_replications.py - Paper-grade independent replications (Jain 25.5.1)

Key improvements:
- Supports evaluation mode (--capa-load, --no-learning)
- Per-replication logging
- Atomic file operations
- Fail-fast on errors
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
    stabilization_sec: int = 300,
    results_base: str = "./results",
    exp_name: str = "meta_experiment",
    # NEW: Evaluation mode parameters
    capa_checkpoint: str = "",
    no_learning: bool = False,
):
    """
    Run R independent paired experiments.
    
    For paper-grade results:
    - Use capa_checkpoint + no_learning=True for evaluation mode
    - This ensures CAPA uses fixed policy (no within-run non-stationarity)
    """
    
    results_dir = os.path.join(results_base, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    replication_log = []
    failed_replications = []
    
    # Validate evaluation mode
    if no_learning and not capa_checkpoint:
        print("WARNING: --no-learning without --capa-load means untrained policy!")
        print("         Consider training first or providing checkpoint.")
    
    for i in range(num_replications):
        seed = base_seed + i
        rep_id = f"rep_{i:02d}_seed_{seed}"
        rep_dir = os.path.join(results_dir, rep_id)
        os.makedirs(rep_dir, exist_ok=True)
        
        log_file = os.path.join(rep_dir, "runner.log")
        
        print(f"\n{'='*60}")
        print(f"REPLICATION {i+1}/{num_replications} (seed={seed})")
        print(f"Output: {rep_dir}")
        print(f"Log: {log_file}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            sys.executable, "unified_experiment_v3_clean.py", "paired",
            "--seed", str(seed),
            "--patterns", patterns,
            "--duration", str(duration),
            "--reset-replicas",
            "--results-dir", results_base,
            "--exp-name", os.path.join(exp_name, rep_id),
        ]
        
        # Add evaluation mode flags
        if capa_checkpoint:
            cmd.extend(["--capa-load", capa_checkpoint])
        if no_learning:
            cmd.append("--no-learning")
        
        start_time = datetime.now()
        
        # Run with logging
        with open(log_file, "w") as log_fh:
            log_fh.write(f"Command: {' '.join(cmd)}\n")
            log_fh.write(f"Start: {start_time.isoformat()}\n")
            log_fh.write("="*60 + "\n\n")
            log_fh.flush()
            
            result = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        end_time = datetime.now()
        
        rep_info = {
            "replication": i,
            "seed": seed,
            "rep_dir": rep_dir,
            "log_file": log_file,
            "return_code": result.returncode,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_sec": (end_time - start_time).total_seconds(),
        }
        replication_log.append(rep_info)
        
        if result.returncode != 0:
            print(f"❌ Replication {i} FAILED (code {result.returncode})")
            print(f"   Check log: {log_file}")
            failed_replications.append(i)
            continue  # Continue with other replications
        
        print(f"✅ Replication {i} completed")
        
        # Run per-replication analysis
        analysis_cmd = [
            sys.executable, "generate_unified_comparison_academic.py",
            "--data-dir", rep_dir,
            "--output-dir", rep_dir,
        ]
        
        analysis_result = subprocess.run(analysis_cmd, capture_output=True, text=True)
        
        if analysis_result.returncode != 0:
            print(f"⚠️  Analysis failed for replication {i}")
            with open(os.path.join(rep_dir, "analysis_error.log"), "w") as f:
                f.write(analysis_result.stderr)
        
        # Move output for meta-analysis (atomic replace)
        src = os.path.join(rep_dir, "academic_report.json")
        dst = os.path.join(results_dir, f"academic_summary_{rep_id}.json")
        if os.path.exists(src):
            os.replace(src, dst)  # Atomic, works even if dst exists
            print(f"   Summary: {dst}")
        
        # Stabilization period (skip after last)
        if i < num_replications - 1:
            print(f"⏳ Stabilization: {stabilization_sec}s...")
            time.sleep(stabilization_sec)
    
    # Save replication log
    log_path = os.path.join(results_dir, "replication_log.json")
    
    meta_info = {
        "num_replications": num_replications,
        "successful": num_replications - len(failed_replications),
        "failed": failed_replications,
        "base_seed": base_seed,
        "patterns": patterns,
        "duration": duration,
        "stabilization_sec": stabilization_sec,
        "evaluation_mode": no_learning,
        "capa_checkpoint": capa_checkpoint or None,
        "replications": replication_log,
    }
    
    with open(log_path, "w") as f:
        json.dump(meta_info, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"COMPLETED: {num_replications - len(failed_replications)}/{num_replications} replications")
    if failed_replications:
        print(f"FAILED: {failed_replications}")
    print(f"Results: {results_dir}")
    print(f"Log: {log_path}")
    print(f"\nNext step:")
    print(f"  python meta_analysis_replications.py --path '{results_dir}/academic_summary_*.json'")
    print(f"{'='*60}")
    
    return len(failed_replications) == 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run independent replications for meta-analysis")
    
    # Replication parameters
    ap.add_argument("-R", "--replications", type=int, default=5,
                    help="Number of independent replications (default: 5)")
    ap.add_argument("--base-seed", type=int, default=1000,
                    help="Starting seed (default: 1000)")
    ap.add_argument("--stabilization", type=int, default=300,
                    help="Seconds between replications (default: 300)")
    
    # Experiment parameters
    ap.add_argument("--patterns", default="warmup,step",
                    help="Comma-separated load patterns")
    ap.add_argument("--duration", type=int, default=1800,
                    help="Duration per paired run in seconds")
    
    # Output
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--exp-name", default="meta_experiment")
    
    # Evaluation mode (CRITICAL for paper-grade results)
    ap.add_argument("--capa-load", default="",
                    help="Directory with trained CAPA checkpoint")
    ap.add_argument("--no-learning", action="store_true",
                    help="Disable CAPA learning (evaluation mode)")
    
    args = ap.parse_args()
    
    success = run_replications(
        num_replications=args.replications,
        base_seed=args.base_seed,
        patterns=args.patterns,
        duration=args.duration,
        stabilization_sec=args.stabilization,
        results_base=args.results_dir,
        exp_name=args.exp_name,
        capa_checkpoint=args.capa_load,
        no_learning=args.no_learning,
    )
    
    sys.exit(0 if success else 1)
