# meta_analysis_replications.py

import glob
import json
import math
import numpy as np
from scipy import stats

def meta_analysis(differences, confidence=0.95):
    """
    Jain 25.5.1 Independent Replications Meta-Analysis
    differences: list of per-replication mean differences (Baseline − CAPA)
    """
    R = len(differences)
    if R < 2:
        raise ValueError("At least 2 replications required")

    mu_bar = np.mean(differences)
    sB2 = np.var(differences, ddof=1)

    alpha = 1 - confidence
    tval = stats.t.ppf(1 - alpha/2, df=R-1)

    half_width = tval * math.sqrt(sB2 / R)

    return {
        "replications": R,
        "mean_difference": mu_bar,
        "ci_low": mu_bar - half_width,
        "ci_high": mu_bar + half_width,
        "significant": not (mu_bar - half_width <= 0 <= mu_bar + half_width),
        "confidence": confidence
    }


def load_replication_results(path_pattern):
    """
    Load per-run academic_summary_*.json files
    """
    diffs = []
    for f in glob.glob(path_pattern):
        with open(f) as fh:
            data = json.load(fh)
            # pick system-level p95 latency diff
            for row in data:
                if row["scope"] == "system" and row["metric"] == "latency_p95_ms":
                    diffs.append(row["diff_mean"])
                    break
    return diffs


if __name__ == "__main__":
    diffs = load_replication_results("./analysis_output/academic_summary_*.json")
    res = meta_analysis(diffs)

    print("===================================================")
    print("META-ANALYSIS ACROSS INDEPENDENT REPLICATIONS")
    print("===================================================")
    print(f"Replications: {res['replications']}")
    print(f"Mean diff (Baseline − CAPA): {res['mean_difference']:.3f} ms")
    print(f"{int(res['confidence']*100)}% CI: [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print(f"Statistically significant: {res['significant']}")
