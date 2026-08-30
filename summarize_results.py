"""Reduce an lm_eval results directory to the headline metrics.

lm_eval writes results to <output_path>/<sanitized-model-name>/results_<ISO-timestamp>.json,
so the path is not known until after the run. This finds the newest one, prints a compact
table, and rewrites it to a fixed path so a caller (run_and_report.sh, or anything driving
`sbatch --wait`) has something deterministic to read.

    python summarize_results.py <output_path> [--json metrics.json]
"""

import argparse
import glob
import json
import os
import sys


def newest_results(output_path):
    hits = glob.glob(os.path.join(output_path, "**", "results_*.json"), recursive=True)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def extract(blob):
    """{task: {metric: value}} keeping only real metrics.

    lm_eval suffixes every key with the filter name (",none" for these tasks) and emits a
    matching "<metric>_stderr,none" plus a cosmetic "alias". Strip all three.
    """
    out = {}
    for task, res in blob.get("results", {}).items():
        metrics = {}
        for k, v in res.items():
            if k == "alias" or "_stderr" in k or not isinstance(v, (int, float)):
                continue
            name = k.split(",")[0]
            metrics[name] = v
            stderr = res.get(f"{name}_stderr,none")
            if isinstance(stderr, (int, float)):
                metrics[f"{name}_stderr"] = stderr
        if metrics:
            out[task] = metrics
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_path")
    ap.add_argument("--json", default=None, help="also write the metrics to this path")
    args = ap.parse_args()

    path = newest_results(args.output_path)
    if path is None:
        # Not fatal on its own, but the caller must not read a stale metrics.json and
        # believe it came from this run.
        print(f"ERROR: no results_*.json under {args.output_path}", file=sys.stderr)
        return 1

    blob = json.load(open(path))
    metrics = extract(blob)

    print(f"\n=== downstream metrics ({os.path.basename(path)}) ===")
    header = f"{'task':<18}{'metric':<14}{'value':>12}{'stderr':>12}"
    print(header)
    print("-" * len(header))
    for task in sorted(metrics):
        for name in sorted(metrics[task]):
            if name.endswith("_stderr"):
                continue
            se = metrics[task].get(f"{name}_stderr")
            se_s = f"{se:12.4f}" if se is not None else f"{'-':>12}"
            print(f"{task:<18}{name:<14}{metrics[task][name]:12.4f}{se_s}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"source": path,
                       "model": blob.get("config", {}).get("model_args"),
                       "metrics": metrics}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
