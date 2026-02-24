#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authority_analysis.posthoc_analysis import run_posthoc_analysis_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc margin and distribution analysis")
    parser.add_argument("--baseline-samples", required=True)
    parser.add_argument("--intervention-samples", required=True)
    parser.add_argument("--behavioral-gt-jsonl", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--margins", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_posthoc_analysis_from_files(
        baseline_samples_json=args.baseline_samples,
        intervention_samples_json=args.intervention_samples,
        out_dir=args.out_dir,
        behavioral_gt_jsonl=args.behavioral_gt_jsonl,
        margins=args.margins,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )

    print(f"Wrote post-hoc analysis to: {args.out_dir}")
    artifacts = report.get("artifacts", {})
    for key in (
        "posthoc_analysis_json",
        "margin_sweep_csv",
        "authority_unsafe_ecdf_csv",
    ):
        path = artifacts.get(key)
        if path:
            print(f"- {path}")


if __name__ == "__main__":
    main()
