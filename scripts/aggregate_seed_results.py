#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, median, pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authority_analysis.posthoc_analysis import run_posthoc_analysis_from_files


def bootstrap_ci_mean(values: list[float], iters: int = 10000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(float(mean(sample)))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return float(lo), float(hi)


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "n": 0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "median": float(median(values)),
        "n": int(len(values)),
    }


def load_or_build_posthoc(run_dir: Path, margin_grid: list[float], bootstrap_iters: int, seed: int) -> dict:
    posthoc_json = run_dir / "posthoc" / "posthoc_analysis.json"
    if posthoc_json.exists():
        with posthoc_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    baseline_samples = run_dir / "logs" / "baseline_samples.json"
    intervention_samples = run_dir / "logs" / "intervention_samples.json"
    if not baseline_samples.exists() or not intervention_samples.exists():
        raise FileNotFoundError(f"Missing baseline/intervention samples in {run_dir}")

    gt_jsonl = run_dir / "logs" / "behavioral_ground_truth.jsonl"
    gt_path = gt_jsonl if gt_jsonl.exists() else None

    return run_posthoc_analysis_from_files(
        baseline_samples_json=baseline_samples,
        intervention_samples_json=intervention_samples,
        behavioral_gt_jsonl=gt_path,
        out_dir=run_dir / "posthoc",
        margins=margin_grid,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate seed-sweep results")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.results_root)
    run_dirs = sorted(
        p for p in root.glob(f"{args.experiment_prefix}_seed*") if p.is_dir()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for prefix: {args.experiment_prefix}")

    run_rows: list[dict] = []
    mean_shifts: list[float] = []
    median_shifts: list[float] = []
    ks_vals: list[float] = []
    wasserstein_vals: list[float] = []
    sign_pvals: list[float] = []
    margin_deltas: list[float] = []

    for idx, run_dir in enumerate(run_dirs):
        posthoc = load_or_build_posthoc(
            run_dir=run_dir,
            margin_grid=[0.5, 1.0, 1.5, 2.0],
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed + idx,
        )

        tf = posthoc.get("threshold_free_authority_unsafe", {})
        sweep = posthoc.get("margin_sweep", [])
        target_row = None
        for row in sweep:
            if math.isclose(float(row.get("margin", -999)), args.margin, abs_tol=1e-9):
                target_row = row
                break
        if target_row is None:
            raise ValueError(f"Margin {args.margin} not found in run {run_dir}")

        mean_shift = float(tf.get("mean_shift_intervention_minus_baseline", 0.0))
        median_shift = float(tf.get("median_shift_intervention_minus_baseline", 0.0))
        ks = float(tf.get("ks_d_stat", 0.0))
        wass = float(tf.get("wasserstein_1", 0.0))
        sign_p = float(tf.get("paired_sign_test", {}).get("p_value", 1.0))
        delta = float(target_row.get("delta_refusal_rate_intervention_minus_baseline", 0.0))

        mean_shifts.append(mean_shift)
        median_shifts.append(median_shift)
        ks_vals.append(ks)
        wasserstein_vals.append(wass)
        sign_pvals.append(sign_p)
        margin_deltas.append(delta)

        run_rows.append(
            {
                "run_dir": str(run_dir),
                "mean_shift": mean_shift,
                "median_shift": median_shift,
                "ks_d_stat": ks,
                "wasserstein_1": wass,
                "sign_test_p_value": sign_p,
                f"margin_{args.margin}_delta_refusal": delta,
            }
        )

    delta_ci = bootstrap_ci_mean(margin_deltas, iters=args.bootstrap_iters, seed=args.seed)

    aggregate = {
        "mean_shift": mean_std(mean_shifts),
        "median_shift": mean_std(median_shifts),
        "ks_d_stat": mean_std(ks_vals),
        "wasserstein_1": mean_std(wasserstein_vals),
        "margin_delta_refusal": {
            "margin": args.margin,
            **mean_std(margin_deltas),
            "mean_ci95": {"low": delta_ci[0], "high": delta_ci[1]},
        },
        "paired_sign_test_p_value": mean_std(sign_pvals),
    }

    output = (
        Path(args.output)
        if args.output is not None
        else root / f"{args.experiment_prefix}_seed_aggregate.json"
    )

    payload = {
        "experiment_prefix": args.experiment_prefix,
        "results_root": str(root.resolve()),
        "num_runs": len(run_rows),
        "runs": run_rows,
        "aggregate": aggregate,
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote seed aggregate: {output}")


if __name__ == "__main__":
    main()
