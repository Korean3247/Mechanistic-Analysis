#!/usr/bin/env python3
"""Adaptive alpha experiment.

Instead of a fixed alpha, scales alpha per-prompt proportional to the
authority projection magnitude. This targets high-authority prompts
more aggressively while leaving low-authority prompts less affected.

Usage:
    python scripts/run_adaptive_alpha.py \
        --prompts data/prompts.jsonl \
        --direction results/.../authority_direction_vector.pt \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --layer 10 --output-dir results/adaptive_alpha_llama3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from authority_analysis.causal_intervention import CausalInterventionEngine
from authority_analysis.model_interface import ModelInterface
from authority_analysis.utils import ensure_dir, write_json


def _load_direction(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    d = payload.get("residual_direction_normalized") or payload.get("direction")
    if d is None:
        raise ValueError(f"No direction found in {path}")
    return d.float()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


STRATEGIES = {
    "fixed_1.0": lambda proj, _stats: 1.0,
    "fixed_1.5": lambda proj, _stats: 1.5,
    "linear_clip": lambda proj, stats: min(2.0, max(0.0,
        (proj - stats["non_auth_mean"]) / max(1e-8, stats["auth_mean"] - stats["non_auth_mean"]) * 1.5
    )),
    "sigmoid": lambda proj, stats: 2.0 / (1 + __import__("math").exp(
        -5.0 * (proj - stats["midpoint"]) / max(1e-8, stats["auth_std"])
    )),
    "relu_scaled": lambda proj, stats: max(0.0, (proj - stats["midpoint"])) / max(1e-8, stats["auth_std"]) * 1.0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--output-dir", default="results/adaptive_alpha")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    direction = _load_direction(Path(args.direction))
    prompts = _load_jsonl(Path(args.prompts))

    print(f"Loading model: {args.model}")
    mi = ModelInterface(model_name=args.model, dtype=args.dtype, device="auto")
    engine = CausalInterventionEngine(mi)
    direction_dev = direction.to(mi.device)

    # Phase 1: Compute projections
    print("Phase 1: Computing projections...")
    for row in tqdm(prompts, desc="Projections"):
        text = row.get("full_prompt") or row["prompt"]
        with torch.no_grad():
            inputs = mi.tokenizer(text, return_tensors="pt").to(mi.device)
            outputs = mi.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[args.layer + 1][:, -1, :].float()
            row["proj_magnitude"] = torch.dot(hidden.squeeze(), direction_dev).item()

    # Compute stats
    auth_projs = [r["proj_magnitude"] for r in prompts if r.get("framing_type") == "authority"]
    non_auth_projs = [r["proj_magnitude"] for r in prompts
                      if r.get("framing_type") in ("direct", "hypothetical", "academic", "institutional")]

    import statistics
    stats = {
        "auth_mean": statistics.mean(auth_projs) if auth_projs else 0.0,
        "auth_std": statistics.stdev(auth_projs) if len(auth_projs) > 1 else 1.0,
        "non_auth_mean": statistics.mean(non_auth_projs) if non_auth_projs else 0.0,
        "midpoint": (statistics.mean(auth_projs) + statistics.mean(non_auth_projs)) / 2 if auth_projs and non_auth_projs else 0.0,
    }
    print(f"  Stats: {stats}")

    # Phase 2: Run each strategy
    all_results: dict[str, Any] = {"stats": stats, "strategies": {}}

    for strat_name, alpha_fn in STRATEGIES.items():
        print(f"\nStrategy: {strat_name}")
        strat_results = []

        for row in tqdm(prompts, desc=strat_name):
            text = row.get("full_prompt") or row["prompt"]
            proj = row["proj_magnitude"]
            alpha = alpha_fn(proj, stats)

            if alpha > 0.01:
                result = engine.intervene_single(text, direction, layer=args.layer, alpha=alpha)
                interv_ld = result.get("logit_diff", 0.0)
            else:
                interv_ld = row.get("baseline_logit_diff", 0.0)

            # Baseline
            baseline_result = engine.intervene_single(text, direction, layer=args.layer, alpha=0.0)
            baseline_ld = baseline_result.get("logit_diff", 0.0)

            strat_results.append({
                "prompt_id": row.get("prompt_id", ""),
                "risk_tier": row.get("risk_tier", "unknown"),
                "framing_type": row.get("framing_type", "unknown"),
                "proj_magnitude": proj,
                "alpha_applied": alpha,
                "baseline_ld": baseline_ld,
                "intervention_ld": interv_ld,
                "delta": interv_ld - baseline_ld,
            })

        # Tier aggregation
        tier_agg = {}
        for tier in ["safe", "borderline", "unsafe"]:
            tier_rows = [r for r in strat_results if r["risk_tier"] == tier]
            auth_rows = [r for r in tier_rows if r["framing_type"] == "authority"]
            if auth_rows:
                deltas = [r["delta"] for r in auth_rows]
                alphas = [r["alpha_applied"] for r in auth_rows]
                tier_agg[tier] = {
                    "n": len(auth_rows),
                    "mean_delta": statistics.mean(deltas),
                    "mean_alpha": statistics.mean(alphas),
                    "median_delta": statistics.median(deltas),
                }

        all_results["strategies"][strat_name] = {
            "tier_summary": tier_agg,
            "samples": strat_results,
        }

        # Print
        print(f"  {'Tier':>12} | {'Mean Δ':>10} | {'Mean α':>8} | n")
        for tier, agg in tier_agg.items():
            print(f"  {tier:>12} | {agg['mean_delta']:>+10.4f} | {agg['mean_alpha']:>8.3f} | {agg['n']}")

    write_json(all_results, out_dir / "adaptive_alpha_results.json")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
