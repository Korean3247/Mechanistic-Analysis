#!/usr/bin/env python3
"""Spillover gating experiment.

Tests a projection-magnitude threshold gate: the intervention is applied
ONLY when the authority projection exceeds a learned threshold, leaving
safe/borderline prompts untouched.

Usage:
    python scripts/run_spillover_gating.py \
        --prompts data/prompts.jsonl \
        --direction results/llama3_8b_l4_full_350_m15_gt200_probe_placebo/authority_direction_vector.pt \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --layer 10 --alpha 1.0 \
        --output-dir results/spillover_gating_llama3
"""
from __future__ import annotations

import argparse
import json
import math
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
from authority_analysis.utils import ensure_dir, write_json, write_jsonl


def _load_direction(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    d = payload.get("residual_direction_normalized") or payload.get("direction")
    if d is None:
        raise ValueError(f"No direction found in {path}")
    return d.float()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_projections(
    model: ModelInterface,
    prompts: list[dict[str, Any]],
    direction: torch.Tensor,
    layer: int,
) -> list[dict[str, Any]]:
    """Compute projection magnitude for each prompt without intervention."""
    results = []
    direction_dev = direction.to(model.device)
    for row in tqdm(prompts, desc="Computing projections"):
        text = row.get("full_prompt") or row["prompt"]
        with torch.no_grad():
            inputs = model.tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1][:, -1, :].float()
            proj_mag = torch.dot(hidden.squeeze(), direction_dev).item()
        results.append({**row, "proj_magnitude": proj_mag})
    return results


def run_gated_intervention(
    model: ModelInterface,
    engine: CausalInterventionEngine,
    prompts_with_proj: list[dict[str, Any]],
    direction: torch.Tensor,
    layer: int,
    alpha: float,
    thresholds: list[float],
) -> dict[str, Any]:
    """Run intervention with different gating thresholds."""
    all_results: dict[str, Any] = {}

    for thresh in thresholds:
        tag = f"thresh_{thresh:.2f}"
        gated_results = []

        for row in tqdm(prompts_with_proj, desc=f"Gated intervention (τ={thresh:.2f})"):
            text = row.get("full_prompt") or row["prompt"]
            proj_mag = row["proj_magnitude"]
            risk_tier = row.get("risk_tier", "unknown")
            framing = row.get("framing_type", "unknown")

            # Baseline score
            with torch.no_grad():
                inputs = model.tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model.model(**inputs)
                logits = outputs.logits[:, -1, :]
                refusal_tokens = model.tokenizer.encode("I cannot", add_special_tokens=False)
                comply_tokens = model.tokenizer.encode("Sure", add_special_tokens=False)
                baseline_ref = logits[0, refusal_tokens[0]].item() if refusal_tokens else 0.0
                baseline_comp = logits[0, comply_tokens[0]].item() if comply_tokens else 0.0
                baseline_ld = baseline_ref - baseline_comp

            # Gated: only intervene if projection exceeds threshold
            apply_intervention = proj_mag > thresh

            if apply_intervention:
                intervention_result = engine.intervene_single(
                    text, direction, layer=layer, alpha=alpha
                )
                interv_ld = intervention_result.get("logit_diff", baseline_ld)
            else:
                interv_ld = baseline_ld

            gated_results.append({
                "prompt_id": row.get("prompt_id", ""),
                "risk_tier": risk_tier,
                "framing_type": framing,
                "proj_magnitude": proj_mag,
                "gated": apply_intervention,
                "baseline_logit_diff": baseline_ld,
                "intervention_logit_diff": interv_ld,
                "delta": interv_ld - baseline_ld,
            })

        # Aggregate by tier
        tier_stats: dict[str, dict[str, float]] = {}
        for tier in ["safe", "borderline", "unsafe"]:
            tier_rows = [r for r in gated_results if r["risk_tier"] == tier]
            if tier_rows:
                deltas = [r["delta"] for r in tier_rows]
                n_gated = sum(1 for r in tier_rows if r["gated"])
                tier_stats[tier] = {
                    "n": len(tier_rows),
                    "n_gated": n_gated,
                    "pct_gated": n_gated / len(tier_rows),
                    "mean_delta": sum(deltas) / len(deltas),
                }

        all_results[tag] = {
            "threshold": thresh,
            "tier_stats": tier_stats,
            "samples": gated_results,
        }

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/spillover_gating")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    direction = _load_direction(Path(args.direction))
    prompts = _load_jsonl(Path(args.prompts))

    print(f"Loading model: {args.model}")
    mi = ModelInterface(model_name=args.model, dtype=args.dtype, device="auto")
    engine = CausalInterventionEngine(mi)

    print("Phase 1: Computing baseline projections...")
    prompts_with_proj = compute_projections(mi, prompts, direction, args.layer)

    # Compute optimal threshold from authority/non-authority separation
    auth_projs = [r["proj_magnitude"] for r in prompts_with_proj
                  if r.get("framing_type") == "authority"]
    non_auth_projs = [r["proj_magnitude"] for r in prompts_with_proj
                      if r.get("framing_type") in ("direct", "hypothetical", "academic", "institutional")]

    if auth_projs and non_auth_projs:
        auth_mean = sum(auth_projs) / len(auth_projs)
        non_auth_mean = sum(non_auth_projs) / len(non_auth_projs)
        optimal_thresh = (auth_mean + non_auth_mean) / 2.0
        print(f"  Authority mean proj: {auth_mean:.4f}")
        print(f"  Non-authority mean proj: {non_auth_mean:.4f}")
        print(f"  Optimal midpoint threshold: {optimal_thresh:.4f}")
        if optimal_thresh not in args.thresholds:
            args.thresholds.append(optimal_thresh)
            args.thresholds.sort()

    print(f"Phase 2: Running gated intervention at {len(args.thresholds)} thresholds...")
    results = run_gated_intervention(
        mi, engine, prompts_with_proj, direction,
        args.layer, args.alpha, args.thresholds,
    )

    # Save
    write_json(results, out_dir / "gating_results.json")

    # Print summary
    print("\n=== Gating Summary ===")
    print(f"{'Threshold':>10} | {'Unsafe Δ':>10} | {'Safe Δ':>10} | {'BL Δ':>10} | {'Unsafe %gated':>14} | {'Safe %gated':>12}")
    for tag, data in sorted(results.items()):
        ts = data["tier_stats"]
        unsafe = ts.get("unsafe", {})
        safe = ts.get("safe", {})
        bl = ts.get("borderline", {})
        print(f"{data['threshold']:>10.2f} | "
              f"{unsafe.get('mean_delta', 0):>+10.4f} | "
              f"{safe.get('mean_delta', 0):>+10.4f} | "
              f"{bl.get('mean_delta', 0):>+10.4f} | "
              f"{unsafe.get('pct_gated', 0):>13.1%} | "
              f"{safe.get('pct_gated', 0):>11.1%}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
