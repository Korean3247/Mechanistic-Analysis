from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .model_interface import ModelInterface
from .utils import read_jsonl, write_json


class CausalInterventionEngine:
    def __init__(self, model_interface: ModelInterface) -> None:
        self.model_interface = model_interface

    @staticmethod
    def make_projection_removal_fn(direction: torch.Tensor, alpha: float = 1.0):
        direction = direction.detach().to(dtype=torch.float32)
        direction_is_finite = bool(torch.isfinite(direction).all().item())
        direction_l2 = float(torch.linalg.norm(direction).item()) if direction_is_finite else 0.0
        degenerate_direction = (not direction_is_finite) or direction_l2 <= 1e-12

        if degenerate_direction:
            stats = {
                "direction_is_finite": direction_is_finite,
                "direction_l2": direction_l2,
                "degenerate_direction": True,
                "calls": 0,
                "non_finite_coeff_calls": 0,
                "non_finite_output_calls": 0,
                "identity_fallback_calls": 0,
            }

            def identity(hidden: torch.Tensor) -> torch.Tensor:
                stats["calls"] += 1
                stats["identity_fallback_calls"] += 1
                return hidden

            setattr(identity, "_debug_stats", stats)
            return identity

        stats = {
            "direction_is_finite": True,
            "direction_l2": direction_l2,
            "degenerate_direction": False,
            "calls": 0,
            "non_finite_coeff_calls": 0,
            "non_finite_output_calls": 0,
            "identity_fallback_calls": 0,
        }
        alpha_f = float(alpha)
        eps = 1e-12

        def intervention(hidden: torch.Tensor) -> torch.Tensor:
            stats["calls"] += 1
            d = direction.to(device=hidden.device, dtype=torch.float32)
            hidden_f32 = hidden.to(dtype=torch.float32)
            denom = torch.sum(d * d).clamp_min(eps)
            coeff = torch.sum(hidden_f32 * d, dim=-1, keepdim=True) / denom
            if not torch.isfinite(coeff).all():
                stats["non_finite_coeff_calls"] += 1
                stats["identity_fallback_calls"] += 1
                return hidden

            projected = coeff * d
            updated = hidden_f32 - alpha_f * projected
            if not torch.isfinite(updated).all():
                stats["non_finite_output_calls"] += 1
                stats["identity_fallback_calls"] += 1
                return hidden

            return updated.to(dtype=hidden.dtype)

        setattr(intervention, "_debug_stats", stats)
        return intervention

    def run(
        self,
        prompts: list[dict[str, Any]],
        layer_idx: int,
        direction: torch.Tensor,
        alpha: float,
        max_tokens: int,
        capture_layers: set[int] | None = None,
        capture_attentions: bool = False,
    ) -> list[dict[str, Any]]:
        intervention_fn = self.make_projection_removal_fn(direction, alpha=alpha)
        rows: list[dict[str, Any]] = []

        for row in prompts:
            artifacts = self.model_interface.run_forward(
                prompt_text=row["full_prompt"],
                max_tokens=max_tokens,
                intervention_layer=layer_idx,
                intervention_fn=intervention_fn,
                capture_layers=capture_layers,
                capture_attentions=capture_attentions,
            )
            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "framing_type": row["framing_type"],
                    "semantic_request_id": row["semantic_request_id"],
                    "safety_label": row.get("safety_label", "unknown"),
                    "risk_tier": row.get("risk_tier", "unknown"),
                    "refusal_score": artifacts.refusal_score,
                    "compliance_score": artifacts.compliance_score,
                    "logit_diff": artifacts.logit_diff,
                    "is_refusal": artifacts.is_refusal,
                    "logits_all_finite": artifacts.logits_all_finite,
                    "logits_non_finite_count": artifacts.logits_non_finite_count,
                    "logits_non_finite_ratio": artifacts.logits_non_finite_ratio,
                    "intervention_direction_is_finite": bool(
                        getattr(intervention_fn, "_debug_stats", {}).get("direction_is_finite", True)
                    ),
                    "intervention_direction_l2": float(
                        getattr(intervention_fn, "_debug_stats", {}).get("direction_l2", 0.0)
                    ),
                    "intervention_degenerate_direction": bool(
                        getattr(intervention_fn, "_debug_stats", {}).get("degenerate_direction", False)
                    ),
                    "intervention_total_calls": int(
                        getattr(intervention_fn, "_debug_stats", {}).get("calls", 0)
                    ),
                    "intervention_identity_fallback_calls": int(
                        getattr(intervention_fn, "_debug_stats", {}).get("identity_fallback_calls", 0)
                    ),
                    "intervention_non_finite_coeff_calls": int(
                        getattr(intervention_fn, "_debug_stats", {}).get("non_finite_coeff_calls", 0)
                    ),
                    "intervention_non_finite_output_calls": int(
                        getattr(intervention_fn, "_debug_stats", {}).get("non_finite_output_calls", 0)
                    ),
                }
            )
        return rows



def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal intervention over prompts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, help="Prompt dataset JSONL")
    parser.add_argument("--direction", required=True, help="authority_direction_vector.pt")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--probe-instruction",
        default="Answer with exactly one word: yes or no.",
    )
    parser.add_argument("--refusal-margin", type=float, default=1.0)
    args = parser.parse_args()

    vector_payload = torch.load(args.direction, map_location="cpu")
    direction = vector_payload["residual_direction_normalized"]

    prompts = read_jsonl(args.prompts)
    mi = ModelInterface(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        probe_instruction=args.probe_instruction,
        refusal_margin=args.refusal_margin,
    )
    engine = CausalInterventionEngine(mi)
    rows = engine.run(
        prompts=prompts,
        layer_idx=args.layer,
        direction=direction,
        alpha=args.alpha,
        max_tokens=args.max_tokens,
        capture_layers=set(),
        capture_attentions=False,
    )
    write_json(args.output, {"samples": rows})
    print(f"Saved intervention results to {args.output}")


if __name__ == "__main__":
    main()
