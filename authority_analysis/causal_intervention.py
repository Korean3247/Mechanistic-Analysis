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
        layer_idx: int | None,
        direction: torch.Tensor | None,
        alpha: float,
        max_tokens: int,
        capture_layers: set[int] | None = None,
        capture_attentions: bool = False,
        intervention_fns_by_layer: dict[int, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if intervention_fns_by_layer:
            intervention_fn = None
            intervention_map = dict(intervention_fns_by_layer)
        else:
            if layer_idx is None or direction is None:
                raise ValueError("Single-layer intervention requires layer_idx and direction")
            intervention_fn = self.make_projection_removal_fn(direction, alpha=alpha)
            intervention_map = None
        rows: list[dict[str, Any]] = []

        for row in prompts:
            artifacts = self.model_interface.run_forward(
                prompt_text=row["full_prompt"],
                max_tokens=max_tokens,
                intervention_layer=layer_idx,
                intervention_fn=intervention_fn,
                intervention_fns_by_layer=intervention_map,
                capture_layers=capture_layers,
                capture_attentions=capture_attentions,
            )
            if intervention_map:
                debug_stats = {
                    str(layer): getattr(fn, "_debug_stats", {})
                    for layer, fn in intervention_map.items()
                }
                direction_is_finite = all(
                    bool(stats.get("direction_is_finite", True)) for stats in debug_stats.values()
                )
                direction_l2 = {
                    layer: float(stats.get("direction_l2", 0.0))
                    for layer, stats in debug_stats.items()
                }
                degenerate_direction = any(
                    bool(stats.get("degenerate_direction", False)) for stats in debug_stats.values()
                )
                total_calls = {
                    layer: int(stats.get("calls", 0)) for layer, stats in debug_stats.items()
                }
                identity_fallback_calls = {
                    layer: int(stats.get("identity_fallback_calls", 0))
                    for layer, stats in debug_stats.items()
                }
                non_finite_coeff_calls = {
                    layer: int(stats.get("non_finite_coeff_calls", 0))
                    for layer, stats in debug_stats.items()
                }
                non_finite_output_calls = {
                    layer: int(stats.get("non_finite_output_calls", 0))
                    for layer, stats in debug_stats.items()
                }
            else:
                stats = getattr(intervention_fn, "_debug_stats", {})
                direction_is_finite = bool(stats.get("direction_is_finite", True))
                direction_l2 = float(stats.get("direction_l2", 0.0))
                degenerate_direction = bool(stats.get("degenerate_direction", False))
                total_calls = int(stats.get("calls", 0))
                identity_fallback_calls = int(stats.get("identity_fallback_calls", 0))
                non_finite_coeff_calls = int(stats.get("non_finite_coeff_calls", 0))
                non_finite_output_calls = int(stats.get("non_finite_output_calls", 0))
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
                    "intervention_direction_is_finite": direction_is_finite,
                    "intervention_direction_l2": direction_l2,
                    "intervention_degenerate_direction": degenerate_direction,
                    "intervention_total_calls": total_calls,
                    "intervention_identity_fallback_calls": identity_fallback_calls,
                    "intervention_non_finite_coeff_calls": non_finite_coeff_calls,
                    "intervention_non_finite_output_calls": non_finite_output_calls,
                }
            )
        return rows


def _load_direction(path: str | Path) -> torch.Tensor:
    vector_payload = torch.load(path, map_location="cpu")
    direction = vector_payload.get("residual_direction_normalized")
    if direction is None:
        direction = vector_payload.get("direction")
    if direction is None:
        raise ValueError(f"Direction file missing residual_direction_normalized/direction: {path}")
    return direction


def _parse_direction_spec(raw: str, default_alpha: float) -> tuple[int, Path, float]:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(
            f"Invalid --direction-spec {raw!r}. Expected 'LAYER:PATH' or 'LAYER:PATH:ALPHA'."
        )
    layer = int(parts[0])
    path = Path(parts[1]).expanduser().resolve()
    alpha = float(parts[2]) if len(parts) == 3 else float(default_alpha)
    return layer, path, alpha



def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal intervention over prompts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, help="Prompt dataset JSONL")
    parser.add_argument("--direction", default=None, help="authority_direction_vector.pt")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--direction-spec",
        action="append",
        default=[],
        help="Multi-layer intervention spec 'LAYER:PATH' or 'LAYER:PATH:ALPHA'. Can be repeated.",
    )
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

    if args.direction and args.layer is None:
        raise ValueError("--direction requires --layer")
    if args.layer is not None and not args.direction:
        raise ValueError("--layer requires --direction")

    prompts = read_jsonl(args.prompts)
    mi = ModelInterface(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        probe_instruction=args.probe_instruction,
        refusal_margin=args.refusal_margin,
    )
    engine = CausalInterventionEngine(mi)
    direction_specs = [_parse_direction_spec(raw, args.alpha) for raw in args.direction_spec]
    if args.direction and args.layer is not None:
        direction_specs.insert(
            0,
            (
                int(args.layer),
                Path(args.direction).expanduser().resolve(),
                float(args.alpha),
            ),
        )
    if not direction_specs:
        raise ValueError("Provide --direction/--layer or at least one --direction-spec")

    intervention_fns_by_layer = None
    layer_idx = None
    direction = None
    alpha = float(args.alpha)
    if len(direction_specs) == 1:
        layer_idx, direction_path, alpha = direction_specs[0]
        direction = _load_direction(direction_path)
    else:
        intervention_fns_by_layer = {
            layer: engine.make_projection_removal_fn(_load_direction(direction_path), alpha=spec_alpha)
            for layer, direction_path, spec_alpha in direction_specs
        }

    rows = engine.run(
        prompts=prompts,
        layer_idx=layer_idx,
        direction=direction,
        alpha=alpha,
        max_tokens=args.max_tokens,
        capture_layers=set(),
        capture_attentions=False,
        intervention_fns_by_layer=intervention_fns_by_layer,
    )
    write_json(args.output, {"samples": rows})
    print(f"Saved intervention results to {args.output}")


if __name__ == "__main__":
    main()
