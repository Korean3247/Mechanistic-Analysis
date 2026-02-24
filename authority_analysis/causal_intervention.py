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

        def intervention(hidden: torch.Tensor) -> torch.Tensor:
            d = direction.to(device=hidden.device, dtype=hidden.dtype)
            denom = torch.sum(d * d) + 1e-8
            coeff = torch.sum(hidden * d, dim=-1, keepdim=True) / denom
            projected = coeff * d
            return hidden - alpha * projected

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
    parser.add_argument("--refusal-margin", type=float, default=0.5)
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
