from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .utils import ensure_dir


class ActivationLogger:
    def __init__(self, activation_root: str | Path) -> None:
        self.activation_root = ensure_dir(activation_root)

    def save_sample(
        self,
        prompt_id: str,
        artifacts: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Path:
        # Security policy: metadata excludes raw completion text and can omit full prompts.
        safe_meta = {
            "prompt_id": metadata.get("prompt_id", prompt_id),
            "framing_type": metadata.get("framing_type", "unknown"),
            "semantic_request_id": metadata.get("semantic_request_id", "unknown"),
            "safety_label": metadata.get("safety_label", "unknown"),
            "risk_tier": metadata.get("risk_tier", "unknown"),
            "risk_level": metadata.get("risk_level", "controlled"),
        }
        payload = {
            "residual_stream": artifacts["residual_stream"],
            "attention_outputs": artifacts["attention_outputs"],
            "final_logits": artifacts["final_logits"],
            "refusal_score": float(artifacts["refusal_score"]),
            "compliance_score": float(artifacts["compliance_score"]),
            "logit_diff": float(artifacts["logit_diff"]),
            "is_refusal": bool(artifacts["is_refusal"]),
            "logits_all_finite": bool(artifacts.get("logits_all_finite", True)),
            "logits_non_finite_count": int(artifacts.get("logits_non_finite_count", 0)),
            "logits_non_finite_ratio": float(artifacts.get("logits_non_finite_ratio", 0.0)),
            "metadata": safe_meta,
        }
        out_path = self.activation_root / f"{prompt_id}.pt"
        torch.save(payload, out_path)
        return out_path

    def list_files(self) -> list[Path]:
        return sorted(self.activation_root.glob("*.pt"))

    def load_sample(self, path: str | Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu")
