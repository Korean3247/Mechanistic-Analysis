from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any

import yaml

from .constants import REQUIRED_FRAMINGS


@dataclass
class ExperimentConfig:
    experiment_name: str
    model: str
    semantic_requests_path: str = "data/semantic_requests.jsonl"
    prompt_dataset_path: str = "data/prompts.jsonl"
    results_dir: str = "results"
    activation_dir: str = "activation"
    layer_for_sae: int = 10
    batch_size: int = 16
    max_tokens: int = 128
    alpha_intervention: float = 1.0
    seed: int = 42
    device: str = "auto"
    dtype: str = "float16"
    probe_instruction: str = "Answer with exactly one word: yes or no."
    capture_attentions: bool = False
    capture_all_layers: bool = False
    capture_layers: list[int] | None = None
    framing_types: list[str] = field(default_factory=lambda: REQUIRED_FRAMINGS.copy())
    control_framing_types: list[str] = field(
        default_factory=lambda: ["direct", "hypothetical", "academic", "institutional"]
    )
    dataset_version: str = "v1"
    generate_prompts_if_missing: bool = True
    sae_hidden_multiplier: int = 8
    sae_l1_lambda: float = 1e-3
    sae_lr: float = 1e-3
    sae_epochs: int = 40
    sae_patience: int = 5
    sae_batch_size: int = 64
    top_k_features: int = 24

    def validate(self) -> None:
        missing = [f for f in REQUIRED_FRAMINGS if f not in self.framing_types]
        if missing:
            raise ValueError(f"framing_types is missing required entries: {missing}")
        if "authority" not in self.framing_types:
            raise ValueError("framing_types must include 'authority'")
        if not isinstance(self.probe_instruction, str) or not self.probe_instruction.strip():
            raise ValueError("probe_instruction must be a non-empty string")
        if self.sae_hidden_multiplier < 1:
            raise ValueError("sae_hidden_multiplier must be >= 1")
        if self.alpha_intervention != 1.0:
            # Spec requires fixed alpha=1.0.
            self.alpha_intervention = 1.0
        if self.capture_layers is not None:
            if not isinstance(self.capture_layers, list):
                raise ValueError("capture_layers must be a list of layer indices")
            if any((not isinstance(i, int) or i < 0) for i in self.capture_layers):
                raise ValueError("capture_layers must contain non-negative integers")
        if not self.capture_all_layers:
            layers = list(self.capture_layers) if self.capture_layers else [self.layer_for_sae]
            if self.layer_for_sae not in layers:
                layers.append(self.layer_for_sae)
            # Deduplicate while preserving order.
            deduped: list[int] = []
            for idx in layers:
                if idx not in deduped:
                    deduped.append(idx)
            self.capture_layers = deduped

    def model_slug(self) -> str:
        return self.model.replace("/", "__")

    def prompt_dataset(self) -> Path:
        return Path(self.prompt_dataset_path)

    def semantic_requests(self) -> Path:
        return Path(self.semantic_requests_path)

    def results_root(self) -> Path:
        return Path(self.results_dir) / self.experiment_name

    def activations_root(self) -> Path:
        return Path(self.activation_dir) / self.model_slug()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError("Config YAML must contain a mapping at top level")
    valid_keys = {f.name for f in dataclass_fields(ExperimentConfig)}
    filtered_payload = {k: v for k, v in payload.items() if k in valid_keys}
    cfg = ExperimentConfig(**filtered_payload)
    cfg.validate()
    return cfg
