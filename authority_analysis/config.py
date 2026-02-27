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
    refusal_margin: float = 1.0
    behavioral_ground_truth_samples_per_variant: int = 0
    behavioral_ground_truth_unsafe_min_per_variant: int = 0
    behavioral_ground_truth_max_new_tokens: int = 48
    behavioral_ground_truth_use_probe_instruction: bool = False
    behavioral_ground_truth_temperature: float | None = None
    behavioral_ground_truth_top_p: float | None = None
    posthoc_margins: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    posthoc_bootstrap_iters: int = 5000
    placebo_modes: list[str] = field(default_factory=list)
    placebo_low_importance_features: int = 32
    placebo_shuffle_seed_offset: int = 2000
    classifier_behavior_samples_per_variant: int = 0
    classifier_behavior_unsafe_min_per_variant: int = 0
    classifier_behavior_max_new_tokens: int = 48
    classifier_behavior_use_probe_instruction: bool = False
    classifier_behavior_temperature: float | None = None
    classifier_behavior_top_p: float | None = None
    classifier_behavior_model: str = "facebook/bart-large-mnli"
    classifier_behavior_device: str = "cpu"
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
        if self.refusal_margin < 0:
            raise ValueError("refusal_margin must be >= 0")
        if self.behavioral_ground_truth_samples_per_variant < 0:
            raise ValueError("behavioral_ground_truth_samples_per_variant must be >= 0")
        if self.behavioral_ground_truth_unsafe_min_per_variant < 0:
            raise ValueError("behavioral_ground_truth_unsafe_min_per_variant must be >= 0")
        if (
            self.behavioral_ground_truth_samples_per_variant > 0
            and self.behavioral_ground_truth_unsafe_min_per_variant
            > self.behavioral_ground_truth_samples_per_variant
        ):
            raise ValueError(
                "behavioral_ground_truth_unsafe_min_per_variant cannot exceed "
                "behavioral_ground_truth_samples_per_variant"
            )
        if self.behavioral_ground_truth_max_new_tokens < 1:
            raise ValueError("behavioral_ground_truth_max_new_tokens must be >= 1")
        if self.behavioral_ground_truth_temperature is not None and self.behavioral_ground_truth_temperature <= 0:
            raise ValueError("behavioral_ground_truth_temperature must be > 0 when set")
        if self.behavioral_ground_truth_top_p is not None and not (0 < self.behavioral_ground_truth_top_p <= 1):
            raise ValueError("behavioral_ground_truth_top_p must be in (0, 1] when set")
        if not self.posthoc_margins:
            raise ValueError("posthoc_margins must include at least one margin value")
        if any(m < 0 for m in self.posthoc_margins):
            raise ValueError("posthoc_margins must contain non-negative values")
        if self.posthoc_bootstrap_iters < 1:
            raise ValueError("posthoc_bootstrap_iters must be >= 1")
        valid_placebo_modes = {"random", "low_importance", "orthogonal", "shuffled_latent"}
        if any(mode not in valid_placebo_modes for mode in self.placebo_modes):
            raise ValueError(f"placebo_modes must be subset of {sorted(valid_placebo_modes)}")
        if self.placebo_low_importance_features < 1:
            raise ValueError("placebo_low_importance_features must be >= 1")
        if self.placebo_shuffle_seed_offset < 0:
            raise ValueError("placebo_shuffle_seed_offset must be >= 0")
        if self.classifier_behavior_samples_per_variant < 0:
            raise ValueError("classifier_behavior_samples_per_variant must be >= 0")
        if self.classifier_behavior_unsafe_min_per_variant < 0:
            raise ValueError("classifier_behavior_unsafe_min_per_variant must be >= 0")
        if (
            self.classifier_behavior_samples_per_variant > 0
            and self.classifier_behavior_unsafe_min_per_variant > self.classifier_behavior_samples_per_variant
        ):
            raise ValueError(
                "classifier_behavior_unsafe_min_per_variant cannot exceed "
                "classifier_behavior_samples_per_variant"
            )
        if self.classifier_behavior_max_new_tokens < 1:
            raise ValueError("classifier_behavior_max_new_tokens must be >= 1")
        if (
            self.classifier_behavior_temperature is not None
            and self.classifier_behavior_temperature <= 0
        ):
            raise ValueError("classifier_behavior_temperature must be > 0 when set")
        if self.classifier_behavior_top_p is not None and not (0 < self.classifier_behavior_top_p <= 1):
            raise ValueError("classifier_behavior_top_p must be in (0, 1] when set")
        if not isinstance(self.classifier_behavior_model, str) or not self.classifier_behavior_model.strip():
            raise ValueError("classifier_behavior_model must be a non-empty string")
        if not isinstance(self.classifier_behavior_device, str) or not self.classifier_behavior_device.strip():
            raise ValueError("classifier_behavior_device must be a non-empty string")
        if self.sae_hidden_multiplier < 1:
            raise ValueError("sae_hidden_multiplier must be >= 1")
        if self.alpha_intervention < 0:
            raise ValueError("alpha_intervention must be >= 0")
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
