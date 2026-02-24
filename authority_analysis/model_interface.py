from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .behavior_evaluator import (
    COMPLIANCE_CUES,
    REFUSAL_CUES,
    cue_first_token_ids,
    cue_scores_from_final_logits,
)


InterventionFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ModelForwardArtifacts:
    residual_stream: dict[str, torch.Tensor]
    attention_outputs: list[torch.Tensor]
    final_logits: torch.Tensor
    refusal_score: float
    compliance_score: float
    logit_diff: float
    is_refusal: bool
    prompt_token_count: int


class ModelInterface:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        probe_instruction: str = "Answer with exactly one word: yes or no.",
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        torch_dtype = self._resolve_dtype(dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.layers = self._decoder_layers()
        self.refusal_cue_token_ids = cue_first_token_ids(self.tokenizer, REFUSAL_CUES)
        self.compliance_cue_token_ids = cue_first_token_ids(self.tokenizer, COMPLIANCE_CUES)
        self.probe_instruction = probe_instruction.strip()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
        if dtype == "float16" and device.type == "cuda":
            return torch.float16
        if dtype == "bfloat16" and device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    def _decoder_layers(self) -> Sequence[torch.nn.Module]:
        model = self.model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox.layers
        raise ValueError("Unsupported architecture: unable to locate decoder layers")

    @staticmethod
    def _to_fp16_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(dtype=torch.float16, device="cpu")

    def compose_prompt(self, prompt_text: str) -> str:
        base = prompt_text.rstrip()
        return f"{base}\n\n{self.probe_instruction}"

    def run_forward(
        self,
        prompt_text: str,
        max_tokens: int = 128,
        intervention_layer: int | None = None,
        intervention_fn: InterventionFn | None = None,
        capture_layers: set[int] | None = None,
        capture_attentions: bool = False,
    ) -> ModelForwardArtifacts:
        prompt_text = self.compose_prompt(prompt_text)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        residual_stream: dict[str, torch.Tensor] = {}
        hook_handles: list[Any] = []
        capture_all = capture_layers is None
        capture_layer_set = set(capture_layers or [])
        hook_layer_set = set(range(len(self.layers))) if capture_all else set(capture_layer_set)
        if intervention_layer is not None:
            hook_layer_set.add(intervention_layer)

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx not in hook_layer_set:
                continue
            should_capture_layer = capture_all or layer_idx in capture_layer_set

            def pre_hook_factory(
                idx: int, should_capture: bool
            ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], Any]:
                def pre_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> Any:
                    hidden = inputs[0]
                    if should_capture:
                        residual_stream[f"blocks.{idx}.hook_resid_pre"] = self._to_fp16_cpu(hidden)
                    if intervention_layer is not None and intervention_fn is not None and idx == intervention_layer:
                        updated = intervention_fn(hidden)
                        if len(inputs) == 1:
                            return (updated,)
                        return (updated, *inputs[1:])
                    return None

                return pre_hook

            def post_hook_factory(
                idx: int, should_capture: bool
            ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], Any], None]:
                def post_hook(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: Any,
                ) -> None:
                    if not should_capture:
                        return
                    hidden = output[0] if isinstance(output, tuple) else output
                    residual_stream[f"blocks.{idx}.hook_resid_post"] = self._to_fp16_cpu(hidden)

                return post_hook

            hook_handles.append(layer.register_forward_pre_hook(pre_hook_factory(layer_idx, should_capture_layer)))
            hook_handles.append(layer.register_forward_hook(post_hook_factory(layer_idx, should_capture_layer)))

        try:
            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_attentions=capture_attentions,
                    use_cache=False,
                )
        finally:
            for handle in hook_handles:
                handle.remove()

        final_token_logits = outputs.logits[:, -1, :]
        scores = cue_scores_from_final_logits(
            final_logits=final_token_logits,
            refusal_token_ids=self.refusal_cue_token_ids,
            compliance_token_ids=self.compliance_cue_token_ids,
        )

        attentions = (
            [self._to_fp16_cpu(attn) for attn in (outputs.attentions or [])]
            if capture_attentions
            else []
        )

        return ModelForwardArtifacts(
            residual_stream=residual_stream,
            attention_outputs=attentions,
            final_logits=self._to_fp16_cpu(final_token_logits),
            refusal_score=float(scores["refusal_score"]),
            compliance_score=float(scores["compliance_score"]),
            logit_diff=float(scores["logit_diff"]),
            is_refusal=bool(scores["is_refusal"]),
            prompt_token_count=int(encoded["input_ids"].shape[1]),
        )
