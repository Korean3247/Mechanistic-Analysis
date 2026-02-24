from __future__ import annotations

import re
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
        refusal_margin: float = 1.0,
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
        self.refusal_margin = float(refusal_margin)

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

    @staticmethod
    def _trim_generated_text(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        trimmed = " ".join(sentences[:max_sentences]).strip()
        if len(trimmed) > max_chars:
            trimmed = trimmed[:max_chars].rstrip()
        return trimmed

    @staticmethod
    def _remove_hooks(hook_handles: list[Any]) -> None:
        for handle in hook_handles:
            handle.remove()

    def _register_layer_hooks(
        self,
        residual_stream: dict[str, torch.Tensor] | None,
        capture_layers: set[int] | None,
        intervention_layer: int | None,
        intervention_fn: InterventionFn | None,
    ) -> list[Any]:
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
                    if should_capture and residual_stream is not None:
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
                    if not should_capture or residual_stream is None:
                        return
                    hidden = output[0] if isinstance(output, tuple) else output
                    residual_stream[f"blocks.{idx}.hook_resid_post"] = self._to_fp16_cpu(hidden)

                return post_hook

            hook_handles.append(layer.register_forward_pre_hook(pre_hook_factory(layer_idx, should_capture_layer)))
            hook_handles.append(layer.register_forward_hook(post_hook_factory(layer_idx, should_capture_layer)))

        return hook_handles

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
        hook_handles = self._register_layer_hooks(
            residual_stream=residual_stream,
            capture_layers=capture_layers,
            intervention_layer=intervention_layer,
            intervention_fn=intervention_fn,
        )

        try:
            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_attentions=capture_attentions,
                    use_cache=False,
                )
        finally:
            self._remove_hooks(hook_handles)

        final_token_logits = outputs.logits[:, -1, :]
        scores = cue_scores_from_final_logits(
            final_logits=final_token_logits,
            refusal_token_ids=self.refusal_cue_token_ids,
            compliance_token_ids=self.compliance_cue_token_ids,
            margin=self.refusal_margin,
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

    def generate_short_completion(
        self,
        prompt_text: str,
        max_tokens: int = 128,
        max_new_tokens: int = 48,
        use_probe_instruction: bool = False,
        intervention_layer: int | None = None,
        intervention_fn: InterventionFn | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        source_prompt = self.compose_prompt(prompt_text) if use_probe_instruction else prompt_text.rstrip()
        encoded = self.tokenizer(
            source_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        hook_handles = self._register_layer_hooks(
            residual_stream=None,
            capture_layers=set(),
            intervention_layer=intervention_layer,
            intervention_fn=intervention_fn,
        )

        prev_det_state: bool | None = None
        det_temporarily_disabled = False
        try:
            with torch.inference_mode():
                do_sample = bool((temperature is not None) or (top_p is not None))
                generate_kwargs: dict[str, Any] = {
                    **encoded,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "num_beams": 1,
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if do_sample:
                    try:
                        prev_det_state = torch.are_deterministic_algorithms_enabled()
                        if prev_det_state:
                            torch.use_deterministic_algorithms(False)
                            det_temporarily_disabled = True
                    except Exception:
                        prev_det_state = None
                    if temperature is not None:
                        generate_kwargs["temperature"] = float(temperature)
                    if top_p is not None:
                        generate_kwargs["top_p"] = float(top_p)
                generated = self.model.generate(
                    **generate_kwargs,
                )
        finally:
            if det_temporarily_disabled and prev_det_state is not None:
                try:
                    torch.use_deterministic_algorithms(prev_det_state)
                except Exception:
                    pass
            self._remove_hooks(hook_handles)

        prompt_len = int(encoded["input_ids"].shape[1])
        completion_ids = generated[0, prompt_len:]
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return self._trim_generated_text(decoded)
