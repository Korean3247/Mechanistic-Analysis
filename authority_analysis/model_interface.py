from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


InterventionFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ModelForwardArtifacts:
    residual_stream: dict[str, torch.Tensor]
    attention_outputs: list[torch.Tensor]
    final_logits: torch.Tensor
    refusal_logit: float
    compliance_logit: float
    refusal_prob: float
    compliance_prob: float
    logit_diff: float
    prompt_token_count: int


class ModelInterface:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        refusal_token: str = " no",
        compliance_token: str = " yes",
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
        self.refusal_token_id = self._resolve_token_id(refusal_token)
        self.compliance_token_id = self._resolve_token_id(compliance_token)

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

    def _resolve_token_id(self, token_text: str) -> int:
        token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Token '{token_text}' produced no token ids")
        # When multi-token strings are provided, use the first token logit for
        # deterministic scalar refusal/compliance probing.
        return token_ids[0]

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

    def run_forward(
        self,
        prompt_text: str,
        max_tokens: int = 128,
        intervention_layer: int | None = None,
        intervention_fn: InterventionFn | None = None,
    ) -> ModelForwardArtifacts:
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        residual_stream: dict[str, torch.Tensor] = {}
        hook_handles: list[Any] = []

        for layer_idx, layer in enumerate(self.layers):
            def pre_hook_factory(idx: int) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], Any]:
                def pre_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> Any:
                    hidden = inputs[0]
                    residual_stream[f"blocks.{idx}.hook_resid_pre"] = self._to_fp16_cpu(hidden)
                    if intervention_layer is not None and intervention_fn is not None and idx == intervention_layer:
                        updated = intervention_fn(hidden)
                        if len(inputs) == 1:
                            return (updated,)
                        return (updated, *inputs[1:])
                    return None

                return pre_hook

            def post_hook_factory(idx: int) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], Any], None]:
                def post_hook(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: Any,
                ) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    residual_stream[f"blocks.{idx}.hook_resid_post"] = self._to_fp16_cpu(hidden)

                return post_hook

            hook_handles.append(layer.register_forward_pre_hook(pre_hook_factory(layer_idx)))
            hook_handles.append(layer.register_forward_hook(post_hook_factory(layer_idx)))

        try:
            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_attentions=True,
                    use_cache=False,
                )
        finally:
            for handle in hook_handles:
                handle.remove()

        final_token_logits = outputs.logits[:, -1, :]
        probs = torch.softmax(final_token_logits, dim=-1)

        refusal_logit = float(final_token_logits[0, self.refusal_token_id].item())
        compliance_logit = float(final_token_logits[0, self.compliance_token_id].item())
        refusal_prob = float(probs[0, self.refusal_token_id].item())
        compliance_prob = float(probs[0, self.compliance_token_id].item())

        attentions = [self._to_fp16_cpu(attn) for attn in (outputs.attentions or [])]

        return ModelForwardArtifacts(
            residual_stream=residual_stream,
            attention_outputs=attentions,
            final_logits=self._to_fp16_cpu(final_token_logits),
            refusal_logit=refusal_logit,
            compliance_logit=compliance_logit,
            refusal_prob=refusal_prob,
            compliance_prob=compliance_prob,
            logit_diff=refusal_logit - compliance_logit,
            prompt_token_count=int(encoded["input_ids"].shape[1]),
        )
