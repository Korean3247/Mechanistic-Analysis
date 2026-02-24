from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .sae_module import load_layer_residual_matrix, load_sae_checkpoint
from .utils import write_json



def compute_feature_analysis(
    activation_dir: str | Path,
    sae_ckpt_path: str | Path,
    layer_idx: int,
    hook_point: str = "post",
    top_k: int = 24,
) -> dict[str, Any]:
    matrix, metadata = load_layer_residual_matrix(activation_dir, layer_idx=layer_idx, hook_point=hook_point)
    sae_model, _ = load_sae_checkpoint(sae_ckpt_path)

    with torch.inference_mode():
        latent = sae_model.encode(matrix)

    framings = [str(m.get("framing_type", "unknown")) for m in metadata]
    framing_set = sorted(set(framings))

    per_framing_mean: dict[str, torch.Tensor] = {}
    for framing in framing_set:
        idx = [i for i, f in enumerate(framings) if f == framing]
        if not idx:
            continue
        per_framing_mean[framing] = latent[idx].mean(dim=0)

    if "authority" not in per_framing_mean:
        raise ValueError("No authority samples found in activation set")

    control_idx = [i for i, f in enumerate(framings) if f != "authority"]
    if not control_idx:
        raise ValueError("No control samples found in activation set")

    authority_mean = latent[[i for i, f in enumerate(framings) if f == "authority"]].mean(dim=0)
    control_mean = latent[control_idx].mean(dim=0)
    latent_direction = authority_mean - control_mean

    decoder_weight = sae_model.decoder.weight.detach().cpu()  # [input_dim, hidden_dim]
    residual_direction = torch.matmul(latent_direction, decoder_weight.T)
    residual_norm = torch.linalg.norm(residual_direction) + 1e-8
    residual_direction_normalized = residual_direction / residual_norm

    top_k = min(top_k, latent_direction.shape[0])
    top_indices = torch.topk(torch.abs(latent_direction), k=top_k).indices.tolist()

    return {
        "authority_mean_latent": authority_mean.detach().cpu(),
        "control_mean_latent": control_mean.detach().cpu(),
        "latent_direction": latent_direction.detach().cpu(),
        "residual_direction": residual_direction.detach().cpu(),
        "residual_direction_normalized": residual_direction_normalized.detach().cpu(),
        "top_feature_indices": top_indices,
        "framing_mean_latent": {k: v.detach().cpu() for k, v in per_framing_mean.items()},
        "layer_idx": layer_idx,
        "hook_point": hook_point,
    }



def save_feature_analysis(
    output_dir: str | Path,
    analysis: dict[str, Any],
    vector_filename: str = "authority_direction_vector.pt",
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vector_path = out_dir / vector_filename
    torch.save(analysis, vector_path)

    summary = {
        "layer_idx": int(analysis["layer_idx"]),
        "hook_point": str(analysis["hook_point"]),
        "top_feature_indices": [int(i) for i in analysis["top_feature_indices"]],
        "latent_direction_l2": float(torch.linalg.norm(analysis["latent_direction"]).item()),
        "residual_direction_l2": float(torch.linalg.norm(analysis["residual_direction"]).item()),
        "framings": sorted(list(analysis["framing_mean_latent"].keys())),
    }
    summary_path = out_dir / "feature_analysis_summary.json"
    write_json(summary_path, summary)
    return vector_path, summary_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAE latent features")
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook-point", default="post", choices=["pre", "post"])
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    analysis = compute_feature_analysis(
        activation_dir=args.activation_dir,
        sae_ckpt_path=args.sae_checkpoint,
        layer_idx=args.layer,
        hook_point=args.hook_point,
        top_k=args.top_k,
    )
    vector_path, summary_path = save_feature_analysis(args.output_dir, analysis)
    print(f"Saved direction vector: {vector_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
