from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import ensure_dir, write_json



def _load_projection_by_layer(
    activation_dir: str | Path,
    direction: torch.Tensor,
    hook_point: str = "post",
) -> dict[int, dict[str, list[float]]]:
    out: dict[int, dict[str, list[float]]] = {}
    d = direction.detach().to(dtype=torch.float32)

    for file in sorted(Path(activation_dir).glob("*.pt")):
        payload = torch.load(file, map_location="cpu")
        framing = payload.get("metadata", {}).get("framing_type", "unknown")
        residual_stream: dict[str, torch.Tensor] = payload.get("residual_stream", {})

        for key, tensor in residual_stream.items():
            if not key.endswith(f"hook_resid_{hook_point}"):
                continue
            try:
                layer_idx = int(key.split(".")[1])
            except Exception:
                continue
            if tensor.ndim != 3:
                continue
            vec = tensor[:, -1, :].reshape(-1).to(dtype=torch.float32)
            if vec.shape[0] != d.shape[0]:
                continue
            projection = float(torch.dot(vec, d).item())
            layer_bucket = out.setdefault(layer_idx, {})
            layer_bucket.setdefault(framing, []).append(projection)
    return out



def _plot_feature_heatmap(
    framing_mean_latent: dict[str, torch.Tensor],
    top_features: list[int],
    out_path: str | Path,
) -> None:
    framings = sorted(framing_mean_latent.keys())
    if not framings or not top_features:
        return

    matrix = []
    for framing in framings:
        row = framing_mean_latent[framing][top_features].numpy()
        matrix.append(row)

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(top_features) * 0.35), 4))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(framings)))
    ax.set_yticklabels(framings)
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels([str(i) for i in top_features], rotation=45, ha="right")
    ax.set_title("Top SAE Feature Activations by Framing")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Framing Type")
    fig.colorbar(im, ax=ax, label="Mean Latent Activation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def _plot_layer_suppression(
    projection_by_layer: dict[int, dict[str, list[float]]],
    out_path: str | Path,
) -> dict[str, list[float]]:
    layers = sorted(projection_by_layer.keys())
    if not layers:
        return {"layers": [], "suppression": []}

    suppression_values: list[float] = []
    for layer in layers:
        bucket = projection_by_layer[layer]
        authority = bucket.get("authority", [])
        control = []
        for framing, vals in bucket.items():
            if framing != "authority":
                control.extend(vals)
        authority_mean = float(np.mean(authority)) if authority else 0.0
        control_mean = float(np.mean(control)) if control else 0.0
        suppression_values.append(authority_mean - control_mean)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, suppression_values, marker="o")
    ax.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.set_title("Layer-wise Authority Suppression Score")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Projection (Authority - Control)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {"layers": layers, "suppression": suppression_values}



def generate_report(
    result_dir: str | Path,
    baseline_summary: dict[str, Any],
    intervention_summary: dict[str, Any],
    feature_payload: dict[str, Any],
    activation_dir: str | Path,
    hook_point: str = "post",
) -> dict[str, Any]:
    result_root = ensure_dir(result_dir)
    plot_dir = ensure_dir(Path(result_root) / "plots")
    log_dir = ensure_dir(Path(result_root) / "logs")

    direction = feature_payload["residual_direction_normalized"]
    projection_by_layer = _load_projection_by_layer(activation_dir, direction, hook_point=hook_point)

    heatmap_path = Path(plot_dir) / "feature_activation_heatmap.png"
    _plot_feature_heatmap(
        framing_mean_latent=feature_payload.get("framing_mean_latent", {}),
        top_features=feature_payload.get("top_feature_indices", []),
        out_path=heatmap_path,
    )

    suppression_path = Path(plot_dir) / "layer_wise_suppression.png"
    suppression_series = _plot_layer_suppression(projection_by_layer, suppression_path)

    direct_rate = None
    framing_metrics = baseline_summary.get("framing_metrics", {})
    if "direct" in framing_metrics:
        direct_rate = framing_metrics["direct"].get("refusal_rate")

    metrics = {
        "baseline_refusal_rate": direct_rate if direct_rate is not None else baseline_summary.get("control_refusal_rate", 0.0),
        "authority_refusal_rate": baseline_summary.get("authority_refusal_rate", 0.0),
        "intervention_refusal_rate": intervention_summary.get("authority_refusal_rate", intervention_summary.get("overall_refusal_rate", 0.0)),
        "baseline_kl_control_vs_authority": baseline_summary.get("kl_divergence_control_vs_authority", 0.0),
        "layer_wise_suppression": suppression_series,
    }

    write_json(Path(result_root) / "metrics.json", metrics)
    write_json(Path(log_dir) / "baseline_summary.json", baseline_summary)
    write_json(Path(log_dir) / "intervention_summary.json", intervention_summary)

    return {
        "metrics": metrics,
        "plot_paths": {
            "feature_activation_heatmap": str(heatmap_path),
            "layer_wise_suppression": str(suppression_path),
        },
    }
