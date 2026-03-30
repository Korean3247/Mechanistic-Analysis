#!/usr/bin/env python3
"""SAE latent decomposition visualization.

Shows which SAE features contribute most to the authority direction,
producing per-feature bar charts and a top-k decomposition figure.

Usage:
    python scripts/visualize_sae_decomposition.py \
        --sae results/.../authority_sae.pt \
        --direction results/.../authority_direction_vector.pt \
        --output-dir results/sae_decomposition
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from authority_analysis.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae", required=True, help="Path to authority_sae.pt")
    parser.add_argument("--direction", required=True, help="Path to authority_direction_vector.pt")
    parser.add_argument("--output-dir", default="results/sae_decomposition")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Load SAE
    sae_data = torch.load(args.sae, map_location="cpu")
    if isinstance(sae_data, dict):
        # Try to extract decoder weights
        W_dec = sae_data.get("decoder.weight") or sae_data.get("W_dec")
        W_enc = sae_data.get("encoder.weight") or sae_data.get("W_enc")
        if W_dec is None:
            # It might be a state_dict
            for key in sae_data:
                if "dec" in key.lower() and "weight" in key.lower():
                    W_dec = sae_data[key]
                    break
                if "W_dec" in key:
                    W_dec = sae_data[key]
                    break
        if W_enc is None:
            for key in sae_data:
                if "enc" in key.lower() and "weight" in key.lower():
                    W_enc = sae_data[key]
                    break
    else:
        # Assume it's the model itself
        W_dec = sae_data.decoder.weight.data if hasattr(sae_data, "decoder") else None
        W_enc = sae_data.encoder.weight.data if hasattr(sae_data, "encoder") else None

    if W_dec is None:
        print("WARNING: Could not find decoder weights, trying full state_dict keys:")
        if isinstance(sae_data, dict):
            for k in sae_data:
                print(f"  {k}: {type(sae_data[k])}")
        return

    # Load direction
    dir_data = torch.load(args.direction, map_location="cpu")
    direction = dir_data.get("residual_direction_normalized") or dir_data.get("direction")
    if direction is None:
        raise ValueError("No direction found")
    direction = direction.float()

    # Compute alignment of each SAE feature with the authority direction
    # W_dec columns (or rows) are the feature directions
    W_dec = W_dec.float()
    if W_dec.shape[0] == direction.shape[0]:
        # W_dec: [d_model, n_features] -- each column is a feature
        feature_dirs = W_dec.T  # [n_features, d_model]
    else:
        # W_dec: [n_features, d_model]
        feature_dirs = W_dec

    # Normalize feature directions
    norms = feature_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    feature_dirs_normed = feature_dirs / norms

    # Cosine similarity with authority direction
    direction_normed = direction / direction.norm().clamp(min=1e-8)
    cosine_sims = feature_dirs_normed @ direction_normed  # [n_features]

    # Projection magnitude (contribution)
    projections = (feature_dirs @ direction)  # unnormalized alignment

    n_features = len(cosine_sims)
    print(f"SAE features: {n_features}")
    print(f"Direction shape: {direction.shape}")

    # Top-k by absolute cosine similarity
    abs_cos = cosine_sims.abs()
    topk_vals, topk_idx = abs_cos.topk(min(args.top_k, n_features))

    print(f"\nTop-{args.top_k} SAE features aligned with authority direction:")
    print(f"{'Rank':>4} | {'Feature':>8} | {'Cosine':>8} | {'Projection':>11} | {'Norm':>8}")
    feature_data = []
    for rank, (val, idx) in enumerate(zip(topk_vals, topk_idx), 1):
        cos = cosine_sims[idx].item()
        proj = projections[idx].item()
        norm = norms[idx].item()
        print(f"{rank:>4} | {idx.item():>8} | {cos:>+8.4f} | {proj:>+11.4f} | {norm:>8.4f}")
        feature_data.append({
            "rank": rank,
            "feature_idx": idx.item(),
            "cosine_sim": cos,
            "projection": proj,
            "norm": norm,
        })

    # Save data
    with (out_dir / "top_features.json").open("w") as f:
        json.dump(feature_data, f, indent=2)

    # Save full cosine distribution
    np.save(str(out_dir / "cosine_sims.npy"), cosine_sims.numpy())

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Top-k bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = [f"F{d['feature_idx']}" for d in feature_data]
        cosines = [d["cosine_sim"] for d in feature_data]
        colors = ["#d62728" if c > 0 else "#1f77b4" for c in cosines]
        ax.barh(indices[::-1], cosines[::-1], color=colors[::-1])
        ax.set_xlabel("Cosine similarity with authority direction")
        ax.set_title(f"Top-{args.top_k} SAE features by authority alignment")
        ax.axvline(0, color="k", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(out_dir / "topk_features_bar.pdf", dpi=150)
        fig.savefig(out_dir / "topk_features_bar.png", dpi=150)
        plt.close(fig)

        # 2. Full cosine distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(cosine_sims.numpy(), bins=100, color="#2ca02c", alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.set_xlabel("Cosine similarity with authority direction")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of SAE feature–authority alignment")
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        # Mark top features
        for d in feature_data[:5]:
            ax.axvline(d["cosine_sim"], color="red", linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        fig.savefig(out_dir / "cosine_distribution.pdf", dpi=150)
        fig.savefig(out_dir / "cosine_distribution.png", dpi=150)
        plt.close(fig)

        # 3. Cumulative contribution
        sorted_projs, sorted_idx = projections.abs().sort(descending=True)
        cumsum = sorted_projs.cumsum(0)
        cumsum_pct = cumsum / cumsum[-1] * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(cumsum_pct) + 1), cumsum_pct.numpy(), color="#1f77b4")
        ax.set_xlabel("Number of SAE features (ranked by contribution)")
        ax.set_ylabel("Cumulative % of total projection")
        ax.set_title("Cumulative authority direction reconstruction")
        ax.axhline(90, color="red", linestyle="--", alpha=0.5, label="90%")
        ax.axhline(95, color="orange", linestyle="--", alpha=0.5, label="95%")
        # Find 90% and 95% points
        n90 = (cumsum_pct >= 90).nonzero(as_tuple=True)[0][0].item() + 1
        n95 = (cumsum_pct >= 95).nonzero(as_tuple=True)[0][0].item() + 1
        ax.axvline(n90, color="red", linestyle=":", alpha=0.3)
        ax.axvline(n95, color="orange", linestyle=":", alpha=0.3)
        ax.legend()
        ax.set_xlim(0, min(200, n_features))
        plt.tight_layout()
        fig.savefig(out_dir / "cumulative_contribution.pdf", dpi=150)
        fig.savefig(out_dir / "cumulative_contribution.png", dpi=150)
        plt.close(fig)

        print(f"\n90% reconstruction: {n90} features")
        print(f"95% reconstruction: {n95} features")
        print(f"Plots saved to {out_dir}")

    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
