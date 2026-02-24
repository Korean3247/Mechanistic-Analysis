from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .utils import write_json


@dataclass
class SAETrainConfig:
    hidden_multiplier: int = 8
    lr: float = 1e-3
    l1_lambda: float = 1e-3
    epochs: int = 40
    patience: int = 5
    batch_size: int = 64
    seed: int = 42


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))



def load_layer_residual_matrix(
    activation_dir: str | Path,
    layer_idx: int,
    hook_point: str = "post",
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    key = f"blocks.{layer_idx}.hook_resid_{hook_point}"
    vectors: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    for file in sorted(Path(activation_dir).glob("*.pt")):
        payload = torch.load(file, map_location="cpu")
        residual_stream = payload.get("residual_stream", {})
        if key not in residual_stream:
            continue
        tensor = residual_stream[key]
        if tensor.ndim != 3:
            continue
        last_token_vec = tensor[:, -1, :].reshape(-1).to(dtype=torch.float32)
        vectors.append(last_token_vec)
        metadata.append(payload.get("metadata", {}))

    if not vectors:
        raise ValueError(f"No residual vectors found for key={key} in {activation_dir}")

    matrix = torch.stack(vectors, dim=0)
    return matrix, metadata



def train_sae(
    data: torch.Tensor,
    cfg: SAETrainConfig,
    device: torch.device,
) -> tuple[SparseAutoencoder, dict[str, Any]]:
    torch.manual_seed(cfg.seed)

    input_dim = int(data.shape[1])
    hidden_dim = int(input_dim * cfg.hidden_multiplier)
    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    dataset = TensorDataset(data)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        train_losses: list[float] = []
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            recon, z = model(x_batch)
            recon_loss = mse(recon, x_batch)
            sparse_loss = cfg.l1_lambda * torch.mean(torch.abs(z))
            loss = recon_loss + sparse_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                recon, z = model(x_batch)
                recon_loss = mse(recon, x_batch)
                sparse_loss = cfg.l1_lambda * torch.mean(torch.abs(z))
                val_losses.append(float((recon_loss + sparse_loss).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("Failed to train SAE: no best state found")

    model.load_state_dict(best_state)
    summary = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "history": history,
    }
    return model, summary



def save_sae_checkpoint(
    out_path: str | Path,
    model: SparseAutoencoder,
    layer_idx: int,
    hook_point: str,
    train_summary: dict[str, Any],
) -> None:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "layer_idx": layer_idx,
        "hook_point": hook_point,
        "input_dim": train_summary["input_dim"],
        "hidden_dim": train_summary["hidden_dim"],
        "train_summary": train_summary,
    }
    torch.save(payload, out_path)



def load_sae_checkpoint(path: str | Path) -> tuple[SparseAutoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    model = SparseAutoencoder(payload["input_dim"], payload["hidden_dim"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = {
        "layer_idx": int(payload["layer_idx"]),
        "hook_point": str(payload.get("hook_point", "post")),
        "train_summary": payload.get("train_summary", {}),
    }
    return model, meta



def main() -> None:
    parser = argparse.ArgumentParser(description="Train sparse autoencoder on residual activations")
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hook-point", default="post", choices=["pre", "post"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--hidden-multiplier", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-lambda", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data, _ = load_layer_residual_matrix(args.activation_dir, args.layer, hook_point=args.hook_point)
    cfg = SAETrainConfig(
        hidden_multiplier=args.hidden_multiplier,
        lr=args.lr,
        l1_lambda=args.l1_lambda,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    device = torch.device(args.device)
    model, summary = train_sae(data, cfg, device)
    save_sae_checkpoint(args.output, model, args.layer, args.hook_point, summary)
    write_json(args.summary_output, summary)
    print(f"Saved SAE checkpoint to {args.output}")


if __name__ == "__main__":
    main()
