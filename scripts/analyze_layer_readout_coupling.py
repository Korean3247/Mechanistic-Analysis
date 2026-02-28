#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_samples(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"expected 'samples' list in {path}")
    return [row for row in samples if isinstance(row, dict)]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _finite_pairs(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    return [(x, y) for x, y in zip(xs, ys) if _is_finite(x) and _is_finite(y)]


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = _finite_pairs(xs, ys)
    if len(pairs) < 2:
        return 0.0
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    mean_a = float(sum(a) / len(a))
    mean_b = float(sum(b) / len(b))
    var_a = float(sum((x - mean_a) ** 2 for x in a) / len(a))
    var_b = float(sum((y - mean_b) ** 2 for y in b) / len(b))
    if var_a <= 1e-12 or var_b <= 1e-12:
        return 0.0
    cov = float(sum((x - mean_a) * (y - mean_b) for x, y in pairs) / len(pairs))
    return float(cov / math.sqrt(var_a * var_b))


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    pairs = _finite_pairs(xs, ys)
    if len(pairs) < 2:
        return 0.0
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    return _pearson(_rankdata(a), _rankdata(b))


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    pairs = _finite_pairs(xs, ys)
    if len(pairs) < 2:
        return 0.0
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    mean_a = float(sum(a) / len(a))
    mean_b = float(sum(b) / len(b))
    var_a = float(sum((x - mean_a) ** 2 for x in a) / len(a))
    if var_a <= 1e-12:
        return 0.0
    cov = float(sum((x - mean_a) * (y - mean_b) for x, y in pairs) / len(pairs))
    return float(cov / var_a)


def _load_direction(path: Path) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("torch is required for analyze_layer_readout_coupling.py")
    payload = torch.load(path, map_location="cpu")
    direction = payload.get("residual_direction_normalized")
    if not isinstance(direction, torch.Tensor):
        raise ValueError(f"missing residual_direction_normalized in {path}")
    return direction.detach().to(dtype=torch.float32, device="cpu")


def _activation_dir_for_run(run_root: Path) -> Path:
    manifest = _read_json(run_root / "logs" / "run_manifest.json")
    cfg = manifest.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError(f"missing config in run manifest: {run_root}")
    model = str(cfg.get("model", ""))
    activation_base = Path(str(cfg.get("activation_dir", "activation")))
    return activation_base / model.replace("/", "__")


def _hook_key_for_run(run_root: Path, hook_point: str) -> str:
    manifest = _read_json(run_root / "logs" / "run_manifest.json")
    cfg = manifest.get("config", {})
    layer = int(cfg.get("layer_for_sae", 10))
    return f"blocks.{layer}.hook_resid_{hook_point}"


def _layer_for_run(run_root: Path) -> int:
    manifest = _read_json(run_root / "logs" / "run_manifest.json")
    cfg = manifest.get("config", {})
    return int(cfg.get("layer_for_sae", 10))


def _paired_logit_deltas(run_root: Path) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    baseline_rows = _read_samples(run_root / "logs" / "baseline_samples.json")
    intervention_rows = _read_samples(run_root / "logs" / "intervention_samples.json")

    baseline_map = {
        str(row.get("prompt_id")): _safe_float(row.get("logit_diff"))
        for row in baseline_rows
        if str(row.get("framing_type")) == "authority" and str(row.get("risk_tier")) == "unsafe"
    }
    intervention_map = {
        str(row.get("prompt_id")): _safe_float(row.get("logit_diff"))
        for row in intervention_rows
        if str(row.get("risk_tier")) == "unsafe"
    }
    delta_map = {
        prompt_id: intervention_map[prompt_id] - baseline_map[prompt_id]
        for prompt_id in baseline_map
        if prompt_id in intervention_map and _is_finite(baseline_map[prompt_id]) and _is_finite(intervention_map[prompt_id])
    }
    return baseline_map, intervention_map, delta_map


def _projection_map(activation_dir: Path, direction: torch.Tensor, hook_key: str, prompt_ids: set[str]) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for analyze_layer_readout_coupling.py")
    projections: dict[str, float] = {}
    for path in sorted(activation_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        prompt_id = str(meta.get("prompt_id", path.stem))
        if prompt_id not in prompt_ids:
            continue
        residual_stream = payload.get("residual_stream", {})
        if not isinstance(residual_stream, dict) or hook_key not in residual_stream:
            continue
        hidden = residual_stream[hook_key]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
            continue
        vec = hidden[0, -1, :].to(dtype=torch.float32, device="cpu")
        if vec.numel() != direction.numel():
            continue
        projections[prompt_id] = float(torch.dot(vec, direction).item())
    return projections


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_projection_delta(path_pdf: Path, path_png: Path, prompt_rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    labels = sorted({str(row["label"]) for row in prompt_rows})
    if not labels:
        return
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        rows = [row for row in prompt_rows if str(row["label"]) == label]
        xs = [_safe_float(row["projection_value"]) for row in rows]
        ys = [_safe_float(row["delta_logit_diff"]) for row in rows]
        ax.scatter(xs, ys, alpha=0.7, s=18)
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="#444444", linestyle=":", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("baseline projection")
        ax.set_ylabel("delta logit_diff")
        ax.grid(alpha=0.25)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=240)
    plt.close(fig)


def _plot_layer_pair(path_pdf: Path, path_png: Path, comparison_rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    if not comparison_rows:
        return
    first = comparison_rows[0]
    label_a = str(first["label_a"])
    label_b = str(first["label_b"])
    a_map = {str(row["prompt_id"]): _safe_float(row["delta_logit_diff"]) for row in prompt_rows if str(row["label"]) == label_a}
    b_map = {str(row["prompt_id"]): _safe_float(row["delta_logit_diff"]) for row in prompt_rows if str(row["label"]) == label_b}
    common = sorted(pid for pid in a_map if pid in b_map and _is_finite(a_map[pid]) and _is_finite(b_map[pid]))
    if not common:
        return
    xs = [a_map[pid] for pid in common]
    ys = [b_map[pid] for pid in common]
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.scatter(xs, ys, alpha=0.7, s=18)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"delta logit_diff: {label_a}")
    ax.set_ylabel(f"delta logit_diff: {label_b}")
    ax.set_title(f"{label_a} vs {label_b}")
    ax.grid(alpha=0.25)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=240)
    plt.close(fig)


def _write_latex_table(path: Path, summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Projection/readout coupling by layer. Projection values come from baseline authority-unsafe activations; delta logit_diff is intervention minus baseline.}",
        "\\label{tab:layer_readout_coupling}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Run & Layer & $n$ & Pearson $r$ & Spearman $\\rho$ & Slope \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['label']} & {int(row['layer_for_sae'])} & {int(row['n_paired'])} & {float(row['pearson_projection_vs_delta']):+.4f} & {float(row['spearman_projection_vs_delta']):+.4f} & {float(row['linear_slope_delta_on_projection']):+.4f} \\\\"
        )
    if comparison_rows:
        lines.append("\\midrule")
        for row in comparison_rows:
            lines.append(
                f"{row['label_a']} vs {row['label_b']} & -- & {int(row['n_overlap'])} & {float(row['pearson_delta_a_vs_delta_b']):+.4f} & -- & {float(row['sign_agreement_rate']):.4f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze projection/readout coupling across layer-specific runs.")
    parser.add_argument("--run-root", action="append", required=True, help="Result directory with logs and authority_direction_vector.pt")
    parser.add_argument("--label", action="append", default=None, help="Label for corresponding run root")
    parser.add_argument("--activation-root", default=None, help="Optional activation root override")
    parser.add_argument("--hook-point", choices=["pre", "post"], default="post")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_roots = [Path(path).expanduser().resolve() for path in args.run_root]
    labels = list(args.label or [])
    if labels and len(labels) != len(run_roots):
        raise ValueError("--label count must match --run-root count when provided")
    if not labels:
        labels = [run_root.name for run_root in run_roots]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    delta_maps: dict[str, dict[str, float]] = {}

    for label, run_root in zip(labels, run_roots):
        activation_dir = Path(args.activation_root).expanduser().resolve() if args.activation_root else _activation_dir_for_run(run_root)
        direction = _load_direction(run_root / "authority_direction_vector.pt")
        hook_key = _hook_key_for_run(run_root, args.hook_point)
        layer = _layer_for_run(run_root)
        _baseline_map, _intervention_map, delta_map = _paired_logit_deltas(run_root)
        projection_map = _projection_map(activation_dir, direction, hook_key, set(delta_map.keys()))
        common = sorted(pid for pid in delta_map if pid in projection_map and _is_finite(delta_map[pid]) and _is_finite(projection_map[pid]))

        projections = [projection_map[pid] for pid in common]
        deltas = [delta_map[pid] for pid in common]
        delta_maps[label] = {pid: delta_map[pid] for pid in common}

        for prompt_id in common:
            prompt_rows.append(
                {
                    "label": label,
                    "run_root": str(run_root),
                    "layer_for_sae": int(layer),
                    "prompt_id": prompt_id,
                    "projection_value": float(projection_map[prompt_id]),
                    "delta_logit_diff": float(delta_map[prompt_id]),
                }
            )

        summary_rows.append(
            {
                "label": label,
                "run_root": str(run_root),
                "activation_dir": str(activation_dir),
                "hook_key": hook_key,
                "layer_for_sae": int(layer),
                "n_paired": int(len(common)),
                "mean_projection": float(mean(projections)) if projections else 0.0,
                "mean_delta_logit_diff": float(mean(deltas)) if deltas else 0.0,
                "pearson_projection_vs_delta": _pearson(projections, deltas),
                "spearman_projection_vs_delta": _spearman(projections, deltas),
                "linear_slope_delta_on_projection": _linear_slope(projections, deltas),
            }
        )
        print(f"loaded {label}: layer={layer} paired={len(common)} activation_dir={activation_dir}")

    comparison_rows: list[dict[str, Any]] = []
    for idx in range(len(labels)):
        for jdx in range(idx + 1, len(labels)):
            label_a = labels[idx]
            label_b = labels[jdx]
            map_a = delta_maps.get(label_a, {})
            map_b = delta_maps.get(label_b, {})
            common = sorted(pid for pid in map_a if pid in map_b and _is_finite(map_a[pid]) and _is_finite(map_b[pid]))
            deltas_a = [map_a[pid] for pid in common]
            deltas_b = [map_b[pid] for pid in common]
            sign_agreement = 0.0
            if common:
                sign_agreement = float(
                    sum(1 for a, b in zip(deltas_a, deltas_b) if (a == 0 and b == 0) or (a > 0 and b > 0) or (a < 0 and b < 0))
                    / len(common)
                )
            comparison_rows.append(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "n_overlap": int(len(common)),
                    "mean_delta_a": float(mean(deltas_a)) if deltas_a else 0.0,
                    "mean_delta_b": float(mean(deltas_b)) if deltas_b else 0.0,
                    "pearson_delta_a_vs_delta_b": _pearson(deltas_a, deltas_b),
                    "sign_agreement_rate": sign_agreement,
                }
            )

    _write_csv(
        out_dir / "layer_readout_coupling.csv",
        [
            "label",
            "run_root",
            "activation_dir",
            "hook_key",
            "layer_for_sae",
            "n_paired",
            "mean_projection",
            "mean_delta_logit_diff",
            "pearson_projection_vs_delta",
            "spearman_projection_vs_delta",
            "linear_slope_delta_on_projection",
        ],
        summary_rows,
    )
    _write_csv(
        out_dir / "prompt_level_coupling.csv",
        ["label", "run_root", "layer_for_sae", "prompt_id", "projection_value", "delta_logit_diff"],
        prompt_rows,
    )
    _write_csv(
        out_dir / "layer_pair_comparison.csv",
        ["label_a", "label_b", "n_overlap", "mean_delta_a", "mean_delta_b", "pearson_delta_a_vs_delta_b", "sign_agreement_rate"],
        comparison_rows,
    )
    summary = {
        "n_runs": len(summary_rows),
        "runs": summary_rows,
        "layer_pair_comparison": comparison_rows,
    }
    (out_dir / "layer_readout_coupling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_latex_table(out_dir / "table_layer_readout_coupling.tex", summary_rows, comparison_rows)
    _plot_projection_delta(
        out_dir / "figure_projection_vs_delta_coupling.pdf",
        out_dir / "figure_projection_vs_delta_coupling.png",
        prompt_rows,
    )
    _plot_layer_pair(
        out_dir / "figure_layer10_vs_layer12_coupling.pdf",
        out_dir / "figure_layer10_vs_layer12_coupling.png",
        comparison_rows,
        prompt_rows,
    )
    print(f"Wrote: {out_dir / 'layer_readout_coupling.csv'}")
    print(f"Wrote: {out_dir / 'prompt_level_coupling.csv'}")
    print(f"Wrote: {out_dir / 'layer_pair_comparison.csv'}")
    print(f"Wrote: {out_dir / 'layer_readout_coupling_summary.json'}")
    print(f"Wrote: {out_dir / 'table_layer_readout_coupling.tex'}")


if __name__ == "__main__":
    main()
