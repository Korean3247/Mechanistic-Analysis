#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(v: float) -> bool:
    return math.isfinite(v)


def _safe_mean(vals: list[float]) -> float:
    x = [v for v in vals if _is_finite(v)]
    return float(mean(x)) if x else float("nan")


def _safe_std(vals: list[float]) -> float:
    x = [v for v in vals if _is_finite(v)]
    return float(pstdev(x)) if len(x) >= 2 else 0.0


def _format(v: float, nd: int = 4) -> str:
    if not _is_finite(v):
        return "N/A"
    return f"{v:.{nd}f}"


def _build_latex_table(out_path: Path, summary: dict[str, Any]) -> None:
    rows = summary["layer_alpha1_summary_rows"]
    signflip = summary["sign_flip_l10_l12_alpha1"]
    cos_info = summary["cosine_diagnostics"]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Layer transfer diagnostics at $\\alpha=1.0$.}",
        "\\label{tab:layer_transfer_diagnostics}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Layer & Mean Shift & Std (seed) & Sign \\\\",
        "\\midrule",
    ]
    for r in rows:
        sign = "+" if _to_float(r["mean_shift_mean"]) > 0 else "-"
        lines.append(
            f"L{int(r['layer_for_sae'])} & {_format(_to_float(r['mean_shift_mean']))} & "
            f"{_format(_to_float(r['mean_shift_std']))} & {sign} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"L10$\\leftrightarrow$L12 sign flip & \\multicolumn{{3}}{{c}}{{{('Yes' if signflip else 'No')}}} \\\\",
        ]
    )

    if cos_info.get("status") == "ok":
        c = cos_info.get("cosine_mean", {})
        lines.append(
            "Cosine $\\cos(d_{10},d_{12})$ & \\multicolumn{3}{c}{"
            + _format(_to_float(c.get("l10_l12")))
            + "} \\\\"
        )
    else:
        lines.append(
            "Cosine $\\cos(d_{10},d_{12})$ & \\multicolumn{3}{c}{N/A (run-level direction vectors unavailable in this archive)} \\\\"
        )

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_direction_path(run_dir_raw: str, results_root: Path | None) -> Path:
    run_dir = Path(run_dir_raw)
    if run_dir.exists():
        return run_dir / "authority_direction_vector.pt"
    if results_root is not None:
        return results_root / run_dir.name / "authority_direction_vector.pt"
    return run_dir / "authority_direction_vector.pt"


def _cosine_diagnostics(alpha1_rows: list[dict[str, str]], results_root: Path | None) -> dict[str, Any]:
    # Optional dependency: this works only when run-level direction vectors are available.
    try:
        import torch
    except Exception:
        return {
            "status": "unavailable",
            "reason": "torch_not_available",
            "n_seed_triplets": 0,
            "cosine_rows": [],
            "cosine_mean": {},
        }

    by_seed_layer: dict[tuple[int, int], Path] = {}
    for row in alpha1_rows:
        seed = int(float(row["seed"]))
        layer = int(float(row["layer_for_sae"]))
        path = _resolve_direction_path(row.get("run_dir", ""), results_root)
        if path.exists():
            by_seed_layer[(seed, layer)] = path

    seeds = sorted({seed for seed, _ in by_seed_layer.keys()})
    cosine_rows: list[dict[str, float | int]] = []
    for seed in seeds:
        need = [(seed, 8), (seed, 10), (seed, 12)]
        if not all(k in by_seed_layer for k in need):
            continue
        vectors: dict[int, Any] = {}
        ok = True
        for _, layer in need:
            payload = torch.load(by_seed_layer[(seed, layer)], map_location="cpu")
            vec = payload.get("direction")
            if vec is None:
                ok = False
                break
            vec = vec.float().flatten()
            if not torch.isfinite(vec).all():
                ok = False
                break
            vectors[layer] = vec
        if not ok:
            continue

        def cos(a: int, b: int) -> float:
            va = vectors[a]
            vb = vectors[b]
            na = torch.linalg.norm(va).item()
            nb = torch.linalg.norm(vb).item()
            if na <= 1e-12 or nb <= 1e-12:
                return float("nan")
            return float(torch.dot(va, vb).item() / (na * nb))

        cosine_rows.append(
            {
                "seed": seed,
                "l8_l10": cos(8, 10),
                "l10_l12": cos(10, 12),
                "l8_l12": cos(8, 12),
            }
        )

    if not cosine_rows:
        return {
            "status": "unavailable",
            "reason": "direction_vectors_missing_for_layer_triplets",
            "n_seed_triplets": 0,
            "cosine_rows": [],
            "cosine_mean": {},
        }

    keys = ("l8_l10", "l10_l12", "l8_l12")
    cosine_mean = {}
    for k in keys:
        vals = [_to_float(r[k]) for r in cosine_rows]
        cosine_mean[k] = _safe_mean(vals)

    return {
        "status": "ok",
        "reason": "computed",
        "n_seed_triplets": len(cosine_rows),
        "cosine_rows": cosine_rows,
        "cosine_mean": cosine_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze layer transfer diagnostics from layer/alpha sweep results.")
    parser.add_argument("--robustness-dir", required=True, help="Directory containing sweep_runs.csv and layer_summary.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for diagnostics artifacts")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional local results root to resolve run_dir basenames when absolute run_dir paths are not available",
    )
    args = parser.parse_args()

    robustness_dir = Path(args.robustness_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else None

    sweep_rows = _read_csv(robustness_dir / "sweep_runs.csv")
    alpha1_rows = [r for r in sweep_rows if abs(_to_float(r.get("alpha_intervention")) - 1.0) < 1e-12]

    layer_to_vals: dict[int, list[float]] = {}
    for r in alpha1_rows:
        layer = int(float(r["layer_for_sae"]))
        v = _to_float(r.get("mean_shift"))
        if _is_finite(v):
            layer_to_vals.setdefault(layer, []).append(v)

    layer_rows: list[dict[str, Any]] = []
    for layer in sorted(layer_to_vals.keys()):
        vals = layer_to_vals[layer]
        layer_rows.append(
            {
                "layer_for_sae": layer,
                "n": len(vals),
                "mean_shift_mean": _safe_mean(vals),
                "mean_shift_std": _safe_std(vals),
            }
        )

    by_layer = {int(r["layer_for_sae"]): _to_float(r["mean_shift_mean"]) for r in layer_rows}
    sign_flip_l10_l12 = (
        10 in by_layer
        and 12 in by_layer
        and _is_finite(by_layer[10])
        and _is_finite(by_layer[12])
        and by_layer[10] * by_layer[12] < 0
    )

    cosine_info = _cosine_diagnostics(alpha1_rows=alpha1_rows, results_root=results_root)

    summary = {
        "inputs": {
            "robustness_dir": str(robustness_dir),
            "results_root": str(results_root) if results_root else None,
        },
        "alpha_fixed": 1.0,
        "n_runs_alpha1": len(alpha1_rows),
        "layer_alpha1_summary_rows": layer_rows,
        "sign_flip_l10_l12_alpha1": bool(sign_flip_l10_l12),
        "cosine_diagnostics": cosine_info,
    }

    (out_dir / "layer_transfer_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "layer_transfer_alpha1_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer_for_sae", "n", "mean_shift_mean", "mean_shift_std"],
        )
        writer.writeheader()
        writer.writerows(layer_rows)

    _build_latex_table(out_path=out_dir / "table_appendix_layer_transfer_diagnostics.tex", summary=summary)
    print(f"Wrote: {out_dir / 'layer_transfer_diagnostics.json'}")
    print(f"Wrote: {out_dir / 'layer_transfer_alpha1_summary.csv'}")
    print(f"Wrote: {out_dir / 'table_appendix_layer_transfer_diagnostics.tex'}")


if __name__ == "__main__":
    main()
