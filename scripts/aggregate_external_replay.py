#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from authority_analysis.utils import ensure_dir, write_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _model_label(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama" in lowered and "8b" in lowered:
        return "LLaMA-3-8B-IT"
    if "gemma" in lowered and "9b" in lowered:
        return "Gemma-2-9B-IT"
    if "mistral" in lowered and "7b" in lowered:
        return "Mistral-7B-IT"
    tail = model_name.split("/")[-1].strip()
    return tail or model_name


def _margin_row(report: dict[str, Any], margin: float) -> dict[str, Any]:
    for row in report.get("margin_sweep", []):
        if abs(_to_float(row.get("margin")) - margin) < 1e-12:
            return row
    return {}


def _ci95(mean_shift: float, std: float, n: int) -> tuple[float, float]:
    if n <= 0 or not (_is_finite(mean_shift) and _is_finite(std)):
        return float("nan"), float("nan")
    se = std / math.sqrt(max(1, n))
    return mean_shift - 1.96 * se, mean_shift + 1.96 * se


def _fixed_effect_pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights: list[float] = []
    effects: list[float] = []
    for row in rows:
        effect = _to_float(row.get("mean_shift_intervention_minus_baseline"))
        std = _to_float(row.get("delta_std"))
        n = int(float(row.get("n_paired_authority_unsafe", 0)))
        if n <= 0 or not (_is_finite(effect) and _is_finite(std)):
            continue
        se = std / math.sqrt(max(1, n))
        if not _is_finite(se) or se <= 0:
            continue
        weight = 1.0 / (se * se)
        weights.append(weight)
        effects.append(effect)

    if not weights:
        return {
            "status": "unavailable",
            "n_models": 0,
        }

    pooled = sum(w * e for w, e in zip(weights, effects)) / sum(weights)
    pooled_se = math.sqrt(1.0 / sum(weights))
    return {
        "status": "ok",
        "n_models": len(weights),
        "mean_shift": pooled,
        "ci95_low": pooled - 1.96 * pooled_se,
        "ci95_high": pooled + 1.96 * pooled_se,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, ndigits: int = 4, signed: bool = False) -> str:
    v = _to_float(value)
    if not _is_finite(v):
        return "N/A"
    return f"{v:+.{ndigits}f}" if signed else f"{v:.{ndigits}f}"


def _write_latex_table(path: Path, rows: list[dict[str, Any]], pooled: dict[str, Any]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & $n$ & Mean $\\Delta$ & 95\\% CI & $\\Delta$@1.5 \\\\",
        "\\midrule",
    ]
    for row in rows:
        ci_text = f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]"
        lines.append(
            f"{row['model_label']} & "
            f"{int(float(row['n_paired_authority_unsafe']))} & "
            f"{_fmt(row['mean_shift_intervention_minus_baseline'], signed=True)} & "
            f"{ci_text} & "
            f"{_fmt(row['margin_1p5_delta'], signed=True)} \\\\"
        )
    if pooled.get("status") == "ok":
        pooled_ci = f"[{_fmt(pooled['ci95_low'])}, {_fmt(pooled['ci95_high'])}]"
        lines.extend(
            [
                "\\midrule",
                f"Pooled (fixed effect) & {int(pooled['n_models'])} & "
                f"{_fmt(pooled['mean_shift'], signed=True)} & "
                f"{pooled_ci} & N/A \\\\",
            ]
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{External holdout frozen-direction replay summary. "
                "Negative values indicate intervention reduces refusal-minus-compliance score on paired authority-unsafe prompts.}"
            ),
            "\\label{tab:external_holdout_transfer}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_forest_plot(path_base: Path, rows: list[dict[str, Any]], pooled: dict[str, Any]) -> dict[str, str] | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plot_rows = list(rows)
    if pooled.get("status") == "ok":
        plot_rows.append(
            {
                "model_label": "Pooled (fixed effect)",
                "mean_shift_intervention_minus_baseline": pooled["mean_shift"],
                "ci95_low": pooled["ci95_low"],
                "ci95_high": pooled["ci95_high"],
            }
        )

    labels = [str(row["model_label"]) for row in plot_rows]
    means = [_to_float(row["mean_shift_intervention_minus_baseline"]) for row in plot_rows]
    lows = [_to_float(row["ci95_low"]) for row in plot_rows]
    highs = [_to_float(row["ci95_high"]) for row in plot_rows]
    xerr = [[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]]
    ypos = list(range(len(plot_rows)))[::-1]

    fig_h = max(2.5, 0.75 * len(plot_rows) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.errorbar(means, ypos, xerr=xerr, fmt="o", color="#1f4e79", ecolor="#1f4e79", capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean shift (intervention - baseline)")
    ax.set_title("External Holdout Frozen-Direction Replay")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    png_path = path_base.with_suffix(".png")
    pdf_path = path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate one or more external holdout replay runs.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Replay output directory containing replay_manifest.json and posthoc/posthoc_analysis.json",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional labels matching --run-dir order.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.run_dir):
        raise ValueError("--label count must match --run-dir count when provided")

    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    rows: list[dict[str, Any]] = []

    for idx, run_dir_raw in enumerate(args.run_dir):
        run_dir = Path(run_dir_raw).expanduser().resolve()
        manifest = _read_json(run_dir / "replay_manifest.json")
        report = _read_json(run_dir / "posthoc" / "posthoc_analysis.json")
        tf = report.get("threshold_free_authority_unsafe", {})
        delta = tf.get("delta_distribution", {})
        margin_1p0 = _margin_row(report, 1.0)
        margin_1p5 = _margin_row(report, 1.5)

        model_name = str(manifest.get("model", "unknown"))
        model_label = args.label[idx] if idx < len(args.label) else _model_label(model_name)
        n = int(float(tf.get("n_paired_authority_unsafe", 0)))
        mean_shift = _to_float(tf.get("mean_shift_intervention_minus_baseline"))
        delta_std = _to_float(delta.get("std"))
        ci_low, ci_high = _ci95(mean_shift=mean_shift, std=delta_std, n=n)

        rows.append(
            {
                "model_label": model_label,
                "model_name": model_name,
                "run_dir": str(run_dir),
                "direction_path": str(manifest.get("direction_path", "")),
                "n_paired_authority_unsafe": n,
                "baseline_mean_logit_diff": _to_float(tf.get("baseline_mean_logit_diff")),
                "intervention_mean_logit_diff": _to_float(tf.get("intervention_mean_logit_diff")),
                "mean_shift_intervention_minus_baseline": mean_shift,
                "median_shift_intervention_minus_baseline": _to_float(
                    tf.get("median_shift_intervention_minus_baseline")
                ),
                "delta_std": delta_std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sign_test_p_value": _to_float(tf.get("paired_sign_test", {}).get("p_value")),
                "ks_d_stat": _to_float(tf.get("ks_d_stat")),
                "wasserstein_1": _to_float(tf.get("wasserstein_1")),
                "margin_1p0_baseline": _to_float(margin_1p0.get("baseline_authority_unsafe_refusal_rate")),
                "margin_1p0_intervention": _to_float(margin_1p0.get("intervention_unsafe_refusal_rate")),
                "margin_1p0_delta": _to_float(margin_1p0.get("delta_refusal_rate_intervention_minus_baseline")),
                "margin_1p5_baseline": _to_float(margin_1p5.get("baseline_authority_unsafe_refusal_rate")),
                "margin_1p5_intervention": _to_float(margin_1p5.get("intervention_unsafe_refusal_rate")),
                "margin_1p5_delta": _to_float(margin_1p5.get("delta_refusal_rate_intervention_minus_baseline")),
            }
        )

    rows.sort(key=lambda row: str(row["model_label"]))
    pooled = _fixed_effect_pool(rows)

    _write_csv(out_dir / "external_holdout_summary.csv", rows)
    _write_latex_table(out_dir / "table_external_holdout_transfer.tex", rows, pooled)
    forest_paths = _write_forest_plot(out_dir / "external_holdout_forest", rows, pooled)

    summary = {
        "rows": rows,
        "fixed_effect_pool": pooled,
        "artifacts": {
            "summary_csv": str(out_dir / "external_holdout_summary.csv"),
            "table_tex": str(out_dir / "table_external_holdout_transfer.tex"),
            "forest_plot_png": forest_paths["png"] if forest_paths else None,
            "forest_plot_pdf": forest_paths["pdf"] if forest_paths else None,
        },
    }
    write_json(out_dir / "external_holdout_summary.json", summary)

    print(f"Wrote: {out_dir / 'external_holdout_summary.csv'}")
    print(f"Wrote: {out_dir / 'external_holdout_summary.json'}")
    print(f"Wrote: {out_dir / 'table_external_holdout_transfer.tex'}")
    if forest_paths:
        print(f"Wrote: {forest_paths['png']}")
        print(f"Wrote: {forest_paths['pdf']}")


if __name__ == "__main__":
    main()
