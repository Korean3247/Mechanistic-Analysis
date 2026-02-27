#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


@dataclass
class ConditionData:
    key: str
    label: str
    run_dir: Path
    metrics: dict[str, Any]
    threshold: dict[str, Any]
    margin_by_value: dict[float, dict[str, Any]]
    classifier_unsafe: dict[str, Any]
    direction_meta: dict[str, Any] | None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _label_from_key(key: str) -> str:
    mapping = {
        "main": "Main",
        "random": "Placebo (Random)",
        "low_importance": "Placebo (Low-Importance)",
        "orthogonal": "Placebo (Orthogonal)",
        "shuffled_latent": "Placebo (Shuffled-Latent)",
    }
    return mapping.get(key, key)


def _load_direction_meta(run_dir: Path, key: str) -> dict[str, Any] | None:
    if key == "main":
        return None
    p = run_dir / "placebo_direction_vector.pt"
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu")
    return payload.get("metadata", {})


def _load_condition(run_dir: Path, key: str) -> ConditionData:
    metrics = _load_json(run_dir / "metrics.json")
    threshold = metrics.get("threshold_free_authority_unsafe", {})
    margins = metrics.get("margin_sweep", [])
    margin_by_value: dict[float, dict[str, Any]] = {}
    for row in margins:
        margin_by_value[_safe_float(row.get("margin"))] = row
    cls = metrics.get("classifier_behavior_endpoint", {}).get("variant_unsafe_summary", {})
    classifier_unsafe = {
        "baseline": cls.get("baseline", {}),
        "authority": cls.get("authority", {}),
        "intervention": cls.get("intervention", {}),
    }
    return ConditionData(
        key=key,
        label=_label_from_key(key),
        run_dir=run_dir,
        metrics=metrics,
        threshold=threshold,
        margin_by_value=margin_by_value,
        classifier_unsafe=classifier_unsafe,
        direction_meta=_load_direction_meta(run_dir, key),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_threshold_rows(conds: list[ConditionData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in conds:
        t = c.threshold
        rows.append(
            {
                "condition_key": c.key,
                "condition": c.label,
                "n_paired_authority_unsafe": int(t.get("n_paired_authority_unsafe", 0)),
                "mean_shift": _safe_float(t.get("mean_shift_intervention_minus_baseline")),
                "median_shift": _safe_float(t.get("median_shift_intervention_minus_baseline")),
                "sign_test_p_value": _safe_float(t.get("paired_sign_test", {}).get("p_value")),
                "ks_d_stat": _safe_float(t.get("ks_d_stat")),
                "wasserstein_1": _safe_float(t.get("wasserstein_1")),
                "cliffs_delta": _safe_float(t.get("cliffs_delta_intervention_vs_baseline")),
                "p_logit_diff_gt_1_0_baseline": _safe_float(t.get("p_logit_diff_gt_1.0", {}).get("baseline")),
                "p_logit_diff_gt_1_0_intervention": _safe_float(
                    t.get("p_logit_diff_gt_1.0", {}).get("intervention")
                ),
                "p_logit_diff_gt_1_5_baseline": _safe_float(t.get("p_logit_diff_gt_1.5", {}).get("baseline")),
                "p_logit_diff_gt_1_5_intervention": _safe_float(
                    t.get("p_logit_diff_gt_1.5", {}).get("intervention")
                ),
            }
        )
    return rows


def _make_margin_rows(conds: list[ConditionData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in conds:
        for m in (0.5, 1.0, 1.5, 2.0):
            r = c.margin_by_value.get(float(m), {})
            rows.append(
                {
                    "condition_key": c.key,
                    "condition": c.label,
                    "margin": float(m),
                    "baseline_refusal_rate": _safe_float(r.get("baseline_authority_unsafe_refusal_rate")),
                    "intervention_refusal_rate": _safe_float(r.get("intervention_unsafe_refusal_rate")),
                    "delta_refusal_rate": _safe_float(r.get("delta_refusal_rate_intervention_minus_baseline")),
                    "delta_refusal_ci95_low": _safe_float(r.get("delta_refusal_ci95_low")),
                    "delta_refusal_ci95_high": _safe_float(r.get("delta_refusal_ci95_high")),
                }
            )
    return rows


def _make_classifier_rows(conds: list[ConditionData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in conds:
        b = c.classifier_unsafe.get("baseline", {})
        a = c.classifier_unsafe.get("authority", {})
        i = c.classifier_unsafe.get("intervention", {})
        rows.append(
            {
                "condition_key": c.key,
                "condition": c.label,
                "baseline_unsafe_refusal_rate": _safe_float(b.get("refusal_rate")),
                "baseline_unsafe_compliance_rate": _safe_float(b.get("compliance_rate")),
                "authority_unsafe_refusal_rate": _safe_float(a.get("refusal_rate")),
                "authority_unsafe_compliance_rate": _safe_float(a.get("compliance_rate")),
                "intervention_unsafe_refusal_rate": _safe_float(i.get("refusal_rate")),
                "intervention_unsafe_compliance_rate": _safe_float(i.get("compliance_rate")),
                "intervention_minus_authority_refusal_delta": _safe_float(i.get("refusal_rate"))
                - _safe_float(a.get("refusal_rate")),
                "intervention_minus_baseline_refusal_delta": _safe_float(i.get("refusal_rate"))
                - _safe_float(b.get("refusal_rate")),
                "intervention_match_rate": _safe_float(i.get("match_rate")),
            }
        )
    return rows


def _make_direction_rows(conds: list[ConditionData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in conds:
        if c.direction_meta is None:
            continue
        md = c.direction_meta
        rows.append(
            {
                "condition_key": c.key,
                "condition": c.label,
                "placebo_mode": md.get("placebo_mode", c.key),
                "actual_norm": _safe_float(md.get("actual_norm")),
                "target_norm": _safe_float(md.get("target_norm")),
                "dot_with_base": _safe_float(md.get("dot_with_base")),
                "direction_is_finite": bool(md.get("direction_is_finite", True)),
                "direction_is_degenerate": bool(md.get("direction_is_degenerate", False)),
                "residual_l2_before_normalize": _safe_float(md.get("residual_l2_before_normalize")),
                "low_feature_count_selected": md.get("low_feature_count_selected"),
                "latent_cosine_with_true_direction": _safe_float(md.get("latent_cosine_with_true_direction")),
            }
        )
    return rows


def _write_latex_threshold_table(path: Path, threshold_rows: list[dict[str, Any]], margin_rows: list[dict[str, Any]]) -> None:
    by_cond_margin_10 = {
        r["condition_key"]: r
        for r in margin_rows
        if abs(_safe_float(r.get("margin")) - 1.0) < 1e-9
    }
    by_cond_margin_15 = {
        r["condition_key"]: r
        for r in margin_rows
        if abs(_safe_float(r.get("margin")) - 1.5) < 1e-9
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Condition & Mean $\\Delta$ & Median $\\Delta$ & Sign-$p$ & KS $D$ & $W_1$ & $\\Delta R@1.0$ & $\\Delta R@1.5$ \\\\",
        "\\midrule",
    ]
    for row in threshold_rows:
        k = row["condition_key"]
        m10 = by_cond_margin_10.get(k, {})
        m15 = by_cond_margin_15.get(k, {})
        lines.append(
            f"{row['condition']} & "
            f"{_safe_float(row['mean_shift']):+.4f} & "
            f"{_safe_float(row['median_shift']):+.4f} & "
            f"{_safe_float(row['sign_test_p_value']):.2e} & "
            f"{_safe_float(row['ks_d_stat']):.4f} & "
            f"{_safe_float(row['wasserstein_1']):.4f} & "
            f"{_safe_float(m10.get('delta_refusal_rate')):+.4f} & "
            f"{_safe_float(m15.get('delta_refusal_rate')):+.4f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Strong-placebo comparison on authority-unsafe paired samples. Threshold-free shifts are primary; margin deltas are sensitivity metrics.}",
        "\\label{tab:strong_placebo_5way}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_threshold_bar(path_png: Path, path_pdf: Path, rows: list[dict[str, Any]]) -> None:
    labels = [r["condition"] for r in rows]
    values = [_safe_float(r["mean_shift"]) for r in rows]
    plt.figure(figsize=(9.5, 4.5))
    bars = plt.bar(labels, values, color=["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"])
    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.ylabel("mean shift (intervention - baseline)")
    plt.title("Authority-Unsafe Mean Shift: Main vs Strong Placebos")
    plt.xticks(rotation=15, ha="right")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_pdf)
    plt.close()


def _plot_margin_overlay(path_png: Path, path_pdf: Path, margin_rows: list[dict[str, Any]]) -> None:
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for row in margin_rows:
        by_cond.setdefault(str(row["condition"]), []).append(row)
    color_map = {
        "Main": "#1f77b4",
        "Placebo (Random)": "#2ca02c",
        "Placebo (Low-Importance)": "#d62728",
        "Placebo (Orthogonal)": "#9467bd",
        "Placebo (Shuffled-Latent)": "#ff7f0e",
    }
    plt.figure(figsize=(9.5, 4.5))
    for cond, rows in by_cond.items():
        rows_sorted = sorted(rows, key=lambda x: _safe_float(x.get("margin")))
        xs = [_safe_float(r["margin"]) for r in rows_sorted]
        ys = [_safe_float(r["delta_refusal_rate"]) for r in rows_sorted]
        plt.plot(xs, ys, marker="o", label=cond, color=color_map.get(cond))
    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("margin")
    plt.ylabel("delta refusal rate")
    plt.title("Margin Sweep Overlay (Main vs Strong Placebos)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_pdf)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize strong placebo experiment outputs")
    parser.add_argument("--main-run", required=True, help="Main run directory")
    parser.add_argument("--placebo-root", required=True, help="Placebo root directory")
    parser.add_argument("--output-dir", required=True, help="Output summary directory")
    args = parser.parse_args()

    main_run = Path(args.main_run).expanduser().resolve()
    placebo_root = Path(args.placebo_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    comp_dir = out_dir / "comparison"
    fig_dir = out_dir / "figures"
    comp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    conds = [
        _load_condition(main_run, "main"),
        _load_condition(placebo_root / "random", "random"),
        _load_condition(placebo_root / "low_importance", "low_importance"),
        _load_condition(placebo_root / "orthogonal", "orthogonal"),
        _load_condition(placebo_root / "shuffled_latent", "shuffled_latent"),
    ]

    threshold_rows = _make_threshold_rows(conds)
    margin_rows = _make_margin_rows(conds)
    classifier_rows = _make_classifier_rows(conds)
    direction_rows = _make_direction_rows(conds)

    _write_csv(
        comp_dir / "strong_placebo_threshold_free.csv",
        threshold_rows,
        [
            "condition_key",
            "condition",
            "n_paired_authority_unsafe",
            "mean_shift",
            "median_shift",
            "sign_test_p_value",
            "ks_d_stat",
            "wasserstein_1",
            "cliffs_delta",
            "p_logit_diff_gt_1_0_baseline",
            "p_logit_diff_gt_1_0_intervention",
            "p_logit_diff_gt_1_5_baseline",
            "p_logit_diff_gt_1_5_intervention",
        ],
    )
    _write_csv(
        comp_dir / "strong_placebo_margin_sweep.csv",
        margin_rows,
        [
            "condition_key",
            "condition",
            "margin",
            "baseline_refusal_rate",
            "intervention_refusal_rate",
            "delta_refusal_rate",
            "delta_refusal_ci95_low",
            "delta_refusal_ci95_high",
        ],
    )
    _write_csv(
        comp_dir / "strong_placebo_classifier_unsafe.csv",
        classifier_rows,
        [
            "condition_key",
            "condition",
            "baseline_unsafe_refusal_rate",
            "baseline_unsafe_compliance_rate",
            "authority_unsafe_refusal_rate",
            "authority_unsafe_compliance_rate",
            "intervention_unsafe_refusal_rate",
            "intervention_unsafe_compliance_rate",
            "intervention_minus_authority_refusal_delta",
            "intervention_minus_baseline_refusal_delta",
            "intervention_match_rate",
        ],
    )
    _write_csv(
        comp_dir / "strong_placebo_direction_metadata.csv",
        direction_rows,
        [
            "condition_key",
            "condition",
            "placebo_mode",
            "actual_norm",
            "target_norm",
            "dot_with_base",
            "direction_is_finite",
            "direction_is_degenerate",
            "residual_l2_before_normalize",
            "low_feature_count_selected",
            "latent_cosine_with_true_direction",
        ],
    )

    _write_latex_threshold_table(comp_dir / "table_strong_placebo_5way.tex", threshold_rows, margin_rows)
    _plot_threshold_bar(
        fig_dir / "strong_placebo_mean_shift_bar.png",
        fig_dir / "strong_placebo_mean_shift_bar.pdf",
        threshold_rows,
    )
    _plot_margin_overlay(
        fig_dir / "strong_placebo_margin_overlay.png",
        fig_dir / "strong_placebo_margin_overlay.pdf",
        margin_rows,
    )

    summary = {
        "main_run": str(main_run),
        "placebo_root": str(placebo_root),
        "conditions": [c.key for c in conds],
        "artifacts": {
            "threshold_csv": str(comp_dir / "strong_placebo_threshold_free.csv"),
            "margin_csv": str(comp_dir / "strong_placebo_margin_sweep.csv"),
            "classifier_csv": str(comp_dir / "strong_placebo_classifier_unsafe.csv"),
            "direction_csv": str(comp_dir / "strong_placebo_direction_metadata.csv"),
            "latex_table": str(comp_dir / "table_strong_placebo_5way.tex"),
            "figure_mean_shift_png": str(fig_dir / "strong_placebo_mean_shift_bar.png"),
            "figure_margin_overlay_png": str(fig_dir / "strong_placebo_margin_overlay.png"),
        },
    }
    (out_dir / "summary_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
