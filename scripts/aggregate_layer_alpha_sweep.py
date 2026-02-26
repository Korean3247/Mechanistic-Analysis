#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


@dataclass
class RunRow:
    run_name: str
    run_dir: Path
    layer_for_sae: int
    alpha_intervention: float
    seed: int
    mean_shift: float
    median_shift: float
    ks_d_stat: float
    wasserstein_1: float
    cliffs_delta: float
    sign_test_p_value: float
    delta_refusal_at_1_0: float
    delta_refusal_at_1_5: float
    delta_refusal_at_1_0_ci_low: float
    delta_refusal_at_1_0_ci_high: float
    delta_refusal_at_1_5_ci_low: float
    delta_refusal_at_1_5_ci_high: float
    safe_refusal_delta: float
    borderline_refusal_delta: float
    unsafe_refusal_delta: float
    spillover_refusal_delta_mean_safe_borderline: float


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _f(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def _load_margin_row(rows: list[dict[str, Any]], margin: float) -> dict[str, Any] | None:
    for row in rows:
        if math.isclose(_f(row.get("margin"), default=-1.0), margin, abs_tol=1e-9):
            return row
    return None


def _summary(values: list[float]) -> dict[str, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(mean(finite)),
        "median": float(median(finite)),
        "std": float(pstdev(finite)) if len(finite) > 1 else 0.0,
        "n": int(len(finite)),
    }


def _collect_runs(results_root: Path, prefix: str) -> list[RunRow]:
    run_dirs = sorted(d for d in results_root.glob(f"{prefix}_l*_a*_s*") if d.is_dir())
    if not run_dirs:
        run_dirs = sorted(d for d in results_root.glob(f"{prefix}*") if d.is_dir())

    rows: list[RunRow] = []
    for run_dir in run_dirs:
        mf = run_dir / "logs" / "run_manifest.json"
        metrics = run_dir / "metrics.json"
        posthoc = run_dir / "posthoc" / "posthoc_analysis.json"
        if not (mf.exists() and metrics.exists() and posthoc.exists()):
            continue

        manifest = _read_json(mf)
        metrics_obj = _read_json(metrics)
        posthoc_obj = _read_json(posthoc)

        cfg = manifest.get("config", {})
        layer = int(cfg.get("layer_for_sae", 0))
        alpha = _f(cfg.get("alpha_intervention", 1.0), default=1.0)
        seed = int(cfg.get("seed", 0))

        tf = posthoc_obj.get("threshold_free_authority_unsafe", {})
        margin_rows = posthoc_obj.get("margin_sweep", [])
        m10 = _load_margin_row(margin_rows, 1.0) or {}
        m15 = _load_margin_row(margin_rows, 1.5) or {}

        base_tiers = metrics_obj.get("tier_summary", {})
        int_tiers = metrics_obj.get("intervention_tier_summary", {})

        def tier_delta(name: str) -> float:
            b = _f((base_tiers.get(name) or {}).get("refusal_rate"), default=0.0)
            i = _f((int_tiers.get(name) or {}).get("refusal_rate"), default=0.0)
            return float(i - b)

        safe_delta = tier_delta("safe")
        borderline_delta = tier_delta("borderline")
        unsafe_delta = tier_delta("unsafe")

        rows.append(
            RunRow(
                run_name=run_dir.name,
                run_dir=run_dir,
                layer_for_sae=layer,
                alpha_intervention=alpha,
                seed=seed,
                mean_shift=_f(tf.get("mean_shift_intervention_minus_baseline"), default=0.0),
                median_shift=_f(tf.get("median_shift_intervention_minus_baseline"), default=0.0),
                ks_d_stat=_f(tf.get("ks_d_stat"), default=0.0),
                wasserstein_1=_f(tf.get("wasserstein_1"), default=0.0),
                cliffs_delta=_f(tf.get("cliffs_delta_intervention_vs_baseline"), default=0.0),
                sign_test_p_value=_f((tf.get("paired_sign_test") or {}).get("p_value"), default=1.0),
                delta_refusal_at_1_0=_f(m10.get("delta_refusal_rate_intervention_minus_baseline"), default=0.0),
                delta_refusal_at_1_5=_f(m15.get("delta_refusal_rate_intervention_minus_baseline"), default=0.0),
                delta_refusal_at_1_0_ci_low=_f(m10.get("delta_refusal_ci95_low"), default=0.0),
                delta_refusal_at_1_0_ci_high=_f(m10.get("delta_refusal_ci95_high"), default=0.0),
                delta_refusal_at_1_5_ci_low=_f(m15.get("delta_refusal_ci95_low"), default=0.0),
                delta_refusal_at_1_5_ci_high=_f(m15.get("delta_refusal_ci95_high"), default=0.0),
                safe_refusal_delta=safe_delta,
                borderline_refusal_delta=borderline_delta,
                unsafe_refusal_delta=unsafe_delta,
                spillover_refusal_delta_mean_safe_borderline=float((safe_delta + borderline_delta) / 2.0),
            )
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _group_summary(rows: list[RunRow], key_name: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[RunRow]] = {}
    for r in rows:
        key = getattr(r, key_name)
        groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for key in sorted(groups.keys()):
        grp = groups[key]
        mean_shift_vals = [r.mean_shift for r in grp]
        med_shift_vals = [r.median_shift for r in grp]
        ks_vals = [r.ks_d_stat for r in grp]
        wass_vals = [r.wasserstein_1 for r in grp]
        cliff_vals = [r.cliffs_delta for r in grp]
        m10_vals = [r.delta_refusal_at_1_0 for r in grp]
        m15_vals = [r.delta_refusal_at_1_5 for r in grp]
        unsafe_vals = [r.unsafe_refusal_delta for r in grp]
        spill_vals = [r.spillover_refusal_delta_mean_safe_borderline for r in grp]

        ms = _summary(mean_shift_vals)
        mds = _summary(med_shift_vals)
        ks = _summary(ks_vals)
        ws = _summary(wass_vals)
        cd = _summary(cliff_vals)
        m10 = _summary(m10_vals)
        m15 = _summary(m15_vals)
        unsafe = _summary(unsafe_vals)
        spill = _summary(spill_vals)

        out.append(
            {
                key_name: key,
                "n_runs": ms["n"],
                "mean_shift_mean": ms["mean"],
                "mean_shift_std": ms["std"],
                "median_shift_mean": mds["mean"],
                "ks_mean": ks["mean"],
                "wasserstein_mean": ws["mean"],
                "cliffs_delta_mean": cd["mean"],
                "delta_refusal_m1_0_mean": m10["mean"],
                "delta_refusal_m1_0_std": m10["std"],
                "delta_refusal_m1_5_mean": m15["mean"],
                "delta_refusal_m1_5_std": m15["std"],
                "unsafe_refusal_delta_mean": unsafe["mean"],
                "spillover_safe_borderline_delta_mean": spill["mean"],
            }
        )
    return out


def _latex_table(path: Path, caption: str, label: str, rows: list[dict[str, Any]], key_col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        f"{key_col} & Mean Shift & KS & W1 & Cliff's $\\delta$ & $\\Delta$Ref@1.0 & $\\Delta$Ref@1.5 \\\\",
        "\\midrule",
    ]
    for r in rows:
        key = r[key_col]
        lines.append(
            f"{key} & {r['mean_shift_mean']:.4f} & {r['ks_mean']:.4f} & {r['wasserstein_mean']:.4f} & "
            f"{r['cliffs_delta_mean']:.4f} & {r['delta_refusal_m1_0_mean']:.4f} & {r['delta_refusal_m1_5_mean']:.4f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_line(path: Path, xs: list[float], ys: list[float], xlabel: str, ylabel: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not xs or not ys:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _is_monotonic_nonincreasing(xs: list[float], ys: list[float]) -> bool:
    if len(xs) <= 1:
        return True
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    for (_, a), (_, b) in zip(pairs[:-1], pairs[1:]):
        if b > a + 1e-9:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate layer/alpha sweep runs for paper-ready robustness tables.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--primary-layer", type=int, default=10)
    parser.add_argument("--primary-alpha", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else results_root / f"{args.experiment_prefix}_layer_alpha_aggregate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(results_root=results_root, prefix=args.experiment_prefix)
    if not runs:
        raise FileNotFoundError(f"No sweep runs found for prefix '{args.experiment_prefix}' in {results_root}")

    run_rows: list[dict[str, Any]] = []
    for r in runs:
        run_rows.append(
            {
                "run_name": r.run_name,
                "layer_for_sae": r.layer_for_sae,
                "alpha_intervention": r.alpha_intervention,
                "seed": r.seed,
                "mean_shift": r.mean_shift,
                "median_shift": r.median_shift,
                "ks_d_stat": r.ks_d_stat,
                "wasserstein_1": r.wasserstein_1,
                "cliffs_delta": r.cliffs_delta,
                "sign_test_p_value": r.sign_test_p_value,
                "delta_refusal_at_1_0": r.delta_refusal_at_1_0,
                "delta_refusal_at_1_0_ci_low": r.delta_refusal_at_1_0_ci_low,
                "delta_refusal_at_1_0_ci_high": r.delta_refusal_at_1_0_ci_high,
                "delta_refusal_at_1_5": r.delta_refusal_at_1_5,
                "delta_refusal_at_1_5_ci_low": r.delta_refusal_at_1_5_ci_low,
                "delta_refusal_at_1_5_ci_high": r.delta_refusal_at_1_5_ci_high,
                "safe_refusal_delta": r.safe_refusal_delta,
                "borderline_refusal_delta": r.borderline_refusal_delta,
                "unsafe_refusal_delta": r.unsafe_refusal_delta,
                "spillover_safe_borderline_delta_mean": r.spillover_refusal_delta_mean_safe_borderline,
                "run_dir": str(r.run_dir),
            }
        )

    _write_csv(
        out_dir / "sweep_runs.csv",
        run_rows,
        fieldnames=list(run_rows[0].keys()),
    )

    layer_subset = [r for r in runs if math.isclose(r.alpha_intervention, args.primary_alpha, abs_tol=1e-9)]
    alpha_subset = [r for r in runs if r.layer_for_sae == args.primary_layer]

    layer_summary = _group_summary(layer_subset, "layer_for_sae")
    alpha_summary = _group_summary(alpha_subset, "alpha_intervention")

    if layer_summary:
        _write_csv(
            out_dir / "layer_summary.csv",
            layer_summary,
            fieldnames=list(layer_summary[0].keys()),
        )
    if alpha_summary:
        _write_csv(
            out_dir / "alpha_summary.csv",
            alpha_summary,
            fieldnames=list(alpha_summary[0].keys()),
        )

    _latex_table(
        path=out_dir / "table_layer_robustness.tex",
        caption=(
            f"Layer robustness at fixed alpha={args.primary_alpha}. "
            "Values are averaged over runs/seeds."
        ),
        label="tab:layer_robustness",
        rows=layer_summary,
        key_col="layer_for_sae",
    )
    _latex_table(
        path=out_dir / "table_alpha_robustness.tex",
        caption=(
            f"Intervention-strength robustness at fixed layer={args.primary_layer}. "
            "Values are averaged over runs/seeds."
        ),
        label="tab:alpha_robustness",
        rows=alpha_summary,
        key_col="alpha_intervention",
    )

    alpha_x = [float(r["alpha_intervention"]) for r in alpha_summary]
    alpha_y = [float(r["mean_shift_mean"]) for r in alpha_summary]
    layer_x = [float(r["layer_for_sae"]) for r in layer_summary]
    layer_y = [float(r["mean_shift_mean"]) for r in layer_summary]

    _plot_line(
        out_dir / "alpha_vs_mean_shift.pdf",
        alpha_x,
        alpha_y,
        xlabel="alpha",
        ylabel="mean shift (intervention - baseline)",
        title=f"Alpha sweep at layer={args.primary_layer}",
    )
    _plot_line(
        out_dir / "layer_vs_mean_shift.pdf",
        layer_x,
        layer_y,
        xlabel="layer",
        ylabel="mean shift (intervention - baseline)",
        title=f"Layer sweep at alpha={args.primary_alpha}",
    )

    summary = {
        "experiment_prefix": args.experiment_prefix,
        "results_root": str(results_root),
        "n_runs": len(runs),
        "unique_layers": sorted({r.layer_for_sae for r in runs}),
        "unique_alphas": sorted({r.alpha_intervention for r in runs}),
        "primary_layer": int(args.primary_layer),
        "primary_alpha": float(args.primary_alpha),
        "layer_subset_run_count": len(layer_subset),
        "alpha_subset_run_count": len(alpha_subset),
        "alpha_monotonic_nonincreasing_mean_shift": _is_monotonic_nonincreasing(alpha_x, alpha_y),
        "spillover_summary": {
            "mean_safe_delta": _summary([r.safe_refusal_delta for r in runs])["mean"],
            "mean_borderline_delta": _summary([r.borderline_refusal_delta for r in runs])["mean"],
            "mean_unsafe_delta": _summary([r.unsafe_refusal_delta for r in runs])["mean"],
            "mean_safe_borderline_spillover": _summary(
                [r.spillover_refusal_delta_mean_safe_borderline for r in runs]
            )["mean"],
        },
        "artifacts": {
            "sweep_runs_csv": str((out_dir / "sweep_runs.csv").resolve()),
            "layer_summary_csv": str((out_dir / "layer_summary.csv").resolve()),
            "alpha_summary_csv": str((out_dir / "alpha_summary.csv").resolve()),
            "table_layer_robustness_tex": str((out_dir / "table_layer_robustness.tex").resolve()),
            "table_alpha_robustness_tex": str((out_dir / "table_alpha_robustness.tex").resolve()),
            "alpha_vs_mean_shift_pdf": str((out_dir / "alpha_vs_mean_shift.pdf").resolve()),
            "layer_vs_mean_shift_pdf": str((out_dir / "layer_vs_mean_shift.pdf").resolve()),
        },
    }

    (out_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote aggregate artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
