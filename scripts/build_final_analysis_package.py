#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_RELATIVE_FILES = [
    "metrics.json",
    "posthoc/posthoc_analysis.json",
    "posthoc/margin_sweep.csv",
    "posthoc/authority_unsafe_ecdf.csv",
    "logs/run_manifest.json",
    "logs/baseline_samples.json",
    "logs/intervention_samples.json",
]

OPTIONAL_RELATIVE_FILES = [
    "logs/prompts.jsonl",
    "logs/baseline_summary.json",
    "logs/intervention_summary.json",
    "logs/behavioral_ground_truth.jsonl",
    "logs/behavioral_ground_truth_summary.json",
    "plots/feature_activation_heatmap.png",
    "plots/layer_wise_suppression.png",
    "authority_direction_vector.pt",
    "placebo_direction_vector.pt",
]

MARGINS_FOR_TABLE = [1.0, 1.5]


@dataclass
class ConditionRun:
    key: str
    label: str
    run_dir: Path
    metrics: dict[str, Any]
    posthoc: dict[str, Any]
    manifest: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _margin_lookup(rows: list[dict[str, Any]], margin: float) -> dict[str, Any] | None:
    for row in rows:
        try:
            if math.isclose(float(row.get("margin", -999)), margin, abs_tol=1e-9):
                return row
        except (TypeError, ValueError):
            continue
    return None


def _detect_project_roots(run_dirs: list[Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for run_dir in run_dirs:
        for anc in [run_dir] + list(run_dir.parents):
            if not anc.exists():
                continue
            if (anc / "data").exists() or (anc / "configs").exists() or (anc / "configs_data").exists():
                resolved = anc.resolve()
                if resolved not in seen:
                    roots.append(resolved)
                    seen.add(resolved)
    return roots


def _resolve_path(path_str: str, candidate_roots: list[Path]) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if not p.is_absolute():
        for root in candidate_roots:
            candidate = root / p
            if candidate.exists():
                return candidate.resolve()
    if p.exists():
        return p.resolve()
    basename = p.name
    for root in candidate_roots:
        for sub in ("data", "configs_data/data", "configs", "configs_data/configs"):
            candidate = root / sub / basename
            if candidate.exists():
                return candidate.resolve()
    return None


def _read_first_prompt_meta(prompts_path: Path) -> dict[str, Any]:
    if not prompts_path.exists():
        return {}
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_run_dir(run_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_RELATIVE_FILES if not (run_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts in {run_dir}: {missing}")


def _load_condition_run(key: str, label: str, run_dir: Path) -> ConditionRun:
    _validate_run_dir(run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    posthoc = _read_json(run_dir / "posthoc" / "posthoc_analysis.json")
    manifest = _read_json(run_dir / "logs" / "run_manifest.json")
    return ConditionRun(
        key=key,
        label=label,
        run_dir=run_dir,
        metrics=metrics,
        posthoc=posthoc,
        manifest=manifest,
    )


def _build_summary_rows(conditions: list[ConditionRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        tf = cond.posthoc.get("threshold_free_authority_unsafe", {})
        sweep = cond.posthoc.get("margin_sweep", [])
        m10 = _margin_lookup(sweep, 1.0) or {}
        m15 = _margin_lookup(sweep, 1.5) or {}
        rows.append(
            {
                "condition_key": cond.key,
                "condition_label": cond.label,
                "run_dir": str(cond.run_dir),
                "git_commit": cond.manifest.get("git_commit", ""),
                "mean_shift_intervention_minus_baseline": _safe_float(
                    tf.get("mean_shift_intervention_minus_baseline")
                ),
                "median_shift_intervention_minus_baseline": _safe_float(
                    tf.get("median_shift_intervention_minus_baseline")
                ),
                "paired_sign_test_p_value": _safe_float(
                    (tf.get("paired_sign_test") or {}).get("p_value"),
                    default=1.0,
                ),
                "ks_d_stat": _safe_float(tf.get("ks_d_stat")),
                "wasserstein_1": _safe_float(tf.get("wasserstein_1")),
                "delta_refusal_rate_margin_1_0": _safe_float(
                    m10.get("delta_refusal_rate_intervention_minus_baseline")
                ),
                "delta_refusal_rate_margin_1_5": _safe_float(
                    m15.get("delta_refusal_rate_intervention_minus_baseline")
                ),
                "n_paired_authority_unsafe": int(_safe_float(tf.get("n_paired_authority_unsafe"), default=0)),
                "intervention_non_finite_logit_diff_rate": _safe_float(
                    cond.metrics.get("intervention_non_finite_logit_diff_rate")
                ),
                "intervention_rows_with_non_finite_logits_rate": _safe_float(
                    cond.metrics.get("intervention_rows_with_non_finite_logits_rate")
                ),
            }
        )
    return rows


def _float_fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_latex_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\hline\n"
        "Condition & Mean Shift & Median Shift & Sign p & KS D & W1 & $\\Delta$Ref@1.0 & $\\Delta$Ref@1.5 \\\\\n"
        "\\hline\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"{r['condition_label']} & "
            f"{_float_fmt(r['mean_shift_intervention_minus_baseline'])} & "
            f"{_float_fmt(r['median_shift_intervention_minus_baseline'])} & "
            f"{_float_fmt(r['paired_sign_test_p_value'])} & "
            f"{_float_fmt(r['ks_d_stat'])} & "
            f"{_float_fmt(r['wasserstein_1'])} & "
            f"{_float_fmt(r['delta_refusal_rate_margin_1_0'])} & "
            f"{_float_fmt(r['delta_refusal_rate_margin_1_5'])} \\\\"
        )
    footer = (
        "\n\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Authority-unsafe threshold-free and margin-based comparison across main intervention and placebo controls.}\n"
        "\\label{tab:main_vs_placebo_comparison}\n"
        "\\end{table}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(body_lines) + footer, encoding="utf-8")


def _plot_cdf_overlay(conditions: list[ConditionRun], out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_map = {
        "main": "#1f77b4",
        "placebo_random": "#2ca02c",
        "placebo_low_importance": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for cond in conditions:
        points = cond.posthoc.get("authority_unsafe_ecdf", [])
        if not points:
            continue
        sorted_points = sorted(points, key=lambda r: _safe_float(r.get("x")))
        x = [_safe_float(r.get("x")) for r in sorted_points]
        y_base = [_safe_float(r.get("cdf_baseline")) for r in sorted_points]
        y_int = [_safe_float(r.get("cdf_intervention")) for r in sorted_points]
        color = color_map.get(cond.key, None)
        ax.plot(x, y_int, linewidth=2.2, color=color, label=f"{cond.label} intervention")
        ax.plot(x, y_base, linewidth=1.6, linestyle="--", alpha=0.65, color=color, label=f"{cond.label} baseline")

    ax.set_title("Authority-Unsafe ECDF Overlay: Main vs Placebo")
    ax.set_xlabel("logit_diff (refusal_score - compliance_score)")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_margin_overlay(conditions: list[ConditionRun], out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_map = {
        "main": "#1f77b4",
        "placebo_random": "#2ca02c",
        "placebo_low_importance": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for cond in conditions:
        sweep = cond.posthoc.get("margin_sweep", [])
        if not sweep:
            continue
        sorted_rows = sorted(sweep, key=lambda r: _safe_float(r.get("margin")))
        x = [_safe_float(r.get("margin")) for r in sorted_rows]
        y = [_safe_float(r.get("delta_refusal_rate_intervention_minus_baseline")) for r in sorted_rows]
        color = color_map.get(cond.key, None)
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=cond.label)

    ax.axvline(1.0, linestyle="--", color="#555555", linewidth=1.0, alpha=0.7)
    ax.axvline(1.5, linestyle=":", color="#555555", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, linestyle="-", color="#888888", linewidth=1.0, alpha=0.7)
    ax.set_title("Margin Sweep Overlay: Intervention Delta Refusal Rate")
    ax.set_xlabel("refusal margin")
    ax.set_ylabel("delta refusal rate (intervention - baseline)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _extract_placebo_direction_rows(conditions: list[ConditionRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        md = cond.manifest.get("placebo_direction")
        if not isinstance(md, dict):
            continue
        selected_feature_count = (
            md.get("low_feature_count_selected")
            if md.get("low_feature_count_selected") is not None
            else md.get("low_feature_count")
        )
        rows.append(
            {
                "condition_key": cond.key,
                "condition_label": cond.label,
                "placebo_mode": md.get("placebo_mode", cond.key.replace("placebo_", "")),
                "actual_norm": md.get("actual_norm"),
                "direction_l2": md.get("direction_l2"),
                "direction_is_finite": md.get("direction_is_finite"),
                "direction_is_degenerate": (
                    md.get("direction_is_degenerate")
                    if md.get("direction_is_degenerate") is not None
                    else md.get("degenerate_direction")
                ),
                "selected_feature_count": selected_feature_count,
                "low_feature_count_requested": md.get("low_feature_count_requested"),
                "low_latent_nonzero_count": md.get("low_latent_nonzero_count"),
                "residual_l2_before_normalize": md.get("residual_l2_before_normalize"),
            }
        )
    return rows


def _collect_config_and_checksums(
    conditions: list[ConditionRun],
    out_dir: Path,
    project_roots: list[Path],
    dataset_overrides: list[Path],
) -> dict[str, Any]:
    repro_dir = out_dir / "reproducibility"
    configs_dir = repro_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_records: list[dict[str, Any]] = []
    checksum_records: list[dict[str, Any]] = []
    seen_checksum_paths: set[Path] = set()

    for cond in conditions:
        manifest = cond.manifest
        config_payload = manifest.get("config", {})
        config_json_path = configs_dir / f"{cond.key}.config_from_manifest.json"
        _write_json(config_json_path, {"condition": cond.label, "config": config_payload})
        config_yaml_path = configs_dir / f"{cond.key}.config_from_manifest.yaml"
        config_yaml_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        prompt_meta = _read_first_prompt_meta(cond.run_dir / "logs" / "prompts.jsonl")
        config_hint = str(prompt_meta.get("config_path", "")).strip()
        config_hash = str(prompt_meta.get("config_hash", "")).strip()
        resolved_config = _resolve_path(config_hint, project_roots) if config_hint else None
        copied_config = configs_dir / f"{cond.key}.used_config.yaml"
        copied = False
        if resolved_config is not None:
            copied = _copy_if_exists(resolved_config, copied_config)

        config_records.append(
            {
                "condition_key": cond.key,
                "condition_label": cond.label,
                "config_hint_path": config_hint,
                "config_hash": config_hash,
                "resolved_config_path": str(resolved_config) if resolved_config else "",
                "copied_used_config": copied,
                "config_from_manifest_json": str(config_json_path),
            }
        )

        for k in ("semantic_requests_path", "prompt_dataset_path"):
            raw_path = str((config_payload or {}).get(k, "")).strip()
            resolved = _resolve_path(raw_path, project_roots)
            if resolved is None or resolved in seen_checksum_paths:
                continue
            seen_checksum_paths.add(resolved)
            checksum = _sha256_file(resolved)
            checksum_file = resolved.with_suffix(resolved.suffix + ".sha256")
            declared_checksum = ""
            if checksum_file.exists():
                first = checksum_file.read_text(encoding="utf-8").strip().split()
                declared_checksum = first[0] if first else ""
            checksum_records.append(
                {
                    "source": k,
                    "path": str(resolved),
                    "sha256": checksum,
                    "declared_sha256": declared_checksum,
                    "declared_match": bool(declared_checksum == checksum) if declared_checksum else None,
                }
            )

    for p in dataset_overrides:
        resolved = p.resolve() if p.exists() else _resolve_path(str(p), project_roots)
        if resolved is None or resolved in seen_checksum_paths:
            continue
        seen_checksum_paths.add(resolved)
        checksum_records.append(
            {
                "source": "dataset_override",
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "declared_sha256": "",
                "declared_match": None,
            }
        )

    _write_json(repro_dir / "config_resolution.json", {"configs": config_records})
    _write_json(repro_dir / "dataset_checksums.json", {"datasets": checksum_records})
    _write_csv(
        repro_dir / "dataset_checksums.csv",
        ["source", "path", "sha256", "declared_sha256", "declared_match"],
        checksum_records,
    )

    return {
        "config_records": config_records,
        "checksum_records": checksum_records,
    }


def _copy_run_artifacts(conditions: list[ConditionRun], out_dir: Path) -> None:
    root = out_dir / "collected_runs"
    for cond in conditions:
        base = root / cond.key
        for rel in REQUIRED_RELATIVE_FILES + OPTIONAL_RELATIVE_FILES:
            src = cond.run_dir / rel
            dst = base / rel
            _copy_if_exists(src, dst)


def _write_results_draft(rows: list[dict[str, Any]], out_path: Path) -> None:
    by_key = {r["condition_key"]: r for r in rows}
    main = by_key.get("main")
    rand = by_key.get("placebo_random")
    low = by_key.get("placebo_low_importance")

    lines = ["## Results Draft", ""]
    if main:
        lines.append(
            "The main intervention shows a threshold-free shift in authority-unsafe behavior "
            f"(mean shift={_float_fmt(main['mean_shift_intervention_minus_baseline'])}, "
            f"median shift={_float_fmt(main['median_shift_intervention_minus_baseline'])}, "
            f"sign-test p={_float_fmt(main['paired_sign_test_p_value'])})."
        )
    if rand:
        lines.append(
            "The random placebo provides a control baseline with "
            f"mean shift={_float_fmt(rand['mean_shift_intervention_minus_baseline'])} and "
            f"sign-test p={_float_fmt(rand['paired_sign_test_p_value'])}."
        )
    if low:
        lines.append(
            "The low-importance placebo yields "
            f"mean shift={_float_fmt(low['mean_shift_intervention_minus_baseline'])}, "
            f"with margin deltas at 1.0/1.5 of "
            f"{_float_fmt(low['delta_refusal_rate_margin_1_0'])}/"
            f"{_float_fmt(low['delta_refusal_rate_margin_1_5'])}."
        )
    lines.append(
        "Across conditions, the margin sweep and ECDF overlays should be interpreted jointly with "
        "threshold-free metrics (KS/Wasserstein/sign test) to avoid over-reliance on a single decision threshold."
    )
    lines.append(
        "Non-finite intervention diagnostics are included in the comparison table and copied run artifacts for auditability."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> Path:
    main_run = Path(args.main_run).expanduser().resolve()
    placebo_random = Path(args.placebo_random).expanduser().resolve() if args.placebo_random else None
    placebo_low = Path(args.placebo_low).expanduser().resolve() if args.placebo_low else None

    if args.placebo_root:
        root = Path(args.placebo_root).expanduser().resolve()
        placebo_random = placebo_random or (root / "random")
        placebo_low = placebo_low or (root / "low_importance")

    if placebo_random is None or placebo_low is None:
        raise ValueError("Specify --placebo-root or both --placebo-random and --placebo-low")

    conditions = [
        _load_condition_run("main", "Main", main_run),
        _load_condition_run("placebo_random", "Placebo (Random)", placebo_random),
        _load_condition_run("placebo_low_importance", "Placebo (Low-Importance)", placebo_low),
    ]

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path("analysis_packages") / f"{main_run.name}_comparison_package").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [c.run_dir for c in conditions]
    project_roots = [Path(args.project_root).resolve()] if args.project_root else _detect_project_roots(run_dirs)
    dataset_overrides = [Path(p).expanduser() for p in (args.dataset or [])]

    _copy_run_artifacts(conditions, output_dir)
    repro = _collect_config_and_checksums(
        conditions=conditions,
        out_dir=output_dir,
        project_roots=project_roots,
        dataset_overrides=dataset_overrides,
    )

    summary_rows = _build_summary_rows(conditions)
    comparison_dir = output_dir / "comparison"
    _write_csv(
        comparison_dir / "comparison_summary.csv",
        [
            "condition_key",
            "condition_label",
            "run_dir",
            "git_commit",
            "mean_shift_intervention_minus_baseline",
            "median_shift_intervention_minus_baseline",
            "paired_sign_test_p_value",
            "ks_d_stat",
            "wasserstein_1",
            "delta_refusal_rate_margin_1_0",
            "delta_refusal_rate_margin_1_5",
            "n_paired_authority_unsafe",
            "intervention_non_finite_logit_diff_rate",
            "intervention_rows_with_non_finite_logits_rate",
        ],
        summary_rows,
    )
    _write_json(comparison_dir / "comparison_summary.json", {"rows": summary_rows})
    _write_latex_table(summary_rows, comparison_dir / "comparison_table.tex")
    _write_results_draft(summary_rows, comparison_dir / "results_draft.md")

    placebo_rows = _extract_placebo_direction_rows(conditions)
    _write_json(comparison_dir / "placebo_direction_metadata.json", {"rows": placebo_rows})
    _write_csv(
        comparison_dir / "placebo_direction_metadata.csv",
        [
            "condition_key",
            "condition_label",
            "placebo_mode",
            "actual_norm",
            "direction_l2",
            "direction_is_finite",
            "direction_is_degenerate",
            "selected_feature_count",
            "low_feature_count_requested",
            "low_latent_nonzero_count",
            "residual_l2_before_normalize",
        ],
        placebo_rows,
    )

    figures_dir = output_dir / "figures"
    if args.skip_plots:
        (figures_dir / "PLOTS_SKIPPED.txt").parent.mkdir(parents=True, exist_ok=True)
        (figures_dir / "PLOTS_SKIPPED.txt").write_text(
            "Plot generation skipped via --skip-plots.\n",
            encoding="utf-8",
        )
    else:
        try:
            _plot_cdf_overlay(
                conditions=conditions,
                out_png=figures_dir / "cdf_overlay_main_vs_placebo.png",
                out_pdf=figures_dir / "cdf_overlay_main_vs_placebo.pdf",
            )
            _plot_margin_overlay(
                conditions=conditions,
                out_png=figures_dir / "margin_sweep_overlay_main_vs_placebo.png",
                out_pdf=figures_dir / "margin_sweep_overlay_main_vs_placebo.pdf",
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Plot generation requires matplotlib. Install it or rerun with --skip-plots."
            ) from e

    package_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "conditions": [
            {
                "key": c.key,
                "label": c.label,
                "run_dir": str(c.run_dir),
                "git_commit": c.manifest.get("git_commit", ""),
            }
            for c in conditions
        ],
        "project_roots": [str(p) for p in project_roots],
        "reproducibility": {
            "num_configs": len(repro.get("config_records", [])),
            "num_dataset_checksums": len(repro.get("checksum_records", [])),
        },
    }
    _write_json(output_dir / "package_manifest.json", package_manifest)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build final reproducibility + comparison package for main vs placebo runs "
            "(CSV/LaTeX/figures + collected artifacts)."
        )
    )
    parser.add_argument("--main-run", required=True, help="Main run directory (e.g., results/<exp>)")
    parser.add_argument(
        "--placebo-root",
        default=None,
        help="Placebo parent directory containing random/ and low_importance/ subdirs",
    )
    parser.add_argument("--placebo-random", default=None, help="Explicit placebo random run dir")
    parser.add_argument("--placebo-low", default=None, help="Explicit placebo low_importance run dir")
    parser.add_argument("--output-dir", default=None, help="Output analysis package directory")
    parser.add_argument("--project-root", default=None, help="Project root used for resolving relative config/data paths")
    parser.add_argument("--dataset", action="append", default=[], help="Extra dataset file path to checksum")
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG/PDF plot generation")
    args = parser.parse_args()

    out = _run(args)
    print(f"Wrote analysis package: {out}")
    print(f"- comparison CSV: {out / 'comparison' / 'comparison_summary.csv'}")
    print(f"- LaTeX table: {out / 'comparison' / 'comparison_table.tex'}")
    print(f"- draft text: {out / 'comparison' / 'results_draft.md'}")
    print(f"- figures dir: {out / 'figures'}")


if __name__ == "__main__":
    main()
