#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format(value: float, ndigits: int = 4) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{ndigits}f}"


def _margin_delta(margin_rows: list[dict[str, Any]], target: float) -> float:
    for row in margin_rows:
        margin = _safe_float(row.get("margin"))
        if math.isfinite(margin) and abs(margin - target) < 1e-9:
            return _safe_float(row.get("delta_refusal_rate_intervention_minus_baseline"))
    return float("nan")


def _load_direction(path: Path) -> Any:
    if torch is None:
        return None
    payload = torch.load(path, map_location="cpu")
    direction = payload.get("residual_direction_normalized")
    if direction is None:
        direction = payload.get("direction")
    if direction is None or not isinstance(direction, torch.Tensor):
        return None
    return direction.detach().to(dtype=torch.float32, device="cpu").flatten()


def _resolve_direction_path(run_dir: Path, explicit_path: str | None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        return path if path.exists() else None

    candidate = run_dir / "authority_direction_vector.pt"
    if candidate.exists():
        return candidate

    replay_manifest = run_dir / "replay_manifest.json"
    if replay_manifest.exists():
        payload = _read_json(replay_manifest)
        direct = payload.get("direction_path")
        if isinstance(direct, str) and direct:
            path = Path(direct).expanduser().resolve()
            if path.exists():
                return path
        specs = payload.get("direction_specs", [])
        if isinstance(specs, list) and specs:
            first = specs[0]
            if isinstance(first, dict):
                raw = first.get("direction_path")
                if isinstance(raw, str) and raw:
                    path = Path(raw).expanduser().resolve()
                    if path.exists():
                        return path

    run_manifest = run_dir / "logs" / "run_manifest.json"
    if run_manifest.exists():
        payload = _read_json(run_manifest)
        artifacts = payload.get("artifacts", {})
        if isinstance(artifacts, dict):
            raw = artifacts.get("direction_vector")
            if isinstance(raw, str) and raw:
                path = Path(raw).expanduser().resolve()
                if path.exists():
                    return path
    return None


def _load_run_payload(run_dir: Path) -> tuple[dict[str, Any], str]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return _read_json(metrics_path), "refit"

    replay_manifest = run_dir / "replay_manifest.json"
    if replay_manifest.exists():
        return _read_json(replay_manifest), "replay"

    posthoc_path = run_dir / "posthoc" / "posthoc_analysis.json"
    if posthoc_path.exists():
        return _read_json(posthoc_path), "posthoc_only"

    raise FileNotFoundError(f"Could not find metrics.json or replay_manifest.json in {run_dir}")


def _extract_row(
    label: str,
    run_dir: Path,
    explicit_direction: str | None,
) -> tuple[dict[str, Any], Path | None]:
    payload, run_kind = _load_run_payload(run_dir)

    threshold = payload.get("threshold_free_authority_unsafe", {})
    if not isinstance(threshold, dict):
        threshold = {}

    margin_rows = payload.get("margin_sweep", [])
    if not isinstance(margin_rows, list):
        margin_rows = []

    prompt_source = None
    model_name = None
    if run_kind == "replay":
        prompt_source = payload.get("prompts_path")
        model_name = payload.get("model")
    else:
        run_manifest_path = run_dir / "logs" / "run_manifest.json"
        if run_manifest_path.exists():
            manifest = _read_json(run_manifest_path)
            config = manifest.get("config", {})
            if isinstance(config, dict):
                prompt_source = config.get("prompt_dataset_path")
                model_name = config.get("model")

    direction_path = _resolve_direction_path(run_dir, explicit_direction)
    row = {
        "label": label,
        "run_kind": run_kind,
        "run_dir": str(run_dir),
        "model": model_name,
        "prompt_source": prompt_source,
        "mean_shift": _safe_float(threshold.get("mean_shift_intervention_minus_baseline")),
        "median_shift": _safe_float(threshold.get("median_shift_intervention_minus_baseline")),
        "sign_test_p": _safe_float((threshold.get("paired_sign_test") or {}).get("p_value")),
        "ks_d": _safe_float(threshold.get("ks_d_stat")),
        "w1": _safe_float(threshold.get("wasserstein_1")),
        "delta_refusal_m1_0": _margin_delta(margin_rows, 1.0),
        "delta_refusal_m1_5": _margin_delta(margin_rows, 1.5),
        "direction_path": str(direction_path) if direction_path else None,
    }
    return row, direction_path


def _sign_label(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _direction_diagnostics(a_path: Path | None, b_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable",
        "reason": "direction_missing",
        "direction_a_path": str(a_path) if a_path else None,
        "direction_b_path": str(b_path) if b_path else None,
    }
    if a_path is None or b_path is None:
        return result
    if torch is None:
        result["reason"] = "torch_not_available"
        return result

    a_vec = _load_direction(a_path)
    b_vec = _load_direction(b_path)
    if a_vec is None or b_vec is None:
        result["reason"] = "direction_tensor_missing"
        return result

    result["dim_a"] = int(a_vec.numel())
    result["dim_b"] = int(b_vec.numel())
    if a_vec.numel() != b_vec.numel():
        result["reason"] = "dimension_mismatch"
        return result

    norm_a = float(torch.linalg.norm(a_vec).item())
    norm_b = float(torch.linalg.norm(b_vec).item())
    result["norm_a"] = norm_a
    result["norm_b"] = norm_b
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        result["reason"] = "near_zero_norm"
        return result

    cosine = float(torch.dot(a_vec, b_vec).item() / (norm_a * norm_b))
    result.update(
        {
            "status": "ok",
            "reason": "computed",
            "cosine": cosine,
        }
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "run_kind",
                "model",
                "prompt_source",
                "mean_shift",
                "median_shift",
                "sign_test_p",
                "ks_d",
                "w1",
                "delta_refusal_m1_0",
                "delta_refusal_m1_5",
                "direction_path",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(path: Path, rows: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    note = "Prompt sources differ between rows." if not comparison.get("same_prompt_source", False) else "Prompt source matches between rows."
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Gemma sign-reversal diagnostics comparing frozen replay against Gemma refit. "
        + note
        + "}",
        "\\label{tab:gemma_sign_reversal}",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Condition & Mean $\\Delta$ & Median $\\Delta$ & Sign-$p$ & KS $D$ & $\\Delta$Ref@1.0 & $\\Delta$Ref@1.5 \\\\",
        "\\midrule",
    ]
    for row in rows:
        label = str(row["label"]).replace("_", "\\_")
        lines.append(
            f"{label} & "
            f"{_format(_safe_float(row['mean_shift']))} & "
            f"{_format(_safe_float(row['median_shift']))} & "
            f"{_format(_safe_float(row['sign_test_p']))} & "
            f"{_format(_safe_float(row['ks_d']))} & "
            f"{_format(_safe_float(row['delta_refusal_m1_0']))} & "
            f"{_format(_safe_float(row['delta_refusal_m1_5']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gemma frozen replay against Gemma refit for sign-reversal diagnostics.")
    parser.add_argument("--frozen-run", required=True, help="Run directory for frozen exact-main replay on Gemma")
    parser.add_argument("--refit-run", required=True, help="Run directory for Gemma refit experiment")
    parser.add_argument("--out-dir", required=True, help="Output directory for comparison artifacts")
    parser.add_argument("--frozen-label", default="gemma_frozen")
    parser.add_argument("--refit-label", default="gemma_refit")
    parser.add_argument("--frozen-direction", default=None, help="Optional explicit direction path for frozen replay")
    parser.add_argument("--refit-direction", default=None, help="Optional explicit direction path for refit run")
    args = parser.parse_args()

    frozen_run = Path(args.frozen_run).expanduser().resolve()
    refit_run = Path(args.refit_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_row, frozen_direction_path = _extract_row(args.frozen_label, frozen_run, args.frozen_direction)
    refit_row, refit_direction_path = _extract_row(args.refit_label, refit_run, args.refit_direction)
    rows = [frozen_row, refit_row]

    comparison = {
        "mean_shift_gap_refit_minus_frozen": _safe_float(refit_row["mean_shift"]) - _safe_float(frozen_row["mean_shift"]),
        "sign_label_frozen": _sign_label(_safe_float(frozen_row["mean_shift"])),
        "sign_label_refit": _sign_label(_safe_float(refit_row["mean_shift"])),
        "sign_flip_between_runs": _sign_label(_safe_float(frozen_row["mean_shift"])) != _sign_label(_safe_float(refit_row["mean_shift"])),
        "same_prompt_source": frozen_row.get("prompt_source") == refit_row.get("prompt_source"),
        "direction_diagnostics": _direction_diagnostics(frozen_direction_path, refit_direction_path),
    }

    summary = {
        "inputs": {
            "frozen_run": str(frozen_run),
            "refit_run": str(refit_run),
        },
        "rows": rows,
        "comparison": comparison,
    }

    (out_dir / "gemma_sign_reversal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "gemma_sign_reversal_rows.csv", rows)
    _write_latex(out_dir / "table_gemma_sign_reversal.tex", rows, comparison)
    print(f"Wrote: {out_dir / 'gemma_sign_reversal_summary.json'}")
    print(f"Wrote: {out_dir / 'gemma_sign_reversal_rows.csv'}")
    print(f"Wrote: {out_dir / 'table_gemma_sign_reversal.tex'}")


if __name__ == "__main__":
    main()
