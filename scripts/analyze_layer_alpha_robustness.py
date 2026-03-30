#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _collect_run_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mf in sorted(results_root.glob("*/logs/run_manifest.json")):
        run_dir = mf.parent.parent
        metrics_path = run_dir / "metrics.json"
        posthoc_path = run_dir / "posthoc" / "posthoc_analysis.json"
        if not metrics_path.exists():
            continue
        manifest = _load_json(mf)
        metrics = _load_json(metrics_path)
        posthoc = _load_json(posthoc_path) if posthoc_path.exists() else {}

        cfg = manifest.get("config", {})
        tf = posthoc.get("threshold_free_authority_unsafe", {})
        rows.append(
            {
                "run_name": run_dir.name,
                "layer_for_sae": cfg.get("layer_for_sae"),
                "alpha_intervention": cfg.get("alpha_intervention"),
                "refusal_margin": cfg.get("refusal_margin"),
                "seed": cfg.get("seed"),
                "mean_shift_intervention_minus_baseline": tf.get("mean_shift_intervention_minus_baseline"),
                "median_shift_intervention_minus_baseline": tf.get("median_shift_intervention_minus_baseline"),
                "ks_d_stat": tf.get("ks_d_stat"),
                "wasserstein_1": tf.get("wasserstein_1"),
                "cliffs_delta_intervention_vs_baseline": tf.get("cliffs_delta_intervention_vs_baseline"),
                "baseline_refusal_rate": metrics.get("baseline_refusal_rate"),
                "authority_refusal_rate": metrics.get("authority_refusal_rate"),
                "intervention_refusal_rate": metrics.get("intervention_refusal_rate"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate layer/alpha robustness runs from results directory.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_run_rows(results_root)
    _write_csv(out_dir / "layer_alpha_run_summary.csv", rows)

    layers = sorted({r["layer_for_sae"] for r in rows if r.get("layer_for_sae") is not None})
    alphas = sorted({r["alpha_intervention"] for r in rows if r.get("alpha_intervention") is not None})
    status = {
        "n_runs": len(rows),
        "unique_layers": layers,
        "unique_alphas": alphas,
        "has_layer_sweep": len(layers) >= 3,
        "has_alpha_sweep": len(alphas) >= 3,
        "notes": (
            "Layer/alpha sweep is considered available only when at least 3 unique values are present."
        ),
    }
    (out_dir / "layer_alpha_coverage.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"Wrote: {out_dir / 'layer_alpha_run_summary.csv'}")
    print(f"Wrote: {out_dir / 'layer_alpha_coverage.json'}")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
