#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _alpha_tag(alpha: float) -> str:
    # Stable filesystem-safe tag, e.g., 1.0 -> a1p0, 0.25 -> a0p25
    s = f"{alpha:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _run_one(runner: Path, cfg_path: Path, dry_run: bool) -> int:
    cmd = [sys.executable, str(runner), "--config", str(cfg_path)]
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a layer/alpha robustness sweep by materializing temporary configs "
            "and invoking scripts/run_experiment.py for each combination."
        )
    )
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="Run name prefix (default: base config experiment_name)",
    )
    parser.add_argument(
        "--runner",
        default="scripts/run_experiment.py",
        help="Runner script path",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run when results/<experiment_name>/metrics.json already exists",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional JSON summary path (default: results/<prefix>_sweep_plan.json)",
    )
    args = parser.parse_args()

    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for run_layer_alpha_sweep.py") from exc

    config_path = Path(args.config).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if not isinstance(base_cfg, dict):
        raise ValueError("Config must be a YAML object")

    base_name = str(args.experiment_prefix or base_cfg.get("experiment_name") or "layer_alpha")
    seeds = list(args.seeds) if args.seeds else [int(base_cfg.get("seed", 42))]
    results_dir = Path(str(base_cfg.get("results_dir", "results"))).expanduser().resolve()

    planned: list[dict[str, object]] = []
    completed = 0
    skipped = 0
    failed = 0

    for layer in args.layers:
        for alpha in args.alphas:
            for seed in seeds:
                exp_name = f"{base_name}_l{int(layer)}_a{_alpha_tag(float(alpha))}_s{int(seed)}"
                run_root = results_dir / exp_name
                metrics_path = run_root / "metrics.json"

                cfg = copy.deepcopy(base_cfg)
                cfg["experiment_name"] = exp_name
                cfg["layer_for_sae"] = int(layer)
                cfg["capture_all_layers"] = False
                cfg["capture_layers"] = [int(layer)]
                cfg["alpha_intervention"] = float(alpha)
                cfg["seed"] = int(seed)

                row: dict[str, object] = {
                    "experiment_name": exp_name,
                    "layer_for_sae": int(layer),
                    "alpha_intervention": float(alpha),
                    "seed": int(seed),
                    "results_root": str(run_root),
                    "status": "planned",
                }

                if args.skip_existing and metrics_path.exists():
                    row["status"] = "skipped_existing"
                    planned.append(row)
                    skipped += 1
                    print(f"[skip] {exp_name} (metrics already exists)")
                    continue

                with tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".yaml",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    yaml.safe_dump(cfg, tmp, sort_keys=False)
                    temp_cfg = Path(tmp.name)

                try:
                    code = _run_one(runner=runner, cfg_path=temp_cfg, dry_run=args.dry_run)
                    if code == 0:
                        row["status"] = "ok" if not args.dry_run else "dry_run"
                        completed += 0 if args.dry_run else 1
                    else:
                        row["status"] = f"failed_exit_{code}"
                        failed += 1
                finally:
                    temp_cfg.unlink(missing_ok=True)

                planned.append(row)

                if failed > 0 and not args.dry_run:
                    print("Stopping after first failure.")
                    break
            if failed > 0 and not args.dry_run:
                break
        if failed > 0 and not args.dry_run:
            break

    summary = {
        "base_config": str(config_path),
        "runner": str(runner),
        "layers": [int(x) for x in args.layers],
        "alphas": [float(x) for x in args.alphas],
        "seeds": [int(x) for x in seeds],
        "total_planned": len(planned),
        "completed": int(completed),
        "skipped_existing": int(skipped),
        "failed": int(failed),
        "runs": planned,
    }

    default_summary = results_dir / f"{base_name}_sweep_plan.json"
    summary_path = Path(args.summary_out).expanduser().resolve() if args.summary_out else default_summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote sweep summary: {summary_path}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
