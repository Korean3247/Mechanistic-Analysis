#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment across multiple seeds")
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="Optional override prefix for experiment_name (default: config experiment_name)",
    )
    parser.add_argument(
        "--runner",
        default="scripts/run_experiment.py",
        help="Runner script path",
    )
    args = parser.parse_args()

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for run_seed_sweep.py") from exc

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if not isinstance(base_cfg, dict):
        raise ValueError("Config must be a YAML object")

    base_name = str(args.experiment_prefix or base_cfg.get("experiment_name") or "seed_sweep")
    runner = Path(args.runner)

    for seed in args.seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = int(seed)
        cfg["experiment_name"] = f"{base_name}_seed{seed}"

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            temp_cfg_path = Path(tmp.name)

        cmd = [sys.executable, str(runner), "--config", str(temp_cfg_path)]
        print(f"Running seed={seed}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        finally:
            temp_cfg_path.unlink(missing_ok=True)

    print("Seed sweep complete")


if __name__ == "__main__":
    main()
