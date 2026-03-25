#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for run_cloud_campaign.py") from exc

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError("Campaign manifest must be a YAML mapping")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Campaign manifest must define a non-empty 'steps' list")
    return payload


def _normalize_cmd(raw: Any) -> list[str]:
    if isinstance(raw, list) and raw and all(isinstance(part, (str, int, float)) for part in raw):
        return [str(part) for part in raw]
    if isinstance(raw, str) and raw.strip():
        return shlex.split(raw)
    raise ValueError("Each campaign step must define 'run' as a non-empty string or list")


def _resolve_path(base: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a YAML-defined cloud experiment campaign.")
    parser.add_argument("--manifest", required=True, help="YAML campaign manifest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional JSON summary path (default: <manifest>.summary.json)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    if manifest_path.parent.name == "cloud" and len(manifest_path.parents) >= 3:
        repo_root = manifest_path.parents[2]
    else:
        repo_root = manifest_path.parent

    base_env = os.environ.copy()
    for key, value in (manifest.get("env") or {}).items():
        base_env[str(key)] = str(value)

    results: list[dict[str, Any]] = []
    failed = False

    for idx, raw_step in enumerate(manifest["steps"], start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(f"Step {idx} must be a mapping")

        name = str(raw_step.get("name") or f"step_{idx}")
        cmd = _normalize_cmd(raw_step.get("run"))
        cwd = _resolve_path(repo_root, raw_step.get("cwd")) or repo_root
        skip_if_exists = _resolve_path(cwd, raw_step.get("skip_if_exists"))
        allow_failure = bool(raw_step.get("allow_failure", False))
        step_env = dict(base_env)
        for key, value in (raw_step.get("env") or {}).items():
            step_env[str(key)] = str(value)

        row: dict[str, Any] = {
            "name": name,
            "cwd": str(cwd),
            "cmd": cmd,
            "skip_if_exists": str(skip_if_exists) if skip_if_exists else None,
            "status": "planned",
        }

        if skip_if_exists and skip_if_exists.exists():
            row["status"] = "skipped_existing"
            results.append(row)
            print(f"[skip] {name} ({skip_if_exists} exists)")
            continue

        print("$", " ".join(shlex.quote(part) for part in cmd))
        if args.dry_run:
            row["status"] = "dry_run"
            results.append(row)
            continue

        proc = subprocess.run(cmd, cwd=str(cwd), env=step_env)
        row["returncode"] = int(proc.returncode)
        row["status"] = "ok" if proc.returncode == 0 else f"failed_exit_{proc.returncode}"
        results.append(row)

        if proc.returncode != 0 and not allow_failure:
            failed = True
            break

    summary = {
        "manifest": str(manifest_path),
        "dry_run": bool(args.dry_run),
        "failed": bool(failed),
        "steps": results,
    }
    summary_path = (
        Path(args.summary_out).expanduser().resolve()
        if args.summary_out
        else manifest_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote campaign summary: {summary_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
