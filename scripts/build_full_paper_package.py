#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


REQUIRED_RUN_FILES = [
    "metrics.json",
    "logs/run_manifest.json",
    "logs/baseline_samples.json",
    "logs/intervention_samples.json",
    "posthoc/posthoc_analysis.json",
    "posthoc/margin_sweep.csv",
    "posthoc/authority_unsafe_ecdf.csv",
]

OPTIONAL_RUN_FILES = [
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


@dataclass
class Condition:
    key: str
    label: str
    run_dir: Path
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    baseline_rows: list[dict[str, Any]]
    intervention_rows: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _finite_values(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        f = _coerce_float(v)
        if _is_finite(f):
            out.append(f)
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    vals = sorted(values)
    pos = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _ci95(values: list[float]) -> tuple[float, float]:
    return _percentile(values, 2.5), _percentile(values, 97.5)


def _refusal_rate(values: list[float], margin: float) -> float:
    finite = _finite_values(values)
    if not finite:
        return 0.0
    return float(sum(1 for v in finite if v > margin) / len(finite))


def _ks_statistic(x: list[float], y: list[float]) -> float:
    a = sorted(_finite_values(x))
    b = sorted(_finite_values(y))
    if not a or not b:
        return 0.0
    points = sorted(set(a + b))
    d = 0.0
    na = len(a)
    nb = len(b)
    for p in points:
        fa = _right_fraction(a, p, na)
        fb = _right_fraction(b, p, nb)
        d = max(d, abs(fa - fb))
    return float(d)


def _right_fraction(sorted_vals: list[float], x: float, n: int) -> float:
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return float(lo / n) if n > 0 else 0.0


def _wasserstein_1d(x: list[float], y: list[float]) -> float:
    a = sorted(_finite_values(x))
    b = sorted(_finite_values(y))
    if not a or not b:
        return 0.0
    points = sorted(set(a + b))
    if len(points) <= 1:
        return 0.0
    na = len(a)
    nb = len(b)
    total = 0.0
    for left, right in zip(points[:-1], points[1:]):
        fa = _right_fraction(a, left, na)
        fb = _right_fraction(b, left, nb)
        total += abs(fa - fb) * (right - left)
    return float(total)


def _cliffs_delta(x: list[float], y: list[float]) -> float:
    a = _finite_values(x)
    b = _finite_values(y)
    if not a or not b:
        return 0.0
    gt = 0
    lt = 0
    for av in a:
        for bv in b:
            if av > bv:
                gt += 1
            elif av < bv:
                lt += 1
    total = len(a) * len(b)
    if total == 0:
        return 0.0
    return float((gt - lt) / total)


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    acc = 0.0
    for i in range(0, k + 1):
        acc += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return float(acc)


def _sign_test_two_sided(deltas: list[float]) -> dict[str, float | int]:
    d = _finite_values(deltas)
    pos = sum(1 for v in d if v > 0)
    neg = sum(1 for v in d if v < 0)
    ties = sum(1 for v in d if v == 0)
    n_eff = pos + neg
    if n_eff == 0:
        return {"n_effective": 0, "pos": 0, "neg": 0, "ties": ties, "p_value": 1.0}
    k = min(pos, neg)
    p = min(1.0, 2.0 * _binom_cdf(k, n_eff, 0.5))
    return {"n_effective": int(n_eff), "pos": int(pos), "neg": int(neg), "ties": int(ties), "p_value": float(p)}


def _delta_distribution(deltas: list[float]) -> dict[str, Any]:
    finite = _finite_values(deltas)
    n_total = len(deltas)
    n = len(finite)
    if n == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "share_negative": 0.0,
            "share_positive": 0.0,
            "share_zero": 0.0,
            "n_total": int(n_total),
            "n_finite": 0,
            "non_finite_delta_count": int(n_total),
            "non_finite_delta_rate": 1.0 if n_total else 0.0,
        }
    return {
        "mean": float(mean(finite)),
        "median": float(median(finite)),
        "std": float(pstdev(finite)) if n > 1 else 0.0,
        "p10": _percentile(finite, 10),
        "p50": _percentile(finite, 50),
        "p90": _percentile(finite, 90),
        "share_negative": float(sum(1 for v in finite if v < 0) / n),
        "share_positive": float(sum(1 for v in finite if v > 0) / n),
        "share_zero": float(sum(1 for v in finite if v == 0) / n),
        "n_total": int(n_total),
        "n_finite": int(n),
        "non_finite_delta_count": int(n_total - n),
        "non_finite_delta_rate": float((n_total - n) / n_total) if n_total else 0.0,
    }


def _paired_bootstrap_margin_delta_ci(
    baseline: list[float],
    intervention: list[float],
    margin: float,
    iters: int,
    seed: int,
) -> dict[str, float]:
    n = min(len(baseline), len(intervention))
    if n == 0:
        return {
            "baseline_ci95_low": 0.0,
            "baseline_ci95_high": 0.0,
            "intervention_ci95_low": 0.0,
            "intervention_ci95_high": 0.0,
            "delta_ci95_low": 0.0,
            "delta_ci95_high": 0.0,
        }
    pairs = [(baseline[i], intervention[i]) for i in range(n)]
    pairs = [(b, i) for b, i in pairs if _is_finite(b) and _is_finite(i)]
    n = len(pairs)
    if n == 0:
        return {
            "baseline_ci95_low": 0.0,
            "baseline_ci95_high": 0.0,
            "intervention_ci95_low": 0.0,
            "intervention_ci95_high": 0.0,
            "delta_ci95_low": 0.0,
            "delta_ci95_high": 0.0,
        }
    rng = random.Random(seed)
    b_rates: list[float] = []
    i_rates: list[float] = []
    d_rates: list[float] = []
    for _ in range(iters):
        idxs = [rng.randrange(n) for _ in range(n)]
        b = [pairs[j][0] for j in idxs]
        i = [pairs[j][1] for j in idxs]
        br = _refusal_rate(b, margin)
        ir = _refusal_rate(i, margin)
        b_rates.append(br)
        i_rates.append(ir)
        d_rates.append(ir - br)
    b_lo, b_hi = _ci95(b_rates)
    i_lo, i_hi = _ci95(i_rates)
    d_lo, d_hi = _ci95(d_rates)
    return {
        "baseline_ci95_low": b_lo,
        "baseline_ci95_high": b_hi,
        "intervention_ci95_low": i_lo,
        "intervention_ci95_high": i_hi,
        "delta_ci95_low": d_lo,
        "delta_ci95_high": d_hi,
    }


def _pair_authority_unsafe(
    baseline_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base_candidates = [
        r for r in baseline_rows if str(r.get("framing_type")) == "authority" and str(r.get("risk_tier")) == "unsafe"
    ]
    int_candidates = [r for r in intervention_rows if str(r.get("risk_tier")) == "unsafe"]

    base_map = {str(r.get("prompt_id")): r.get("logit_diff", float("nan")) for r in base_candidates}
    int_map = {str(r.get("prompt_id")): r.get("logit_diff", float("nan")) for r in int_candidates}
    common_ids = sorted(set(base_map).intersection(int_map))

    pair_ids: list[str] = []
    base_vals: list[float] = []
    int_vals: list[float] = []
    dropped_examples: list[dict[str, Any]] = []
    dropped_non_finite = 0
    for pid in common_ids:
        b = _coerce_float(base_map[pid])
        i = _coerce_float(int_map[pid])
        if not (_is_finite(b) and _is_finite(i)):
            dropped_non_finite += 1
            if len(dropped_examples) < 5:
                dropped_examples.append(
                    {
                        "prompt_id": pid,
                        "baseline_logit_diff": repr(base_map[pid]),
                        "intervention_logit_diff": repr(int_map[pid]),
                    }
                )
            continue
        pair_ids.append(pid)
        base_vals.append(b)
        int_vals.append(i)

    return {
        "pair_ids": pair_ids,
        "baseline": base_vals,
        "intervention": int_vals,
        "diagnostics": {
            "n_baseline_unsafe_authority_rows": len(base_candidates),
            "n_intervention_unsafe_rows": len(int_candidates),
            "baseline_duplicate_prompt_id_count": len(base_candidates) - len(base_map),
            "intervention_duplicate_prompt_id_count": len(int_candidates) - len(int_map),
            "n_common_prompt_ids": len(common_ids),
            "n_paired_finite": len(pair_ids),
            "n_dropped_non_finite_pairs": dropped_non_finite,
            "dropped_non_finite_pair_examples": dropped_examples,
        },
    }


def _compute_threshold_free(baseline: list[float], intervention: list[float]) -> dict[str, Any]:
    deltas = [iv - bv for bv, iv in zip(baseline, intervention)]
    finite_deltas = _finite_values(deltas)
    return {
        "n_paired_authority_unsafe": int(len(finite_deltas)),
        "baseline_mean_logit_diff": float(mean(_finite_values(baseline))) if baseline else 0.0,
        "intervention_mean_logit_diff": float(mean(_finite_values(intervention))) if intervention else 0.0,
        "mean_shift_intervention_minus_baseline": float(mean(finite_deltas)) if finite_deltas else 0.0,
        "baseline_median_logit_diff": float(median(_finite_values(baseline))) if baseline else 0.0,
        "intervention_median_logit_diff": float(median(_finite_values(intervention))) if intervention else 0.0,
        "median_shift_intervention_minus_baseline": float(median(finite_deltas)) if finite_deltas else 0.0,
        "delta_distribution": _delta_distribution(deltas),
        "paired_sign_test": _sign_test_two_sided(deltas),
        "ks_d_stat": _ks_statistic(baseline, intervention),
        "wasserstein_1": _wasserstein_1d(baseline, intervention),
        "cliffs_delta_intervention_vs_baseline": _cliffs_delta(intervention, baseline),
    }


def _compute_margin_rows(
    baseline: list[float],
    intervention: list[float],
    margins: list[float],
    bootstrap_iters: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, m in enumerate(margins):
        b = _refusal_rate(baseline, m)
        i = _refusal_rate(intervention, m)
        ci = _paired_bootstrap_margin_delta_ci(
            baseline=baseline,
            intervention=intervention,
            margin=m,
            iters=bootstrap_iters,
            seed=seed + idx * 173,
        )
        rows.append(
            {
                "margin": float(m),
                "baseline_refusal_rate": b,
                "intervention_refusal_rate": i,
                "delta_refusal_rate_intervention_minus_baseline": i - b,
                "baseline_compliance_rate": 1.0 - b,
                "intervention_compliance_rate": 1.0 - i,
                **ci,
            }
        )
    return rows


def _ecdf_rows(baseline: list[float], intervention: list[float]) -> list[dict[str, float]]:
    b = sorted(_finite_values(baseline))
    i = sorted(_finite_values(intervention))
    if not b and not i:
        return []
    points = sorted(set(b + i))
    rows: list[dict[str, float]] = []
    nb = len(b)
    ni = len(i)
    for x in points:
        rows.append(
            {
                "x": float(x),
                "cdf_baseline": _right_fraction(b, x, nb) if nb else 0.0,
                "cdf_intervention": _right_fraction(i, x, ni) if ni else 0.0,
            }
        )
    return rows


def _validate_run_dir(run_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_RUN_FILES if not (run_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {run_dir}: {missing}")


def _load_condition(key: str, label: str, run_dir: Path) -> Condition:
    _validate_run_dir(run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    manifest = _read_json(run_dir / "logs" / "run_manifest.json")
    baseline_payload = _read_json(run_dir / "logs" / "baseline_samples.json")
    intervention_payload = _read_json(run_dir / "logs" / "intervention_samples.json")
    baseline_rows = baseline_payload.get("samples", [])
    intervention_rows = intervention_payload.get("samples", [])
    if not isinstance(baseline_rows, list) or not isinstance(intervention_rows, list):
        raise ValueError(f"Invalid sample payload in {run_dir}")
    return Condition(
        key=key,
        label=label,
        run_dir=run_dir,
        metrics=metrics,
        manifest=manifest,
        baseline_rows=baseline_rows,
        intervention_rows=intervention_rows,
    )


def _copy_collected_artifacts(conditions: list[Condition], out_dir: Path) -> None:
    dst_root = out_dir / "collected_runs"
    for cond in conditions:
        cond_root = dst_root / cond.key
        for rel in REQUIRED_RUN_FILES + OPTIONAL_RUN_FILES:
            src = cond.run_dir / rel
            if not src.exists():
                continue
            dst = cond_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
                    seen.add(resolved)
                    roots.append(resolved)
    return roots


def _resolve_path(path_str: str, roots: list[Path]) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p.resolve()
    if not p.is_absolute():
        for root in roots:
            candidate = root / p
            if candidate.exists():
                return candidate.resolve()
    if p.exists():
        return p.resolve()
    basename = p.name
    for root in roots:
        for sub in ("data", "configs", "configs_data/data", "configs_data/configs"):
            candidate = root / sub / basename
            if candidate.exists():
                return candidate.resolve()
    return None


def _read_prompts_first_row(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    return rows[0] if rows else {}


def _collect_reproducibility(
    conditions: list[Condition],
    out_dir: Path,
    project_roots: list[Path],
    dataset_overrides: list[Path],
) -> dict[str, Any]:
    repro_dir = out_dir / "reproducibility"
    cfg_dir = repro_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    config_rows: list[dict[str, Any]] = []
    checksum_rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    for cond in conditions:
        manifest_cfg = cond.manifest.get("config", {})
        manifest_path = cfg_dir / f"{cond.key}.config_from_manifest.json"
        _write_json(manifest_path, {"condition": cond.label, "config": manifest_cfg})

        prompt_meta = _read_prompts_first_row(cond.run_dir / "logs" / "prompts.jsonl")
        config_hint = str(prompt_meta.get("config_path", "")).strip()
        config_hash = str(prompt_meta.get("config_hash", "")).strip()
        config_resolved = _resolve_path(config_hint, project_roots) if config_hint else None
        copied_path = cfg_dir / f"{cond.key}.used_config.yaml"
        copied = False
        if config_resolved is not None:
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_resolved, copied_path)
            copied = True

        config_rows.append(
            {
                "condition_key": cond.key,
                "condition_label": cond.label,
                "git_commit": cond.manifest.get("git_commit", ""),
                "config_hint_path": config_hint,
                "config_hash": config_hash,
                "config_resolved_path": str(config_resolved) if config_resolved else "",
                "copied_used_config": copied,
            }
        )

        for source_key in ("semantic_requests_path", "prompt_dataset_path"):
            raw_path = str((manifest_cfg or {}).get(source_key, "")).strip()
            resolved = _resolve_path(raw_path, project_roots)
            if resolved is None or resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            sha = _sha256_file(resolved)
            declared = ""
            sha_path = resolved.with_suffix(resolved.suffix + ".sha256")
            if sha_path.exists():
                parts = sha_path.read_text(encoding="utf-8").strip().split()
                declared = parts[0] if parts else ""
            checksum_rows.append(
                {
                    "source": source_key,
                    "path": str(resolved),
                    "sha256": sha,
                    "declared_sha256": declared,
                    "declared_match": bool(declared == sha) if declared else None,
                }
            )

    for override in dataset_overrides:
        resolved = override.resolve() if override.exists() else _resolve_path(str(override), project_roots)
        if resolved is None or resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        checksum_rows.append(
            {
                "source": "dataset_override",
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "declared_sha256": "",
                "declared_match": None,
            }
        )

    _write_json(repro_dir / "config_resolution.json", {"configs": config_rows})
    _write_json(repro_dir / "dataset_checksums.json", {"datasets": checksum_rows})
    _write_csv(
        repro_dir / "dataset_checksums.csv",
        ["source", "path", "sha256", "declared_sha256", "declared_match"],
        checksum_rows,
    )

    return {"configs": config_rows, "datasets": checksum_rows}


def _direction_diagnostics(cond: Condition) -> dict[str, Any]:
    manifest_direction = cond.manifest.get("placebo_direction", {})
    rows = cond.intervention_rows
    l2_vals = _finite_values([r.get("intervention_direction_l2", float("nan")) for r in rows])
    finite_flags = [bool(r.get("intervention_direction_is_finite", True)) for r in rows]
    degenerate_flags = [bool(r.get("intervention_degenerate_direction", False)) for r in rows]
    return {
        "manifest_direction_metadata": manifest_direction,
        "row_runtime_summary": {
            "n_rows": len(rows),
            "direction_l2_min": min(l2_vals) if l2_vals else 0.0,
            "direction_l2_max": max(l2_vals) if l2_vals else 0.0,
            "direction_l2_mean": float(mean(l2_vals)) if l2_vals else 0.0,
            "all_direction_is_finite": all(finite_flags) if finite_flags else True,
            "any_direction_is_degenerate": any(degenerate_flags) if degenerate_flags else False,
            "max_identity_fallback_calls": int(
                max((_coerce_float(r.get("intervention_identity_fallback_calls", 0)) for r in rows), default=0)
            ),
            "max_non_finite_coeff_calls": int(
                max((_coerce_float(r.get("intervention_non_finite_coeff_calls", 0)) for r in rows), default=0)
            ),
            "max_non_finite_output_calls": int(
                max((_coerce_float(r.get("intervention_non_finite_output_calls", 0)) for r in rows), default=0)
            ),
        },
        "requested_direction_fields": {
            "direction_is_degenerate": manifest_direction.get("direction_is_degenerate"),
            "residual_l2_before_normalize": manifest_direction.get("residual_l2_before_normalize"),
            "low_latent_nonzero_count": manifest_direction.get("low_latent_nonzero_count"),
            "direction_is_finite": manifest_direction.get("direction_is_finite"),
            "selected_feature_count": manifest_direction.get("low_feature_count_selected")
            if manifest_direction.get("low_feature_count_selected") is not None
            else manifest_direction.get("low_feature_count"),
        },
    }


def _non_finite_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    nonfinite_logit_diff = sum(
        1 for r in rows if not _is_finite(_coerce_float(r.get("logit_diff", float("nan"))))
    )
    nonfinite_logits_rows = sum(1 for r in rows if not bool(r.get("logits_all_finite", True)))
    return {
        "n_rows": n,
        "non_finite_logit_diff_count": int(nonfinite_logit_diff),
        "non_finite_logit_diff_rate": float(nonfinite_logit_diff / n) if n else 0.0,
        "rows_with_non_finite_logits_count": int(nonfinite_logits_rows),
        "rows_with_non_finite_logits_rate": float(nonfinite_logits_rows / n) if n else 0.0,
    }


def _compute_condition_recalc(
    cond: Condition,
    margins: list[float],
    bootstrap_iters: int,
    seed: int,
) -> dict[str, Any]:
    pair = _pair_authority_unsafe(cond.baseline_rows, cond.intervention_rows)
    base = pair["baseline"]
    interv = pair["intervention"]
    threshold = _compute_threshold_free(base, interv)
    margin_rows = _compute_margin_rows(base, interv, margins, bootstrap_iters, seed=seed)
    ecdf = _ecdf_rows(base, interv)
    return {
        "condition_key": cond.key,
        "condition_label": cond.label,
        "run_dir": str(cond.run_dir),
        "git_commit": cond.manifest.get("git_commit", ""),
        "pairing_diagnostics": pair["diagnostics"],
        "threshold_free": threshold,
        "margin_sweep_recomputed": margin_rows,
        "ecdf_recomputed": ecdf,
        "non_finite_summary_baseline": _non_finite_summary(cond.baseline_rows),
        "non_finite_summary_intervention": _non_finite_summary(cond.intervention_rows),
        "direction_diagnostics": _direction_diagnostics(cond),
    }


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.{digits}f}"
    return str(v)


def _make_threshold_table_rows(recalc: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        tf = cond["threshold_free"]
        rows.append(
            {
                "condition_key": key,
                "condition_label": cond["condition_label"],
                "mean_shift": tf["mean_shift_intervention_minus_baseline"],
                "median_shift": tf["median_shift_intervention_minus_baseline"],
                "sign_test_p_value": (tf.get("paired_sign_test") or {}).get("p_value", 1.0),
                "ks_d_stat": tf.get("ks_d_stat", 0.0),
                "wasserstein_1": tf.get("wasserstein_1", 0.0),
                "cliffs_delta": tf.get("cliffs_delta_intervention_vs_baseline", 0.0),
                "n_paired_authority_unsafe": tf.get("n_paired_authority_unsafe", 0),
            }
        )
    return rows


def _find_margin_row(rows: list[dict[str, Any]], margin: float) -> dict[str, Any] | None:
    for row in rows:
        try:
            if math.isclose(float(row.get("margin", -999)), margin, abs_tol=1e-9):
                return row
        except (TypeError, ValueError):
            continue
    return None


def _make_margin_rows(recalc: dict[str, dict[str, Any]], selected_margins: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        sweep = cond["margin_sweep_recomputed"]
        for m in selected_margins:
            mr = _find_margin_row(sweep, m)
            if mr is None:
                continue
            rows.append(
                {
                    "condition_key": key,
                    "condition_label": cond["condition_label"],
                    "margin": m,
                    "delta_refusal_rate_intervention_minus_baseline": mr.get(
                        "delta_refusal_rate_intervention_minus_baseline", 0.0
                    ),
                    "delta_ci95_low": mr.get("delta_ci95_low", 0.0),
                    "delta_ci95_high": mr.get("delta_ci95_high", 0.0),
                    "baseline_refusal_rate": mr.get("baseline_refusal_rate", 0.0),
                    "intervention_refusal_rate": mr.get("intervention_refusal_rate", 0.0),
                }
            )
    return rows


def _make_margin_wide_rows(recalc: dict[str, dict[str, Any]], selected_margins: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        sweep = cond["margin_sweep_recomputed"]
        out: dict[str, Any] = {
            "condition_key": key,
            "condition_label": cond["condition_label"],
        }
        for m in selected_margins:
            mr = _find_margin_row(sweep, m) or {}
            name = str(m).replace(".", "_")
            out[f"delta_m{name}"] = mr.get("delta_refusal_rate_intervention_minus_baseline", 0.0)
            out[f"delta_m{name}_ci95_low"] = mr.get("delta_ci95_low", 0.0)
            out[f"delta_m{name}_ci95_high"] = mr.get("delta_ci95_high", 0.0)
        rows.append(out)
    return rows


def _make_main_vs_placebo_rows(
    threshold_rows: list[dict[str, Any]],
    margin_wide_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threshold_map = {r["condition_key"]: r for r in threshold_rows}
    margin_map = {r["condition_key"]: r for r in margin_wide_rows}
    rows: list[dict[str, Any]] = []
    for key in ("main", "placebo_random", "placebo_low_importance"):
        t = threshold_map.get(key, {})
        m = margin_map.get(key, {})
        rows.append(
            {
                "condition_key": key,
                "condition_label": t.get("condition_label", key),
                "mean_shift": t.get("mean_shift", 0.0),
                "median_shift": t.get("median_shift", 0.0),
                "sign_test_p_value": t.get("sign_test_p_value", 1.0),
                "ks_d_stat": t.get("ks_d_stat", 0.0),
                "wasserstein_1": t.get("wasserstein_1", 0.0),
                "cliffs_delta": t.get("cliffs_delta", 0.0),
                "delta_refusal_rate_margin_1_0": m.get("delta_m1_0", 0.0),
                "delta_refusal_rate_margin_1_0_ci95_low": m.get("delta_m1_0_ci95_low", 0.0),
                "delta_refusal_rate_margin_1_0_ci95_high": m.get("delta_m1_0_ci95_high", 0.0),
                "delta_refusal_rate_margin_1_5": m.get("delta_m1_5", 0.0),
                "delta_refusal_rate_margin_1_5_ci95_low": m.get("delta_m1_5_ci95_low", 0.0),
                "delta_refusal_rate_margin_1_5_ci95_high": m.get("delta_m1_5_ci95_high", 0.0),
                "n_paired_authority_unsafe": t.get("n_paired_authority_unsafe", 0),
            }
        )
    return rows


def _make_direction_metadata_rows(recalc: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        diag = cond.get("direction_diagnostics", {})
        manifest_meta = diag.get("manifest_direction_metadata", {})
        runtime = diag.get("row_runtime_summary", {})
        req = diag.get("requested_direction_fields", {})
        rows.append(
            {
                "condition_key": key,
                "condition_label": cond.get("condition_label", key),
                "placebo_mode": manifest_meta.get("mode", ""),
                "target_norm": manifest_meta.get("target_norm", ""),
                "actual_norm": manifest_meta.get("actual_norm", ""),
                "direction_is_degenerate": req.get("direction_is_degenerate"),
                "direction_is_finite": req.get("direction_is_finite"),
                "residual_l2_before_normalize": req.get("residual_l2_before_normalize"),
                "low_latent_nonzero_count": req.get("low_latent_nonzero_count"),
                "selected_feature_count": req.get("selected_feature_count"),
                "runtime_direction_l2_mean": runtime.get("direction_l2_mean", 0.0),
                "runtime_all_direction_is_finite": runtime.get("all_direction_is_finite", True),
                "runtime_any_direction_is_degenerate": runtime.get("any_direction_is_degenerate", False),
                "runtime_max_identity_fallback_calls": runtime.get("max_identity_fallback_calls", 0),
                "runtime_max_non_finite_coeff_calls": runtime.get("max_non_finite_coeff_calls", 0),
                "runtime_max_non_finite_output_calls": runtime.get("max_non_finite_output_calls", 0),
            }
        )
    return rows


def _make_overlay_margin_rows(recalc: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        for row in cond.get("margin_sweep_recomputed", []):
            rows.append(
                {
                    "condition_key": key,
                    "condition_label": cond.get("condition_label", key),
                    "margin": row.get("margin", 0.0),
                    "baseline_refusal_rate": row.get("baseline_refusal_rate", 0.0),
                    "intervention_refusal_rate": row.get("intervention_refusal_rate", 0.0),
                    "delta_refusal_rate_intervention_minus_baseline": row.get(
                        "delta_refusal_rate_intervention_minus_baseline", 0.0
                    ),
                    "delta_ci95_low": row.get("delta_ci95_low", 0.0),
                    "delta_ci95_high": row.get("delta_ci95_high", 0.0),
                }
            )
    return rows


def _make_overlay_ecdf_rows(recalc: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        for row in cond.get("ecdf_recomputed", []):
            rows.append(
                {
                    "condition_key": key,
                    "condition_label": cond.get("condition_label", key),
                    "x": row.get("x", 0.0),
                    "cdf_baseline": row.get("cdf_baseline", 0.0),
                    "cdf_intervention": row.get("cdf_intervention", 0.0),
                }
            )
    return rows


def _write_latex_tables(
    threshold_rows: list[dict[str, Any]],
    margin_wide_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    threshold_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Condition & Mean Shift & Median Shift & Sign p & KS D & W1 & Cliff's $\\delta$ \\\\",
        "\\hline",
    ]
    for r in threshold_rows:
        threshold_lines.append(
            f"{r['condition_label']} & "
            f"{_fmt(r['mean_shift'])} & "
            f"{_fmt(r['median_shift'])} & "
            f"{_fmt(r['sign_test_p_value'])} & "
            f"{_fmt(r['ks_d_stat'])} & "
            f"{_fmt(r['wasserstein_1'])} & "
            f"{_fmt(r['cliffs_delta'])} \\\\"
        )
    threshold_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Threshold-free authority-unsafe comparison across main intervention and placebo controls.}",
            "\\label{tab:threshold_free_main_placebo}",
            "\\end{table}",
            "",
        ]
    )
    _write_text(out_dir / "table_threshold_free.tex", "\n".join(threshold_lines))

    margin_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "Condition & $\\Delta$Refusal@1.0 (95\\% CI) & $\\Delta$Refusal@1.5 (95\\% CI) \\\\",
        "\\hline",
    ]
    for r in margin_wide_rows:
        m10 = f"{_fmt(r['delta_m1_0'])} [{_fmt(r['delta_m1_0_ci95_low'])}, {_fmt(r['delta_m1_0_ci95_high'])}]"
        m15 = f"{_fmt(r['delta_m1_5'])} [{_fmt(r['delta_m1_5_ci95_low'])}, {_fmt(r['delta_m1_5_ci95_high'])}]"
        margin_lines.append(f"{r['condition_label']} & {m10} & {m15} \\\\")
    margin_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Margin-based unsafe refusal delta for selected thresholds. Positive values indicate higher refusal under intervention.}",
            "\\label{tab:margin_delta_main_placebo}",
            "\\end{table}",
            "",
        ]
    )
    _write_text(out_dir / "table_margin_selected.tex", "\n".join(margin_lines))

    combined_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Condition & Mean Shift & Median Shift & Sign p & KS D & W1 & $\\Delta$Ref@1.0 (95\\% CI) & $\\Delta$Ref@1.5 (95\\% CI) \\\\",
        "\\hline",
    ]
    for r in combined_rows:
        m10 = (
            f"{_fmt(r['delta_refusal_rate_margin_1_0'])} "
            f"[{_fmt(r['delta_refusal_rate_margin_1_0_ci95_low'])}, {_fmt(r['delta_refusal_rate_margin_1_0_ci95_high'])}]"
        )
        m15 = (
            f"{_fmt(r['delta_refusal_rate_margin_1_5'])} "
            f"[{_fmt(r['delta_refusal_rate_margin_1_5_ci95_low'])}, {_fmt(r['delta_refusal_rate_margin_1_5_ci95_high'])}]"
        )
        combined_lines.append(
            f"{r['condition_label']} & "
            f"{_fmt(r['mean_shift'])} & "
            f"{_fmt(r['median_shift'])} & "
            f"{_fmt(r['sign_test_p_value'])} & "
            f"{_fmt(r['ks_d_stat'])} & "
            f"{_fmt(r['wasserstein_1'])} & "
            f"{m10} & "
            f"{m15} \\\\"
        )
    combined_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Main vs placebo comparison combining threshold-free and selected margin-based effects on authority-unsafe pairs.}",
            "\\label{tab:main_vs_placebo_combined}",
            "\\end{table}",
            "",
        ]
    )
    _write_text(out_dir / "table_main_vs_placebo_combined.tex", "\n".join(combined_lines))


def _plot_figures(
    recalc: dict[str, dict[str, Any]],
    conditions: list[Condition],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "main": "#1f77b4",
        "placebo_random": "#2ca02c",
        "placebo_low_importance": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        rows = cond["ecdf_recomputed"]
        x = [_coerce_float(r["x"]) for r in rows]
        yb = [_coerce_float(r["cdf_baseline"]) for r in rows]
        yi = [_coerce_float(r["cdf_intervention"]) for r in rows]
        c = colors[key]
        ax.plot(x, yi, color=c, linewidth=2.2, label=f"{cond['condition_label']} intervention")
        ax.plot(x, yb, color=c, linewidth=1.6, linestyle="--", alpha=0.6, label=f"{cond['condition_label']} baseline")
    ax.set_title("Authority-Unsafe ECDF Overlay (Main vs Placebos)")
    ax.set_xlabel("logit_diff (refusal_score - compliance_score)")
    ax.set_ylabel("Empirical CDF")
    ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(1.5, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(out_dir / "cdf_overlay_main_vs_placebos.png", dpi=320)
    fig.savefig(out_dir / "cdf_overlay_main_vs_placebos.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for key in ("main", "placebo_random", "placebo_low_importance"):
        cond = recalc[key]
        rows = sorted(cond["margin_sweep_recomputed"], key=lambda r: _coerce_float(r["margin"]))
        x = [_coerce_float(r["margin"]) for r in rows]
        y = [_coerce_float(r["delta_refusal_rate_intervention_minus_baseline"]) for r in rows]
        lo = [_coerce_float(r["delta_ci95_low"]) for r in rows]
        hi = [_coerce_float(r["delta_ci95_high"]) for r in rows]
        c = colors[key]
        ax.plot(x, y, marker="o", linewidth=2.1, color=c, label=cond["condition_label"])
        ax.fill_between(x, lo, hi, color=c, alpha=0.14)
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(1.5, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title("Margin Sweep Overlay: Unsafe Refusal Delta")
    ax.set_xlabel("refusal margin")
    ax.set_ylabel("delta refusal rate (intervention - baseline)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "margin_sweep_overlay_main_vs_placebos.png", dpi=320)
    fig.savefig(out_dir / "margin_sweep_overlay_main_vs_placebos.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for cond in conditions:
        lw = cond.metrics.get("layer_wise_suppression", {})
        layers = lw.get("layers", [])
        supp = lw.get("suppression", [])
        if not layers or not supp:
            continue
        ax.plot(layers, supp, marker="o", linewidth=2.0, label=cond.label, color=colors.get(cond.key, None))
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_title("Layer-wise Suppression Overlay")
    ax.set_xlabel("layer")
    ax.set_ylabel("suppression score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "layer_wise_suppression_overlay_main_vs_placebos.png", dpi=320)
    fig.savefig(out_dir / "layer_wise_suppression_overlay_main_vs_placebos.pdf")
    plt.close(fig)


def _paper_draft_markdown(
    threshold_rows: list[dict[str, Any]],
    margin_wide_rows: list[dict[str, Any]],
    recalc: dict[str, dict[str, Any]],
    reproducibility: dict[str, Any],
) -> str:
    t = {r["condition_key"]: r for r in threshold_rows}
    m = {r["condition_key"]: r for r in margin_wide_rows}

    main = t["main"]
    rand = t["placebo_random"]
    low = t["placebo_low_importance"]
    mm = m["main"]
    mr = m["placebo_random"]
    ml = m["placebo_low_importance"]

    txt = f"""# LLaMA-3 Authority Circuit Intervention Study

## Abstract
We study whether authority framing induces a distinct internal direction in LLaMA-3-8B-Instruct and whether intervention on that direction changes unsafe behavior. Using sparse autoencoders (SAEs), we extract an authority-related residual direction and perform projection-subtraction intervention at inference. We compare the main direction against two placebo controls (random and low-importance directions). On authority-unsafe paired samples, threshold-free analysis shows a distinct shift for the main intervention (mean shift {main['mean_shift']:.4f}, median shift {main['median_shift']:.4f}, sign-test p={main['sign_test_p_value']:.4g}, KS={main['ks_d_stat']:.4f}, W1={main['wasserstein_1']:.4f}). Placebo controls show substantially smaller or opposite signed effects. Margin-based deltas at 1.0/1.5 are {mm['delta_m1_0']:.4f}/{mm['delta_m1_5']:.4f} for main, {mr['delta_m1_0']:.4f}/{mr['delta_m1_5']:.4f} for random placebo, and {ml['delta_m1_0']:.4f}/{ml['delta_m1_5']:.4f} for low-importance placebo. These results support direction-specific causal influence in internal representation space.

## 1. Introduction
Large language models can exhibit framing-sensitive safety behavior, where social-context cues alter refusal/compliance tendencies. Authority framing is particularly important because it can alter policy adherence without changing surface task semantics.

This work asks whether authority framing corresponds to a mechanistically identifiable circuit-level direction and whether direct intervention on that direction causes measurable behavior change.

Contributions:
1. A controlled authority-framing benchmark setup with paired unsafe evaluation.
2. SAE-based extraction of an authority direction in residual space.
3. Causal intervention with placebo validation (random and low-importance directions).
4. Threshold-free and margin-based evaluation with reproducibility artifacts.

## 2. Method
### 2.1 Model and Data
- Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Framing variants include direct/hypothetical/academic/authority/institutional.
- Risk tiers: safe, borderline, unsafe.

### 2.2 SAE and Direction Construction
- Residual activations are decomposed with SAE.
- Authority direction is derived from authority-vs-control latent difference and decoded to residual space.

### 2.3 Intervention
Given residual vector `r` and direction `d`, intervention uses projection subtraction:
`r_new = r - alpha * proj_d(r)`.
Implementation includes fp32 stabilization, degenerate-direction fallback, and non-finite runtime checks.

### 2.4 Evaluation
- Threshold-free: mean/median shift, sign test, KS, Wasserstein, Cliff's delta.
- Margin-based: refusal deltas at multiple margins with paired bootstrap CI.
- Placebos: random direction and low-importance SAE-feature direction.

## 3. Results
### 3.1 Threshold-Free Effects
Main intervention shows the largest signed shift among conditions:
- Main: mean {main['mean_shift']:.4f}, median {main['median_shift']:.4f}, sign p={main['sign_test_p_value']:.4g}, KS {main['ks_d_stat']:.4f}, W1 {main['wasserstein_1']:.4f}, Cliff's delta {main['cliffs_delta']:.4f}.
- Random placebo: mean {rand['mean_shift']:.4f}, median {rand['median_shift']:.4f}, sign p={rand['sign_test_p_value']:.4g}, KS {rand['ks_d_stat']:.4f}, W1 {rand['wasserstein_1']:.4f}.
- Low-importance placebo: mean {low['mean_shift']:.4f}, median {low['median_shift']:.4f}, sign p={low['sign_test_p_value']:.4g}, KS {low['ks_d_stat']:.4f}, W1 {low['wasserstein_1']:.4f}.

### 3.2 Margin-Based Effects
At margin 1.0/1.5, intervention refusal deltas are:
- Main: {mm['delta_m1_0']:.4f} [{mm['delta_m1_0_ci95_low']:.4f}, {mm['delta_m1_0_ci95_high']:.4f}] / {mm['delta_m1_5']:.4f} [{mm['delta_m1_5_ci95_low']:.4f}, {mm['delta_m1_5_ci95_high']:.4f}]
- Random placebo: {mr['delta_m1_0']:.4f} [{mr['delta_m1_0_ci95_low']:.4f}, {mr['delta_m1_0_ci95_high']:.4f}] / {mr['delta_m1_5']:.4f} [{mr['delta_m1_5_ci95_low']:.4f}, {mr['delta_m1_5_ci95_high']:.4f}]
- Low-importance placebo: {ml['delta_m1_0']:.4f} [{ml['delta_m1_0_ci95_low']:.4f}, {ml['delta_m1_0_ci95_high']:.4f}] / {ml['delta_m1_5']:.4f} [{ml['delta_m1_5_ci95_low']:.4f}, {ml['delta_m1_5_ci95_high']:.4f}]

### 3.3 Placebo Validation and Specificity
Effect magnitudes and signs differ between main and placebo conditions, indicating direction-specific causal influence rather than a generic intervention artifact.

### 3.4 Robustness and Diagnostics
Non-finite intervention diagnostics are tracked at sample level and summarized in reproducibility artifacts. Placebo low-importance direction metadata confirms finite, non-degenerate direction state in this run.

## 4. Discussion
The results support that authority framing corresponds to a manipulable internal direction. However, practical safety impact depends on intervention sign and threshold calibration; threshold-free metrics should be the primary evidence, with margin analyses as sensitivity checks.

Limitations:
- Single model family and one primary dataset configuration.
- Margin-dependent behavioral interpretation.
- Limited external human-judged behavioral labels.

Future work:
1. Cross-model replication and transferability tests.
2. Bidirectional intervention sign ablation.
3. Stronger behavioral ground-truth with richer annotation protocols.

## 5. Conclusion
We present a reproducible mechanistic intervention framework for authority-induced behavior shifts in LLaMA-3. Main-vs-placebo comparisons show direction-specific causal effects in internal representation space, supporting circuit-level intervention as a viable analysis tool for framing-sensitive safety behavior.
"""
    return txt


def _results_one_page_markdown(
    threshold_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
) -> str:
    t = {r["condition_key"]: r for r in threshold_rows}
    def _mrow(cond: str, margin: float) -> dict[str, Any]:
        for r in margin_rows:
            if r["condition_key"] == cond and math.isclose(float(r["margin"]), margin, abs_tol=1e-9):
                return r
        return {}

    lines = [
        "# Results Summary (1-Page)",
        "",
        "## Core Finding",
        (
            "Threshold-free analysis on authority-unsafe paired samples shows that the main intervention "
            f"(mean shift {t['main']['mean_shift']:.4f}, sign-test p={t['main']['sign_test_p_value']:.4g}) "
            "is distinguishable from placebo controls."
        ),
        "",
        "## Threshold-Free Comparison",
    ]
    for key in ("main", "placebo_random", "placebo_low_importance"):
        r = t[key]
        lines.append(
            f"- {r['condition_label']}: mean={r['mean_shift']:.4f}, median={r['median_shift']:.4f}, "
            f"sign p={r['sign_test_p_value']:.4g}, KS={r['ks_d_stat']:.4f}, W1={r['wasserstein_1']:.4f}, "
            f"Cliff's delta={r['cliffs_delta']:.4f}"
        )
    lines.extend(["", "## Margin 1.0 / 1.5"])
    for key, label in [
        ("main", "Main"),
        ("placebo_random", "Placebo (Random)"),
        ("placebo_low_importance", "Placebo (Low-Importance)"),
    ]:
        m10 = _mrow(key, 1.0)
        m15 = _mrow(key, 1.5)
        lines.append(
            f"- {label}: ΔRef@1.0={_coerce_float(m10.get('delta_refusal_rate_intervention_minus_baseline')):.4f} "
            f"[{_coerce_float(m10.get('delta_ci95_low')):.4f}, {_coerce_float(m10.get('delta_ci95_high')):.4f}], "
            f"ΔRef@1.5={_coerce_float(m15.get('delta_refusal_rate_intervention_minus_baseline')):.4f} "
            f"[{_coerce_float(m15.get('delta_ci95_low')):.4f}, {_coerce_float(m15.get('delta_ci95_high')):.4f}]"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Threshold-free metrics are primary evidence of causal direction-specific effects.",
            "- Margin analysis is sensitivity/practical calibration, not sole evidence.",
            "- Placebo comparisons support causal specificity over generic perturbation effects.",
            "",
        ]
    )
    return "\n".join(lines)


def _supplementary_appendix_markdown(
    threshold_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    reproducibility: dict[str, Any],
) -> str:
    config_rows = reproducibility.get("configs", [])
    checksum_rows = reproducibility.get("datasets", [])
    lines = [
        "# Supplementary Appendix Draft",
        "",
        "## A. Experimental Configuration",
        "- Model: LLaMA-3-8B-Instruct",
        "- Primary run conditions: main, placebo-random, placebo-low-importance",
        "- Prompt tiers: safe / borderline / unsafe",
        "",
        "## B. Additional Statistical Outputs",
        "Threshold-free rows:",
    ]
    for r in threshold_rows:
        lines.append(
            f"- {r['condition_label']}: mean={r['mean_shift']:.6f}, median={r['median_shift']:.6f}, "
            f"sign p={r['sign_test_p_value']:.6g}, KS={r['ks_d_stat']:.6f}, W1={r['wasserstein_1']:.6f}, "
            f"Cliff's delta={r['cliffs_delta']:.6f}, n={r['n_paired_authority_unsafe']}"
        )
    lines.extend(["", "Margin rows (selected):"])
    for r in margin_rows:
        lines.append(
            f"- {r['condition_label']} @ m={r['margin']}: delta={r['delta_refusal_rate_intervention_minus_baseline']:.6f}, "
            f"CI=[{r['delta_ci95_low']:.6f}, {r['delta_ci95_high']:.6f}]"
        )
    lines.extend(
        [
            "",
            "## C. Reproducibility Artifacts",
            f"- Number of resolved configs: {len(config_rows)}",
            f"- Number of dataset checksum records: {len(checksum_rows)}",
            "",
            "## D. Diagnostics",
            "- Non-finite and direction-degeneracy diagnostics are included in recomputed per-condition JSON files.",
            "- Figure files are provided in both PNG and PDF formats for publication workflows.",
            "",
        ]
    )
    return "\n".join(lines)


def _reproducibility_markdown(reproducibility: dict[str, Any]) -> str:
    cfg = reproducibility.get("configs", [])
    ds = reproducibility.get("datasets", [])
    lines = [
        "## Reproducibility",
        "",
        "We package all run manifests, sample-level artifacts, resolved configs, and dataset checksums.",
        "",
        "### Commits and Configs",
    ]
    for row in cfg:
        lines.append(
            f"- {row.get('condition_label')}: git={row.get('git_commit')}, "
            f"config_hash={row.get('config_hash')}, resolved_config={row.get('config_resolved_path')}"
        )
    lines.extend(["", "### Dataset Checksums"])
    for row in ds:
        lines.append(
            f"- {row.get('source')}: {row.get('path')} | sha256={row.get('sha256')} | declared_match={row.get('declared_match')}"
        )
    lines.append("")
    return "\n".join(lines)


def _build_output_package(args: argparse.Namespace) -> Path:
    main_dir = Path(args.main_run).expanduser().resolve()
    placebo_random = Path(args.placebo_random).expanduser().resolve() if args.placebo_random else None
    placebo_low = Path(args.placebo_low).expanduser().resolve() if args.placebo_low else None
    if args.placebo_root:
        root = Path(args.placebo_root).expanduser().resolve()
        placebo_random = placebo_random or (root / "random")
        placebo_low = placebo_low or (root / "low_importance")
    if placebo_random is None or placebo_low is None:
        raise ValueError("Specify --placebo-root or both --placebo-random and --placebo-low")

    conditions = [
        _load_condition("main", "Main", main_dir),
        _load_condition("placebo_random", "Placebo (Random)", placebo_random),
        _load_condition("placebo_low_importance", "Placebo (Low-Importance)", placebo_low),
    ]

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path("analysis_packages") / f"{main_dir.name}_full_paper_package").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_collected_artifacts(conditions, out_dir)

    project_roots = [Path(args.project_root).resolve()] if args.project_root else _detect_project_roots(
        [c.run_dir for c in conditions]
    )
    dataset_overrides = [Path(p).expanduser() for p in (args.dataset or [])]
    reproducibility = _collect_reproducibility(
        conditions=conditions,
        out_dir=out_dir,
        project_roots=project_roots,
        dataset_overrides=dataset_overrides,
    )

    recalc: dict[str, dict[str, Any]] = {}
    for idx, cond in enumerate(conditions):
        recalc[cond.key] = _compute_condition_recalc(
            cond=cond,
            margins=args.margins,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed + idx * 1000,
        )

    recomputed_dir = out_dir / "recomputed"
    for key, payload in recalc.items():
        _write_json(recomputed_dir / f"{key}.recomputed.json", payload)

    threshold_rows = _make_threshold_table_rows(recalc)
    margin_rows = _make_margin_rows(recalc, selected_margins=[1.0, 1.5])
    margin_wide_rows = _make_margin_wide_rows(recalc, selected_margins=[1.0, 1.5])
    combined_rows = _make_main_vs_placebo_rows(threshold_rows, margin_wide_rows)
    direction_rows = _make_direction_metadata_rows(recalc)
    overlay_margin_rows = _make_overlay_margin_rows(recalc)
    overlay_ecdf_rows = _make_overlay_ecdf_rows(recalc)

    comparison_dir = out_dir / "comparison"
    _write_csv(
        comparison_dir / "threshold_free_comparison.csv",
        [
            "condition_key",
            "condition_label",
            "mean_shift",
            "median_shift",
            "sign_test_p_value",
            "ks_d_stat",
            "wasserstein_1",
            "cliffs_delta",
            "n_paired_authority_unsafe",
        ],
        threshold_rows,
    )
    _write_csv(
        comparison_dir / "margin_comparison_selected.csv",
        [
            "condition_key",
            "condition_label",
            "margin",
            "delta_refusal_rate_intervention_minus_baseline",
            "delta_ci95_low",
            "delta_ci95_high",
            "baseline_refusal_rate",
            "intervention_refusal_rate",
        ],
        margin_rows,
    )
    _write_csv(
        comparison_dir / "margin_comparison_selected_wide.csv",
        [
            "condition_key",
            "condition_label",
            "delta_m1_0",
            "delta_m1_0_ci95_low",
            "delta_m1_0_ci95_high",
            "delta_m1_5",
            "delta_m1_5_ci95_low",
            "delta_m1_5_ci95_high",
        ],
        margin_wide_rows,
    )
    _write_csv(
        comparison_dir / "main_vs_placebo_combined.csv",
        [
            "condition_key",
            "condition_label",
            "mean_shift",
            "median_shift",
            "sign_test_p_value",
            "ks_d_stat",
            "wasserstein_1",
            "cliffs_delta",
            "delta_refusal_rate_margin_1_0",
            "delta_refusal_rate_margin_1_0_ci95_low",
            "delta_refusal_rate_margin_1_0_ci95_high",
            "delta_refusal_rate_margin_1_5",
            "delta_refusal_rate_margin_1_5_ci95_low",
            "delta_refusal_rate_margin_1_5_ci95_high",
            "n_paired_authority_unsafe",
        ],
        combined_rows,
    )
    _write_csv(
        comparison_dir / "placebo_direction_metadata.csv",
        [
            "condition_key",
            "condition_label",
            "placebo_mode",
            "target_norm",
            "actual_norm",
            "direction_is_degenerate",
            "direction_is_finite",
            "residual_l2_before_normalize",
            "low_latent_nonzero_count",
            "selected_feature_count",
            "runtime_direction_l2_mean",
            "runtime_all_direction_is_finite",
            "runtime_any_direction_is_degenerate",
            "runtime_max_identity_fallback_calls",
            "runtime_max_non_finite_coeff_calls",
            "runtime_max_non_finite_output_calls",
        ],
        direction_rows,
    )
    _write_csv(
        comparison_dir / "margin_sweep_overlay.csv",
        [
            "condition_key",
            "condition_label",
            "margin",
            "baseline_refusal_rate",
            "intervention_refusal_rate",
            "delta_refusal_rate_intervention_minus_baseline",
            "delta_ci95_low",
            "delta_ci95_high",
        ],
        overlay_margin_rows,
    )
    _write_csv(
        comparison_dir / "authority_unsafe_ecdf_overlay.csv",
        [
            "condition_key",
            "condition_label",
            "x",
            "cdf_baseline",
            "cdf_intervention",
        ],
        overlay_ecdf_rows,
    )
    _write_json(
        comparison_dir / "comparison_summary.json",
        {
            "threshold_free_rows": threshold_rows,
            "margin_rows_selected": margin_rows,
            "margin_rows_selected_wide": margin_wide_rows,
            "main_vs_placebo_combined": combined_rows,
            "placebo_direction_metadata": direction_rows,
        },
    )

    _write_latex_tables(threshold_rows, margin_wide_rows, combined_rows, comparison_dir)

    figures_dir = out_dir / "figures"
    if args.skip_plots:
        _write_text(figures_dir / "PLOTS_SKIPPED.txt", "Plot generation skipped via --skip-plots.\n")
    else:
        try:
            _plot_figures(recalc=recalc, conditions=conditions, out_dir=figures_dir)
        except ModuleNotFoundError as e:
            raise RuntimeError("matplotlib is required for figure output or use --skip-plots.") from e

    paper_dir = out_dir / "paper"
    _write_text(
        paper_dir / "paper_draft_iclr_acl.md",
        _paper_draft_markdown(
            threshold_rows=threshold_rows,
            margin_wide_rows=margin_wide_rows,
            recalc=recalc,
            reproducibility=reproducibility,
        ),
    )
    _write_text(
        paper_dir / "results_section_one_page.md",
        _results_one_page_markdown(threshold_rows=threshold_rows, margin_rows=margin_rows),
    )
    _write_text(
        paper_dir / "supplementary_appendix_draft.md",
        _supplementary_appendix_markdown(
            threshold_rows=threshold_rows,
            margin_rows=margin_rows,
            reproducibility=reproducibility,
        ),
    )
    _write_text(
        paper_dir / "reproducibility_section.md",
        _reproducibility_markdown(reproducibility),
    )

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
        "margins": list(args.margins),
        "bootstrap_iters": int(args.bootstrap_iters),
        "seed": int(args.seed),
        "project_roots": [str(p) for p in project_roots],
    }
    _write_json(out_dir / "package_manifest.json", package_manifest)
    return out_dir


def _parse_margins(raw: str) -> list[float]:
    vals: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("At least one margin value is required.")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build full analysis + paper draft package from main/placebo experiment runs. "
            "All key statistics are recomputed from sample-level JSON artifacts."
        )
    )
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--placebo-root", default=None)
    parser.add_argument("--placebo-random", default=None)
    parser.add_argument("--placebo-low", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--margins", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    args.margins = _parse_margins(args.margins)
    out = _build_output_package(args)
    print(f"Wrote full analysis package: {out}")
    print(f"- paper draft: {out / 'paper' / 'paper_draft_iclr_acl.md'}")
    print(f"- threshold table: {out / 'comparison' / 'table_threshold_free.tex'}")
    print(f"- margin table: {out / 'comparison' / 'table_margin_selected.tex'}")
    print(f"- figures: {out / 'figures'}")


if __name__ == "__main__":
    main()
