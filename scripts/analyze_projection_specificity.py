#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class GroupComparison:
    key: str
    label: str
    a_filter: Callable[[dict[str, Any]], bool]
    b_filter: Callable[[dict[str, Any]], bool]


@dataclass
class KSTestResult:
    statistic: float
    pvalue: float


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _isfinite(v: float) -> bool:
    return math.isfinite(v)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, (len(a) + len(b) - 2))
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    gt = 0
    lt = 0
    for av in a:
        gt += int(np.sum(av > b))
        lt += int(np.sum(av < b))
    total = len(a) * len(b)
    return float((gt - lt) / total) if total else 0.0


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> KSTestResult:
    if len(a) == 0 or len(b) == 0:
        return KSTestResult(statistic=0.0, pvalue=1.0)

    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    values = np.sort(np.concatenate([a, b]))

    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))

    # Smirnov asymptotic approximation, adequate here for ranking/effect reporting.
    n1 = len(a)
    n2 = len(b)
    en = math.sqrt(n1 * n2 / (n1 + n2))
    if en <= 0 or d <= 0:
        return KSTestResult(statistic=d, pvalue=1.0)

    lam = (en + 0.12 + 0.11 / en) * d
    series = 0.0
    for k in range(1, 101):
        term = (-1) ** (k - 1) * math.exp(-2.0 * (lam**2) * (k**2))
        series += term
        if abs(term) < 1e-12:
            break
    pvalue = max(0.0, min(1.0, 2.0 * series))
    return KSTestResult(statistic=d, pvalue=pvalue)


def _bootstrap_ci_mean_diff_and_d(
    a: np.ndarray,
    b: np.ndarray,
    iters: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {
            "mean_diff_ci95_low": 0.0,
            "mean_diff_ci95_high": 0.0,
            "cohens_d_ci95_low": 0.0,
            "cohens_d_ci95_high": 0.0,
        }
    rng = np.random.default_rng(seed)
    mean_diffs: list[float] = []
    ds: list[float] = []
    for _ in range(iters):
        aa = a[rng.integers(0, len(a), size=len(a))]
        bb = b[rng.integers(0, len(b), size=len(b))]
        mean_diffs.append(float(np.mean(aa) - np.mean(bb)))
        ds.append(_cohens_d(aa, bb))
    return {
        "mean_diff_ci95_low": float(np.percentile(mean_diffs, 2.5)),
        "mean_diff_ci95_high": float(np.percentile(mean_diffs, 97.5)),
        "cohens_d_ci95_low": float(np.percentile(ds, 2.5)),
        "cohens_d_ci95_high": float(np.percentile(ds, 97.5)),
    }


def _permutation_p_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    iters: int = 20000,
    seed: int = 42,
) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1.0
    obs = abs(float(np.mean(a) - np.mean(b)))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(iters):
        perm = rng.permutation(combined)
        stat = abs(float(np.mean(perm[:n_a]) - np.mean(perm[n_a:])))
        if stat >= obs - 1e-12:
            ge += 1
    return float((ge + 1) / (iters + 1))


def _bh_fdr(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    arr = np.asarray(pvals, dtype=np.float64)
    m = len(arr)
    order = np.argsort(arr)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank_idx in range(m - 1, -1, -1):
        j = int(order[rank_idx])
        rank = rank_idx + 1
        val = float(arr[j] * m / rank)
        prev = min(prev, val)
        q[j] = min(prev, 1.0)
    return [float(x) for x in q]


def _load_direction(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if "residual_direction_normalized" not in payload:
        raise ValueError(f"Direction file missing residual_direction_normalized: {path}")
    d = payload["residual_direction_normalized"]
    if not isinstance(d, torch.Tensor):
        raise ValueError("residual_direction_normalized is not a tensor")
    return d.detach().to(dtype=torch.float32)


def _collect_projection_rows(
    activation_dir: Path,
    direction: torch.Tensor,
    hook_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d = direction
    for path in sorted(activation_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("metadata", {})
        rs = payload.get("residual_stream", {})
        if hook_key not in rs:
            continue
        t = rs[hook_key]
        if not isinstance(t, torch.Tensor) or t.ndim != 3 or t.shape[0] != 1:
            continue
        v = t[0, -1, :].to(dtype=torch.float32)
        if v.numel() != d.numel():
            continue
        proj = float(torch.dot(v, d).item())  # direction already normalized
        row = {
            "source_file": str(path),
            "prompt_id": str(meta.get("prompt_id", path.stem)),
            "semantic_request_id": str(meta.get("semantic_request_id", "")),
            "framing_type": str(meta.get("framing_type", "")),
            "risk_tier": str(meta.get("risk_tier", "")),
            "safety_label": str(meta.get("safety_label", "")),
            "projection_value": proj,
        }
        rows.append(row)
    return rows


def _fit_ols_authority_unsafe(rows: list[dict[str, Any]]) -> dict[str, float]:
    # projection ~ 1 + authority + unsafe + authority*unsafe
    X = []
    y = []
    for r in rows:
        p = _safe_float(r.get("projection_value"))
        if not _isfinite(p):
            continue
        authority = 1.0 if str(r.get("framing_type")) == "authority" else 0.0
        unsafe = 1.0 if str(r.get("risk_tier")) == "unsafe" else 0.0
        X.append([1.0, authority, unsafe, authority * unsafe])
        y.append(p)
    if not X:
        return {
            "beta_intercept": 0.0,
            "beta_authority": 0.0,
            "beta_unsafe": 0.0,
            "beta_authority_x_unsafe": 0.0,
            "n_rows": 0.0,
        }
    x = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    return {
        "beta_intercept": float(beta[0]),
        "beta_authority": float(beta[1]),
        "beta_unsafe": float(beta[2]),
        "beta_authority_x_unsafe": float(beta[3]),
        "n_rows": float(len(y)),
    }


def _group_stats(rows: list[dict[str, Any]], cmp: GroupComparison) -> dict[str, Any]:
    a_vals = np.asarray(
        [_safe_float(r["projection_value"]) for r in rows if cmp.a_filter(r)],
        dtype=np.float64,
    )
    b_vals = np.asarray(
        [_safe_float(r["projection_value"]) for r in rows if cmp.b_filter(r)],
        dtype=np.float64,
    )
    a_vals = a_vals[np.isfinite(a_vals)]
    b_vals = b_vals[np.isfinite(b_vals)]
    if len(a_vals) == 0 or len(b_vals) == 0:
        return {
            "comparison_key": cmp.key,
            "comparison_label": cmp.label,
            "n_a": int(len(a_vals)),
            "n_b": int(len(b_vals)),
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_diff_a_minus_b": 0.0,
            "cohens_d": 0.0,
            "cliffs_delta": 0.0,
            "ks_stat": 0.0,
            "ks_pvalue": 1.0,
            "perm_p_mean_diff": 1.0,
            "mean_diff_ci95_low": 0.0,
            "mean_diff_ci95_high": 0.0,
            "cohens_d_ci95_low": 0.0,
            "cohens_d_ci95_high": 0.0,
        }
    ks = _ks_2samp(a_vals, b_vals)
    ci = _bootstrap_ci_mean_diff_and_d(a_vals, b_vals, iters=10000, seed=42)
    return {
        "comparison_key": cmp.key,
        "comparison_label": cmp.label,
        "n_a": int(len(a_vals)),
        "n_b": int(len(b_vals)),
        "mean_a": float(np.mean(a_vals)),
        "mean_b": float(np.mean(b_vals)),
        "mean_diff_a_minus_b": float(np.mean(a_vals) - np.mean(b_vals)),
        "cohens_d": _cohens_d(a_vals, b_vals),
        "cliffs_delta": _cliffs_delta(a_vals, b_vals),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "perm_p_mean_diff": _permutation_p_mean_diff(a_vals, b_vals, iters=20000, seed=43),
        **ci,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Comparison & Mean Diff & Cohen $d$ & KS $D$ & Perm-$p$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['comparison_label']} & "
            f"{float(r['mean_diff_a_minus_b']):.4f} "
            f"[{float(r['mean_diff_ci95_low']):.4f}, {float(r['mean_diff_ci95_high']):.4f}] & "
            f"{float(r['cohens_d']):.4f} "
            f"[{float(r['cohens_d_ci95_low']):.4f}, {float(r['cohens_d_ci95_high']):.4f}] & "
            f"{float(r['ks_stat']):.4f} & "
            f"{float(r['perm_p_mean_diff']):.2e} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Projection specificity tests using $\\langle r, d_{auth}\\rangle$ on baseline activations (layer 10, post-residual).}",
            "\\label{tab:projection_specificity}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_distributions(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vals = np.asarray([_safe_float(r["projection_value"]) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    auth = np.asarray(
        [_safe_float(r["projection_value"]) for r in rows if str(r["framing_type"]) == "authority"],
        dtype=np.float64,
    )
    non = np.asarray(
        [_safe_float(r["projection_value"]) for r in rows if str(r["framing_type"]) != "authority"],
        dtype=np.float64,
    )
    auth = auth[np.isfinite(auth)]
    non = non[np.isfinite(non)]

    bins = np.linspace(float(np.min(vals)), float(np.max(vals)), 50)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(non, bins=bins, density=True, alpha=0.4, label="Non-authority", color="#1f77b4")
    ax.hist(auth, bins=bins, density=True, alpha=0.4, label="Authority", color="#d62728")
    ax.axvline(float(np.mean(non)), color="#1f77b4", linestyle="--", linewidth=1.5)
    ax.axvline(float(np.mean(auth)), color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Projection Distribution: Authority vs Non-authority")
    ax.set_xlabel(r"$\langle r, d_{auth} \rangle$")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "projection_authority_vs_non_authority.pdf")
    fig.savefig(out_dir / "projection_authority_vs_non_authority.png", dpi=300)
    plt.close(fig)

    tiers = ["safe", "borderline", "unsafe"]
    data_auth = [
        [
            _safe_float(r["projection_value"])
            for r in rows
            if str(r["framing_type"]) == "authority" and str(r["risk_tier"]) == tier
        ]
        for tier in tiers
    ]
    data_non = [
        [
            _safe_float(r["projection_value"])
            for r in rows
            if str(r["framing_type"]) != "authority" and str(r["risk_tier"]) == tier
        ]
        for tier in tiers
    ]
    # keep finite only
    data_auth = [list(np.asarray(v, dtype=np.float64)[np.isfinite(np.asarray(v, dtype=np.float64))]) for v in data_auth]
    data_non = [list(np.asarray(v, dtype=np.float64)[np.isfinite(np.asarray(v, dtype=np.float64))]) for v in data_non]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(tiers))
    width = 0.34
    means_auth = [float(np.mean(v)) if v else 0.0 for v in data_auth]
    means_non = [float(np.mean(v)) if v else 0.0 for v in data_non]
    ax.bar(x - width / 2, means_non, width, label="Non-authority", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, means_auth, width, label="Authority", color="#d62728", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel(r"Mean $\langle r, d_{auth} \rangle$")
    ax.set_title("Projection Means by Risk Tier and Framing")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "projection_means_by_tier_and_framing.pdf")
    fig.savefig(out_dir / "projection_means_by_tier_and_framing.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze authority-direction projection specificity.")
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--direction-pt", required=True)
    parser.add_argument("--hook-key", default="blocks.10.hook_resid_post")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    activation_dir = Path(args.activation_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    direction = _load_direction(Path(args.direction_pt).expanduser().resolve())
    rows = _collect_projection_rows(activation_dir, direction, hook_key=args.hook_key)
    _write_csv(out_dir / "projection_samples.csv", rows)

    comps = [
        GroupComparison(
            key="authority_vs_non_authority_all",
            label="Authority vs Non-authority (all tiers)",
            a_filter=lambda r: str(r["framing_type"]) == "authority",
            b_filter=lambda r: str(r["framing_type"]) != "authority",
        ),
        GroupComparison(
            key="authority_vs_non_authority_unsafe",
            label="Authority vs Non-authority (unsafe)",
            a_filter=lambda r: str(r["framing_type"]) == "authority" and str(r["risk_tier"]) == "unsafe",
            b_filter=lambda r: str(r["framing_type"]) != "authority" and str(r["risk_tier"]) == "unsafe",
        ),
        GroupComparison(
            key="authority_vs_non_authority_safe",
            label="Authority vs Non-authority (safe)",
            a_filter=lambda r: str(r["framing_type"]) == "authority" and str(r["risk_tier"]) == "safe",
            b_filter=lambda r: str(r["framing_type"]) != "authority" and str(r["risk_tier"]) == "safe",
        ),
        GroupComparison(
            key="unsafe_vs_safe_within_authority",
            label="Unsafe vs Safe (authority only)",
            a_filter=lambda r: str(r["framing_type"]) == "authority" and str(r["risk_tier"]) == "unsafe",
            b_filter=lambda r: str(r["framing_type"]) == "authority" and str(r["risk_tier"]) == "safe",
        ),
        GroupComparison(
            key="unsafe_vs_safe_within_non_authority",
            label="Unsafe vs Safe (non-authority)",
            a_filter=lambda r: str(r["framing_type"]) != "authority" and str(r["risk_tier"]) == "unsafe",
            b_filter=lambda r: str(r["framing_type"]) != "authority" and str(r["risk_tier"]) == "safe",
        ),
    ]

    stats_rows = [_group_stats(rows, cmp) for cmp in comps]
    pvals_perm = [float(r["perm_p_mean_diff"]) for r in stats_rows]
    pvals_ks = [float(r["ks_pvalue"]) for r in stats_rows]
    q_perm = _bh_fdr(pvals_perm)
    q_ks = _bh_fdr(pvals_ks)
    for i in range(len(stats_rows)):
        stats_rows[i]["perm_p_fdr_bh"] = q_perm[i]
        stats_rows[i]["ks_pvalue_fdr_bh"] = q_ks[i]

    ols = _fit_ols_authority_unsafe(rows)
    payload = {
        "n_projection_rows": len(rows),
        "hook_key": args.hook_key,
        "group_comparisons": stats_rows,
        "ols_authority_unsafe_interaction": ols,
    }
    (out_dir / "projection_specificity_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(out_dir / "projection_specificity_stats.csv", stats_rows)
    _write_latex_table(out_dir / "table_appendix_projection_specificity.tex", stats_rows[:3])
    _plot_distributions(rows, out_dir)
    print(f"Wrote projection analysis to {out_dir}")


if __name__ == "__main__":
    main()
