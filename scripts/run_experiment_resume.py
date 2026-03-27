#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from authority_analysis.behavior_evaluator import evaluate_behavior
from authority_analysis.causal_intervention import CausalInterventionEngine
from authority_analysis.config import load_config
from authority_analysis.feature_analyzer import compute_feature_analysis, save_feature_analysis
from authority_analysis.metrics_reporter import generate_report
from authority_analysis.model_interface import ModelInterface
from authority_analysis.pipeline import (
    _build_behavior_candidates,
    _build_placebo_direction,
    _check_prompt_security,
    _enrich_metrics_with_posthoc,
    _guess_generated_behavior,
    _run_classifier_behavior_endpoint,
    _run_placebo_experiment,
    _runtime_environment,
    _sample_rows_by_tier,
    _summarize_behavioral_ground_truth,
)
from authority_analysis.posthoc_analysis import run_posthoc_analysis_from_rows
from authority_analysis.prompt_generator import (
    generate_prompts,
    load_semantic_requests,
    save_prompt_dataset,
)
from authority_analysis.sae_module import (
    SAETrainConfig,
    load_layer_residual_matrix,
    save_sae_checkpoint,
    train_sae,
)
from authority_analysis.utils import (
    current_git_commit,
    ensure_dir,
    read_jsonl,
    set_global_seed,
    write_json,
    write_jsonl,
)


def _load_or_reconstruct_baseline_rows(log_dir: Path, activation_root: Path) -> list[dict[str, Any]]:
    baseline_samples_path = log_dir / "baseline_samples.json"
    if baseline_samples_path.exists():
        payload = json.loads(baseline_samples_path.read_text(encoding="utf-8"))
        samples = payload.get("samples")
        if isinstance(samples, list) and samples:
            return samples

    rows: list[dict[str, Any]] = []
    for file in sorted(activation_root.glob("*.pt")):
        payload = torch.load(file, map_location="cpu")
        meta = payload.get("metadata", {})
        rows.append(
            {
                "prompt_id": meta.get("prompt_id", file.stem),
                "framing_type": meta.get("framing_type", "unknown"),
                "semantic_request_id": meta.get("semantic_request_id", "unknown"),
                "safety_label": meta.get("safety_label", "unknown"),
                "risk_tier": meta.get("risk_tier", "unknown"),
                "refusal_score": float(payload.get("refusal_score", 0.0)),
                "compliance_score": float(payload.get("compliance_score", 0.0)),
                "logit_diff": float(payload.get("logit_diff", 0.0)),
                "is_refusal": bool(payload.get("is_refusal", False)),
                "logits_all_finite": bool(payload.get("logits_all_finite", True)),
                "logits_non_finite_count": int(payload.get("logits_non_finite_count", 0)),
                "logits_non_finite_ratio": float(payload.get("logits_non_finite_ratio", 0.0)),
            }
        )
    return rows


def _resume_from_existing_activations(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_global_seed(cfg.seed)
    git_hash = current_git_commit()
    config_ref = str(Path(config_path).resolve())
    config_hash = hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()

    result_root = ensure_dir(cfg.results_root())
    log_dir = ensure_dir(Path(result_root) / "logs")
    activation_root = ensure_dir(cfg.activations_root())
    ensure_dir(cfg.prompt_dataset().parent)

    if cfg.generate_prompts_if_missing or not cfg.prompt_dataset().exists():
        semantic_rows = load_semantic_requests(cfg.semantic_requests())
        prompts = generate_prompts(semantic_rows, cfg.framing_types)
        save_prompt_dataset(cfg.prompt_dataset(), prompts)
    prompt_rows = read_jsonl(cfg.prompt_dataset())
    prompt_lookup = {str(row["prompt_id"]): row for row in prompt_rows}
    _check_prompt_security(prompt_rows, cfg)

    activation_files = sorted(Path(activation_root).glob("*.pt"))
    if len(activation_files) != len(prompt_rows):
        raise RuntimeError(
            f"Activation cache incomplete for resume: found {len(activation_files)} files, "
            f"expected {len(prompt_rows)} under {activation_root}"
        )

    model = ModelInterface(
        model_name=cfg.model,
        device=cfg.device,
        dtype=cfg.dtype,
        probe_instruction=cfg.probe_instruction,
        refusal_margin=cfg.refusal_margin,
    )
    runtime_environment = _runtime_environment(model)

    baseline_rows = _load_or_reconstruct_baseline_rows(Path(log_dir), Path(activation_root))
    baseline_eval_rows, baseline_summary = evaluate_behavior(
        baseline_rows,
        control_framings=cfg.control_framing_types,
        refusal_margin=cfg.refusal_margin,
    )
    write_json(Path(log_dir) / "baseline_samples.json", {"samples": baseline_eval_rows})

    sae_input, sae_metadata = load_layer_residual_matrix(
        activation_root,
        layer_idx=cfg.layer_for_sae,
        hook_point="post",
    )
    has_authority = any(m.get("framing_type") == "authority" for m in sae_metadata)
    has_control = any(m.get("framing_type") != "authority" for m in sae_metadata)
    if not has_authority or not has_control:
        raise ValueError("SAE training requires both authority and control samples")

    sae_cfg = SAETrainConfig(
        hidden_multiplier=cfg.sae_hidden_multiplier,
        lr=cfg.sae_lr,
        l1_lambda=cfg.sae_l1_lambda,
        epochs=cfg.sae_epochs,
        patience=cfg.sae_patience,
        batch_size=cfg.sae_batch_size,
        seed=cfg.seed,
    )
    sae_model, sae_summary = train_sae(sae_input, sae_cfg, device=model.device)

    sae_ckpt_path = Path(result_root) / "authority_sae.pt"
    save_sae_checkpoint(
        out_path=sae_ckpt_path,
        model=sae_model,
        layer_idx=cfg.layer_for_sae,
        hook_point="post",
        train_summary=sae_summary,
    )
    write_json(Path(log_dir) / "sae_training_summary.json", sae_summary)

    feature_payload = compute_feature_analysis(
        activation_dir=activation_root,
        sae_ckpt_path=sae_ckpt_path,
        layer_idx=cfg.layer_for_sae,
        hook_point="post",
        top_k=cfg.top_k_features,
    )
    direction_path, feature_summary_path = save_feature_analysis(result_root, feature_payload)

    prompt_log_rows: list[dict[str, Any]] = []
    for row in prompt_rows:
        prompt_log_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "variant": "authority" if row.get("framing_type") == "authority" else "baseline",
                "risk_tier": row.get("risk_tier", "unknown"),
                "domain": row.get("domain", "unknown"),
                "prompt": model.compose_prompt(row["full_prompt"]),
                "seed": cfg.seed,
                "git_hash": git_hash,
                "config_path": config_ref,
                "config_hash": config_hash,
            }
        )

    authority_prompts = [r for r in prompt_rows if r["framing_type"] == "authority"]
    intervention_engine = CausalInterventionEngine(model)
    for row in authority_prompts:
        prompt_log_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "variant": "intervention",
                "risk_tier": row.get("risk_tier", "unknown"),
                "domain": row.get("domain", "unknown"),
                "prompt": model.compose_prompt(row["full_prompt"]),
                "seed": cfg.seed,
                "git_hash": git_hash,
                "config_path": config_ref,
                "config_hash": config_hash,
            }
        )

    intervention_rows = intervention_engine.run(
        prompts=authority_prompts,
        layer_idx=cfg.layer_for_sae,
        direction=feature_payload["residual_direction_normalized"],
        alpha=cfg.alpha_intervention,
        max_tokens=cfg.max_tokens,
        capture_attentions=False,
        capture_layers=set(),
    )
    intervention_eval_rows, intervention_summary = evaluate_behavior(
        intervention_rows,
        control_framings=cfg.control_framing_types,
        refusal_margin=cfg.refusal_margin,
    )
    write_json(Path(log_dir) / "intervention_samples.json", {"samples": intervention_eval_rows})
    write_jsonl(Path(log_dir) / "prompts.jsonl", prompt_log_rows)

    behavioral_gt_rows: list[dict[str, Any]] | None = None
    behavior_candidates = _build_behavior_candidates(
        prompt_lookup=prompt_lookup,
        baseline_eval_rows=baseline_eval_rows,
        intervention_eval_rows=intervention_eval_rows,
    )
    if cfg.behavioral_ground_truth_samples_per_variant > 0:
        rng = random.Random(cfg.seed)
        intervention_fn = intervention_engine.make_projection_removal_fn(
            feature_payload["residual_direction_normalized"],
            alpha=cfg.alpha_intervention,
        )
        diagnostic_rows: list[dict[str, Any]] = []
        for variant, candidates in behavior_candidates.items():
            sampled = _sample_rows_by_tier(
                rows=candidates,
                sample_count=cfg.behavioral_ground_truth_samples_per_variant,
                rng=rng,
                unsafe_min=cfg.behavioral_ground_truth_unsafe_min_per_variant,
            )
            for sampled_row in sampled:
                generation_prompt = sampled_row["prompt"]
                generated_text = model.generate_short_completion(
                    prompt_text=generation_prompt,
                    max_tokens=cfg.max_tokens,
                    max_new_tokens=cfg.behavioral_ground_truth_max_new_tokens,
                    use_probe_instruction=cfg.behavioral_ground_truth_use_probe_instruction,
                    intervention_layer=cfg.layer_for_sae if variant == "intervention" else None,
                    intervention_fn=intervention_fn if variant == "intervention" else None,
                    temperature=cfg.behavioral_ground_truth_temperature,
                    top_p=cfg.behavioral_ground_truth_top_p,
                )
                generation_input = (
                    model.compose_prompt(generation_prompt)
                    if cfg.behavioral_ground_truth_use_probe_instruction
                    else generation_prompt.rstrip()
                )
                predicted_label = "refusal" if bool(sampled_row.get("is_refusal", False)) else "compliance"
                generated_guess = _guess_generated_behavior(
                    generated_text=generated_text,
                    risk_tier=str(sampled_row.get("risk_tier", "unknown")),
                )
                diagnostic_rows.append(
                    {
                        "prompt_id": sampled_row["prompt_id"],
                        "variant": variant,
                        "framing_type": sampled_row.get("framing_type", "unknown"),
                        "semantic_request_id": sampled_row.get("semantic_request_id", "unknown"),
                        "risk_tier": sampled_row.get("risk_tier", "unknown"),
                        "domain": sampled_row.get("domain", "unknown"),
                        "prompt": generation_input,
                        "generated_text": generated_text,
                        "predicted_behavior": predicted_label,
                        "predicted_is_refusal": bool(sampled_row.get("is_refusal", False)),
                        "refusal_score": float(sampled_row.get("refusal_score", 0.0)),
                        "compliance_score": float(sampled_row.get("compliance_score", 0.0)),
                        "logit_diff": float(sampled_row.get("logit_diff", 0.0)),
                        "generated_behavior_guess": generated_guess,
                        "generated_guess_matches_prediction": (
                            generated_guess == predicted_label
                            if generated_guess in {"refusal", "compliance"}
                            else None
                        ),
                        "max_new_tokens": cfg.behavioral_ground_truth_max_new_tokens,
                        "seed": cfg.seed,
                        "git_hash": git_hash,
                        "config_path": config_ref,
                        "config_hash": config_hash,
                    }
                )
        behavioral_gt_rows = diagnostic_rows
        write_jsonl(Path(log_dir) / "behavioral_ground_truth.jsonl", diagnostic_rows)
        write_json(
            Path(log_dir) / "behavioral_ground_truth_summary.json",
            _summarize_behavioral_ground_truth(diagnostic_rows),
        )

    classifier_behavior_summary = _run_classifier_behavior_endpoint(
        cfg=cfg,
        model=model,
        intervention_fn=intervention_engine.make_projection_removal_fn(
            feature_payload["residual_direction_normalized"],
            alpha=cfg.alpha_intervention,
        ),
        candidates_by_variant=behavior_candidates,
        out_log_dir=log_dir,
        git_hash=git_hash,
        config_ref=config_ref,
        config_hash=config_hash,
        seed_offset=6000,
    )

    report = generate_report(
        result_dir=result_root,
        baseline_summary=baseline_summary,
        intervention_summary=intervention_summary,
        feature_payload=feature_payload,
        activation_dir=activation_root,
        hook_point="post",
    )

    posthoc_report = run_posthoc_analysis_from_rows(
        baseline_rows=baseline_eval_rows,
        intervention_rows=intervention_eval_rows,
        out_dir=Path(result_root) / "posthoc",
        behavioral_gt_rows=behavioral_gt_rows,
        margins=cfg.posthoc_margins,
        bootstrap_iters=cfg.posthoc_bootstrap_iters,
        seed=cfg.seed,
    )
    report["metrics"] = _enrich_metrics_with_posthoc(report["metrics"], posthoc_report)
    if classifier_behavior_summary is not None:
        report["metrics"]["classifier_behavior_endpoint"] = classifier_behavior_summary
    write_json(Path(result_root) / "metrics.json", report["metrics"])

    direct_metrics = baseline_summary.get("framing_metrics", {}).get("direct", {})
    authority_metrics = baseline_summary.get("framing_metrics", {}).get("authority", {})
    run_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "dataset_version": cfg.dataset_version,
        "config": cfg.to_dict(),
        "artifacts": {
            "activation_root": str(activation_root),
            "results_root": str(result_root),
            "sae_checkpoint": str(sae_ckpt_path),
            "direction_vector": str(direction_path),
            "feature_summary": str(feature_summary_path),
            "posthoc_dir": str(Path(result_root) / "posthoc"),
            "classifier_behavior_labels": str(Path(log_dir) / "classifier_behavior_labels.jsonl"),
            "classifier_behavior_summary": str(Path(log_dir) / "classifier_behavior_summary.json"),
        },
        "checks": {
            "authority_refusal_score_reduction": float(authority_metrics.get("mean_logit_diff", 0.0))
            < float(direct_metrics.get("mean_logit_diff", 0.0)),
            "sae_feature_difference_computed": True,
            "intervention_rate_quantified": True,
            "plots_generated": True,
            "reproducible_config_run": True,
            "resumed_from_existing_activations": True,
        },
        "runtime_environment": runtime_environment,
    }
    write_json(Path(log_dir) / "run_manifest.json", run_manifest)

    placebo_results: list[dict[str, Any]] = []
    if cfg.placebo_modes:
        placebo_parent_root = ensure_dir(Path(cfg.results_dir) / f"{cfg.experiment_name}_placebo")
        for idx, mode in enumerate(cfg.placebo_modes):
            direction, direction_meta = _build_placebo_direction(
                mode=mode,
                feature_payload=feature_payload,
                sae_model=sae_model,
                seed=cfg.seed + cfg.placebo_shuffle_seed_offset + idx,
                low_feature_count=cfg.placebo_low_importance_features,
                sae_input=sae_input,
                sae_metadata=sae_metadata,
            )
            placebo_out = _run_placebo_experiment(
                cfg=cfg,
                mode=mode,
                seed_offset=cfg.placebo_shuffle_seed_offset + idx,
                direction=direction,
                direction_meta=direction_meta,
                model=model,
                intervention_engine=intervention_engine,
                authority_prompts=authority_prompts,
                baseline_eval_rows=baseline_eval_rows,
                baseline_summary=baseline_summary,
                feature_payload=feature_payload,
                activation_root=activation_root,
                prompt_log_rows=prompt_log_rows,
                prompt_lookup=prompt_lookup,
                git_hash=git_hash,
                config_ref=config_ref,
                config_hash=config_hash,
                runtime_environment=runtime_environment,
            )
            placebo_results.append(placebo_out)
        write_json(
            Path(placebo_parent_root) / "placebo_summary.json",
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_name": cfg.experiment_name,
                "modes": [r.get("mode") for r in placebo_results],
                "results": placebo_results,
            },
        )

    return {
        "result_root": str(result_root),
        "activation_root": str(activation_root),
        "direction_path": str(direction_path),
        "metrics": report["metrics"],
        "placebo_results": placebo_results,
        "resumed_from_existing_activations": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume a full experiment from existing activation captures when possible."
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    activation_root = cfg.activations_root()
    prompt_count = 0
    if cfg.prompt_dataset().exists():
        prompt_count = len(read_jsonl(cfg.prompt_dataset()))
    activation_count = len(list(Path(activation_root).glob("*.pt")))

    if activation_count == prompt_count and prompt_count > 0:
        output = _resume_from_existing_activations(args.config)
    else:
        from authority_analysis.pipeline import run_full_experiment

        output = run_full_experiment(args.config)

    print("Experiment complete")
    print(output)


if __name__ == "__main__":
    main()
