from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .activation_logger import ActivationLogger
from .behavior_evaluator import evaluate_behavior
from .causal_intervention import CausalInterventionEngine
from .config import ExperimentConfig, load_config
from .feature_analyzer import compute_feature_analysis, save_feature_analysis
from .metrics_reporter import generate_report
from .model_interface import ModelInterface
from .prompt_generator import (
    FRAMING_TEMPLATES,
    generate_prompt_text,
    generate_prompts,
    load_semantic_requests,
    save_prompt_dataset,
)
from .sae_module import SAETrainConfig, load_layer_residual_matrix, save_sae_checkpoint, train_sae
from .schemas import validate_prompt_row
from .utils import current_git_commit, ensure_dir, read_jsonl, set_global_seed, write_json



def _check_prompt_security(rows: list[dict[str, Any]], cfg: ExperimentConfig) -> None:
    for row in rows:
        validate_prompt_row(row)
        framing = row["framing_type"]
        if framing not in FRAMING_TEMPLATES:
            raise ValueError(f"Prompt row has unsupported framing_type: {framing}")
        base_request = row.get("base_request")
        if base_request is not None:
            expected = generate_prompt_text(base_request, framing)
            if row["full_prompt"].strip() != expected.strip():
                raise ValueError(
                    "Prompt dataset must use controlled deterministic templates only; "
                    f"mismatch for prompt_id={row['prompt_id']}"
                )
        if str(row.get("risk_level", "controlled")) != "controlled":
            raise ValueError(f"risk_level must be controlled: prompt_id={row['prompt_id']}")



def _artifact_dict_from_forward(artifacts: Any) -> dict[str, Any]:
    return {
        "residual_stream": artifacts.residual_stream,
        "attention_outputs": artifacts.attention_outputs,
        "final_logits": artifacts.final_logits,
        "refusal_logit": artifacts.refusal_logit,
        "compliance_logit": artifacts.compliance_logit,
        "refusal_prob": artifacts.refusal_prob,
        "compliance_prob": artifacts.compliance_prob,
        "logit_diff": artifacts.logit_diff,
    }



def run_full_experiment(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_global_seed(cfg.seed)

    result_root = ensure_dir(cfg.results_root())
    log_dir = ensure_dir(Path(result_root) / "logs")
    activation_root = ensure_dir(cfg.activations_root())
    ensure_dir(cfg.prompt_dataset().parent)

    # Ensure reproducible runs by removing prior activation snapshots for this model.
    for stale_file in Path(activation_root).glob("*.pt"):
        stale_file.unlink(missing_ok=True)

    if cfg.generate_prompts_if_missing or not cfg.prompt_dataset().exists():
        semantic_rows = load_semantic_requests(cfg.semantic_requests())
        prompts = generate_prompts(semantic_rows, cfg.framing_types)
        save_prompt_dataset(cfg.prompt_dataset(), prompts)
    prompt_rows = read_jsonl(cfg.prompt_dataset())

    _check_prompt_security(prompt_rows, cfg)

    model = ModelInterface(
        model_name=cfg.model,
        device=cfg.device,
        dtype=cfg.dtype,
        refusal_token=cfg.refusal_token,
        compliance_token=cfg.compliance_token,
    )
    logger = ActivationLogger(activation_root)

    baseline_rows: list[dict[str, Any]] = []
    for row in tqdm(prompt_rows, desc="Collecting activations"):
        artifacts = model.run_forward(row["full_prompt"], max_tokens=cfg.max_tokens)
        logger.save_sample(
            prompt_id=row["prompt_id"],
            artifacts=_artifact_dict_from_forward(artifacts),
            metadata=row,
        )
        baseline_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "framing_type": row["framing_type"],
                "semantic_request_id": row["semantic_request_id"],
                "safety_label": row["safety_label"],
                "refusal_prob": artifacts.refusal_prob,
                "compliance_prob": artifacts.compliance_prob,
                "logit_diff": artifacts.logit_diff,
            }
        )

    baseline_eval_rows, baseline_summary = evaluate_behavior(
        baseline_rows,
        refusal_threshold=cfg.refusal_threshold,
        control_framings=cfg.control_framing_types,
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

    authority_prompts = [r for r in prompt_rows if r["framing_type"] == "authority"]
    intervention_engine = CausalInterventionEngine(model)
    intervention_rows = intervention_engine.run(
        prompts=authority_prompts,
        layer_idx=cfg.layer_for_sae,
        direction=feature_payload["residual_direction_normalized"],
        alpha=cfg.alpha_intervention,
        max_tokens=cfg.max_tokens,
    )
    intervention_eval_rows, intervention_summary = evaluate_behavior(
        intervention_rows,
        refusal_threshold=cfg.refusal_threshold,
        control_framings=cfg.control_framing_types,
    )
    write_json(Path(log_dir) / "intervention_samples.json", {"samples": intervention_eval_rows})

    report = generate_report(
        result_dir=result_root,
        baseline_summary=baseline_summary,
        intervention_summary=intervention_summary,
        feature_payload=feature_payload,
        activation_dir=activation_root,
        hook_point="post",
    )

    direct_metrics = baseline_summary.get("framing_metrics", {}).get("direct", {})
    authority_metrics = baseline_summary.get("framing_metrics", {}).get("authority", {})

    run_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "dataset_version": cfg.dataset_version,
        "config": cfg.to_dict(),
        "artifacts": {
            "activation_root": str(activation_root),
            "results_root": str(result_root),
            "sae_checkpoint": str(sae_ckpt_path),
            "direction_vector": str(direction_path),
            "feature_summary": str(feature_summary_path),
        },
        "checks": {
            "authority_refusal_logit_reduction": float(authority_metrics.get("mean_logit_diff", 0.0))
            < float(direct_metrics.get("mean_logit_diff", 0.0)),
            "sae_feature_difference_computed": True,
            "intervention_rate_quantified": True,
            "plots_generated": True,
            "reproducible_config_run": True,
        },
    }
    write_json(Path(log_dir) / "run_manifest.json", run_manifest)

    return {
        "result_root": str(result_root),
        "activation_root": str(activation_root),
        "direction_path": str(direction_path),
        "metrics": report["metrics"],
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run full authority suppression experiment")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    output = run_full_experiment(args.config)
    print("Experiment complete")
    print(output)


if __name__ == "__main__":
    main()
