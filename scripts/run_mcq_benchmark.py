#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from authority_analysis.causal_intervention import CausalInterventionEngine
from authority_analysis.model_interface import ModelInterface
from authority_analysis.utils import ensure_dir, write_json, write_jsonl


LETTER_OPTIONS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class DirectionSpec:
    layer: int
    path: Path
    alpha: float


@dataclass
class BenchmarkExample:
    example_id: str
    task: str
    group: str
    prompt: str
    option_labels: list[str]
    correct_label: str
    metadata: dict[str, Any]


def _load_direction(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    direction = payload.get("residual_direction_normalized")
    if direction is None:
        direction = payload.get("direction")
    if direction is None or not isinstance(direction, torch.Tensor):
        raise ValueError(f"Direction file missing tensor payload: {path}")
    return direction


def _parse_direction_spec(raw: str, default_alpha: float) -> DirectionSpec:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(
            f"Invalid --direction-spec {raw!r}. Expected 'LAYER:PATH' or 'LAYER:PATH:ALPHA'."
        )
    layer = int(parts[0])
    alpha = float(parts[2]) if len(parts) == 3 else float(default_alpha)
    path = Path(parts[1]).expanduser().resolve()
    return DirectionSpec(layer=layer, path=path, alpha=alpha)


def _build_intervention_map(
    specs: Sequence[DirectionSpec],
) -> dict[int, Any]:
    intervention_map: dict[int, Any] = {}
    for spec in specs:
        direction = _load_direction(spec.path)
        intervention_map[int(spec.layer)] = CausalInterventionEngine.make_projection_removal_fn(
            direction,
            alpha=spec.alpha,
        )
    return intervention_map


def _format_choices(choices: Sequence[str]) -> tuple[list[str], str]:
    if len(choices) > len(LETTER_OPTIONS):
        raise ValueError(f"Too many answer options ({len(choices)}); max supported is {len(LETTER_OPTIONS)}")
    labels = list(LETTER_OPTIONS[: len(choices)])
    rendered = "\n".join(
        f"{label}. {choice.strip()}" for label, choice in zip(labels, choices)
    )
    return labels, rendered


def _subject_label(subject: str) -> str:
    return str(subject).replace("_", " ")


def _build_mmlu_prompt(
    question: str,
    choices: Sequence[str],
    subject: str,
    fewshot_rows: Sequence[dict[str, Any]],
) -> tuple[str, list[str]]:
    label_lines: list[str] = [
        f"The following are multiple choice questions (with answers) about {_subject_label(subject)}.",
        "",
    ]

    for row in fewshot_rows:
        labels, rendered = _format_choices(row["choices"])
        answer_idx = int(row["answer"])
        label_lines.extend(
            [
                row["question"].strip(),
                rendered,
                f"Answer: {labels[answer_idx]}",
                "",
            ]
        )

    option_labels, rendered_choices = _format_choices(choices)
    label_lines.extend(
        [
            question.strip(),
            rendered_choices,
            "Answer:",
        ]
    )
    return "\n".join(label_lines).strip(), option_labels


def _build_truthfulqa_prompt(
    question: str,
    choices: Sequence[str],
) -> tuple[str, list[str]]:
    option_labels, rendered_choices = _format_choices(choices)
    prompt = "\n".join(
        [
            "Answer the following multiple choice question by selecting the most truthful answer.",
            "",
            question.strip(),
            rendered_choices,
            "Answer:",
        ]
    )
    return prompt, option_labels


def _prepare_mmlu_examples(
    split: str,
    fewshot: int,
    max_examples: int | None,
    seed: int,
    subjects: Sequence[str] | None,
) -> list[BenchmarkExample]:
    try:
        from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required for MMLU benchmarking. "
            "Install it with `pip install datasets`."
        ) from exc

    candidate_subjects = [
        cfg
        for cfg in get_dataset_config_names("cais/mmlu")
        if cfg and cfg not in {"all", "default"}
    ]
    available_subjects: list[str] = []
    for cfg_name in candidate_subjects:
        try:
            builder = load_dataset_builder("cais/mmlu", cfg_name)
        except Exception:
            continue
        split_names = set((builder.info.splits or {}).keys())
        if {"dev", split}.issubset(split_names):
            available_subjects.append(cfg_name)
    selected_subjects = list(subjects) if subjects else available_subjects
    missing = sorted(set(selected_subjects) - set(available_subjects))
    if missing:
        raise ValueError(
            f"Unknown or unsupported MMLU subjects for split={split!r}: {missing}. "
            f"Available subjects with dev/{split}: {available_subjects[:10]}..."
        )

    examples: list[BenchmarkExample] = []
    for subject in selected_subjects:
        dev_rows = load_dataset("cais/mmlu", subject, split="dev")
        eval_rows = load_dataset("cais/mmlu", subject, split=split)
        fewshot_rows = [dev_rows[i] for i in range(min(fewshot, len(dev_rows)))]

        for idx, row in enumerate(eval_rows):
            prompt, option_labels = _build_mmlu_prompt(
                question=row["question"],
                choices=row["choices"],
                subject=subject,
                fewshot_rows=fewshot_rows,
            )
            correct_label = option_labels[int(row["answer"])]
            examples.append(
                BenchmarkExample(
                    example_id=f"mmlu::{subject}::{idx}",
                    task="mmlu",
                    group=subject,
                    prompt=prompt,
                    option_labels=option_labels,
                    correct_label=correct_label,
                    metadata={
                        "subject": subject,
                        "split": split,
                        "question": row["question"],
                    },
                )
            )

    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    return examples


def _prepare_truthfulqa_examples(
    split: str,
    max_examples: int | None,
    seed: int,
) -> list[BenchmarkExample]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required for TruthfulQA benchmarking. "
            "Install it with `pip install datasets`."
        ) from exc

    ds = load_dataset("truthful_qa", "multiple_choice", split=split)
    examples: list[BenchmarkExample] = []
    for idx, row in enumerate(ds):
        targets = row.get("mc1_targets") or {}
        choices = list(targets.get("choices") or [])
        labels = list(targets.get("labels") or [])
        if not choices or not labels or len(choices) != len(labels):
            raise ValueError("TruthfulQA mc1_targets schema mismatch; expected aligned choices/labels")
        if 1 not in labels:
            raise ValueError("TruthfulQA MC1 row missing positive label")
        correct_idx = labels.index(1)
        prompt, option_labels = _build_truthfulqa_prompt(row["question"], choices)
        examples.append(
            BenchmarkExample(
                example_id=f"truthfulqa_mc1::{idx}",
                task="truthfulqa_mc1",
                group="truthfulqa_mc1",
                prompt=prompt,
                option_labels=option_labels,
                correct_label=option_labels[int(correct_idx)],
                metadata={
                    "split": split,
                    "question": row["question"],
                },
            )
        )

    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    return examples


def _argmax_label(scores: dict[str, float], option_labels: Sequence[str]) -> str:
    return max(option_labels, key=lambda label: (scores[label], label))


def _evaluate_examples(
    model: ModelInterface,
    examples: Sequence[BenchmarkExample],
    intervention_layer: int | None,
    intervention_fn: Any | None,
    intervention_fns_by_layer: dict[int, Any] | None,
    max_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc="benchmark", ncols=100):
        scores = model.score_option_letters(
            prompt_text=ex.prompt,
            option_labels=ex.option_labels,
            max_tokens=max_tokens,
            use_probe_instruction=False,
            intervention_layer=intervention_layer,
            intervention_fn=intervention_fn,
            intervention_fns_by_layer=intervention_fns_by_layer,
        )
        predicted = _argmax_label(scores, ex.option_labels)
        rows.append(
            {
                "example_id": ex.example_id,
                "task": ex.task,
                "group": ex.group,
                "correct_label": ex.correct_label,
                "predicted_label": predicted,
                "is_correct": bool(predicted == ex.correct_label),
                "scores": scores,
                **ex.metadata,
            }
        )
    return rows


def _group_summary(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        n = len(group_rows)
        correct = sum(1 for row in group_rows if row["is_correct"])
        summary_rows.append(
            {
                "group": group,
                "n_examples": n,
                "n_correct": correct,
                "accuracy": float(correct / n) if n else 0.0,
            }
        )
    return summary_rows


def _overall_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_correct = sum(1 for row in rows if row["is_correct"])
    return {
        "n_examples": n,
        "n_correct": n_correct,
        "accuracy": float(n_correct / n) if n else 0.0,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCQ utility benchmarks with optional frozen-direction intervention.")
    parser.add_argument("--task", choices=["mmlu", "truthfulqa_mc1"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--fewshot", type=int, default=5, help="MMLU only; ignored for TruthfulQA")
    parser.add_argument("--subjects", nargs="+", default=None, help="Optional MMLU subject subset")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--direction", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--direction-spec",
        action="append",
        default=[],
        help="Multi-layer intervention spec 'LAYER:PATH' or 'LAYER:PATH:ALPHA'. Can be repeated.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    if args.task == "mmlu":
        examples = _prepare_mmlu_examples(
            split=args.split or "test",
            fewshot=max(0, int(args.fewshot)),
            max_examples=args.max_examples,
            seed=args.seed,
            subjects=args.subjects,
        )
    else:
        examples = _prepare_truthfulqa_examples(
            split=args.split or "validation",
            max_examples=args.max_examples,
            seed=args.seed,
        )

    if not examples:
        raise ValueError("Benchmark selection produced zero examples")

    model = ModelInterface(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        probe_instruction="Answer with exactly one letter.",
        refusal_margin=1.0,
    )

    direction_specs = [_parse_direction_spec(raw, args.alpha) for raw in args.direction_spec]
    if args.direction and args.layer is not None:
        direction_specs.insert(
            0,
            DirectionSpec(
                layer=int(args.layer),
                path=Path(args.direction).expanduser().resolve(),
                alpha=float(args.alpha),
            ),
        )
    if args.direction and args.layer is None:
        raise ValueError("--direction requires --layer")
    if args.layer is not None and not args.direction:
        raise ValueError("--layer requires --direction")

    baseline_rows = _evaluate_examples(
        model=model,
        examples=examples,
        intervention_layer=None,
        intervention_fn=None,
        intervention_fns_by_layer=None,
        max_tokens=args.max_tokens,
    )
    baseline_overall = _overall_summary(baseline_rows)
    baseline_groups = _group_summary(baseline_rows)

    intervention_rows: list[dict[str, Any]] | None = None
    intervention_overall: dict[str, Any] | None = None
    intervention_groups: list[dict[str, Any]] | None = None
    if direction_specs:
        intervention_map = _build_intervention_map(direction_specs)
        single_layer = direction_specs[0].layer if len(direction_specs) == 1 else None
        single_fn = intervention_map.get(single_layer) if len(direction_specs) == 1 else None
        multi_map = None if len(direction_specs) == 1 else intervention_map
        intervention_rows = _evaluate_examples(
            model=model,
            examples=examples,
            intervention_layer=single_layer,
            intervention_fn=single_fn,
            intervention_fns_by_layer=multi_map,
            max_tokens=args.max_tokens,
        )
        intervention_overall = _overall_summary(intervention_rows)
        intervention_groups = _group_summary(intervention_rows)

    write_jsonl(out_dir / "baseline_examples.jsonl", baseline_rows)
    _write_csv(out_dir / "baseline_group_summary.csv", baseline_groups)

    if intervention_rows is not None and intervention_overall is not None and intervention_groups is not None:
        write_jsonl(out_dir / "intervention_examples.jsonl", intervention_rows)
        _write_csv(out_dir / "intervention_group_summary.csv", intervention_groups)

    summary = {
        "task": args.task,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "split": args.split or ("test" if args.task == "mmlu" else "validation"),
        "fewshot": int(args.fewshot) if args.task == "mmlu" else 0,
        "max_examples": args.max_examples,
        "subjects": list(args.subjects) if args.subjects else None,
        "seed": int(args.seed),
        "baseline": {
            "overall": baseline_overall,
            "groups": baseline_groups,
        },
        "intervention": None,
    }
    if intervention_rows is not None and intervention_overall is not None and intervention_groups is not None:
        summary["intervention"] = {
            "specs": [
                {
                    "layer": int(spec.layer),
                    "direction_path": str(spec.path),
                    "alpha": float(spec.alpha),
                }
                for spec in direction_specs
            ],
            "overall": intervention_overall,
            "groups": intervention_groups,
            "delta_accuracy": float(intervention_overall["accuracy"] - baseline_overall["accuracy"]),
        }

    write_json(out_dir / "summary.json", summary)
    print(f"Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
