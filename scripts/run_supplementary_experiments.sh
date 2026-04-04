#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Supplementary experiments for NeurIPS 2026 submission
# Run from the project root: cd /path/to/Mechanistic-Analysis && bash scripts/run_supplementary_experiments.sh
# =============================================================================

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

CURATED_DIR="analysis_packages/llama3_full_350_m15_gt200_probe_placebo_full_paper/collected_runs/main/authority_direction_vector.pt"
RESULTS_BASE="results/supplementary"

echo "=========================================="
echo "  Experiment 1: Curated-direction utility"
echo "=========================================="
echo "Running MMLU (5-shot) with curated-set direction..."

python scripts/run_mcq_benchmark.py \
    --task mmlu \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir "${RESULTS_BASE}/curated_mmlu" \
    --layer 10 \
    --direction "${CURATED_DIR}" \
    --alpha 1.0 \
    --fewshot 5 \
    --seed 42 \
    --dtype float16

echo "Running TruthfulQA-MC1 with curated-set direction..."

python scripts/run_mcq_benchmark.py \
    --task truthfulqa_mc1 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir "${RESULTS_BASE}/curated_truthfulqa" \
    --layer 10 \
    --direction "${CURATED_DIR}" \
    --alpha 1.0 \
    --seed 42 \
    --dtype float16

echo ""
echo "=========================================="
echo "  Experiment 2: Sign-reversal ablation"
echo "  (expanded dataset, 4 control framings)"
echo "=========================================="
echo "Running expanded set with restricted control framings..."

python scripts/run_experiment.py \
    --config configs/supplementary/expanded_4control.yaml

echo ""
echo "=========================================="
echo "  Experiment 3: Phi-3 multi-layer sweep"
echo "=========================================="
for LAYER in 8 12 16 20 24; do
    echo "Running Phi-3-Mini layer ${LAYER}..."
    python scripts/run_experiment.py \
        --config configs/supplementary/phi3_layer${LAYER}.yaml
done

echo ""
echo "=========================================="
echo "  All supplementary experiments complete."
echo "=========================================="
