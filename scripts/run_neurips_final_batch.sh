#!/usr/bin/env bash
# NeurIPS 2026 final experiment batch
# Runs all P1+P2+P3 experiments sequentially (~45 GPU-hours on L4)
#
# Usage:
#   export HF_TOKEN=<your_token>
#   bash scripts/run_neurips_final_batch.sh 2>&1 | tee /tmp/neurips_final.log
#
# Or with tmux:
#   tmux new -s final -d 'cd ~/Mechanistic-Analysis && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && export HF_TOKEN=<TOKEN> && bash scripts/run_neurips_final_batch.sh 2>&1 | tee /tmp/neurips_final.log'

set -e
cd "$(dirname "$0")/.."

mkdir -p results/supplementary /tmp/neurips_logs

echo "=========================================="
echo "NeurIPS 2026 final experiment batch"
echo "Start: $(date)"
echo "=========================================="

# Direction paths (set these to actual locations after curated+expanded runs)
CURATED_DIR="results/full_350_m15_gt200_probe_placebo/authority_direction_vector.pt"
EXPANDED_DIR="results/supplementary/llama3_8b_l4_expanded/authority_direction_vector.pt"

# Fallback: find curated direction automatically
if [ ! -f "$CURATED_DIR" ]; then
    CURATED_DIR=$(find results -name "authority_direction_vector.pt" -path "*full_350*" 2>/dev/null | head -1)
fi
if [ ! -f "$EXPANDED_DIR" ]; then
    EXPANDED_DIR=$(find results -name "authority_direction_vector.pt" -path "*expanded*" ! -path "*placebo*" ! -path "*4control*" 2>/dev/null | head -1)
fi

echo "CURATED_DIR=$CURATED_DIR"
echo "EXPANDED_DIR=$EXPANDED_DIR"

# ============================================================
# P1-A: Utility benchmarks multi-seed (MMLU + TruthfulQA)
# ============================================================
echo ""
echo "### P1-A: Utility benchmarks multi-seed (5h) ###"
for SEED in 42 0 1; do
    for DIR_NAME in curated expanded; do
        if [ "$DIR_NAME" = "curated" ]; then DIR_PATH="$CURATED_DIR"; else DIR_PATH="$EXPANDED_DIR"; fi
        if [ -z "$DIR_PATH" ] || [ ! -f "$DIR_PATH" ]; then
            echo "SKIP: $DIR_NAME direction not found"
            continue
        fi
        for TASK in mmlu truthfulqa; do
            echo "-- utility seed=$SEED dir=$DIR_NAME task=$TASK"
            python scripts/run_mcq_benchmark.py \
                --model meta-llama/Meta-Llama-3-8B-Instruct \
                --task $TASK --seed $SEED \
                --direction "$DIR_PATH" --layer 10 --alpha 1.0 \
                --output-dir "results/supplementary/utility_multiseed/${TASK}_${DIR_NAME}_s${SEED}" \
                2>&1 | tee "/tmp/neurips_logs/util_${TASK}_${DIR_NAME}_s${SEED}.log" || echo "FAILED"
        done
    done
done

# ============================================================
# P1-B: Phi-3 probe alternatives (3h)
# ============================================================
echo ""
echo "### P1-B: Phi-3 probe alternatives (3h) ###"
for PROBE in phi3_probe_alt_v2 phi3_probe_alt_v3 phi3_probe_alt_v4; do
    echo "-- phi3 probe: $PROBE"
    python scripts/run_experiment.py --config "configs/supplementary/${PROBE}.yaml" \
        2>&1 | tee "/tmp/neurips_logs/${PROBE}.log" || echo "FAILED"
done

# ============================================================
# P1-C: Main result multi-seed (LLaMA-3-8B curated) (8h)
# ============================================================
echo ""
echo "### P1-C: Main multi-seed (8h) ###"
python scripts/run_seed_sweep.py \
    --config configs/cloud/llama3_8b_l4_safe.yaml \
    --seeds 42 0 1 2 3 \
    --experiment-prefix llama3_8b_curated_seedsweep \
    2>&1 | tee /tmp/neurips_logs/main_seedsweep.log || echo "FAILED"

# ============================================================
# P2-D: Larger LLaMA variants (4h)
# ============================================================
echo ""
echo "### P2-D: LLaMA-3.1/3.2 replication (4h) ###"
python scripts/run_experiment.py --config configs/supplementary/llama3_1_8b.yaml \
    2>&1 | tee /tmp/neurips_logs/llama3_1_8b.log || echo "FAILED"
python scripts/run_experiment.py --config configs/supplementary/llama3_2_3b.yaml \
    2>&1 | tee /tmp/neurips_logs/llama3_2_3b.log || echo "FAILED"

# ============================================================
# P2-F: Alpha sweep on expanded (4h) - 5 alphas
# ============================================================
echo ""
echo "### P2-F: Alpha sweep on expanded (4h) ###"
for ALPHA in 0.25 0.5 1.0 1.5 2.0; do
    echo "-- expanded alpha=$ALPHA"
    # Create temp config with overridden alpha
    TMP_CFG="/tmp/expanded_alpha_${ALPHA}.yaml"
    sed "s/alpha_intervention: 1.0/alpha_intervention: ${ALPHA}/" \
        configs/supplementary/expanded_alpha_sweep_base.yaml > "$TMP_CFG"
    sed -i "s/llama3_8b_expanded_alpha_sweep/llama3_8b_expanded_alpha_${ALPHA}/" "$TMP_CFG"
    python scripts/run_experiment.py --config "$TMP_CFG" \
        2>&1 | tee "/tmp/neurips_logs/expanded_alpha_${ALPHA}.log" || echo "FAILED"
done

# ============================================================
# P2-E: Sign-flip stability via expanded sub-sampling (2h)
# (bootstrap 100 subsamples of n=70 from expanded authority-unsafe pairs)
# ============================================================
echo ""
echo "### P2-E: Sign-flip bootstrap stability (2h) ###"
python - <<'PYEOF' 2>&1 | tee /tmp/neurips_logs/signflip_bootstrap.log || echo "FAILED"
import json, random, numpy as np
from pathlib import Path

# Load expanded posthoc results
posthoc_path = Path("results/supplementary/llama3_8b_l4_expanded/posthoc/posthoc_analysis.json")
if not posthoc_path.exists():
    posthoc_path = Path("results/supplementary/llama3_8b_expanded/posthoc/posthoc_analysis.json")
if not posthoc_path.exists():
    candidates = list(Path("results").rglob("posthoc_analysis.json"))
    candidates = [c for c in candidates if "expanded" in str(c) and "4control" not in str(c) and "placebo" not in str(c) and "alpha" not in str(c)]
    if candidates: posthoc_path = candidates[0]

print(f"Using: {posthoc_path}")
try:
    data = json.loads(posthoc_path.read_text())
    # delta_per_pair might be in different structure - find it
    deltas = None
    for k in ["delta_per_pair", "paired_deltas", "deltas"]:
        if k in data: deltas = data[k]; break
    if deltas is None and "threshold_free_authority_unsafe" in data:
        # Try reconstruction from histograms not possible; use ecdf csv
        import csv
        ecdf_path = posthoc_path.parent / "authority_unsafe_ecdf.csv"
        if ecdf_path.exists():
            rows = list(csv.DictReader(ecdf_path.open()))
            # Each row has baseline and intervention logit_diff
            base_vals = [float(r.get("baseline_logit_diff", r.get("baseline", 0))) for r in rows]
            intv_vals = [float(r.get("intervention_logit_diff", r.get("intervention", 0))) for r in rows]
            deltas = [i - b for i, b in zip(intv_vals, base_vals)]

    if deltas is None:
        print("Could not locate per-pair deltas; skipping")
    else:
        deltas = list(deltas)
        print(f"Loaded {len(deltas)} paired deltas from expanded")
        rng = random.Random(42)
        n_boot = 1000
        sub_size = 70
        subsample_means = []
        for _ in range(n_boot):
            sub = rng.sample(deltas, sub_size)
            subsample_means.append(sum(sub)/len(sub))
        arr = np.array(subsample_means)
        result = {
            "n_bootstrap": n_boot, "subsample_size": sub_size,
            "full_mean": float(np.mean(deltas)),
            "subsample_mean_mean": float(arr.mean()),
            "subsample_mean_std": float(arr.std()),
            "subsample_mean_p2.5": float(np.percentile(arr, 2.5)),
            "subsample_mean_p50": float(np.percentile(arr, 50)),
            "subsample_mean_p97.5": float(np.percentile(arr, 97.5)),
            "share_neg": float((arr < 0).mean()),
            "contains_curated_value": float((arr <= -0.093).mean()),
        }
        out = Path("results/supplementary/signflip_bootstrap.json")
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print(f"Saved: {out}")
except Exception as e:
    print(f"Error: {e}")
PYEOF

# ============================================================
# P3-G: Direction transfer across datasets (3h)
# ============================================================
echo ""
echo "### P3-G: Direction transfer cross-dataset (3h) ###"
# Apply curated direction to expanded data and vice versa
if [ -f "$CURATED_DIR" ] && [ -f "$EXPANDED_DIR" ]; then
    python scripts/run_frozen_direction_replay.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --direction "$CURATED_DIR" --layer 10 --alpha 1.0 \
        --prompt-dataset data/prompts_expanded.jsonl \
        --output-dir results/supplementary/transfer_curated_to_expanded \
        2>&1 | tee /tmp/neurips_logs/transfer_c2e.log || echo "FAILED"

    python scripts/run_frozen_direction_replay.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --direction "$EXPANDED_DIR" --layer 10 --alpha 1.0 \
        --prompt-dataset data/prompts.jsonl \
        --output-dir results/supplementary/transfer_expanded_to_curated \
        2>&1 | tee /tmp/neurips_logs/transfer_e2c.log || echo "FAILED"
fi

# ============================================================
# P3-H: Layer x Alpha heatmap on curated (4h)
# ============================================================
echo ""
echo "### P3-H: Layer x Alpha heatmap sweep (4h) ###"
python scripts/run_layer_alpha_sweep.py \
    --config configs/cloud/llama3_8b_l4_safe.yaml \
    --layers 8 10 12 14 \
    --alphas 0.25 0.5 1.0 1.5 \
    --output-dir results/supplementary/layer_alpha_heatmap \
    2>&1 | tee /tmp/neurips_logs/layer_alpha_heatmap.log || echo "FAILED"

# ============================================================
# P3-I: Qwen-2.5-14B with int8 (3h)
# ============================================================
echo ""
echo "### P3-I: Qwen-2.5-14B int8 (3h) ###"
python scripts/run_experiment.py --config configs/cloud/qwen25_14b_l4_int8.yaml \
    2>&1 | tee /tmp/neurips_logs/qwen25_14b.log || echo "FAILED"

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "End: $(date)"
echo "Logs: /tmp/neurips_logs/"
echo "Results: results/supplementary/"
echo "=========================================="
