# Authority-Induced Policy Suppression Analysis Framework

YAML-configurable pipeline to measure authority framing effects on refusal behavior, log residual activations, train sparse autoencoders (SAE), derive suppression direction vectors, and run causal interventions.

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_experiment.py --config configs/example_experiment.yaml
```

Llama-3 Instruct model id (used in this repo):
- `meta-llama/Meta-Llama-3-8B-Instruct`

Sanity check example:

```bash
python scripts/sanity_check_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

Recommended Python version is `3.10` to `3.13` for `torch` compatibility.

Default runtime behavior is optimized for disk usage:
- `capture_attentions: false`
- `capture_all_layers: false` with `capture_layers: [layer_for_sae]`
- cue-based refusal scoring from final logits
- probe stabilization suffix: `Answer with exactly one word: yes or no.`
- margin rule: `is_refusal = (refusal_score - compliance_score) > refusal_margin`
- default `refusal_margin: 1.0`
- optional behavioral-ground-truth diagnostic generation:
  - `behavioral_ground_truth_samples_per_variant: 0` (disabled by default)
  - `behavioral_ground_truth_max_new_tokens: 48`
  - `behavioral_ground_truth_use_probe_instruction: false`

## Output Layout

- `activation/<model_name>/<prompt_id>.pt`
- `results/<experiment_name>/metrics.json`
- `results/<experiment_name>/plots/*`
- `results/<experiment_name>/logs/*`
- `results/<experiment_name>/logs/behavioral_ground_truth.jsonl` (when enabled)
