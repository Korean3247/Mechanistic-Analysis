# Authority-Induced Policy Suppression Analysis Framework

YAML-configurable pipeline to measure authority framing effects on refusal behavior, log residual activations, train sparse autoencoders (SAE), derive suppression direction vectors, and run causal interventions.

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_experiment.py --config configs/example_experiment.yaml
```

Recommended Python version is `3.10` to `3.13` for `torch` compatibility.

## Output Layout

- `activation/<model_name>/<prompt_id>.pt`
- `results/<experiment_name>/metrics.json`
- `results/<experiment_name>/plots/*`
- `results/<experiment_name>/logs/*`
