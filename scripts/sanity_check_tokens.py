#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from transformers import AutoTokenizer

from authority_analysis.behavior_evaluator import COMPLIANCE_CUES, REFUSAL_CUES



def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check tokenization for probe labels and cues")
    parser.add_argument("--model", default="meta-llama/Llama-3-8B-Instruct")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"model={args.model}")

    failed = False
    print("\n[probe_labels]")
    for label in ["yes", "no"]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        print(f"{label}: len={len(ids)} ids={ids}")
        if len(ids) != 1:
            failed = True

    print("\n[refusal_cues]")
    for cue in REFUSAL_CUES:
        ids = tokenizer.encode(cue, add_special_tokens=False)
        first_id = ids[0] if ids else None
        print(f"cue={cue!r} len={len(ids)} first_id={first_id}")

    print("\n[compliance_cues]")
    for cue in COMPLIANCE_CUES:
        ids = tokenizer.encode(cue, add_special_tokens=False)
        first_id = ids[0] if ids else None
        print(f"cue={cue!r} len={len(ids)} first_id={first_id}")

    if failed:
        print("\nFAILED: yes/no are not single-token labels under this tokenizer", file=sys.stderr)
        sys.exit(2)

    print("\nPASS: yes/no are single-token labels")


if __name__ == "__main__":
    main()
