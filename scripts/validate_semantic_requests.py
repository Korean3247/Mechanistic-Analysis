#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "semantic_request_id",
    "base_request",
    "domain",
    "risk_tier",
    "expected_model_behavior",
    "rationale_short",
    "tags",
}

REQUIRED_DOMAINS = {
    "academic_integrity",
    "privacy",
    "security",
    "medical",
    "legal",
    "coding",
    "general",
}

ALLOWED_RISK = {"safe", "borderline", "unsafe"}
ALLOWED_BEHAVIOR = {"comply", "refuse"}
TARGET_RISK_RATIO = {"safe": 0.40, "borderline": 0.40, "unsafe": 0.20}
RISK_TOLERANCE = 0.02
SIMILARITY_THRESHOLD = 0.85

PROHIBITED_PATTERNS = [
    re.compile(r"\bstep[- ]by[- ]step\b", re.IGNORECASE),
    re.compile(r"\bweapon construction\b", re.IGNORECASE),
    re.compile(r"\bbuild (a|an) weapon\b", re.IGNORECASE),
    re.compile(r"\bmake (a|an) bomb\b", re.IGNORECASE),
    re.compile(r"\bself[- ]harm\b", re.IGNORECASE),
    re.compile(r"\bsuicide\b", re.IGNORECASE),
    re.compile(r"\bexploit code\b", re.IGNORECASE),
    re.compile(r"\bshellcode\b", re.IGNORECASE),
    re.compile(r"\bdoxx\b", re.IGNORECASE),
    re.compile(r"\bextract .*social security number\b", re.IGNORECASE),
]

PAIR_ID_RE = re.compile(r"^pair_id:[A-Z]{2}_[0-9]{3}$")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "against",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "could",
    "describe",
    "discuss",
    "do",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "high",
    "how",
    "if",
    "in",
    "include",
    "into",
    "is",
    "it",
    "its",
    "keep",
    "level",
    "might",
    "no",
    "non",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "outline",
    "over",
    "provide",
    "related",
    "review",
    "summarize",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "used",
    "using",
    "ways",
    "when",
    "while",
    "with",
    "without",
}


@dataclass
class ValidationError:
    check: str
    detail: str



def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            rows.append(obj)
    return rows



def find_pair_id(tags: list[str]) -> str | None:
    for t in tags:
        if t.startswith("pair_id:"):
            return t
    return None



def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()



def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalized(text))
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}



def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)



def check_similarity(rows: list[dict[str, Any]], threshold: float) -> list[ValidationError]:
    errors: list[ValidationError] = []
    by_domain: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(
            (row["semantic_request_id"], content_tokens(row["base_request"]))
        )

    for domain, items in by_domain.items():
        n = len(items)
        for i in range(n):
            id_a, tokens_a = items[i]
            for j in range(i + 1, n):
                id_b, tokens_b = items[j]
                score = jaccard_similarity(tokens_a, tokens_b)
                if score > threshold:
                    errors.append(
                        ValidationError(
                            check="similarity_threshold",
                            detail=(
                                f"{domain}: {id_a} vs {id_b} similarity={score:.3f} exceeds {threshold}"
                            ),
                        )
                    )
                    if len(errors) >= 25:
                        return errors
    return errors



def validate_rows(rows: list[dict[str, Any]]) -> list[ValidationError]:
    errors: list[ValidationError] = []

    if len(rows) < 300:
        errors.append(ValidationError("minimum_count", f"count={len(rows)} < 300"))

    ids = [row.get("semantic_request_id") for row in rows]
    id_counter = Counter(ids)
    dup_ids = [k for k, v in id_counter.items() if v > 1 and k is not None]
    if dup_ids:
        errors.append(ValidationError("unique_ids", f"duplicate semantic_request_id values: {dup_ids[:5]}"))

    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, row in enumerate(rows, start=1):
        missing = REQUIRED_FIELDS - set(row.keys())
        if missing:
            errors.append(ValidationError("required_fields", f"row {idx} missing fields: {sorted(missing)}"))
            continue

        for key in REQUIRED_FIELDS:
            if row[key] is None:
                errors.append(ValidationError("null_values", f"row {idx} field '{key}' is null"))

        for key in [
            "semantic_request_id",
            "base_request",
            "domain",
            "risk_tier",
            "expected_model_behavior",
            "rationale_short",
        ]:
            value = row[key]
            if not isinstance(value, str) or not value.strip():
                errors.append(ValidationError("empty_string", f"row {idx} field '{key}' must be non-empty string"))

        tags = row["tags"]
        if not isinstance(tags, list) or len(tags) < 2:
            errors.append(ValidationError("tags", f"row {idx} tags must be list with >=2 items"))
            continue
        if any((not isinstance(t, str) or not t.strip()) for t in tags):
            errors.append(ValidationError("tags", f"row {idx} tags must contain non-empty strings"))
            continue

        pair_tag = find_pair_id(tags)
        if pair_tag is None:
            errors.append(ValidationError("pair_id", f"row {idx} missing pair_id tag"))
            continue
        if not PAIR_ID_RE.match(pair_tag):
            errors.append(ValidationError("pair_id", f"row {idx} invalid pair_id format: {pair_tag}"))
            continue

        domain = row["domain"]
        risk = row["risk_tier"]
        behavior = row["expected_model_behavior"]
        base_request = row["base_request"]

        if domain not in REQUIRED_DOMAINS:
            errors.append(ValidationError("domain_values", f"row {idx} invalid domain: {domain}"))
        if risk not in ALLOWED_RISK:
            errors.append(ValidationError("risk_values", f"row {idx} invalid risk_tier: {risk}"))
        if behavior not in ALLOWED_BEHAVIOR:
            errors.append(ValidationError("behavior_values", f"row {idx} invalid expected_model_behavior: {behavior}"))

        if risk == "safe" and behavior != "comply":
            errors.append(ValidationError("behavior_mapping", f"row {idx} safe must map to comply"))
        if risk == "unsafe" and behavior != "refuse":
            errors.append(ValidationError("behavior_mapping", f"row {idx} unsafe must map to refuse"))

        sentence_count = len(re.findall(r"[.!?](?:\s|$)", base_request))
        if not (1 <= sentence_count <= 3):
            errors.append(
                ValidationError(
                    "base_request_length",
                    f"row {idx} should contain 1-3 sentences, detected {sentence_count}",
                )
            )

        for pat in PROHIBITED_PATTERNS:
            if pat.search(base_request):
                errors.append(
                    ValidationError(
                        "prohibited_patterns",
                        f"row {idx} matched prohibited pattern '{pat.pattern}'",
                    )
                )
                break

        pair_groups[pair_tag].append(row)

    domain_counter = Counter(row.get("domain") for row in rows)
    total = len(rows)
    for domain in sorted(REQUIRED_DOMAINS):
        ratio = domain_counter.get(domain, 0) / total if total else 0.0
        if ratio < 0.10:
            errors.append(
                ValidationError(
                    "domain_coverage",
                    f"domain {domain} ratio={ratio:.3f} below 0.10",
                )
            )

    risk_counter = Counter(row.get("risk_tier") for row in rows)
    for tier, target in TARGET_RISK_RATIO.items():
        ratio = risk_counter.get(tier, 0) / total if total else 0.0
        if abs(ratio - target) > RISK_TOLERANCE:
            errors.append(
                ValidationError(
                    "risk_distribution",
                    f"tier {tier} ratio={ratio:.3f} target={target:.2f} tol={RISK_TOLERANCE:.2f}",
                )
            )

    for pair_id, group in pair_groups.items():
        if len(group) not in {2, 3}:
            errors.append(
                ValidationError(
                    "pair_group_size",
                    f"{pair_id} has {len(group)} entries; expected 2 or 3",
                )
            )
        group_domains = {g["domain"] for g in group if "domain" in g}
        if len(group_domains) > 1:
            errors.append(
                ValidationError(
                    "pair_group_domain",
                    f"{pair_id} spans multiple domains: {sorted(group_domains)}",
                )
            )

    errors.extend(check_similarity(rows, SIMILARITY_THRESHOLD))

    return errors



def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def write_checksum(path: Path, checksum: str) -> Path:
    out_path = path.with_suffix(path.suffix + ".sha256")
    out_path.write_text(f"{checksum}  {path.name}\n", encoding="utf-8")
    return out_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Validate semantic request dataset")
    parser.add_argument("--input", default="data/semantic_requests.jsonl")
    parser.add_argument("--write-checksum", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(input_path)
    errors = validate_rows(rows)

    total = len(rows)
    risk_counter = Counter(row.get("risk_tier") for row in rows)
    domain_counter = Counter(row.get("domain") for row in rows)

    print(f"Total entries: {total}")
    print(
        "Risk distribution:",
        ", ".join(f"{k}={risk_counter.get(k, 0)}" for k in ["safe", "borderline", "unsafe"]),
    )
    print(
        "Domain counts:",
        ", ".join(f"{d}={domain_counter.get(d, 0)}" for d in sorted(REQUIRED_DOMAINS)),
    )

    if errors:
        print("\nVALIDATION FAILED")
        for err in errors:
            print(f"- [{err.check}] {err.detail}")
        sys.exit(2)

    checksum = sha256_of_file(input_path)
    print("\nVALIDATION PASSED")
    print(f"SHA256: {checksum}")

    if args.write_checksum:
        checksum_path = write_checksum(input_path, checksum)
        print(f"Checksum written to: {checksum_path}")


if __name__ == "__main__":
    main()
