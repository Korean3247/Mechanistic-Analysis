from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PROMPT_REQUIRED_FIELDS = {
    "prompt_id",
    "framing_type",
    "semantic_request_id",
    "full_prompt",
    "safety_label",
}


@dataclass
class SemanticRequest:
    semantic_request_id: str
    base_request: str
    safety_label: str
    risk_level: str = "controlled"
    domain: str = "general"
    risk_tier: str = "safe"
    expected_model_behavior: str = "comply"
    rationale_short: str = "Controlled semantic request for safety analysis."
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def _map_risk_tier_to_safety_label(risk_tier: str) -> str:
        if risk_tier == "safe":
            return "safe"
        # Borderline and unsafe are treated as non-benign for coarse safety labeling.
        return "unsafe"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SemanticRequest":
        missing = [k for k in ["semantic_request_id", "base_request"] if k not in row]
        if missing:
            raise ValueError(f"Semantic request missing fields: {missing}")

        risk_tier = str(row.get("risk_tier", "safe"))
        if "safety_label" in row:
            safety_label = str(row["safety_label"])
        elif "risk_tier" in row:
            safety_label = cls._map_risk_tier_to_safety_label(risk_tier)
        else:
            raise ValueError(
                "Semantic request must include either 'safety_label' or 'risk_tier'"
            )

        expected_model_behavior = str(row.get("expected_model_behavior", "comply"))
        tags = row.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        return cls(
            semantic_request_id=str(row["semantic_request_id"]),
            base_request=str(row["base_request"]),
            safety_label=safety_label,
            risk_level=str(row.get("risk_level", "controlled")),
            domain=str(row.get("domain", "general")),
            risk_tier=risk_tier,
            expected_model_behavior=expected_model_behavior,
            rationale_short=str(
                row.get(
                    "rationale_short",
                    "Controlled semantic request for safety analysis.",
                )
            ),
            tags=[str(t) for t in tags if str(t).strip()],
        )


@dataclass
class PromptRecord:
    prompt_id: str
    base_request: str
    framing_type: str
    semantic_request_id: str
    full_prompt: str
    safety_label: str
    risk_level: str = "controlled"
    domain: str = "general"
    risk_tier: str = "safe"
    expected_model_behavior: str = "comply"
    rationale_short: str = "Controlled semantic request for safety analysis."
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "base_request": self.base_request,
            "framing_type": self.framing_type,
            "semantic_request_id": self.semantic_request_id,
            "full_prompt": self.full_prompt,
            "safety_label": self.safety_label,
            "risk_level": self.risk_level,
            "domain": self.domain,
            "risk_tier": self.risk_tier,
            "expected_model_behavior": self.expected_model_behavior,
            "rationale_short": self.rationale_short,
            "tags": self.tags,
        }



def validate_prompt_row(row: dict[str, Any]) -> None:
    missing = sorted(PROMPT_REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"Prompt row missing required fields: {missing}")
