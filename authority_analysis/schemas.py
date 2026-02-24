from __future__ import annotations

from dataclasses import dataclass
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

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SemanticRequest":
        missing = [k for k in ["semantic_request_id", "base_request", "safety_label"] if k not in row]
        if missing:
            raise ValueError(f"Semantic request missing fields: {missing}")
        return cls(
            semantic_request_id=str(row["semantic_request_id"]),
            base_request=str(row["base_request"]),
            safety_label=str(row["safety_label"]),
            risk_level=str(row.get("risk_level", "controlled")),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "base_request": self.base_request,
            "framing_type": self.framing_type,
            "semantic_request_id": self.semantic_request_id,
            "full_prompt": self.full_prompt,
            "safety_label": self.safety_label,
            "risk_level": self.risk_level,
        }



def validate_prompt_row(row: dict[str, Any]) -> None:
    missing = sorted(PROMPT_REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"Prompt row missing required fields: {missing}")
