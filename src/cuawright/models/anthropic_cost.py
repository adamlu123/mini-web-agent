from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from cuawright.models.openrouter_cost import CostAccountingError, DollarBudgetExceeded

MODEL_PRICING_USD_PER_MILLION = {
    "claude-opus-5": {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.0,
    },
    "claude-fable-5-1": {
        "input": 10.0,
        "output": 50.0,
        "cache_read": 0.25,
        "cache_write_5m": 12.5,
        "cache_write_1h": 20.0,
    },
}


def pricing_for_model(model: str) -> dict[str, float]:
    for prefix, pricing in MODEL_PRICING_USD_PER_MILLION.items():
        if model == prefix or model.startswith(f"{prefix}-"):
            return pricing
    raise CostAccountingError(
        f"Direct Anthropic cost accounting has no pricing for {model!r}"
    )


@dataclass(frozen=True)
class AnthropicGenerationCost:
    generation_id: str
    cost_usd: float
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cache_creation_5m_input_tokens: int
    cache_creation_1h_input_tokens: int
    thinking_tokens: int
    iterations: int


class AnthropicCostTracker:
    def __init__(
        self,
        *,
        max_dollars: float | None,
        artifact_path: Path,
        cache_ttl: Literal["5m", "1h"] = "5m",
        pricing_model: str = "claude-opus-5",
    ) -> None:
        if max_dollars is not None and max_dollars <= 0:
            raise ValueError("max_dollars must be positive")
        self.max_dollars = float(max_dollars) if max_dollars is not None else None
        self.artifact_path = artifact_path
        self.cache_ttl = cache_ttl
        self.pricing = pricing_for_model(pricing_model)
        self._records: list[AnthropicGenerationCost] = []
        self._total_cost_usd = 0.0
        self._write()

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def remaining_dollars(self) -> float | None:
        if self.max_dollars is None:
            return None
        return max(0.0, self.max_dollars - self._total_cost_usd)

    def ensure_request_allowed(self) -> None:
        if self.max_dollars is not None and self._total_cost_usd >= self.max_dollars:
            raise DollarBudgetExceeded(
                "Anthropic dollar budget exhausted: "
                f"${self._total_cost_usd:.6f} spent of ${self.max_dollars:.2f}"
            )

    def record_message(self, message: Any) -> AnthropicGenerationCost:
        record = calculate_message_cost(message, cache_ttl=self.cache_ttl)
        generation_id = record.generation_id
        if not generation_id:
            raise CostAccountingError(
                "Anthropic response did not include a generation ID"
            )
        if any(record.generation_id == generation_id for record in self._records):
            raise CostAccountingError(
                f"Anthropic generation was accounted twice: {generation_id}"
            )
        self._records.append(record)
        self._total_cost_usd += record.cost_usd
        self._write()
        return record

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": "anthropic",
            "pricing_usd_per_million": self.pricing,
            "max_dollars": self.max_dollars,
            "total_cost_usd": self._total_cost_usd,
            "remaining_dollars": self.remaining_dollars,
            "requests": len(self._records),
            "generations": [asdict(record) for record in self._records],
        }

    def _write(self) -> None:
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.artifact_path.with_suffix(self.artifact_path.suffix + ".tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.snapshot(), handle, indent=2)
            handle.write("\n")
        os.replace(temporary, self.artifact_path)
        self.artifact_path.chmod(0o600)


def _optional_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def calculate_message_cost(
    message: Any,
    *,
    cache_ttl: Literal["5m", "1h"] = "5m",
) -> AnthropicGenerationCost:
    generation_id = str(_field(message, "id") or "")
    model = _optional_text(_field(message, "model"))
    usage = _field(message, "usage")
    if not generation_id:
        raise CostAccountingError("Anthropic response did not include a generation ID")
    if model is None:
        raise CostAccountingError("Anthropic response did not include a model")
    pricing = pricing_for_model(model)
    if usage is None:
        raise CostAccountingError(
            f"Anthropic generation {generation_id} did not include usage"
        )

    iterations = list(_field(usage, "iterations") or [])
    usage_rows = iterations or [usage]
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation_total = 0
    cache_creation_5m = 0
    cache_creation_1h = 0
    iteration_thinking = 0
    for row in usage_rows:
        input_tokens += int(_field(row, "input_tokens") or 0)
        output_tokens += int(_field(row, "output_tokens") or 0)
        cache_read += int(_field(row, "cache_read_input_tokens") or 0)
        row_creation_total = int(_field(row, "cache_creation_input_tokens") or 0)
        cache_creation_total += row_creation_total
        cache_creation = _field(row, "cache_creation")
        row_creation_5m = int(_field(cache_creation, "ephemeral_5m_input_tokens") or 0)
        row_creation_1h = int(_field(cache_creation, "ephemeral_1h_input_tokens") or 0)
        detailed_creation = row_creation_5m + row_creation_1h
        if detailed_creation == 0 and row_creation_total:
            if cache_ttl == "1h":
                row_creation_1h = row_creation_total
            else:
                row_creation_5m = row_creation_total
        elif detailed_creation != row_creation_total:
            raise CostAccountingError(
                f"Anthropic generation {generation_id} returned inconsistent "
                "cache-creation usage within one iteration"
            )
        cache_creation_5m += row_creation_5m
        cache_creation_1h += row_creation_1h
        iteration_details = _field(row, "output_tokens_details")
        iteration_thinking += int(_field(iteration_details, "thinking_tokens") or 0)

    output_details = _field(usage, "output_tokens_details")
    thinking_tokens = iteration_thinking or int(
        _field(output_details, "thinking_tokens") or 0
    )
    cost = (
        input_tokens * pricing["input"]
        + cache_read * pricing["cache_read"]
        + cache_creation_5m * pricing["cache_write_5m"]
        + cache_creation_1h * pricing["cache_write_1h"]
        + output_tokens * pricing["output"]
    ) / 1_000_000
    return AnthropicGenerationCost(
        generation_id=generation_id,
        cost_usd=cost,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation_total,
        cache_creation_5m_input_tokens=cache_creation_5m,
        cache_creation_1h_input_tokens=cache_creation_1h,
        thinking_tokens=thinking_tokens,
        iterations=len(usage_rows),
    )
