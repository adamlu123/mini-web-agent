from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


class DollarBudgetExceeded(RuntimeError):
    """Raised before a model request when its local dollar budget is exhausted."""


class CostAccountingError(RuntimeError):
    """Raised when an OpenRouter generation cannot be accounted exactly."""


@dataclass(frozen=True)
class OpenRouterGenerationCost:
    generation_id: str
    cost_usd: float
    model: str | None
    provider: str | None
    native_prompt_tokens: int | None
    native_completion_tokens: int | None
    native_cached_tokens: int | None
    native_reasoning_tokens: int | None


class OpenRouterCostTracker:
    def __init__(
        self,
        *,
        api_key: str,
        max_dollars: float | None,
        artifact_path: Path,
        generation_url: str = "https://openrouter.ai/api/v1/generation",
    ) -> None:
        if max_dollars is not None and max_dollars <= 0:
            raise ValueError("max_dollars must be positive")
        self.max_dollars = float(max_dollars) if max_dollars is not None else None
        self.artifact_path = artifact_path
        self.generation_url = generation_url
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._records: list[OpenRouterGenerationCost] = []
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
                "OpenRouter dollar budget exhausted: "
                f"${self._total_cost_usd:.6f} spent of ${self.max_dollars:.2f}"
            )

    def record_generation(
        self,
        generation_id: str,
    ) -> OpenRouterGenerationCost:
        if not generation_id:
            raise CostAccountingError(
                "OpenRouter response did not include a generation ID"
            )
        if any(record.generation_id == generation_id for record in self._records):
            raise CostAccountingError(
                f"OpenRouter generation was accounted twice: {generation_id}"
            )

        payload: dict[str, Any] | None = None
        last_error = ""
        # Successful generations are usually visible immediately. Malformed
        # streamed tool responses can take several minutes to appear even
        # though OpenRouter has already charged them, so accounting must wait
        # rather than silently omit the cost or abort before the model retry.
        for delay in (0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0):
            if delay:
                time.sleep(delay)
            try:
                response = self._session.get(
                    self.generation_url,
                    params={"id": generation_id},
                    timeout=30,
                )
                if response.status_code == 404:
                    last_error = "generation record not ready"
                    continue
                response.raise_for_status()
                candidate = response.json().get("data")
                if not isinstance(candidate, dict):
                    last_error = "response lacked a data object"
                    continue
                payload = candidate
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        if payload is None:
            raise CostAccountingError(
                f"Could not account OpenRouter generation {generation_id}: "
                f"{last_error}"
            )

        raw_cost = payload.get("total_cost", payload.get("usage"))
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError) as exc:
            raise CostAccountingError(
                f"OpenRouter generation {generation_id} lacked exact cost"
            ) from exc
        if cost < 0:
            raise CostAccountingError(
                f"OpenRouter generation {generation_id} returned negative cost"
            )

        record = OpenRouterGenerationCost(
            generation_id=generation_id,
            cost_usd=cost,
            model=_optional_text(payload.get("model")),
            provider=_optional_text(payload.get("provider_name")),
            native_prompt_tokens=_optional_int(payload.get("native_tokens_prompt")),
            native_completion_tokens=_optional_int(
                payload.get("native_tokens_completion")
            ),
            native_cached_tokens=_optional_int(payload.get("native_tokens_cached")),
            native_reasoning_tokens=_optional_int(
                payload.get("native_tokens_reasoning")
            ),
        )
        self._records.append(record)
        self._total_cost_usd += cost
        self._write()
        return record

    def snapshot(self) -> dict[str, Any]:
        return {
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
