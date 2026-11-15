from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cuawright.models.anthropic_cost import (
    AnthropicCostTracker,
    calculate_message_cost,
)
from cuawright.models.openrouter_cost import CostAccountingError


class AnthropicCostTrackerTests(unittest.TestCase):
    def test_records_native_usage_at_public_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anthropic-cost.json"
            tracker = AnthropicCostTracker(
                max_dollars=1.0,
                artifact_path=path,
            )
            message = SimpleNamespace(
                id="msg_1",
                model="claude-opus-5",
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=20,
                    cache_read_input_tokens=40,
                    cache_creation_input_tokens=30,
                    cache_creation=SimpleNamespace(
                        ephemeral_5m_input_tokens=30,
                        ephemeral_1h_input_tokens=0,
                    ),
                    output_tokens_details=SimpleNamespace(thinking_tokens=12),
                ),
            )

            record = tracker.record_message(message)

            self.assertAlmostEqual(
                record.cost_usd,
                (100 * 5 + 20 * 25 + 40 * 0.5 + 30 * 6.25) / 1_000_000,
            )
            payload = json.loads(path.read_text())
            self.assertEqual(payload["provider"], "anthropic")
            self.assertEqual(payload["requests"], 1)
            self.assertEqual(
                payload["generations"][0]["thinking_tokens"],
                12,
            )
            self.assertEqual(payload["generations"][0]["iterations"], 1)

    def test_sums_server_compaction_iterations(self) -> None:
        message = {
            "id": "msg_compact",
            "model": "claude-opus-5",
            "usage": {
                "input_tokens": 2,
                "output_tokens": 79,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 6008,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 5114,
                    "ephemeral_1h_input_tokens": 0,
                },
                "iterations": [
                    {
                        "input_tokens": 45,
                        "output_tokens": 2307,
                        "cache_read_input_tokens": 148323,
                        "cache_creation_input_tokens": 5114,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 5114,
                            "ephemeral_1h_input_tokens": 0,
                        },
                    },
                    {
                        "input_tokens": 2,
                        "output_tokens": 79,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 6008,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 6008,
                            "ephemeral_1h_input_tokens": 0,
                        },
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AnthropicCostTracker(
                max_dollars=None,
                artifact_path=Path(tmp) / "anthropic-cost.json",
            )

            record = tracker.record_message(message)

        self.assertEqual(record.iterations, 2)
        self.assertEqual(record.input_tokens, 47)
        self.assertEqual(record.output_tokens, 2386)
        self.assertEqual(record.cache_read_input_tokens, 148323)
        self.assertEqual(record.cache_creation_input_tokens, 11122)
        self.assertAlmostEqual(record.cost_usd, 0.203559)

    def test_iterations_own_cache_tiers_and_thinking_without_top_level_counts(
        self,
    ) -> None:
        record = calculate_message_cost(
            {
                "id": "msg_iterations",
                "model": "claude-opus-5",
                "usage": {
                    "input_tokens": 10_000,
                    "output_tokens": 10_000,
                    "cache_read_input_tokens": 10_000,
                    "cache_creation_input_tokens": 10_000,
                    "output_tokens_details": {"thinking_tokens": 10_000},
                    "iterations": [
                        {
                            "input_tokens": 11,
                            "output_tokens": 12,
                            "cache_read_input_tokens": 13,
                            "cache_creation_input_tokens": 50,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": 20,
                                "ephemeral_1h_input_tokens": 30,
                            },
                            "output_tokens_details": {"thinking_tokens": 7},
                        },
                        {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "cache_read_input_tokens": 3,
                            "cache_creation_input_tokens": 10,
                            "output_tokens_details": {"thinking_tokens": 8},
                        },
                    ],
                },
            },
            cache_ttl="1h",
        )

        self.assertEqual(record.input_tokens, 12)
        self.assertEqual(record.output_tokens, 14)
        self.assertEqual(record.cache_read_input_tokens, 16)
        self.assertEqual(record.cache_creation_input_tokens, 60)
        self.assertEqual(record.cache_creation_5m_input_tokens, 20)
        self.assertEqual(record.cache_creation_1h_input_tokens, 40)
        self.assertEqual(record.thinking_tokens, 15)
        self.assertEqual(record.iterations, 2)

    def test_rejects_models_without_known_direct_pricing(self) -> None:
        with self.assertRaisesRegex(CostAccountingError, "no pricing"):
            calculate_message_cost(
                {
                    "id": "msg_other_model",
                    "model": "claude-sonnet-4",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                    },
                }
            )

    def test_records_fable_51_usage_at_public_rates(self) -> None:
        record = calculate_message_cost(
            {
                "id": "msg_fable",
                "model": "claude-fable-5-1",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 40,
                    "cache_creation_input_tokens": 30,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 30,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            }
        )

        self.assertAlmostEqual(
            record.cost_usd,
            (100 * 10 + 20 * 50 + 40 * 0.25 + 30 * 12.5) / 1_000_000,
        )


if __name__ == "__main__":
    unittest.main()
