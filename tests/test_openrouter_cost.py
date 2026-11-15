from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from cuawright.models.openrouter_cost import (
    DollarBudgetExceeded,
    OpenRouterCostTracker,
)


class OpenRouterCostTrackerTests(unittest.TestCase):
    @staticmethod
    def response(status_code: int, payload: dict) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def test_records_exact_cost_atomically_and_enforces_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openrouter-cost.json"
            tracker = OpenRouterCostTracker(
                api_key="secret",
                max_dollars=0.25,
                artifact_path=path,
            )
            tracker._session.get = Mock(
                side_effect=[
                    self.response(404, {}),
                    self.response(
                        200,
                        {
                            "data": {
                                "id": "gen_1",
                                "total_cost": 0.25,
                                "model": "anthropic/claude-opus-5",
                                "provider_name": "provider",
                                "native_tokens_prompt": 100,
                                "native_tokens_completion": 20,
                                "native_tokens_cached": 40,
                                "native_tokens_reasoning": 10,
                            }
                        },
                    ),
                ]
            )

            with unittest.mock.patch("cuawright.models.openrouter_cost.time.sleep"):
                record = tracker.record_generation("gen_1")

            self.assertEqual(record.cost_usd, 0.25)
            self.assertEqual(tracker.total_cost_usd, 0.25)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["generations"][0]["generation_id"], "gen_1")
            with self.assertRaises(DollarBudgetExceeded):
                tracker.ensure_request_allowed()

    def test_records_cost_without_enforcing_a_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openrouter-cost.json"
            tracker = OpenRouterCostTracker(
                api_key="secret",
                max_dollars=None,
                artifact_path=path,
            )
            tracker._session.get = Mock(
                return_value=self.response(
                    200,
                    {
                        "data": {
                            "id": "gen_1",
                            "total_cost": 12.5,
                        }
                    },
                )
            )

            record = tracker.record_generation("gen_1")
            tracker.ensure_request_allowed()

            self.assertEqual(record.cost_usd, 12.5)
            self.assertIsNone(tracker.remaining_dollars)
            payload = json.loads(path.read_text())
            self.assertIsNone(payload["max_dollars"])
            self.assertIsNone(payload["remaining_dollars"])


if __name__ == "__main__":
    unittest.main()
