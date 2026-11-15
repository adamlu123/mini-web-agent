from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cuawright.environments.benchcad.errors import BenchCADInfrastructureError
from cuawright.environments.cadgenbench.errors import (
    CadGenBenchInfrastructureError,
)
from cuawright.environments.control import StatePersistenceError
from cuawright.environments.osworld.lifecycle import InfrastructureError
from cuawright.run import benchcad, cadgenbench, osworld


class StatePersistenceClassificationTests(unittest.TestCase):
    def test_launchers_chain_state_persistence_as_benchmark_infrastructure(
        self,
    ) -> None:
        cases = (
            (
                osworld._run_with_infrastructure_classification,
                InfrastructureError,
            ),
            (
                benchcad._run_with_infrastructure_classification,
                BenchCADInfrastructureError,
            ),
            (
                cadgenbench._run_with_infrastructure_classification,
                CadGenBenchInfrastructureError,
            ),
        )
        for classify, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                persistence_error = StatePersistenceError("mirror failed")
                run = Mock(side_effect=persistence_error)

                with self.assertRaises(expected_type) as caught:
                    classify(run)

                self.assertIs(caught.exception.__cause__, persistence_error)
                run.assert_called_once_with()

    def test_osworld_state_persistence_retries_once_in_a_fresh_vm(self) -> None:
        first = SimpleNamespace(
            run=Mock(side_effect=StatePersistenceError("mirror failed")),
            cost_tracker=None,
        )
        second = SimpleNamespace(
            run=Mock(return_value={"benchmark": "osworld-v2"}),
            cost_tracker=None,
        )

        with (
            patch.object(osworld, "parse_args", return_value=osworld.parse_args([])),
            patch.object(
                osworld, "OSWorldRunner", side_effect=(first, second)
            ) as runner,
            patch("builtins.print"),
        ):
            osworld.main()

        self.assertEqual(runner.call_count, 2)
        first.run.assert_called_once_with()
        second.run.assert_called_once_with()

    def test_benchcad_state_persistence_retries_once_fresh(self) -> None:
        args = SimpleNamespace(
            manifest="manifest.json",
            record_id="record-1",
            mode="full",
            image="benchcad:test",
            benchcad_repo="benchcad-repo",
            api_key_file="key",
            result_root="results",
            model_provider="openai",
            base_url=None,
            model=None,
            reasoning_effort="xhigh",
            total_steps=200,
            compact_every=40,
            defer_score=False,
            log_level="INFO",
        )
        manifest = SimpleNamespace(
            benchmark_commit="commit",
            dataset_revision="revision",
            manifest_sha256="sha256",
        )
        task = SimpleNamespace(instruction="Build the part")
        first = SimpleNamespace(
            run_dir="first",
            run_terminal_benchmark=Mock(
                side_effect=StatePersistenceError("mirror failed")
            ),
        )
        second = SimpleNamespace(
            run_dir="second",
            run_terminal_benchmark=Mock(return_value={"benchmark": "benchcad"}),
        )

        with (
            patch.object(benchcad, "parse_args", return_value=args),
            patch.object(benchcad, "load_task", return_value=(manifest, task)),
            patch.object(benchcad, "TaskRunner", side_effect=(first, second)) as runner,
            patch.object(benchcad, "create_full_plugin", side_effect=(Mock(), Mock())),
            patch("builtins.print"),
        ):
            benchcad.main()

        self.assertEqual(runner.call_count, 2)
        first.run_terminal_benchmark.assert_called_once()
        second.run_terminal_benchmark.assert_called_once()


if __name__ == "__main__":
    unittest.main()
