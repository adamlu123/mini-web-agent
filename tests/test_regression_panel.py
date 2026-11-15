from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cuawright.run.panel import (
    BENCHCAD_TASK_IDS,
    CADGENBENCH_TASK_IDS,
    OSWORLD_TASK_IDS,
    PANEL_TASK_COUNT,
    PanelConfig,
    RegressionPanel,
    _panel_failed,
)


class RegressionPanelTests(unittest.TestCase):
    @staticmethod
    def config(root: Path) -> PanelConfig:
        return PanelConfig(
            result_root=root / "results",
            vm_path=root / "vm.qcow2",
            assets_dir=root / "assets",
            api_key_file=root / "key",
            proxy_config=root / "proxy.json",
            website_host_suffix="127.0.0.1.nip.io",
            benchcad_manifest=root / "benchcad.json",
            benchcad_repo=root / "benchcad",
            benchcad_image="benchcad:test",
            cadgenbench_dataset=root / "cadgen-data",
            cadgenbench_repo=root / "cadgen",
            cadgenbench_image="cadgen:test",
            health_delay_seconds=300,
            checkpoint_delay_seconds=1_800,
        )

    def test_fixed_panel_has_exact_27_unique_tasks(self) -> None:
        self.assertEqual(len(OSWORLD_TASK_IDS), 21)
        self.assertEqual(len(BENCHCAD_TASK_IDS), 3)
        self.assertEqual(len(CADGENBENCH_TASK_IDS), 3)
        self.assertEqual(PANEL_TASK_COUNT, 27)
        labels = {
            *(f"osworld:{value}" for value in OSWORLD_TASK_IDS),
            *(f"benchcad:{value}" for value in BENCHCAD_TASK_IDS),
            *(f"cadgenbench:{value}" for value in CADGENBENCH_TASK_IDS),
        }
        self.assertEqual(len(labels), PANEL_TASK_COUNT)

    def test_process_specs_launch_full_width_with_expected_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)

            specs = RegressionPanel(config)._process_specs()

        self.assertEqual(len(specs), 7)
        osworld = specs[0]
        self.assertTrue(osworld.manages_own_health)
        self.assertIn("--width", osworld.command)
        width_index = osworld.command.index("--width")
        self.assertEqual(osworld.command[width_index + 1], "21")
        self.assertEqual(osworld.command.count("--task-id"), 21)
        for spec in specs[1:]:
            self.assertIn("--model", spec.command)
            self.assertIn("gpt-5.5", spec.command)
            self.assertIn("--reasoning-effort", spec.command)
            self.assertIn("xhigh", spec.command)

    def test_normal_completion_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            panel = RegressionPanel(self.config(Path(temporary)))
            panel._validate_inputs = Mock()
            panel._process_specs = Mock(return_value=())
            panel._prepare_root = Mock()
            panel._install_signal_handlers = Mock()
            panel._restore_signal_handlers = Mock()
            panel._wait_for_processes = Mock()
            panel._write_summary = Mock(return_value={"rows": []})
            with patch(
                "cuawright.run.panel.validate_source_tree",
                return_value="commit",
            ):
                self.assertEqual(panel.run(), 0)

    def test_panel_failure_detects_missing_or_failed_children(self) -> None:
        self.assertTrue(
            _panel_failed(
                {
                    "rows": [
                        {
                            "benchmark": "benchcad",
                            "exit_code": 1,
                            "result": None,
                        }
                    ]
                }
            )
        )
        self.assertTrue(
            _panel_failed(
                {
                    "rows": [
                        {
                            "benchmark": "benchcad",
                            "exit_code": 0,
                            "health": {"objective_stop_required": True},
                            "result": {"score": 1.0},
                        }
                    ]
                }
            )
        )
        self.assertTrue(
            _panel_failed(
                {
                    "rows": [
                        {
                            "benchmark": "cadgenbench",
                            "exit_code": 0,
                            "health": {"objective_stop_required": False},
                            "result": {"deadline_triggered": True},
                        }
                    ]
                }
            )
        )
        self.assertTrue(
            _panel_failed(
                {
                    "rows": [
                        {
                            "benchmark": "osworld",
                            "exit_code": 0,
                            "batch_summary": {
                                "nonzero_exits": 0,
                                "selection_errors": 1,
                                "health_failures": 0,
                            },
                        }
                    ]
                }
            )
        )

    def test_signal_handler_only_sets_stop_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            panel = RegressionPanel(self.config(Path(temporary)))
            panel._stop_all_processes = Mock()

            panel._handle_signal(2, object())

            self.assertTrue(panel.stop_event.is_set())
            panel._stop_all_processes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
