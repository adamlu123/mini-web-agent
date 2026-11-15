from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))
MODULE_PATH = TOOLS_DIR / "build_cli_trace_comparison.py"
SPEC = importlib.util.spec_from_file_location("build_cli_trace_comparison", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load trace comparison builder")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_parse_views_requires_unique_safe_labels(tmp_path):
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    assert comparison.parse_views([f"5.5={first}", f"5.6={second}"]) == [
        ("5.5", first.resolve()),
        ("5.6", second.resolve()),
    ]
    with pytest.raises(RuntimeError, match="Duplicate"):
        comparison.parse_views([f"same={first}", f"same={second}"])
    with pytest.raises(RuntimeError, match="safe"):
        comparison.parse_views([f"../bad={first}", f"good={second}"])


def test_homepage_links_each_view():
    rendered = comparison.homepage(
        [
            {
                "label": "5.5",
                "title": "GPT-5.5",
                "summary": {
                    "mean": 0.6,
                    "evaluated": 105,
                    "taskCount": 105,
                    "scoreCount": 105,
                    "strictPasses": 25,
                },
            },
            {
                "label": "5.6",
                "title": "GPT-5.6 Sol",
                "summary": {
                    "mean": 0.67,
                    "evaluated": 105,
                    "taskCount": 105,
                    "scoreCount": 105,
                    "strictPasses": 26,
                },
            },
        ]
    )

    assert 'href="5.5/index.html"' in rendered
    assert 'href="5.6/index.html"' in rendered
    assert "60.00%" in rendered
    assert "67.00%" in rendered


def test_homepage_uses_aggregate_without_inventing_task_scores():
    rendered = comparison.homepage(
        [
            {
                "label": "cadgenbench",
                "title": "CADGenBench",
                "summaryScore": 0.4793,
                "summaryNote": "Official aggregate; per-fixture scores unavailable",
                "summary": {
                    "mean": None,
                    "evaluated": 81,
                    "taskCount": 81,
                    "scoreCount": 0,
                    "strictPasses": 0,
                },
            }
        ]
    )

    assert "47.93%" in rendered
    assert "per-fixture scores unavailable" in rendered
    assert "strict passes" not in rendered


def test_homepage_embeds_interactive_report():
    rendered = comparison.homepage(
        [
            {
                "label": "5.5",
                "title": "OSWorld GPT-5.5",
                "homepageGroup": "osworld",
                "homepageCost": 9.55,
                "summaryScore": 0.60,
                "summary": {
                    "mean": 0.60,
                    "taskCount": 105,
                    "scoreCount": 105,
                },
            },
            {
                "label": "5.6",
                "title": "OSWorld GPT-5.6",
                "homepageGroup": "osworld",
                "homepageCost": 7.06,
                "summaryScore": 0.67,
                "summary": {
                    "mean": 0.67,
                    "taskCount": 105,
                    "scoreCount": 105,
                },
            },
        ],
        "performance.html",
    )

    assert 'src="performance.html"' in rendered
    assert "See where the runs land." in rendered
    assert "<td>$9.55</td>" in rendered
    assert "<td>$7.06</td>" in rendered


def test_embedded_report_hides_full_explorer_chrome():
    rendered = comparison.embedded_report("<html><head></head><body></body></html>")

    assert 'id="cuawright-homepage-embed"' in rendered
    assert ".layout > aside" in rendered
    assert ".table-panel" in rendered
