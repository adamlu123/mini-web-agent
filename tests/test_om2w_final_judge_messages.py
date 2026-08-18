from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
from pathlib import Path

from miniswewebagent.evaluation.om2w import judge as robust_judge
from om2w_judge.methods import webjudge_online_mind2web as official_judge

# Source: OSU-NLP-Group/Online-Mind2Web at commit 0e00e251cd32ac8f6aa9d08d9e3474b63bd02330.
OFFICIAL_SOURCE_SHA256 = "e3cd499b7c1fc92cbd51d3c4216c95ebab774c2c2cd998c5ad6dad94710780c4"
VENDORED_SOURCE = (
    Path(__file__).resolve().parents[1] / "om2w_judge" / "methods" / "webjudge_online_mind2web.py"
)


async def _fake_identify_key_points(task, model):
    del task, model
    return "**Key Points**:\n1. Apply the requested filter"


class _BombModel:
    def generate(self, *args, **kwargs):
        raise AssertionError(f"unexpected model call: {args!r} {kwargs!r}")


def test_vendored_webjudge_matches_pinned_official_source() -> None:
    source = VENDORED_SOURCE.read_bytes()

    assert len(source) == 9962
    assert hashlib.sha256(source).hexdigest() == OFFICIAL_SOURCE_SHA256


def test_official_import_restores_preloaded_top_level_utils() -> None:
    code = """
import sys
import types

preloaded_utils = types.ModuleType("utils")
sys.modules["utils"] = preloaded_utils

from om2w_judge.methods import webjudge_online_mind2web

assert webjudge_online_mind2web.encode_image.__module__ == "om2w_judge.utils"
assert sys.modules["utils"] is preloaded_utils
"""
    subprocess.run([sys.executable, "-c", code], cwd=VENDORED_SOURCE.parents[2], check=True)


def test_retry_hardened_messages_match_official_without_snapshots(monkeypatch) -> None:
    monkeypatch.setattr(official_judge, "identify_key_points", _fake_identify_key_points)
    args = (
        "Apply the requested filter",
        ["Open filters", "Apply requested value"],
        [],
        _BombModel(),
        3,
    )

    official = asyncio.run(official_judge.WebJudge_Online_Mind2Web_eval(*args))
    hardened = asyncio.run(robust_judge.robust_webjudge_online_mind2web_eval(*args))

    assert official[:3] == hardened[:3]
    assert official[3] == hardened[3] == []
    assert official[4] == hardened[4]
    assert "potentially important snapshots" not in official[1]
    assert official[0][1]["content"] == [{"type": "text", "text": official[1]}]


def test_retry_hardened_messages_match_official_with_snapshot(monkeypatch) -> None:
    response = "**Reasoning**: selected evidence\n\n**Score**: 5"

    async def fake_judge_image(task, image_path, key_points, model):
        del task, image_path, key_points, model
        return response

    async def fake_judge_image_with_retry(task, image_path, key_points, model):
        del task, image_path, key_points, model
        return {
            "Response": response,
            "Score": 5,
            "Reasoning": "selected evidence",
            "Attempts": 1,
            "ParseFailed": False,
        }

    monkeypatch.setattr(official_judge, "identify_key_points", _fake_identify_key_points)
    monkeypatch.setattr(official_judge, "judge_image", fake_judge_image)
    monkeypatch.setattr(robust_judge, "judge_image_with_retry", fake_judge_image_with_retry)
    monkeypatch.setattr(official_judge, "encode_image", lambda image: "encoded")
    monkeypatch.setattr(official_judge.Image, "open", lambda path: object())
    args = (
        "Apply the requested filter",
        ["Open filters", "Apply requested value"],
        ["shot.png"],
        _BombModel(),
        3,
    )

    official = asyncio.run(official_judge.WebJudge_Online_Mind2Web_eval(*args))
    hardened = asyncio.run(robust_judge.robust_webjudge_online_mind2web_eval(*args))

    assert official[:3] == hardened[:3]
    assert "1. selected evidence" in official[1]
    assert official[0][1]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,encoded",
            "detail": "high",
        },
    }
