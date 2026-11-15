from __future__ import annotations

import concurrent.futures
import importlib.util
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tools"
    / "build_cli_trace_viewer.py"
)
SPEC = importlib.util.spec_from_file_location("build_cli_trace_viewer", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load trace viewer builder")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class TraceViewerBuilderTests(unittest.TestCase):
    def test_static_app_always_renders_evaluator_rubric_state(self) -> None:
        source = (builder.STATIC_ROOT / "assets" / "app.js").read_text(encoding="utf-8")
        index = (builder.STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn("renderEvaluatorRubric(evaluation, task)", source)
        self.assertIn("renderPromptContext(task)", source)
        self.assertIn('<div class="prompt-text">', source)
        self.assertIn('id="prompt-toggle"', index)
        self.assertIn(
            "No public per-criterion rubric was retained for this trajectory.",
            source,
        )

    def test_launch_prompts_are_extracted_without_request_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            requests = task_dir / "requests"
            requests.mkdir()
            payload = {
                "model": "private-model-name",
                "input": [
                    {"role": "system", "content": "System contract."},
                    {"role": "user", "content": "Original task instruction."},
                    {"role": "user", "content": "Budget message."},
                ],
                "api_key": "must-not-ship",
            }
            with gzip.open(
                requests / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(payload, handle)

            prompts = builder.load_model_visible_prompts(task_dir)

            self.assertTrue(prompts["available"])
            self.assertEqual(prompts["system"], "System contract.")
            self.assertEqual(
                prompts["user"],
                ["Original task instruction.", "Budget message."],
            )
            self.assertNotIn("private-model-name", json.dumps(prompts))
            self.assertNotIn("must-not-ship", json.dumps(prompts))

    def test_launch_prompt_extraction_redacts_secrets_and_host_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            requests = task_dir / "requests"
            requests.mkdir()
            payload = {
                "system": (
                    "Use /data/iso-osworld-v2/private. "
                    'The sudo password is "guest-secret".'
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": "api_key=secret-value",
                    }
                ],
            }
            with gzip.open(
                requests / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(payload, handle)

            prompts = builder.load_model_visible_prompts(task_dir)

            self.assertEqual(
                prompts["system"],
                'Use [HOST_PATH] The sudo password is "[REDACTED]".',
            )
            self.assertEqual(prompts["user"], ["api_key=[REDACTED]"])

    def test_selection_map_accepts_arbitrary_unique_task_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.tsv"
            path.write_text(
                "task\tscore\tsource\n"
                "001\t1.0\t/results/shard-a\n"
                "002\t0.5\t/results/retry-002\n",
                encoding="utf-8",
            )

            rows = builder.read_selection_map(path)

            self.assertEqual([row["task"] for row in rows], ["001", "002"])

    def test_selection_map_rejects_duplicate_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.tsv"
            path.write_text(
                "task\tscore\tsource\n"
                "001\t0.0\t/results/original\n"
                "001\t1.0\t/results/retry\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                builder.read_selection_map(path)

    def test_package_manifest_omits_source_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.html").write_text("ok", encoding="utf-8")

            builder.write_package_manifest(
                root,
                {"tasks": [{"id": "001"}]},
                ["/data/iso-osworld-v2/results/private/task-001"],
            )

            manifest = json.loads((root / "package-manifest.json").read_text())
            self.assertEqual(manifest["sourceTaskCount"], 1)
            self.assertNotIn("sourceTaskDirectories", manifest)
            self.assertNotIn("/data/iso-osworld-v2", json.dumps(manifest))

    def test_build_task_accepts_direct_trace_and_missing_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "direct"
            task_dir.mkdir()
            (task_dir / "trace.jsonl").write_text("", encoding="utf-8")
            row = {
                "task": "fixture-1",
                "score": "",
                "source": str(task_dir),
                "title": "Fixture 1",
                "instruction": "Create the requested fixture.",
            }

            task, sources = builder.build_task(
                row,
                root / "site",
                root,
                builder.ImageStore(root / "site", 84),
            )

            self.assertIsNone(task["score"])
            self.assertFalse(task["strictPass"])
            self.assertEqual(task["title"], "Fixture 1")
            self.assertEqual(sources, [str(task_dir)])

    def test_redaction_removes_common_secret_forms(self) -> None:
        text = (
            "Authorization: Bearer abcdefghijklmnop "
            "api_key=secret-value sk-abcdefghijklmnop"
        )
        redacted = builder.redact_text(text)
        self.assertNotIn("abcdefghijklmnop", redacted)
        self.assertNotIn("secret-value", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_redaction_removes_host_environment_details(self) -> None:
        text = (
            "/data/iso-osworld-v2/results/run "
            "/home/v-theolee/.a-creds "
            "https://gateway.phyagi.net/api "
            "call_abcdefghijklmnop"
        )

        redacted = builder.redact_text(text)

        self.assertNotIn("/data/iso-osworld-v2", redacted)
        self.assertNotIn("/home/v-theolee", redacted)
        self.assertNotIn("gateway.phyagi.net", redacted)
        self.assertNotIn("call_abcdefghijklmnop", redacted)

    def test_strict_pass_tolerates_evaluator_float_noise(self) -> None:
        self.assertTrue(builder.is_strict_pass(1.0))
        self.assertTrue(builder.is_strict_pass(0.9999999999999999))
        self.assertFalse(builder.is_strict_pass(0.999))

    def test_collection_profile_uses_benchmark_specific_metric(self) -> None:
        profile = builder.collection_profile(
            [{"task": "001", "score": "0.5", "source": "run"}],
            "BenchCAD GPT-5.6 trajectories",
        )

        self.assertEqual(profile["benchmark"], "benchcad")
        self.assertEqual(profile["benchmarkLabel"], "BenchCAD")
        self.assertEqual(profile["metricLabel"], "IoU")
        self.assertEqual(profile["metricKind"], "overlap")

    def test_event_role_prefers_specific_stage_over_legacy_combined_role(self) -> None:
        self.assertEqual(builder.event_role("binder-critic", "critic"), "Critic")
        self.assertEqual(
            builder.event_role("actor-orchestrator", "orchestrate"), "Actor"
        )

    def test_related_software_parser_accepts_json_or_csv(self) -> None:
        self.assertEqual(
            builder.parse_list_field('["thunderbird", "calendar"]'),
            ["thunderbird", "calendar"],
        )
        self.assertEqual(
            builder.parse_list_field("gimp, image viewer"),
            ["gimp", "image viewer"],
        )

    def test_related_software_inference_uses_task_visible_apps(self) -> None:
        task = {
            "instruction": (
                "Open MailHub, cross-check the charge in VaultBank, and submit "
                "the claim through ExpenseFlow in Chrome."
            )
        }

        self.assertEqual(
            builder.infer_related_software(task),
            ["MailHub", "VaultBank", "ExpenseFlow", "Chrome"],
        )

    def test_thumbnail_prefers_result_evidence(self) -> None:
        thumbnail = builder.choose_thumbnail(
            [
                {
                    "images": [
                        {
                            "path": "media/reference.webp",
                            "evidenceKind": "reference",
                            "width": 100,
                            "height": 80,
                        }
                    ]
                },
                {
                    "images": [
                        {
                            "path": "media/final.webp",
                            "evidenceKind": "result",
                            "width": 120,
                            "height": 90,
                        }
                    ]
                },
            ]
        )

        self.assertEqual(thumbnail["path"], "media/final.webp")
        self.assertEqual(thumbnail["evidenceKind"], "result")

    def test_public_evaluation_exposes_compact_rubric_breakdown(self) -> None:
        evaluation = builder.public_evaluation(
            {
                "score": 0.7,
                "partial_scores": {
                    "saved_file": {
                        "score": 1.0,
                        "weight": 0.4,
                        "description": "The required file exists.",
                        "evidence": "The evaluator opened the saved file.",
                    },
                    "exact_format": {
                        "score": 0.5,
                        "weight": 0.6,
                        "description": "Half of the requested formatting matched.",
                        "evidence": {"hidden": "structured evaluator internals"},
                    },
                },
            }
        )

        rows = evaluation["rubric"]["rows"]
        self.assertEqual(rows[0]["status"], "satisfied")
        self.assertEqual(rows[0]["contribution"], 0.4)
        self.assertEqual(rows[0]["evidence"], "The evaluator opened the saved file.")
        self.assertEqual(rows[1]["status"], "partial")
        self.assertEqual(rows[1]["contribution"], 0.3)
        self.assertEqual(rows[1]["evidence"], "")
        self.assertEqual(evaluation["rubric"]["satisfied"], 1)
        self.assertEqual(evaluation["rubric"]["partial"], 1)

    def test_public_evaluation_exposes_safety_and_phase_rows(self) -> None:
        safety = builder.public_evaluation(
            {
                "score": 0.0,
                "safety": {
                    "checks": [
                        {
                            "name": "document_integrity",
                            "result": "PASS",
                            "penalty": 0.0,
                        }
                    ]
                },
            }
        )
        phases = builder.public_evaluation(
            {
                "score": 0.35,
                "phase_results": [
                    {"phase_index": 1, "phase_name": "Create meeting", "score": 0.35}
                ],
            }
        )

        self.assertEqual(safety["rubric"]["rows"][0]["kind"], "safety")
        self.assertEqual(safety["rubric"]["rows"][0]["status"], "satisfied")
        self.assertEqual(phases["rubric"]["rows"][0]["kind"], "phase")
        self.assertEqual(phases["rubric"]["rows"][0]["status"], "credited")

    def test_reconstructed_rubric_replaces_scalar_only_breakdown(self) -> None:
        merged = builder.merge_reconstructed_rubric(
            {"score": 0.5, "rubric": {"rows": []}},
            {
                "sourceSummary": "Two equally weighted file checks.",
                "aggregation": "Average the two component scores.",
                "criteria": [
                    {
                        "id": "first_file",
                        "label": "First file",
                        "description": "The first file matches.",
                        "weight": 0.5,
                    },
                    {
                        "id": "second_file",
                        "label": "Second file",
                        "description": "The second file matches.",
                        "weight": 0.5,
                    },
                ],
                "runs": {
                    "5.5": {
                        "overall": 0.5,
                        "outcomeNote": "Recovered from the selected run cache.",
                        "criteria": [
                            {
                                "id": "first_file",
                                "score": 1.0,
                                "contribution": 0.5,
                                "status": "satisfied",
                                "evidence": "Matched.",
                            },
                            {
                                "id": "second_file",
                                "score": 0.0,
                                "contribution": 0.0,
                                "status": "not_satisfied",
                                "evidence": "Missing.",
                            },
                        ],
                    }
                },
            },
            "5.5",
        )

        self.assertTrue(merged["rubric"]["reconstructed"])
        self.assertEqual(
            merged["rubric"]["outcomeNote"],
            "Recovered from the selected run cache.",
        )
        self.assertEqual(merged["rubric"]["satisfied"], 1)
        self.assertEqual(merged["rubric"]["notSatisfied"], 1)
        self.assertEqual(merged["rubric"]["rows"][0]["contribution"], 0.5)

    def test_reconstructed_rubric_keeps_scalar_when_components_unknown(
        self,
    ) -> None:
        merged = builder.merge_reconstructed_rubric(
            0.75,
            {
                "sourceSummary": "Two checks.",
                "aggregation": "Average.",
                "criteria": [
                    {"id": "first", "weight": 0.5},
                    {"id": "second", "weight": 0.5},
                ],
                "runs": {
                    "current": {
                        "overall": 0.75,
                        "criteria": [
                            {"id": "first", "status": "unknown"},
                            {"id": "second", "status": "unknown"},
                        ],
                    }
                },
            },
            "current",
        )

        self.assertEqual(merged["rubric"]["recorded"], 1)
        self.assertEqual(merged["rubric"]["unknown"], 2)
        self.assertEqual(merged["rubric"]["rows"][0]["kind"], "aggregate")
        self.assertEqual(merged["rubric"]["rows"][0]["score"], 0.75)

    def test_build_task_exposes_identity_and_result_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            task_dir.mkdir()
            (task_dir / "trace.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "event": "submission_accepted",
                        "forced": False,
                        "missing_reports": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (task_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark": "benchcad",
                        "model": "gpt-5.6-sol",
                        "reasoning_effort": "high",
                        "release": "benchcad-v1",
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "orchestration.json").write_text(
                json.dumps(
                    {
                        "completed": True,
                        "stop_reason": "completed",
                        "reports": {"verifier": {"status": "ready"}},
                        "budget": {"used_steps": 12, "by_role": {"actor": 8}},
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "evaluation").mkdir()
            (task_dir / "evaluation" / "result.json").write_text(
                json.dumps({"evaluation": {"status": "ok", "iou": 0.75}}),
                encoding="utf-8",
            )
            row = {
                "task": "washer-1",
                "score": "0.75",
                "source": str(task_dir),
                "instruction": "Build the washer.",
                "benchmark": "benchcad",
                "benchmark_label": "BenchCAD",
                "metric_label": "IoU",
                "metric_kind": "overlap",
            }

            task, _ = builder.build_task(
                row,
                root / "site",
                root,
                builder.ImageStore(root / "site", 84),
            )

            self.assertEqual(task["modelLabel"], "gpt-5.6-sol")
            self.assertEqual(task["reasoningEffort"], "high")
            self.assertEqual(task["metricLabel"], "IoU")
            self.assertEqual(task["evaluationStatus"], "evaluated")
            self.assertEqual(task["resultSummary"]["submissionStatus"], "accepted")
            self.assertEqual(task["resultSummary"]["verificationStatus"], "complete")
            self.assertEqual(
                task["resultSummary"]["specialists"], {"Verifier": "ready"}
            )

    def test_event_normalization_keeps_model_visible_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            task_dir.mkdir()
            image_dir = task_dir / "images"
            image_dir.mkdir()
            Image.new("RGB", (16, 10), "white").save(image_dir / "seen.png")
            rows = [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "event": "shared_budget_step",
                    "global_step": 1,
                    "budget": {
                        "total_steps": 300,
                        "used_steps": 1,
                        "remaining_steps": 299,
                        "progress_percent": 0.333,
                    },
                },
                {
                    "timestamp": "2026-01-01T00:00:01+00:00",
                    "event": "model_step",
                    "role": "actor-orchestrator",
                    "stage": "orchestrate",
                    "step": 1,
                    "reasoning_summaries": ["Inspect the result."],
                    "tool_calls": [
                        {
                            "name": "run_command",
                            "arguments": {"command": "echo ok"},
                        }
                    ],
                    "output_text": "",
                },
                {
                    "timestamp": "2026-01-01T00:00:02+00:00",
                    "event": "tool_result",
                    "role": "actor-orchestrator",
                    "stage": "orchestrate",
                    "step": 1,
                    "tool": "run_command",
                    "output": '{"status":"success","api_key":"secret-value"}',
                    "image_blob": "images/seen.png",
                },
            ]
            with (task_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            store = builder.ImageStore(root / "site", 84)
            events, timing = builder.normalize_events(task_dir, store)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["role"], "Actor")
            self.assertEqual(
                events[0]["reasoning"],
                [
                    {
                        "text": "Inspect the result.",
                        "provenance": "trace_fallback",
                    }
                ],
            )
            self.assertIn("[REDACTED]", events[0]["observations"][0]["output"])
            self.assertEqual(len(events[0]["images"]), 1)
            self.assertTrue((root / "site" / events[0]["images"][0]["path"]).exists())
            self.assertEqual(timing["durationSeconds"], 2.0)

    def test_event_normalization_omits_host_only_command_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            task_dir.mkdir()
            (task_dir / "trace.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "event": "command_result",
                        "role": "actor-orchestrator",
                        "stdout": "x" * 1000,
                        "stderr": "warning",
                        "exit_code": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            events, _ = builder.normalize_events(
                task_dir, builder.ImageStore(root / "site", 84)
            )

            self.assertNotIn("stdout", events[0]["raw"])
            self.assertNotIn("stderr", events[0]["raw"])
            self.assertEqual(events[0]["raw"], {})

    def test_model_turn_uses_recorded_summary_and_correlates_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            responses = task_dir / "responses"
            task_dir.mkdir()
            responses.mkdir()
            response = {
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "Inspect the task before editing.",
                            }
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_abcdefghijklmnop",
                        "name": "run_command",
                        "arguments": json.dumps(
                            {"command": "printf ok", "timeout": 30}
                        ),
                    },
                ]
            }
            with gzip.open(
                responses / "step-0001.json.gz", "wt", encoding="utf-8"
            ) as handle:
                json.dump(response, handle)
            rows = [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "event": "model_turn",
                    "role": "actor",
                    "stage": "orchestrate",
                    "global_step": 1,
                    "billed_responses": 1,
                    "tool_calls": [
                        {
                            "call_id": "call_abcdefghijklmnop",
                            "name": "run_command",
                        }
                    ],
                    "text": None,
                },
                {
                    "timestamp": "2026-01-01T00:00:01+00:00",
                    "event": "terminal_command",
                    "command": "printf ok",
                    "timeout": 30,
                },
                {
                    "timestamp": "2026-01-01T00:00:02+00:00",
                    "event": "command_result",
                    "role": "actor",
                    "stage": "orchestrate",
                    "global_step": 1,
                    "call_id": "call_abcdefghijklmnop",
                    "exit_code": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "attachments": [],
                },
            ]
            with (task_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            events, _ = builder.normalize_events(
                task_dir, builder.ImageStore(root / "site", 84)
            )

            self.assertEqual(len(events), 1)
            self.assertEqual(
                events[0]["reasoning"],
                [
                    {
                        "text": "Inspect the task before editing.",
                        "provenance": "provider_response",
                    }
                ],
            )
            self.assertEqual(
                events[0]["toolCalls"][0]["arguments"]["command"], "printf ok"
            )
            self.assertEqual(events[0]["observations"][0]["output"], "ok")
            self.assertEqual(
                events[0]["observations"][0]["visibility"], "host_recorded"
            )
            self.assertNotIn("call_id", events[0]["toolCalls"][0])

    def test_observation_cap_preserves_head_and_tail(self) -> None:
        source = "head" + ("x" * builder.MAX_OBSERVATION_CHARS) + "tail"

        capped, truncated = builder.cap_observation(source)

        self.assertTrue(truncated)
        self.assertTrue(capped.startswith("head"))
        self.assertTrue(capped.endswith("tail"))
        self.assertIn("recorded characters omitted", capped)

    def test_reused_image_path_preserves_event_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            image_dir = task_dir / "images"
            image_dir.mkdir(parents=True)
            first = image_dir / "first.png"
            second = image_dir / "second.png"
            Image.new("RGB", (4, 4), "white").save(first)
            Image.new("RGB", (4, 4), "black").save(second)
            first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
            second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
            first.rename(image_dir / f"{first_digest}.png")
            second.rename(image_dir / f"{second_digest}.png")
            rows = []
            for step, call_id, digest in (
                (1, "call-1", first_digest),
                (2, "call-2", second_digest),
            ):
                rows.extend(
                    [
                        {
                            "timestamp": f"2026-01-01T00:00:0{step}+00:00",
                            "event": "model_turn",
                            "role": "actor",
                            "stage": "orchestrate",
                            "global_step": step,
                            "tool_calls": [
                                {
                                    "call_id": call_id,
                                    "name": "run_command",
                                    "arguments": {"command": "show"},
                                }
                            ],
                        },
                        {
                            "timestamp": f"2026-01-01T00:00:0{step}+00:00",
                            "event": "image_attached",
                            "source_path": "/tmp/reused.png",
                            "sha256": digest,
                        },
                        {
                            "timestamp": f"2026-01-01T00:00:0{step}+00:00",
                            "event": "command_result",
                            "call_id": call_id,
                            "exit_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "attachments": [
                                {
                                    "kind": "image",
                                    "source_path": "/tmp/reused.png",
                                }
                            ],
                        },
                    ]
                )
            with (task_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            events, _ = builder.normalize_events(
                task_dir, builder.ImageStore(root / "site", 84)
            )

            self.assertEqual(events[0]["images"][0]["sha256"], first_digest)
            self.assertEqual(events[1]["images"][0]["sha256"], second_digest)

    def test_image_store_deduplicates_concurrent_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            Image.new("RGB", (16, 16), "white").save(source)
            store = builder.ImageStore(root / "site", 84)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                images = list(executor.map(store.add, [source] * 16))

            self.assertTrue(all(image == images[0] for image in images))
            self.assertEqual(len(list((root / "site/media").glob("*.webp"))), 1)

    def test_corrupt_image_is_recorded_without_aborting_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            image_dir = task_dir / "images"
            image_dir.mkdir(parents=True)
            raw = b"truncated-png"
            digest = hashlib.sha256(raw).hexdigest()
            (image_dir / f"{digest}.png").write_bytes(raw)
            rows = [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "event": "model_turn",
                    "role": "actor",
                    "stage": "orchestrate",
                    "global_step": 1,
                    "tool_calls": [{"call_id": "call-1", "name": "run_command"}],
                },
                {
                    "timestamp": "2026-01-01T00:00:01+00:00",
                    "event": "image_attached",
                    "source_path": "/tmp/broken.png",
                    "sha256": digest,
                },
                {
                    "timestamp": "2026-01-01T00:00:02+00:00",
                    "event": "command_result",
                    "call_id": "call-1",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "attachments": [
                        {"kind": "image", "source_path": "/tmp/broken.png"}
                    ],
                },
            ]
            with (task_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            events, _ = builder.normalize_events(
                task_dir, builder.ImageStore(root / "site", 84)
            )

            self.assertEqual(events[0]["images"], [])
            self.assertEqual(
                events[0]["attachmentWarnings"][0]["sourcePath"],
                "/tmp/broken.png",
            )

    def test_specialist_artifacts_preserve_phase_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for phase, request in (("phase-001", "first"), ("phase-002", "second")):
                directory = root / "specialists" / phase / "binder-01"
                directory.mkdir(parents=True)
                (directory / "request.txt").write_text(request, encoding="utf-8")
                (directory / "report.json").write_text(
                    json.dumps({"status": "complete", "report": request}),
                    encoding="utf-8",
                )

            artifacts = builder.load_specialist_artifacts(root)

            self.assertEqual(
                [artifact["artifactKey"] for artifact in artifacts],
                [
                    "specialist-001",
                    "specialist-002",
                ],
            )
            self.assertEqual(
                [artifact["request"] for artifact in artifacts],
                ["first", "second"],
            )

    def test_event_chunks_respect_target_size_for_normal_entries(self) -> None:
        events = [{"id": str(index), "text": "x" * 200_000} for index in range(100)]
        chunks = builder.chunk_events(events)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            encoded = json.dumps(
                chunk,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertLessEqual(len(encoded), builder.CHUNK_TARGET_BYTES + 200_000)


if __name__ == "__main__":
    unittest.main()
