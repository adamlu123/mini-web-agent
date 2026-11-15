from __future__ import annotations

import argparse
import json
from typing import Any

from cuawright.benchmarks import BENCHMARKS, benchmark_descriptor
from cuawright.config.store import load_config, save_config
from cuawright.dependencies import require_optional_modules


def setup(args: argparse.Namespace) -> int:
    descriptor = benchmark_descriptor(args.benchmark)
    config = load_config()
    config[descriptor.name] = descriptor.setup_handler(args)
    save_config(config)
    print(json.dumps({"ok": True, "benchmark": descriptor.name}, indent=2))
    return 0


def doctor(args: argparse.Namespace) -> int:
    config = load_config()
    selected = [args.benchmark] if args.benchmark else list(BENCHMARKS)
    results: dict[str, Any] = {}
    for name in selected:
        descriptor = benchmark_descriptor(name)
        item = config.get(name)
        if not isinstance(item, dict):
            results[name] = {"ok": False, "error": "not configured"}
            continue
        try:
            results[name] = descriptor.doctor_handler(item)
        except Exception as exc:
            results[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(result.get("ok") for result in results.values()) else 1


def run(args: argparse.Namespace) -> int:
    descriptor = benchmark_descriptor(args.benchmark)
    item = load_config().get(descriptor.name)
    if not isinstance(item, dict):
        raise RuntimeError(
            f"{descriptor.name} is not configured; run "
            f"`cuawright setup {descriptor.name}`"
        )
    require_optional_modules(
        benchmark=descriptor.display_name,
        extra=descriptor.install_extra,
        modules=descriptor.required_modules,
        minimum_versions=descriptor.minimum_versions,
        requires_separate_runtime=descriptor.requires_separate_runtime,
    )
    return descriptor.run_handler(args, item)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cuawright")
    commands = root.add_subparsers(dest="command", required=True)

    setup_parser = commands.add_parser("setup")
    setup_targets = setup_parser.add_subparsers(dest="benchmark", required=True)
    for descriptor in BENCHMARKS.values():
        target = setup_targets.add_parser(descriptor.name)
        descriptor.add_setup_arguments(target)
        target.set_defaults(func=setup)

    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument(
        "benchmark",
        nargs="?",
        choices=list(BENCHMARKS),
    )
    doctor_parser.set_defaults(func=doctor)

    run_parser = commands.add_parser("run")
    run_targets = run_parser.add_subparsers(dest="benchmark", required=True)
    for descriptor in BENCHMARKS.values():
        target = run_targets.add_parser(descriptor.name)
        descriptor.add_run_arguments(target)
        target.add_argument("--result-root", required=True)
        target.set_defaults(func=run)

    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))
