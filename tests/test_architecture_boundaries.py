from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "cuawright"


def imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_model_adapters_do_not_depend_on_environment_modules() -> None:
    violations = {
        str(path.relative_to(PACKAGE)): sorted(
            module
            for module in imports(path)
            if module.startswith("cuawright.environments")
        )
        for path in (PACKAGE / "models").rglob("*.py")
    }

    assert not {path: values for path, values in violations.items() if values}


def test_agent_loop_does_not_import_benchmark_adapters() -> None:
    benchmark_prefixes = tuple(
        f"cuawright.environments.{name}"
        for name in ("osworld", "weavebench", "benchcad", "cadgenbench", "odysseys")
    )

    assert not {
        module
        for module in imports(PACKAGE / "agents" / "default.py")
        if module.startswith(benchmark_prefixes)
    }


def test_weavebench_does_not_import_osworld_or_run_task() -> None:
    violations = {
        str(path.relative_to(PACKAGE)): sorted(
            module
            for module in imports(path)
            if module.startswith("cuawright.environments.osworld")
            or module == "cuawright.run.task"
        )
        for path in (PACKAGE / "environments" / "weavebench").rglob("*.py")
    }

    assert not {path: values for path, values in violations.items() if values}
