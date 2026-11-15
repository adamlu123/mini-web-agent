from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from cuawright.environments import base as environment_base
from cuawright.environments.benchcad.plugin import BenchCADPlugin
from cuawright.environments.cadgenbench.plugin import CadGenBenchPlugin
from cuawright.environments.desktop.guest_tools import (
    provision_control_clients,
    write_guest_file,
)
from cuawright.environments.desktop.plugin import (
    DesktopCommandPlugin,
    GuestWorkspace,
)
from cuawright.environments.osworld import GuestWorkspace as LegacyGuestWorkspace
from cuawright.environments.osworld.compatibility import (
    OSWORLD_RELEASE_COMPATIBILITY,
)
from cuawright.environments.osworld.environment import OSWorldBenchmarkPlugin
from cuawright.environments.osworld.guest_tools import (
    _provision_control_clients,
    _write_guest_file,
)
from cuawright.environments.osworld.source import (
    DEFAULT_RELEASE_TAG,
    OSWORLD_RELEASES,
    OSWorldSource,
    activate_source,
    resolve_release,
)
from cuawright.environments.weavebench.adapter import WeaveBenchDesktopPlugin
from cuawright.models import base as model_base
from cuawright.runtime import contracts

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def imported_modules(path: Path) -> set[str]:
    package = ".".join(path.relative_to(SOURCE_ROOT).parent.parts)
    return imported_modules_from_source(
        path.read_text(encoding="utf-8"),
        package=package,
    )


def imported_modules_from_source(source: str, *, package: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = resolved_import_from_module(node, package=package)
            if module:
                modules.add(module)
                modules.update(
                    f"{module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
    return modules


def resolved_import_from_module(node: ast.ImportFrom, *, package: str) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = package.split(".") if package else []
    retained = len(package_parts) - (node.level - 1)
    base = package_parts[: max(0, retained)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def imports_namespace(module: str, namespace: str) -> bool:
    return module == namespace or module.startswith(namespace + ".")


class RuntimeBoundaryTests(unittest.TestCase):
    def test_import_parser_detects_operations_boundary_evasions(self) -> None:
        fixtures = (
            "from ...operations import scheduler",
            "from ... import operations",
            "from cuawright import operations as ops",
            "import cuawright.operations.scheduler as scheduler",
        )
        for source in fixtures:
            with self.subTest(source=source):
                modules = imported_modules_from_source(
                    source,
                    package="cuawright.environments.odysseys",
                )
                self.assertTrue(
                    any(
                        imports_namespace(module, "cuawright.operations")
                        for module in modules
                    ),
                    modules,
                )

    def test_weavebench_has_no_osworld_or_run_task_imports(self) -> None:
        weavebench_root = REPOSITORY_ROOT / "src/cuawright/environments/weavebench"
        modules = set().union(
            *(imported_modules(path) for path in weavebench_root.glob("*.py"))
        )
        self.assertFalse(
            any(
                imports_namespace(module, namespace)
                for module in modules
                for namespace in (
                    "cuawright.environments.osworld",
                    "cuawright.run.task",
                )
            )
        )
        prompt_tree = ast.parse(
            (weavebench_root / "prompts.py").read_text(encoding="utf-8")
        )
        imported_names = {
            alias.name
            for node in ast.walk(prompt_tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(
            any(
                name == "OSWORLD_OVERLAY" or name.startswith("OSWORLD_")
                for name in imported_names
            )
        )

    def test_benchmark_packages_do_not_import_other_benchmarks(self) -> None:
        environments_root = REPOSITORY_ROOT / "src/cuawright/environments"
        benchmarks = (
            "osworld",
            "weavebench",
            "benchcad",
            "cadgenbench",
            "odysseys",
        )
        for benchmark in benchmarks:
            modules = set().union(
                *(
                    imported_modules(path)
                    for path in (environments_root / benchmark).rglob("*.py")
                )
            )
            forbidden = {
                f"cuawright.environments.{other}"
                for other in benchmarks
                if other != benchmark
            }
            with self.subTest(benchmark=benchmark):
                self.assertFalse(
                    any(
                        imports_namespace(module, namespace)
                        for module in modules
                        for namespace in forbidden
                    )
                )

    def test_environment_packages_do_not_import_operations(self) -> None:
        environments_root = REPOSITORY_ROOT / "src/cuawright/environments"
        violations = {
            str(path.relative_to(REPOSITORY_ROOT)): sorted(
                module
                for module in imported_modules(path)
                if imports_namespace(module, "cuawright.operations")
            )
            for path in environments_root.rglob("*.py")
        }
        self.assertEqual(
            {},
            {path: modules for path, modules in violations.items() if modules},
        )

    def test_desktop_compatibility_reexports_keep_identity(self) -> None:
        self.assertIs(LegacyGuestWorkspace, GuestWorkspace)
        self.assertIs(_provision_control_clients, provision_control_clients)
        self.assertIs(_write_guest_file, write_guest_file)
        compatibility = OSWORLD_RELEASE_COMPATIBILITY["v2026.06.24"]
        self.assertEqual(compatibility.release, "v2026.06.24")
        self.assertEqual(
            compatibility.install_setup_shim.__module__,
            "cuawright.environments.osworld.lifecycle",
        )
        august_compatibility = OSWORLD_RELEASE_COMPATIBILITY["v2026.08.08"]
        self.assertEqual(august_compatibility.release, "v2026.08.08")
        self.assertEqual(
            august_compatibility.install_setup_shim.__module__,
            "cuawright.environments.osworld.compatibility.v2026_08_08",
        )

    def test_osworld_release_registry_resolves_tags_and_manifest_names(self) -> None:
        self.assertEqual(DEFAULT_RELEASE_TAG, "v2026.06.24")
        self.assertEqual(tuple(OSWORLD_RELEASES), ("v2026.06.24", "v2026.08.08"))
        self.assertEqual(
            resolve_release("osworld-v2-2026.08.08").source_commit,
            "d578d2d4e0dc82b43e270fdaa7fa89d9708cd154",
        )
        with self.assertRaisesRegex(ValueError, "unsupported OSWorld release"):
            resolve_release("v2099.01.01")

    def test_osworld_activation_prioritizes_the_selected_source(self) -> None:
        selected = Path("/tmp/selected-osworld-source")
        release = resolve_release("v2026.08.08")
        inspected = OSWorldSource(
            root=selected,
            commit=release.source_commit,
            release=release,
        )
        original_path = list(sys.path)
        try:
            sys.path[:] = ["/tmp/harness", str(selected), "/tmp/other"]
            with (
                unittest.mock.patch(
                    "cuawright.environments.osworld.source.inspect_source",
                    return_value=inspected,
                ),
                unittest.mock.patch.dict(
                    sys.modules,
                    {"desktop_env": None, "task_loader": None},
                ),
            ):
                activated = activate_source(release=release.tag)
            self.assertIs(activated, inspected)
            self.assertEqual(sys.path[0], str(selected))
            self.assertEqual(sys.path.count(str(selected)), 1)
        finally:
            sys.path[:] = original_path

    def test_all_benchmark_adapter_constructors_remain_available(self) -> None:
        controller = SimpleNamespace(run_python_script=lambda *_args, **_kwargs: None)
        desktop_env = SimpleNamespace(
            controller=controller,
            is_environment_used=False,
            evaluate=lambda: {"score": 1.0},
            close=lambda: None,
        )
        workspace = GuestWorkspace("/home/user/.agent", "/home/user/.agent")
        executor = lambda _command, _timeout: contracts.CommandResult(0)

        desktop = DesktopCommandPlugin(
            desktop_env,
            executor,
            workspace=workspace,
        )
        osworld = OSWorldBenchmarkPlugin(
            desktop_env,
            executor,
            workspace=workspace,
        )
        weavebench = WeaveBenchDesktopPlugin(
            desktop_env,
            executor,
            workspace=workspace,
        )
        benchcad = BenchCADPlugin(SimpleNamespace())
        cadgenbench = CadGenBenchPlugin(SimpleNamespace())

        self.assertIsInstance(desktop, DesktopCommandPlugin)
        self.assertIsInstance(osworld, DesktopCommandPlugin)
        self.assertIsInstance(weavebench, DesktopCommandPlugin)
        self.assertIsInstance(benchcad, BenchCADPlugin)
        self.assertIsInstance(cadgenbench, CadGenBenchPlugin)

    def test_only_benchmark_adapters_own_lifecycle_methods(self) -> None:
        self.assertNotIn("evaluate", DesktopCommandPlugin.__dict__)
        self.assertNotIn("close", DesktopCommandPlugin.__dict__)
        self.assertIn("evaluate", OSWorldBenchmarkPlugin.__dict__)
        self.assertIn("close", OSWorldBenchmarkPlugin.__dict__)

    def test_runtime_contracts_have_no_benchmark_or_provider_imports(self) -> None:
        modules = imported_modules(
            REPOSITORY_ROOT / "src/cuawright/runtime/contracts.py"
        )
        self.assertFalse(
            any(
                imports_namespace(module, namespace)
                for module in modules
                for namespace in (
                    "cuawright.environments",
                    "cuawright.models",
                    "cuawright.providers",
                    "cuawright.run",
                    "cuawright.agents",
                )
            )
        )

    def test_agent_loop_imports_only_neutral_core_surfaces(self) -> None:
        modules = imported_modules(REPOSITORY_ROOT / "src/cuawright/agents/default.py")
        self.assertFalse(
            any(
                imports_namespace(module, namespace)
                for module in modules
                for namespace in (
                    "cuawright.environments",
                    "cuawright.models",
                    "cuawright.providers",
                    "cuawright.config",
                )
            )
        )

    def test_provider_facade_has_no_benchmark_agent_or_run_imports(self) -> None:
        provider_root = REPOSITORY_ROOT / "src/cuawright/providers"
        modules = set().union(
            *(imported_modules(path) for path in provider_root.glob("*.py"))
        )
        self.assertFalse(
            any(
                imports_namespace(module, namespace)
                for module in modules
                for namespace in (
                    "cuawright.agents",
                    "cuawright.run",
                    "cuawright.environments.benchcad",
                    "cuawright.environments.cadgenbench",
                    "cuawright.environments.osworld",
                    "cuawright.environments.weavebench",
                )
            )
        )

    def test_legacy_contract_imports_keep_object_identity(self) -> None:
        self.assertIs(model_base.ModelTurn, contracts.ModelTurn)
        self.assertIs(model_base.ToolCall, contracts.ToolCall)
        self.assertIs(model_base.CommandResult, contracts.CommandResult)
        self.assertIs(model_base.ModelAdapter, contracts.TurnProvider)
        self.assertIs(
            model_base.RUN_COMMAND_TOOL_SCHEMA,
            contracts.RUN_COMMAND_TOOL_SCHEMA,
        )
        self.assertIs(environment_base.CommandResult, contracts.CommandResult)
        self.assertIs(environment_base.ImageAttachment, contracts.ImageAttachment)
        self.assertIs(environment_base.FileAttachment, contracts.FileAttachment)
        self.assertIs(
            environment_base.cap_command_text,
            contracts.cap_command_text,
        )


if __name__ == "__main__":
    unittest.main()
