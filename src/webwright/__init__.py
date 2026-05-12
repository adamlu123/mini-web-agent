from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

try:
    import dotenv
except ModuleNotFoundError:
    class _DotenvShim:
        @staticmethod
        def load_dotenv(*args, **kwargs):
            return False
    dotenv = _DotenvShim()
try:
    from platformdirs import user_config_dir
except ModuleNotFoundError:
    def user_config_dir(appname: str) -> str:
        return str(Path.home() / ".config" / appname)

__version__ = "0.1.0"

package_dir = Path(__file__).resolve().parent
project_config_file = package_dir.parent.parent / ".env"
global_config_dir = Path(
    os.getenv("MSWEBA_GLOBAL_CONFIG_DIR") or user_config_dir("mini-swe-webagent")
)
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = global_config_dir / ".env"
dotenv.load_dotenv(dotenv_path=project_config_file)
dotenv.load_dotenv(dotenv_path=global_config_file)


def _socksio_available() -> bool:
    try:
        import socksio  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _is_socks_proxy(value: str) -> bool:
    return value.strip().lower().startswith(("socks://", "socks4://", "socks5://"))


def _normalize_httpx_proxy_env() -> None:
    """Prefer ClashX's HTTP proxy when httpx cannot use SOCKS proxies."""

    if _socksio_available():
        return

    http_proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or ""
    )
    if not http_proxy or _is_socks_proxy(http_proxy):
        return

    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key, "")
        if value and _is_socks_proxy(value):
            os.environ[key] = http_proxy


_normalize_httpx_proxy_env()


class Model(Protocol):
    config: Any

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]: ...

    def format_message(self, **kwargs) -> dict[str, Any]: ...

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict[str, Any]: ...


class Environment(Protocol):
    config: Any

    def prepare(self, **kwargs) -> None: ...

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class Agent(Protocol):
    config: Any

    def run(self, task: str, **kwargs) -> dict[str, Any]: ...

    def save(self, path: Path | None, *extra_dicts) -> dict[str, Any]: ...


__all__ = [
    "Agent",
    "Environment",
    "Model",
    "__version__",
    "global_config_dir",
    "global_config_file",
    "package_dir",
]
