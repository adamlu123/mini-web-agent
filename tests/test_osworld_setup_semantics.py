from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cuawright.environments.osworld import lifecycle
from cuawright.environments.osworld.compatibility.v2026_08_08 import (
    _install_dynamic_docker_ports,
)


def test_shell_background_detection_preserves_sequential_prerequisites():
    foreground = (
        "cd /tmp/app && npm install",
        "unzip source.zip -d source && rm source.zip",
        "for i in $(seq 1 20); do curl -f localhost:5002 && break; sleep 2; done",
        "mkdir -p /tmp/a /tmp/b && echo ready",
        "python3 test.py >test_run_output.txt 2>&1",
        "python3 test.py &>test_run_output.txt",
        "python3 test.py >&test_run_output.txt",
        "echo 'literal & text'",
        'printf "nohup server"',
    )
    background = (
        "python3 -m http.server 8010 >/tmp/server.log 2>&1 &",
        "sleep 2 && google-chrome http://localhost:3000 &",
        "HOME=/tmp nohup python3 server.py",
        "(cd /tmp/app && node server.js) &",
    )
    for script in foreground:
        assert not lifecycle._shell_requests_background_execution(script), script
    for script in background:
        assert lifecycle._shell_requests_background_execution(script), script


def test_setup_shim_preserves_execute_launch_contract():
    class SetupController:
        def __init__(self):
            self.executed = []
            self.launched = []
            self.launch_setup_commands = []
            self.use_proxy = False

        def execute(self, command, **kwargs):
            self.executed.append((command, kwargs))
            return "executed"

        def launch(self, command):
            self.launched.append(command)
            return "launched"

        def _launch_setup(self, command, shell=False):
            self.launch_setup_commands.append((command, shell))
            return "launch-setup"

    desktop_env = types.ModuleType("desktop_env")
    controllers = types.ModuleType("desktop_env.controllers")
    setup = types.ModuleType("desktop_env.controllers.setup")
    setup.SetupController = SetupController
    original_flag = lifecycle._SETUP_EXECUTE_PATCHED
    original_launch_flag = lifecycle._SETUP_LAUNCH_PATCHED
    try:
        lifecycle._SETUP_EXECUTE_PATCHED = False
        lifecycle._SETUP_LAUNCH_PATCHED = False
        with patch.dict(
            sys.modules,
            {
                "desktop_env": desktop_env,
                "desktop_env.controllers": controllers,
                "desktop_env.controllers.setup": setup,
            },
        ):
            lifecycle.install_setup_compatibility_shim()
            controller = SetupController()
            assert (
                controller.execute(
                    ["bash", "-c", "cd /tmp/app && npm install --silent"]
                )
                == "executed"
            )
            assert (
                controller.execute(
                    [
                        "bash",
                        "-c",
                        "for i in $(seq 1 20); do curl -f localhost:5002 "
                        "&& break; sleep 2; done",
                    ]
                )
                == "executed"
            )
            assert (
                controller.execute(
                    ["bash", "-c", "python3 test.py >test_run_output.txt 2>&1"]
                )
                == "executed"
            )
            assert (
                controller.execute(
                    ["bash", "-c", "python3 -m http.server 8010 >/tmp/x 2>&1 &"]
                )
                == "launched"
            )
            assert (
                controller.execute(
                    [
                        "bash",
                        "-c",
                        "cd /home/user/ds2019-request/ds2019-request && npm start",
                    ]
                )
                == "launched"
            )
            assert (
                controller.execute(["libreoffice", "--calc", "/home/user/source.xlsx"])
                == "launched"
            )
            assert (
                controller.execute(
                    ["libreoffice", "--headless", "--convert-to", "pdf", "source.docx"]
                )
                == "executed"
            )
            assert len(controller.executed) == 4
            assert len(controller.launched) == 3
            controller.use_proxy = True
            with patch.dict(
                os.environ,
                {"WEBSITE_HOST_SUFFIX": "10.0.0.25.nip.io"},
            ):
                assert (
                    controller._launch_setup(
                        ["google-chrome", "--remote-debugging-port=1337"]
                    )
                    == "launch-setup"
                )
            command, shell = controller.launch_setup_commands[-1]
            assert shell is False
            assert (
                "--proxy-bypass-list=localhost;127.0.0.1;*.10.0.0.25.nip.io"
            ) in command
    finally:
        lifecycle._SETUP_EXECUTE_PATCHED = original_flag
        lifecycle._SETUP_LAUNCH_PATCHED = original_launch_flag


def test_august_docker_compatibility_uses_daemon_assigned_ports():
    class Container:
        def __init__(self):
            self.attrs = {
                "NetworkSettings": {
                    "Ports": {
                        "8006/tcp": [{"HostPort": "38006"}],
                        "5000/tcp": [{"HostPort": "35000"}],
                        "9222/tcp": [{"HostPort": "39222"}],
                        "8080/tcp": [{"HostPort": "38080"}],
                    }
                }
            }

        def reload(self):
            return None

        def remove(self, **_kwargs):
            return None

    class Containers:
        def __init__(self):
            self.calls = []

        def run(self, image, **kwargs):
            self.calls.append((image, kwargs))
            return Container()

    class DockerProvider:
        def __init__(self):
            self.client = SimpleNamespace(containers=Containers())
            self.environment = {
                "DISK_SIZE": "80G",
                "RAM_SIZE": "4G",
                "CPU_CORES": "4",
            }
            self.container = None
            self.server_port = None
            self.vnc_port = None
            self.chromium_port = None
            self.vlc_port = None
            self.ready = False

        def _wait_for_vm_ready(self):
            self.ready = True

    desktop_env = types.ModuleType("desktop_env")
    providers = types.ModuleType("desktop_env.providers")
    docker_package = types.ModuleType("desktop_env.providers.docker")
    provider = types.ModuleType("desktop_env.providers.docker.provider")
    provider.DockerProvider = DockerProvider

    with (
        patch.dict(
            sys.modules,
            {
                "desktop_env": desktop_env,
                "desktop_env.providers": providers,
                "desktop_env.providers.docker": docker_package,
                "desktop_env.providers.docker.provider": provider,
            },
        ),
        patch("os.path.exists", return_value=True),
    ):
        _install_dynamic_docker_ports()
        runtime = DockerProvider()
        runtime.start_emulator("/tmp/System.qcow2", True, "Ubuntu")

    image, options = runtime.client.containers.calls[0]
    assert image == "happysixd/osworld-docker"
    assert options["ports"] == {
        8006: None,
        5000: None,
        9222: None,
        8080: None,
    }
    assert runtime.vnc_port == 38006
    assert runtime.server_port == 35000
    assert runtime.chromium_port == 39222
    assert runtime.vlc_port == 38080
    assert runtime.ready


def test_proxy_preflight_rejects_example_credentials_before_network():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proxy.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "host": "gw.dataimpulse.com",
                        "port": 823,
                        "username": "your_username",
                        "password": "your_password",
                        "protocol": "http",
                    }
                ]
            )
        )
        with (
            patch("cuawright.environments.osworld.lifecycle.requests.get") as get,
            pytest.raises(
                lifecycle.ProxyConfigurationError,
                match="example credentials",
            ),
        ):
            lifecycle.preflight_proxy_configuration(path)
        get.assert_not_called()


def test_proxy_preflight_validates_connectivity_without_returning_credentials():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proxy.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "host": "proxy.example",
                        "port": 823,
                        "username": "real-user",
                        "password": "real-password",
                        "protocol": "http",
                        "provider": "example",
                        "country": "US",
                    }
                ]
            )
        )
        path.chmod(0o600)
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            close=lambda: None,
        )
        with patch(
            "cuawright.environments.osworld.lifecycle.requests.get",
            return_value=response,
        ) as get:
            result = lifecycle.preflight_proxy_configuration(path)

        assert result == {
            "status": "ready",
            "host": "proxy.example",
            "port": 823,
            "protocol": "http",
            "provider": "example",
            "country": "US",
        }
        proxy_url = get.call_args.kwargs["proxies"]["https"]
        assert "real-user" in proxy_url
        assert "real-password" in proxy_url
        assert "username" not in result
        assert "password" not in result
