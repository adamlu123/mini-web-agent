"""Compatibility façade for the pinned OSWorld v2026.08.08 release."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from docker.errors import DockerException

from cuawright.environments.osworld.lifecycle import (
    install_setup_compatibility_shim as install_base_setup_compatibility,
)

logger = logging.getLogger("cuawright.environments.osworld.compatibility.v2026_08_08")


def _install_dynamic_docker_ports() -> None:
    from desktop_env.providers.docker.provider import DockerProvider

    if getattr(DockerProvider, "_cuawright_dynamic_ports", False):
        return

    def clear_container_state(self: Any) -> None:
        self.container = None
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None

    def get_published_port(
        self: Any,
        container_port: int,
        timeout: int = 30,
    ) -> int:
        deadline = time.monotonic() + timeout
        key = f"{container_port}/tcp"
        while time.monotonic() < deadline:
            self.container.reload()
            mappings = self.container.attrs["NetworkSettings"]["Ports"].get(key)
            if mappings:
                return int(mappings[0]["HostPort"])
            time.sleep(0.1)
        raise TimeoutError(f"Docker did not publish container port {container_port}")

    def start_emulator(
        self: Any,
        path_to_vm: str,
        headless: bool,
        os_type: str,
    ) -> None:
        del headless, os_type
        try:
            devices = []
            if os.path.exists("/dev/kvm"):
                devices.append("/dev/kvm")
                logger.info("KVM device found, using hardware acceleration")
            else:
                self.environment["KVM"] = "N"
                logger.warning(
                    "KVM device not found, running without hardware acceleration"
                )

            self.container = self.client.containers.run(
                "happysixd/osworld-docker",
                environment=self.environment,
                cap_add=["NET_ADMIN"],
                devices=devices,
                volumes={
                    os.path.abspath(path_to_vm): {
                        "bind": "/System.qcow2",
                        "mode": "ro",
                    }
                },
                ports={
                    8006: None,
                    5000: None,
                    9222: None,
                    8080: None,
                },
                detach=True,
            )
            self.vnc_port = get_published_port(self, 8006)
            self.server_port = get_published_port(self, 5000)
            self.chromium_port = get_published_port(self, 9222)
            self.vlc_port = get_published_port(self, 8080)
            logger.info(
                "Started container with ports - VNC: %s, Server: %s, "
                "Chrome: %s, VLC: %s",
                self.vnc_port,
                self.server_port,
                self.chromium_port,
                self.vlc_port,
            )
            self._wait_for_vm_ready()
        except (DockerException, OSError, RuntimeError, TimeoutError):
            container = self.container
            if container is not None:
                try:
                    container.remove(force=True, v=True)
                except DockerException as cleanup_error:
                    logger.error(
                        "Could not remove failed Docker VM container: %s",
                        cleanup_error,
                    )
                finally:
                    if self.container is container:
                        clear_container_state(self)
            raise

    DockerProvider._get_published_port = get_published_port
    DockerProvider._clear_container_state = clear_container_state
    DockerProvider.start_emulator = start_emulator
    DockerProvider._cuawright_dynamic_ports = True


def install_setup_compatibility_shim() -> None:
    install_base_setup_compatibility()
    _install_dynamic_docker_ports()


__all__ = [
    "_install_dynamic_docker_ports",
    "install_setup_compatibility_shim",
]
