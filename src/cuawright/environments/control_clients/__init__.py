"""Packaged sources for the provisioned CUAWright terminal clients."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from types import MappingProxyType
from typing import Final, Iterator, Mapping


CLIENT_FILENAMES: Final[tuple[str, ...]] = (
    "binder.py",
    "critic.py",
    "verifier.py",
    "image_show.py",
    "ask_user.py",
    "submit.py",
)
CLIENT_INSTALL_ROOT: Final[str] = "/opt/cuawright-tools"
_RESOURCE_ROOT = files(__package__)
CLIENT_SOURCES: Final[Mapping[str, bytes]] = MappingProxyType(
    {name: _RESOURCE_ROOT.joinpath(name).read_bytes() for name in CLIENT_FILENAMES}
)
CLIENT_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        name: hashlib.sha256(source).hexdigest()
        for name, source in CLIENT_SOURCES.items()
    }
)


def client_source_hashes() -> dict[str, str]:
    return dict(CLIENT_SHA256)


def iter_control_clients() -> Iterator[tuple[str, bytes, str]]:
    for name in CLIENT_FILENAMES:
        yield name, CLIENT_SOURCES[name], CLIENT_SHA256[name]
