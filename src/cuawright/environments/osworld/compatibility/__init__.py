from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from cuawright.environments.osworld.compatibility.v2026_06_24 import (
    install_setup_compatibility_shim as install_v2026_06_24_setup_compatibility,
)
from cuawright.environments.osworld.compatibility.v2026_08_08 import (
    install_setup_compatibility_shim as install_v2026_08_08_setup_compatibility,
)

install_setup_compatibility_shim = install_v2026_06_24_setup_compatibility


@dataclass(frozen=True, slots=True)
class OSWorldReleaseCompatibility:
    release: str
    install_setup_shim: Callable[[], None]


OSWORLD_RELEASE_COMPATIBILITY: Final[Mapping[str, OSWorldReleaseCompatibility]] = (
    MappingProxyType(
        {
            "v2026.06.24": OSWorldReleaseCompatibility(
                release="v2026.06.24",
                install_setup_shim=install_v2026_06_24_setup_compatibility,
            ),
            "v2026.08.08": OSWorldReleaseCompatibility(
                release="v2026.08.08",
                install_setup_shim=install_v2026_08_08_setup_compatibility,
            ),
        }
    )
)


def install_release_compatibility(release: str) -> None:
    try:
        compatibility = OSWORLD_RELEASE_COMPATIBILITY[release]
    except KeyError:
        raise ValueError(
            f"unsupported OSWorld compatibility release: {release!r}"
        ) from None
    compatibility.install_setup_shim()


__all__ = [
    "OSWORLD_RELEASE_COMPATIBILITY",
    "OSWorldReleaseCompatibility",
    "install_release_compatibility",
    "install_setup_compatibility_shim",
]
