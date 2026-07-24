"""Registry of available systems. Adding a new one means: write a module
next to this file implementing MDSystem (see base.py), then add one line
below -- nothing else in the codebase needs to change.
"""
from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec
from .cu_deposition import CopperEAMSystem
from .lipid_membrane import LipidMembraneSystem
from .lj_argon import LJArgonSystem
from .mb_water import MBWaterSystem
from .mesomem_hex import MesoMemHexSystem
from .mesomem_sheet import MesoMemSheetSystem
from .nacl import NaClSystem

REGISTRY = {
    CopperEAMSystem.spec.key: CopperEAMSystem,
    LJArgonSystem.spec.key: LJArgonSystem,
    NaClSystem.spec.key: NaClSystem,
    LipidMembraneSystem.spec.key: LipidMembraneSystem,
    MBWaterSystem.spec.key: MBWaterSystem,
    MesoMemHexSystem.spec.key: MesoMemHexSystem,
    MesoMemSheetSystem.spec.key: MesoMemSheetSystem,
}


def list_systems():
    """Ordered (key, SystemSpec) pairs, in registry-declaration order."""
    return [(key, cls.spec) for key, cls in REGISTRY.items()]


def get_system_class(key):
    try:
        return REGISTRY[key]
    except KeyError:
        available = ", ".join(REGISTRY)
        raise KeyError(f"Unknown system {key!r}. Available: {available}") from None


__all__ = [
    "MDSystem", "SystemSpec", "SliderSpec", "ForceFeedbackProfile",
    "REGISTRY", "list_systems", "get_system_class",
]
