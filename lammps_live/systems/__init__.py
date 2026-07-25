"""Registry of hand-written legacy systems.

These are the original monolithic MDSystem classes. New work goes through
`lammps_live/playgrounds/` instead: a playground composes a ForceField, a
Scenario and a Mode declaratively, which is far less code per system and lets
game mode and sim mode be chosen at run time (see lammps_live/playground/).

Resolution is lazy -- `_LAZY` maps a key to "module:ClassName" and the module is
imported only when that system is actually built. The old eager form imported all
eight system modules (and therefore `lammps` and `scipy`) just to answer
`--list-systems`.
"""
import importlib

from .base import ForceFeedbackProfile, MDSystem, MDSystem3D, SliderSpec, SystemSpec

# key -> (module basename, class name), in the order they should be offered.
# Copper EAM and LJ argon used to be here; they are now playgrounds
# (playgrounds/cu_deposition.py, playgrounds/lj_argon.py) built from the shared
# Deposition2D scenario, which is where their calibration notes live too.
_LAZY = {
    "nacl": ("nacl", "NaClSystem"),
    "lipid": ("lipid_membrane", "LipidMembraneSystem"),
    "mb_water": ("mb_water", "MBWaterSystem"),
}


def system_keys():
    return list(_LAZY)


def get_system_class(key):
    try:
        module_name, class_name = _LAZY[key]
    except KeyError:
        available = ", ".join(_LAZY)
        raise KeyError(f"Unknown system {key!r}. Available: {available}") from None
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, class_name)


def list_systems():
    """Ordered (key, SystemSpec) pairs. Imports each system module, since the
    spec is a class attribute -- only call it when the specs are actually
    needed."""
    return [(key, get_system_class(key).spec) for key in _LAZY]


__all__ = [
    "MDSystem", "MDSystem3D", "SystemSpec", "SliderSpec", "ForceFeedbackProfile",
    "system_keys", "list_systems", "get_system_class",
]
