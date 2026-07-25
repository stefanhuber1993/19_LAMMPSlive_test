"""One list of everything runnable: playgrounds and legacy systems.

The app and CLI talk to this instead of to either registry directly, so a
playground and a hand-written MDSystem are interchangeable at the picker level and
neither layer needs to know the other exists.

Playgrounds come first because they are where new work goes. Both listings are
lazy about construction -- a spec is produced without building a LAMMPS instance
-- but the legacy listing does have to import its modules, since those specs are
class attributes.
"""
PLAYGROUND = "playground"
SYSTEM = "system"


def _playground_entries():
    from .playground import registry
    try:
        return [(key, spec, PLAYGROUND) for key, spec in registry.list_playgrounds()]
    except Exception as exc:
        print(f"warning: could not list playgrounds: {exc}")
        return []


def _system_entries():
    from .systems import list_systems
    try:
        return [(key, spec, SYSTEM) for key, spec in list_systems()]
    except Exception as exc:
        print(f"warning: could not list legacy systems: {exc}")
        return []


def list_entries():
    """[(key, SystemSpec, kind), ...] -- playgrounds then legacy systems."""
    return _playground_entries() + _system_entries()


def list_specs():
    """[(key, SystemSpec), ...] -- the shape the app's picker and renderer use."""
    return [(key, spec) for key, spec, _ in list_entries()]


def kind_of(key):
    for k, _spec, kind in list_entries():
        if k == key:
            return kind
    return None


def build(key, mode=None, preset=None):
    """Construct whatever `key` names.

    A key containing a path separator or ending in .py is always a playground
    file, so a researcher's own file works without being registered anywhere.
    """
    import os
    if os.path.sep in key or key.endswith(".py"):
        from .playground.registry import build as build_playground
        return build_playground(key, mode=mode, preset=preset)

    from .playground import registry
    if key in registry.bundled_keys():
        return registry.build(key, mode=mode, preset=preset)

    from .systems import get_system_class
    return get_system_class(key)()


def resolve(key):
    """Validate a key, returning it, or raise SystemExit with the alternatives."""
    import os
    if os.path.sep in key or key.endswith(".py"):
        return key
    known = [k for k, _spec, _kind in list_entries()]
    if key in known:
        return key
    raise SystemExit(
        f"unknown --system/--playground {key!r}.\nAvailable: " + ", ".join(known)
    )
