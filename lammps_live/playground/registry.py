"""Discovery and loading of playground files.

A playground is any importable module exposing a module-level `PLAYGROUND`.
Bundled ones live in `lammps_live/playgrounds/`; a user's own can be passed to the
CLI as a path, so exploring a new idea needs no changes inside the package.

Discovery reads the module (cheap -- a playground file is declarations) but does
NOT construct anything, so listing playgrounds costs no LAMMPS instance.
"""
import importlib
import importlib.util
import os
import pkgutil

_PACKAGE = "lammps_live.playgrounds"


def _load_module(name):
    return importlib.import_module(f"{_PACKAGE}.{name}")


def _from_path(path):
    """Import a playground from a filesystem path, so a researcher can keep their
    own playgrounds anywhere."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no playground file at {path}")
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"_playground_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, stem


def _playground_of(module, fallback_key):
    pg = getattr(module, "PLAYGROUND", None)
    if pg is None:
        raise AttributeError(
            f"{module.__name__} defines no module-level PLAYGROUND. A playground "
            f"file must expose exactly one: PLAYGROUND = Playground(...)"
        )
    if not pg.key:
        # Default the CLI id to the module basename, so a file named
        # mesomem_sheet.py is selectable as --playground mesomem_sheet.
        import dataclasses
        pg = dataclasses.replace(pg, key=fallback_key)
    return pg


# The order the picker, the number keys and Tab walk through. Not alphabetical:
# the MesoMem playgrounds are the point of the demo and build on each other, so
# they come first and in the order a talk goes through them. Anything not named
# here follows, alphabetically, so a new playground file appears without editing
# this list.
#
# The sequence, and what each one adds to the one before it:
#
#   mesomem_bead          one bead: the controls, and no physics at all
#   mesomem_patch         seven, driven by FORCE -- pull one out of the plane
#   mesomem_patch_torque  the same seven, driven by TORQUE -- twist one instead
#   mesomem_sheet         a periodic monolayer of them
#   mesomem_assembly      a box that finds the monolayer for itself
#   mesomem_rod           a second species: something for the membrane to wrap
#   mesomem_remote        the assembly box scaled up on a cluster GPU
#   mesomem_polymer       and closed into a vesicle with a polymer inside it
#
# The two force/torque patches are adjacent deliberately: they are the same seven
# beads and the same force field, and switching between them with Tab is the
# comparison (see mesomem_patch_torque's docstring). The two remote ones are last
# and adjacent for a different reason -- they share one GPU allocation, and moving
# between them is a thing the connect panel has to handle (see ui/remote_panel.py).
_ORDER = ("mesomem_bead", "mesomem_patch", "mesomem_patch_torque",
          "mesomem_sheet", "mesomem_assembly", "mesomem_rod",
          "mesomem_remote", "mesomem_polymer")


def bundled_keys():
    """Names of the bundled playground modules, in presentation order."""
    package = importlib.import_module(_PACKAGE)
    found = {m.name for m in pkgutil.iter_modules(package.__path__)
             if not m.name.startswith("_")}
    first = [k for k in _ORDER if k in found]
    return first + sorted(found - set(first))


def load(ref):
    """A Playground from a bundled name or a filesystem path."""
    if os.path.sep in ref or ref.endswith(".py"):
        module, stem = _from_path(ref)
        return _playground_of(module, stem)
    return _playground_of(_load_module(ref), ref)


def all_playgrounds():
    """[(key, Playground), ...] for every bundled playground."""
    out = []
    for name in bundled_keys():
        try:
            out.append((name, load(name)))
        except Exception as exc:            # a broken user file shouldn't hide the rest
            print(f"warning: skipping playground {name!r}: {exc}")
    return out


def list_playgrounds():
    """[(key, SystemSpec), ...] -- what the CLI listing and the app's picker show.
    Builds specs only; no LAMMPS instance is created."""
    from .system import make_spec
    return [(key, make_spec(pg, pg.mode)) for key, pg in all_playgrounds()]


def build(ref, mode=None, preset=None, remote_override=None):
    """Construct a running system for a playground.

    A playground that declares a `remote` target gets a `RemoteSystem` instead of
    a `PlaygroundSystem`: it builds no LAMMPS here, because its simulation runs on
    a cluster GPU and this process only draws it. Both satisfy MDSystem3D, so the
    app above this line cannot tell them apart -- which is the whole reason the
    remote demo needed no changes to the control loop or the renderer.
    """
    playground = load(ref)
    if playground.remote is not None:
        from ..remote.client import RemoteSystem
        return RemoteSystem(playground, preset=preset,
                            target=remote_override or playground.remote)
    from .system import PlaygroundSystem
    return PlaygroundSystem(playground, mode_name=mode, preset=preset)


def resolve(ref):
    """Validate what the user asked for, returning it, or exit with the
    alternatives. A path is taken on trust -- it is a file the user wrote, and
    `load` will report anything wrong with it far more usefully than a list of
    bundled names would."""
    if os.path.sep in ref or ref.endswith(".py"):
        return ref
    known = bundled_keys()
    if ref in known:
        return ref
    raise SystemExit(f"unknown playground {ref!r}.\nAvailable: " + ", ".join(known))
