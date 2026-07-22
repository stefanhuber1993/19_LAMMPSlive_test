"""Build-and-load helper for the real MesoMem LAMMPS pair style.

The MesoMem membrane force field (Sillano, Marrink & Idema 2026) lives in a
custom C++ LAMMPS pair-style (`pair_membrane_sillano_v2.{cpp,h}`, pair name
`mesomem`), taken verbatim from the authors' repository
(gitlab.tudelft.nl/idema-group/mesomem) except for one dropped unused include
(`math_extra.h`, not shipped in the pip LAMMPS wheel; the code does its cross
products by hand). Rather than rebuild all of LAMMPS, we compile just this one
pair style into a runtime-loadable shared library and pull it into the stock
pip-installed LAMMPS with `plugin load` -- the wheel ships the PLUGIN package,
so no full rebuild is needed. This is the reusable path for every future
MesoMem-based system in this demo.

`ensure_plugin_loaded(lmp)` compiles the plugin once (cached next to the
sources, rebuilt only when a source file is newer than the artifact) and loads
it into the given LAMMPS instance, making `pair_style mesomem` available.
"""
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCES = ["mesomemplugin.cpp", "pair_membrane_sillano_v2.cpp"]
_LIB_NAME = "mesomem.dylib" if sys.platform == "darwin" else "mesomem.so"
_LIB_PATH = os.path.join(_HERE, _LIB_NAME)


def _lammps_include_dir():
    """The LAMMPS headers bundled inside the pip `lammps` package."""
    import lammps
    inc = os.path.join(os.path.dirname(lammps.__file__), "include", "lammps")
    if not os.path.isdir(inc):
        raise RuntimeError(f"LAMMPS headers not found at {inc}")
    return inc


def _mpi_include_dir():
    """Header dir for the MPI implementation LAMMPS was built against.

    The pair style transitively includes <mpi.h> (via LAMMPS' pointers.h), so
    we must compile against the SAME MPI's headers as the loaded liblammps
    (MPICH here). Try, in order: an explicit override, the MPI compiler
    wrapper's reported include dir, then Homebrew's include prefix."""
    env = os.environ.get("MESOMEM_MPI_INCLUDE")
    if env and os.path.isfile(os.path.join(env, "mpi.h")):
        return env
    for wrapper in ("mpicxx", "mpic++", "mpicc"):
        exe = shutil.which(wrapper)
        if not exe:
            continue
        for flag in ("-showme:incdirs", "-show"):
            try:
                out = subprocess.check_output([exe, flag], text=True, stderr=subprocess.DEVNULL)
            except Exception:
                continue
            for tok in out.replace("-I", " -I").split():
                cand = tok[2:] if tok.startswith("-I") else tok
                if os.path.isfile(os.path.join(cand, "mpi.h")):
                    return cand
    for cand in ("/opt/homebrew/include", "/usr/local/include"):
        if os.path.isfile(os.path.join(cand, "mpi.h")):
            return cand
    raise RuntimeError(
        "Could not locate mpi.h. Install the MPI whose runtime LAMMPS uses "
        "(Homebrew: `brew install mpich`) or set MESOMEM_MPI_INCLUDE."
    )


def _needs_build():
    if not os.path.isfile(_LIB_PATH):
        return True
    lib_mtime = os.path.getmtime(_LIB_PATH)
    for name in _SOURCES + ["pair_membrane_sillano_v2.h", "lammpsplugin.h", "version.h"]:
        if os.path.getmtime(os.path.join(_HERE, name)) > lib_mtime:
            return True
    return False


def _build():
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError("No C++ compiler (clang++/g++) found to build the MesoMem plugin.")
    if sys.platform == "darwin":
        link_flags = ["-undefined", "dynamic_lookup"]
    else:
        # ELF resolves undefined plugin symbols against the already-loaded
        # liblammps at dlopen time; allow them to stay unresolved at link.
        link_flags = ["-Wl,--allow-shlib-undefined"]
    cmd = [
        cxx, "-std=c++17", "-O3", "-shared", "-fPIC", *link_flags,
        f"-I{_lammps_include_dir()}", f"-I{_mpi_include_dir()}", f"-I{_HERE}",
        *[os.path.join(_HERE, s) for s in _SOURCES],
        "-o", _LIB_PATH,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to compile the MesoMem pair-style plugin:\n"
            + " ".join(cmd) + "\n" + proc.stderr
        )
    return _LIB_PATH


def ensure_plugin_loaded(lmp):
    """Compile (if needed) and `plugin load` the MesoMem pair style into `lmp`.
    Idempotent per LAMMPS instance -- loading twice is harmless (LAMMPS warns
    and keeps the existing style)."""
    if _needs_build():
        _build()
    lmp.command(f"plugin load {_LIB_PATH}")
    return _LIB_PATH
