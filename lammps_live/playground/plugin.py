"""Compile-on-demand LAMMPS pair-style plugins.

Generalized from systems/mesomem_ff/__init__.py, which did exactly this for one
hardcoded pair style. A custom force field is the normal case for someone
inventing one, so this is the path: drop your `pair_*.cpp` / `.h` plus a small
`*plugin.cpp` registration stub in a directory, point a PluginSpec at it, and the
style is compiled into a runtime-loadable shared library and pulled into the
stock pip-installed LAMMPS with `plugin load` -- no full LAMMPS rebuild.

The artifact is cached next to the sources and rebuilt only when a source file is
newer, so the edit-C++-restart-and-the-sliders-still-work loop costs one compile.
"""
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PluginSpec:
    """A custom pair style to compile and load.

    `directory` holds the sources; `sources` are the .cpp files to compile (the
    pair style plus its plugin registration stub); `headers` are watched for
    staleness but not compiled; `lib_stem` names the cached artifact.
    """
    directory: str
    sources: tuple
    headers: tuple = ()
    lib_stem: str = "plugin"

    @property
    def lib_path(self):
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        return os.path.join(self.directory, self.lib_stem + ext)

    def source_paths(self):
        return [os.path.join(self.directory, s) for s in self.sources]

    def watched_paths(self):
        return self.source_paths() + [os.path.join(self.directory, h)
                                      for h in self.headers]


def _lammps_include_dir():
    """The LAMMPS headers bundled inside the pip `lammps` package."""
    import lammps
    inc = os.path.join(os.path.dirname(lammps.__file__), "include", "lammps")
    if not os.path.isdir(inc):
        raise RuntimeError(f"LAMMPS headers not found at {inc}")
    return inc


def _mpi_include_dir():
    """Header dir for the MPI implementation LAMMPS was built against.

    A pair style transitively includes <mpi.h> (via LAMMPS' pointers.h), so we
    must compile against the SAME MPI's headers as the loaded liblammps (MPICH
    here). Try, in order: an explicit override, the MPI compiler wrapper's
    reported include dir, then the usual per-OS install locations.
    """
    env = os.environ.get("LAMMPS_LIVE_MPI_INCLUDE") or os.environ.get("MESOMEM_MPI_INCLUDE")
    if env and os.path.isfile(os.path.join(env, "mpi.h")):
        return env
    for wrapper in ("mpicxx", "mpic++", "mpicc"):
        exe = shutil.which(wrapper)
        if not exe:
            continue
        for flag in ("-showme:incdirs", "-show"):
            try:
                out = subprocess.check_output([exe, flag], text=True,
                                              stderr=subprocess.DEVNULL)
            except Exception:
                continue
            for tok in out.replace("-I", " -I").split():
                cand = tok[2:] if tok.startswith("-I") else tok
                if os.path.isfile(os.path.join(cand, "mpi.h")):
                    return cand
    # Fallbacks when no MPI wrapper is on PATH: Homebrew (macOS) and the common
    # Linux MPICH header locations. Debian/Ubuntu put mpi.h under a multiarch
    # subdir (/usr/include/<arch>/mpich); Fedora uses /usr/include/mpich-<arch>
    # or /usr/lib64/mpich/include, so those are globbed rather than hardcoded.
    candidates = ["/opt/homebrew/include", "/usr/local/include", "/usr/include/mpich"]
    candidates += glob.glob("/usr/include/*/mpich")     # Debian/Ubuntu multiarch
    candidates += glob.glob("/usr/include/mpich-*")     # Fedora
    candidates += glob.glob("/usr/lib*/mpich/include")  # Fedora modules
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "mpi.h")):
            return cand
    raise RuntimeError(
        "Could not locate mpi.h. Install the MPI whose runtime LAMMPS uses, with "
        "its development headers -- macOS: `brew install mpich`; Debian/Ubuntu: "
        "`sudo apt install mpich libmpich-dev`; Fedora: `sudo dnf install mpich "
        "mpich-devel` -- or set LAMMPS_LIVE_MPI_INCLUDE to the dir containing mpi.h."
    )


def _needs_build(spec):
    lib = spec.lib_path
    if not os.path.isfile(lib):
        return True
    lib_mtime = os.path.getmtime(lib)
    for path in spec.watched_paths():
        if os.path.isfile(path) and os.path.getmtime(path) > lib_mtime:
            return True
    return False


def _build(spec):
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError(
            f"No C++ compiler (clang++/g++) found to build the {spec.lib_stem} plugin."
        )
    if sys.platform == "darwin":
        link_flags = ["-undefined", "dynamic_lookup"]
    else:
        # ELF resolves undefined plugin symbols against the already-loaded
        # liblammps at dlopen time; allow them to stay unresolved at link.
        link_flags = ["-Wl,--allow-shlib-undefined"]
    cmd = [
        cxx, "-std=c++17", "-O3", "-shared", "-fPIC", *link_flags,
        f"-I{_lammps_include_dir()}", f"-I{_mpi_include_dir()}", f"-I{spec.directory}",
        *spec.source_paths(),
        "-o", spec.lib_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to compile the {spec.lib_stem} pair-style plugin:\n"
            + " ".join(cmd) + "\n" + proc.stderr
        )
    return spec.lib_path


def ensure_loaded(spec, lmp):
    """Compile (if a source is newer than the cached library) and `plugin load`
    the style into `lmp`. Idempotent per LAMMPS instance -- loading twice is
    harmless (LAMMPS warns and keeps the existing style)."""
    if _needs_build(spec):
        _build(spec)
    lmp.command(f"plugin load {spec.lib_path}")
    return spec.lib_path
