"""Ask a machine whether it can run the remote server, and report as JSON.

    python -m lammps_live.remote.probe            # readable
    python -m lammps_live.remote.probe --json     # for the connect flow

Run it on the login node to check the build, and inside the allocation to check
the GPU. The connect flow (session.py) runs it automatically and shows what it
found, because every one of these questions has bitten this port already and
each has a very different fix:

  * `import lammps` failing means the build has no Python module -- the `lmp`
    binary alone cannot be driven frame by frame, so there is no demo without it.
    The fix is a rebuild with `-DBUILD_SHARED_LIBS=yes -DPKG_PYTHON=yes`, then
    `make install-python` (or PYTHONPATH to the build's python/ directory).
  * `mesomem` missing means the pair style is not in this build. On the cluster
    that is fatal; locally it is expected, because locally it arrives as a runtime
    plugin -- so the probe asks the question the way the server will build, which
    is what `--profile` selects.
  * `mesomem/kk` missing means the suffix will silently fall back to the host
    style: the run works and is slow, which is the worst failure to diagnose from
    a frame rate alone.
  * `nve/sphere/kk` missing is docs/snellius/README.md point 2 -- the integrator
    then copies positions, dipoles and torques back to the host every step, and
    the pair kernel's gain goes with it. Worth knowing, not fatal.
  * The 9th `pair_coeff` value decides whether this build has the patched splay
    term (see hosts.py). The client hides the slider when it does not.

EVERY LAMMPS QUESTION IS ASKED IN A SUBPROCESS, one per question that can abort.
A LAMMPS error normally raises a Python exception, but "normally" is doing real
work in that sentence -- a build without exceptions enabled calls MPI_Abort, which
takes the whole process with it. A probe that can kill the thing it is probing is
useless, so the answers come back through a pipe.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

# Asked in a child interpreter: build the smallest system the question needs and
# print one JSON line. Anything that goes wrong is an answer, not a crash.
_SNIPPET = r"""
import json, os, sys


class _StopHere(Exception):
    # Not an error: the light level stops before opening the library. A comment and
    # not a docstring, because this source lives inside a triple-quoted template and
    # a nested triple quote would end it.
    pass


out = {"ok": False, "numpy": None, "scipy": None, "version": None, "module": None}
out["executable"] = sys.executable
# Each dependency in its OWN try. Sharing one made a missing numpy report itself
# as "no usable lammps Python module", which sent the reader off to rebuild
# LAMMPS -- the wrong repair for the wrong problem. Now every answer is separate,
# and the verdict below decides which one to complain about first.
try:
    # numpy is required (every readout is an array); scipy is NOT -- the server
    # runs no analysis, and the one thing that needs scipy (the pair list) is
    # built on the client. Worth reporting both, because a cluster python with
    # numpy and no scipy is common and would otherwise look like a risk.
    import numpy
    out["numpy"] = numpy.__version__
except Exception as exc:
    out["numpy_error"] = f"{type(exc).__name__}: {exc}"
try:
    import scipy
    out["scipy"] = scipy.__version__
except Exception:
    pass
try:
    import lammps as lammps_module
    out["module"] = lammps_module.__file__
    # Where liblammps.so would be looked for. Reported WITHOUT opening it: on a
    # CUDA build the library links against the driver, so the file existing and
    # the file loading are two different questions answered on two different
    # machines (see the libcuda note in verdict()).
    import os.path
    here = os.path.dirname(out["module"])
    for candidate in [os.path.join(here, n) for n in ("liblammps.so", "liblammps.dylib")] + [
            os.path.join(d, "liblammps.so")
            for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]:
        if os.path.isfile(candidate):
            out["library"] = candidate
            break
    if not %(open_lammps)r:
        raise _StopHere
    from lammps import lammps
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"] + %(args)r)
    out["version"] = lmp.version()
    if %(plugin)r:
        # The local profile: the pair style is compiled on demand and loaded at
        # run time, so it is not in the build and asking whether it is would be
        # the wrong question.
        try:
            from lammps_live.forcefields.mesomem import MESOMEM_PLUGIN
            from lammps_live.playground.plugin import ensure_loaded
            out["plugin"] = ensure_loaded(MESOMEM_PLUGIN, lmp)
        except Exception as exc:
            out["plugin_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["packages"] = sorted(lmp.installed_packages)
    except Exception:
        out["packages"] = []
    def has(cat, name):
        try:
            return bool(lmp.has_style(cat, name))
        except Exception:
            return None
    out["styles"] = {
        "atom/dipole_sphere_angle": has("atom", "dipole_sphere_angle"),
        "atom/hybrid": has("atom", "hybrid"),
        "pair/mesomem": has("pair", "mesomem"),
        "pair/mesomem/kk": has("pair", "mesomem/kk"),
        "fix/nve/sphere": has("fix", "nve/sphere"),
        "fix/nve/sphere/kk": has("fix", "nve/sphere/kk"),
        "fix/langevin/kk": has("fix", "langevin/kk"),
    }
except _StopHere:
    pass
except Exception as exc:
    out["lammps_error"] = f"{type(exc).__name__}: {exc}"
out["ok"] = bool(out["numpy"] and (out["version"] if %(open_lammps)r
                                   else out["module"]))
print("PROBE " + json.dumps(out))
"""

# The coefficient-count question needs a real pair style on real atoms, because
# "how many values does pair_coeff take" is only answerable by handing it that
# many and seeing whether it complains.
_COEFF_SNIPPET = r"""
import json
out = {"values": None}
try:
    from lammps import lammps
    for count in (9, 8):
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        try:
            if %(plugin)r:
                from lammps_live.forcefields.mesomem import MESOMEM_PLUGIN
                from lammps_live.playground.plugin import ensure_loaded
                ensure_loaded(MESOMEM_PLUGIN, lmp)
            for cmd in ["units lj", "dimension 3", "atom_style %(atom_style)s",
                        "boundary p p p",
                        "region box block 0 10 0 10 0 10 units box",
                        "create_box 1 box",
                        "create_atoms 1 single 1 1 1 units box",
                        "create_atoms 1 single 3 1 1 units box",
                        "set group all diameter 1.0", "set group all density 1.0",
                        # THE MASS IDIOM, exactly as the profile will write it. It
                        # is here because the probe passing while the server failed
                        # on its very first setup command is a real thing that
                        # happened: this snippet reached for `set ... density`, the
                        # per-atom-mass spelling, so the per-TYPE `mass 1 1.0` that
                        # the force field emits was never tried anywhere before the
                        # server tried it on the node -- and a style with `rmass`
                        # rejects it. Whatever the profile declares, it gets tested
                        # here, where failing costs ten seconds instead of an
                        # allocation.
                        %(mass_cmd)r,
                        "pair_style mesomem 2.5",
                        "pair_coeff 1 1 " + " ".join(
                            ["1.0", "1.0", "12.0", "1.0", "2.5", "2.0", "5.0",
                             "0.0", "0.0"][:count]),
                        "run 0"]:
                lmp.command(cmd)
            out["values"] = count
            lmp.close()
            break
        except Exception as exc:
            out.setdefault("rejected", {})[count] = f"{type(exc).__name__}: {exc}"
            try:
                lmp.close()
            except Exception:
                pass
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("PROBE " + json.dumps(out))
"""


def _run_snippet(code, timeout=180):
    """Run `code` in a child interpreter and return the JSON it printed.

    The output is scanned for the PROBE marker rather than parsed whole, because a
    LAMMPS build prints its own banners, MPI warnings and Kokkos configuration to
    both streams and none of that is ours to suppress.
    """
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.startswith("PROBE "):
            try:
                return json.loads(line[len("PROBE "):])
            except json.JSONDecodeError:
                pass
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"error": "no answer from the child interpreter: " + " / ".join(tail)}


def _gpus():
    """What nvidia-smi says, or None where there is no nvidia-smi (a login node,
    or this laptop)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def probe(profile="cluster-gpu", level="full"):
    """Everything the server needs to know about this machine, as a dict.

    `profile` is a hosts.HostProfile name, so the probe asks its questions the way
    the server will actually build: with Kokkos and an in-tree pair style on the
    cluster, with a compiled-on-demand plugin locally.

    `level` is WHERE this is running, and it matters more than it sounds:

      "light"  what a login node can answer: the interpreter, numpy, and whether
               the `lammps` module and its shared library are present.
      "full"   everything, which means OPENING that library -- and a Kokkos/CUDA
               build links against libcuda.so.1, the NVIDIA driver, which exists
               only on a machine that has a GPU. So the full level belongs inside
               the allocation, on the node, and asking for it on a login node
               fails with an error about the driver that says nothing about the
               build being wrong.
    """
    from . import hosts
    host = hosts.get(profile)
    report = {
        "profile": profile,
        "level": level,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "gpus": _gpus(),
    }
    # Kokkos is only started when there is a GPU to start it on: on a login node
    # `-k on g 1` is a hard error, and "the build check failed" would be the wrong
    # thing to conclude from it.
    args = [a for a in host.lammps_args if a != "-sf" and a != "kk"]
    if not report["gpus"]:
        args = []
    report["lammps"] = _run_snippet(_SNIPPET % {"args": args,
                                                "plugin": host.load_plugin,
                                                "open_lammps": level == "full"})
    if level == "full" and report["lammps"].get("ok"):
        styles = report["lammps"].get("styles") or {}
        style = (host.atom_style if (host.atom_style
                                     and styles.get("atom/dipole_sphere_angle"))
                 else "hybrid sphere dipole")
        report["atom_style"] = style
        if styles.get("pair/mesomem"):
            mass_cmd = ("set type 1 mass 1.0" if host.per_atom_mass
                        else "mass 1 1.0")
            report["mass_cmd"] = mass_cmd
            report["coeff"] = _run_snippet(
                _COEFF_SNIPPET % {"atom_style": style,
                                  "plugin": host.load_plugin,
                                  "mass_cmd": mass_cmd})
    return report


_MODULE_ADVICE = (
    "  the module comes from a LAMMPS built with -DBUILD_SHARED_LIBS=yes,",
    "  then `make install-python` -- or just add <build>/python to",
    "  PYTHONPATH. (-DPKG_PYTHON is a different thing: python INSIDE",
    "  LAMMPS, and it is not needed for this.)",
)


def verdict(report):
    """(ok, [line, ...]) -- the report as something to show a user.

    `ok` is about whether the server can run at all, so a missing Kokkos variant
    is a warning: the demo works, it is just slower than the hardware allows.
    """
    lines = []
    lmp = report.get("lammps") or {}
    fatal = []
    gpus = report.get("gpus")
    lines.append(f"host {report.get('host')}, profile {report.get('profile')}")
    # The interpreter's PATH, not just its version: "which python is this?" is the
    # first question every dependency failure below raises.
    lines.append(f"python {report.get('python')} at "
                 f"{lmp.get('executable') or report.get('executable')}")
    if gpus is None and report.get("level") == "light":
        # Expected, and said so plainly: this check deliberately runs on the LOGIN
        # node, before anything is allocated, so that a missing interpreter or a
        # missing numpy costs no queue time. It is not a sign that anything failed.
        lines.append("no GPU here -- expected: this is the login-node check, before "
                     "the allocation")
    elif gpus is None:
        lines.append("no GPU here -- and opening a CUDA build needs one; expect the "
                     "driver error below")
    else:
        lines.append(f"{len(gpus)} GPU(s): " + "; ".join(g[:60] for g in gpus))

    # numpy first, and on its own: it is the far more common failure, and when it is
    # missing the LAMMPS import below usually fails only BECAUSE of it.
    if not lmp.get("numpy"):
        fatal.append("this python has no numpy: " + str(lmp.get("numpy_error")))
        lines.append("FATAL " + fatal[-1])
        lines.append("  the server needs numpy (nothing else -- scipy is not used")
        lines.append("  there). Two ways: load a module stack that has it, e.g.")
        lines.append("  `module load 2023 && module load SciPy-bundle/2023.07-gfbf-2023a`")
        lines.append("  from your env.sh; or point LAMMPS_LIVE_REMOTE_PYTHON at a")
        lines.append("  python that already has numpy AND can import lammps.")
    else:
        lines.append(f"numpy {lmp.get('numpy')}, scipy {lmp.get('scipy') or 'absent'}"
                     f" (scipy not needed -- the analysis runs on the client)")

    if lmp.get("module"):
        lines.append("lammps module: " + str(lmp["module"]))
        lines.append("liblammps at:  " + str(lmp.get("library") or "not found next "
                                             "to the module or on LD_LIBRARY_PATH"))
    detail = str(lmp.get("lammps_error") or "")

    if report.get("level") == "light":
        # The login node's job: the interpreter and the files. Opening the library
        # is somebody else's question.
        if not lmp.get("module"):
            fatal.append("no `lammps` Python module on this PYTHONPATH: " + detail)
            lines.append("FATAL " + fatal[-1])
            lines.extend(_MODULE_ADVICE)
        elif not lmp.get("library"):
            lines.append("WARNING liblammps.so was not found beside the module or "
                         "on LD_LIBRARY_PATH -- it may still be found by rpath")
        if not fatal:
            lines.append("looks right so far; the build itself is checked on the "
                         "GPU node, where the CUDA driver exists")
        return not fatal, lines

    if not lmp.get("version"):
        if not lmp.get("numpy"):
            lines.append(f"`import lammps` also failed ({detail[:60]}) -- probably "
                         f"just numpy; fix that first and re-probe")
        elif "libcuda" in detail:
            # The single most confusing error this port can produce, so it gets
            # named. A Kokkos/CUDA liblammps links against the NVIDIA driver, and
            # the driver is installed on GPU nodes only.
            fatal.append("liblammps needs the CUDA driver (libcuda.so.1), which "
                         "exists only on a GPU node")
            lines.append("FATAL " + fatal[-1])
            lines.append("  This is EXPECTED on a login node and does not mean the")
            lines.append("  build is broken: a Kokkos/CUDA build links against the")
            lines.append("  driver, so its library can only be opened where a GPU")
            lines.append("  is. Run this probe inside an allocation:")
            lines.append("    srun --jobid=<id> python -m lammps_live.remote.probe")
            lines.append("  (the connect flow now does exactly that, on the node.)")
        else:
            fatal.append("no usable `lammps` Python module: " + detail)
            lines.append("FATAL " + fatal[-1])
            lines.extend(_MODULE_ADVICE)
        return False, lines
    lines.append(f"liblammps {lmp.get('version')}, "
                 f"{len(lmp.get('packages') or [])} packages")
    if fatal:
        return False, lines
    styles = lmp.get("styles") or {}
    cluster = str(report.get("profile", "")).startswith("cluster")
    if lmp.get("plugin"):
        lines.append("pair style loaded as a plugin: " + str(lmp["plugin"]))
    elif lmp.get("plugin_error"):
        fatal.append("the mesomem plugin would not build or load: "
                     + str(lmp["plugin_error"]))
        lines.append("FATAL " + fatal[-1])
    if not styles.get("pair/mesomem"):
        fatal.append("the `mesomem` pair style is not in this build")
        lines.append("FATAL " + fatal[-1])
    if cluster:
        if not styles.get("atom/dipole_sphere_angle"):
            lines.append("note: no `dipole_sphere_angle` atom style; "
                         "falling back to `hybrid sphere dipole`")
        if not styles.get("pair/mesomem/kk"):
            lines.append("WARNING no `mesomem/kk`: -sf kk will fall back to the "
                         "host pair style, and the run will be slow but correct")
        if not styles.get("fix/nve/sphere/kk"):
            lines.append("WARNING no `nve/sphere/kk`: the integrator will copy "
                         "state host<->device every step "
                         "(docs/snellius/README.md point 2)")
    coeff = report.get("coeff") or {}
    rejected = (coeff.get("rejected") or {}).values()
    if any("atom mass" in str(r) for r in rejected):
        fatal = True
        lines.append(f"FATAL this build rejected `{report.get('mass_cmd')}`: "
                     f"{next(iter(rejected))}. The profile's `per_atom_mass` is the "
                     f"wrong way round for this atom style (remote/hosts.py)")
    elif report.get("mass_cmd"):
        lines.append(f"mass set with `{report['mass_cmd']}` -- accepted")
    if coeff.get("values") == 9:
        lines.append("pair_coeff takes 9 values -- the patched splay term is here")
    elif coeff.get("values") == 8:
        lines.append("pair_coeff takes 8 values -- no splay_symmetry; "
                     "that slider will be hidden")
    elif "values" in coeff:
        lines.append("WARNING could not settle the pair_coeff arity: "
                     + str(coeff.get("error") or coeff.get("rejected")))
    return not fatal, lines


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lammps_live.remote.probe",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="print the raw report as one JSON object")
    parser.add_argument("--profile", default="cluster-gpu",
                        help="host profile to ask about: cluster-gpu (default), "
                             "cluster-cpu, or local")
    parser.add_argument("--level", default="full", choices=("light", "full"),
                        help="full (default) opens liblammps, and so needs a GPU "
                             "node for a CUDA build; light checks the interpreter, "
                             "numpy and the module files only, which is all a login "
                             "node can answer")
    args = parser.parse_args(argv)
    report = probe(profile=args.profile, level=args.level)
    ok, lines = verdict(report)
    report["ok"] = ok
    report["summary"] = lines
    if args.json:
        print(json.dumps(report))
    else:
        for line in lines:
            print(line)
        print("\nverdict:", "the server can run here" if ok
              else "the server CANNOT run here")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
