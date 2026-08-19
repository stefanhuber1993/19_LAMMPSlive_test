"""What differs about a LAMMPS that is not the pip wheel this app was built on.

The cluster's build is not a drop-in for the local one, in four specific ways, and
all four are here rather than scattered through the force field:

  1. THE PAIR STYLE IS COMPILED IN, not loaded. Locally, `mesomem` is the authors'
     C++ compiled on demand into a shared library and pulled in with `plugin load`
     (playground/plugin.py). The cluster build has it in-tree -- it has to, because
     the Kokkos variant `mesomem/kk` cannot be a runtime plugin -- so the
     `plugin load` of a macOS .dylib must not be attempted there.

  2. THE ATOM STYLE HAS A DIFFERENT NAME. Locally `hybrid sphere dipole`; the GPU
     build carries a purpose-written `dipole_sphere_angle` with a Kokkos variant,
     which is what keeps the atom data on the device (docs/snellius/README.md).
     Same per-particle fields either way -- x, mu, omega, torque, radius, rmass.

  3. AND SO IT TAKES ITS MASS DIFFERENTLY. That `rmass` in the list above is the
     whole story, and it cost a real run to notice: a style that stores mass
     PER ATOM rejects the per-type `mass 1 1.0` command outright --

         ERROR: Cannot set per-type atom mass for atom style dipole_sphere_angle/kk

     -- while `hybrid sphere dipole` accepts it, which is why every local run and
     every loopback test passed. The same number therefore has to be written as
     `set type 1 mass 1.0`, and the profile rewrites the force field's command
     rather than making every force field ask what kind of build it is on.

  4. `pair_coeff` MAY TAKE ONE FEWER VALUE. This repo's patched style has a 9th
     numeric coefficient, `splay_symmetry`; the authors' original stops at 8. Which
     one is on the node is not something to guess -- probe.py finds out (it tries
     the 9-value form on two atoms and watches for an error), and the server passes
     the answer in here. When it is 8, the client's splay-symmetry slider is
     removed rather than left dragging something that is not connected.

A profile ADAPTS A FORCE-FIELD INSTANCE IN PLACE. That is deliberate and it is
why it is safe: `PlaygroundSystem` constructs a fresh force field per system, so
setting `ff.atom_style` on the instance shadows the class attribute for that one
instance and nothing else in the process is affected. The alternative -- a second
`MesoMem` subclass registered under another name -- would have to be named in the
playground file, which would make the local loopback test unrunnable and put a
cluster's build details into a file that is meant to describe physics.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HostProfile:
    """How to build this app's playground on one particular LAMMPS."""

    name: str
    # Extra command-line arguments for the LAMMPS instance.
    lammps_args: tuple = ()
    # Override the force field's atom style, or None to keep its own.
    atom_style: str = None
    # Whether to `plugin load` a custom pair style (False when it is compiled in).
    load_plugin: bool = True
    # Truncate `pair_coeff` to this many numeric values, or None for all of them.
    coeff_values: int = None
    # Whether this host's atom style stores mass per ATOM rather than per type, and
    # so needs `set type I mass M` where the force field wrote `mass I M`.
    per_atom_mass: bool = False
    notes: tuple = field(default_factory=tuple)

    def adapt(self, ff):
        """Adjust a force-field instance for this host. Returns it, for chaining."""
        if self.atom_style:
            ff.atom_style = self.atom_style
        if not self.load_plugin:
            ff.plugin = None
        if self.coeff_values is not None:
            ff.pair_commands = _truncating(ff.pair_commands, self.coeff_values)
            ff.coeff_commands = _truncating(ff.coeff_commands, self.coeff_values)
        if self.per_atom_mass:
            ff.setup_commands = _per_atom_mass(ff.setup_commands)
        return ff

    def with_coeff_values(self, n):
        from dataclasses import replace
        return replace(self, coeff_values=n)


def _truncating(emit, keep):
    """Wrap a command-emitting method so each `pair_coeff` it produces keeps only
    its first `keep` numeric values.

    Both `pair_commands` and `coeff_commands` are wrapped. The base class defines
    the second as the first minus its `pair_style` line -- but a force field is
    free to write it out itself, and MesoMem does, so wrapping only one of them
    truncates the build and leaves every live slider re-issuing the untruncated
    form. Which is a nastier bug than it sounds: it would work at startup and fail
    the first time the user touched k_tilt.
    """
    def wrapped(params):
        out = []
        for cmd in emit(params):
            if cmd.startswith("pair_coeff"):
                bits = cmd.split()
                head, values = bits[:3], bits[3:]     # `pair_coeff I J` + numbers
                cmd = " ".join(head + values[:keep])
            out.append(cmd)
        return out
    return wrapped


def _per_atom_mass(emit):
    """Wrap a command-emitting method so `mass I M` becomes `set type I mass M`.

    A rewrite rather than a flag on the force field, for the same reason as
    `_truncating`: the force field describes the physics ("each bead weighs M"),
    and which of LAMMPS' two spellings expresses that is a property of the build it
    is about to run on.

    `set` needs the atoms to exist, and it is already true that these commands are
    issued after `create_atoms` -- MesoMem's own `set group all diameter` in the
    same list depends on it.
    """
    def wrapped(params):
        out = []
        for cmd in emit(params):
            bits = cmd.split()
            if bits and bits[0] == "mass" and len(bits) == 3:
                cmd = f"set type {bits[1]} mass {bits[2]}"
            out.append(cmd)
        return out
    return wrapped


# The local machine: the pip LAMMPS wheel, the compiled-on-demand plugin, and this
# repo's patched pair style with all 9 coefficients. Also what the loopback test
# uses, which is why "run the server here" is a supported configuration and not a
# special case.
LOCAL = HostProfile("local")

# The cluster's Kokkos/CUDA build on one A100 (or one MIG slice -- same flags).
# `-k on g 1` starts Kokkos with one GPU; `-sf kk` appends /kk to every style that
# has a Kokkos variant. Deliberately no `-pk kokkos newton on neigh half`: the
# ported pair style asks for `full, newton off`, which is the Kokkos default and
# the right list for a GPU kernel -- forcing a half list fights the port (see
# docs/snellius/README.md point 3).
CLUSTER_GPU = HostProfile(
    "cluster-gpu",
    lammps_args=("-k", "on", "g", "1", "-sf", "kk"),
    atom_style="dipole_sphere_angle",
    load_plugin=False,
    per_atom_mass=True,
    notes=("Kokkos/CUDA, one GPU",),
)

# The same in-tree build with the GPU left out: the fallback if Kokkos turns out to
# be the problem, and the way to tell a GPU speedup from a build difference, since
# only the two command-line flags change.
CLUSTER_CPU = HostProfile(
    "cluster-cpu",
    atom_style="dipole_sphere_angle",
    load_plugin=False,
    per_atom_mass=True,
    notes=("in-tree build, host styles only",),
)

PROFILES = {p.name: p for p in (LOCAL, CLUSTER_GPU, CLUSTER_CPU)}


def get(name):
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown host profile {name!r}. Available: {known}") from None
