"""The ForceField interface: one object per force field, reused by every
scenario that runs it.

A ForceField owns three things, and nothing else:

  1. Its parameter declarations (see params.Param) -- the single source of the
     sliders the GUI shows.
  2. The LAMMPS commands that install it (`pair_style`, `pair_coeff`, masses,
     per-particle attributes) and re-install it when a live parameter moves.
  3. Optionally, a vectorized Python expression of its energy, decomposed into
     the additive terms it is built from.

That third item is the part that makes this more than a refactor. It exists once,
so it can (a) drive the live energy-breakdown panels and (b) be checked against
what LAMMPS actually computed (see verify.py). Previously the MesoMem energy
expression was hand-written five times across three system modules -- with
docstrings warning that the copies had to be kept in sync with the C++ by hand.

It is a DIAGNOSTIC, not the force loop. The forces that move particles always
come from the compiled pair style inside LAMMPS.
"""
from abc import ABC, abstractmethod

from .params import ParamSet

_REGISTRY = {}


def register(cls):
    """Class decorator adding a ForceField to the by-name registry."""
    _REGISTRY[cls.name] = cls
    return cls


def get(name):
    """Look up a force field class by name, importing the bundled ones lazily.

    Lazy so `--list-playgrounds` does not import scipy and the LAMMPS bindings
    just to print a table (the old eager systems registry imported all eight
    system modules, and therefore `lammps` + `scipy`, to answer
    `--list-systems`).
    """
    if name not in _REGISTRY:
        from .. import forcefields  # noqa: F401  -- import registers the bundled fields
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown force field {name!r}. Available: {known}") from None


def available():
    from .. import forcefields  # noqa: F401
    return dict(_REGISTRY)


class ForceField(ABC):
    """Base class for a force field.

    Subclasses set the class attributes and implement `pair_commands`. Everything
    else is optional.
    """

    name = ""                    # registry key, used in playground files
    units = "lj"                 # LAMMPS units command
    dimension = 3
    atom_style = "atomic"
    n_types = 1
    # Set when the style is a custom C++ pair style needing compilation (see
    # plugin.PluginSpec); None for stock LAMMPS styles.
    plugin = None
    # Declared parameters, in the order their sliders should appear.
    params = ()
    # Display labels for the additive energy terms, in the order energy_terms
    # returns them. Empty -> this force field offers no decomposition.
    energy_terms_labels = ()
    # Whether particles carry an orientation (LAMMPS `mu`). Drives whether
    # FrameState.directors is populated and whether director-based observables
    # are offered.
    has_directors = False
    # Bar half-range for the energy panels, per particle involved. Scaled by the
    # relevant particle count so the same force field reads sensibly on a 7-bead
    # patch and a 1500-bead box.
    energy_scale_per_particle = 1.0

    def new_params(self, overrides=None):
        return ParamSet.build(self.params, overrides)

    def ensure_available(self, lmp):
        """Load any custom pair-style plugin into this LAMMPS instance."""
        if self.plugin is not None:
            from .plugin import ensure_loaded
            ensure_loaded(self.plugin, lmp)

    def setup_commands(self, params):
        """Per-particle setup issued after the atoms exist: masses, diameters,
        anything the pair style needs on each particle. Returns a list of LAMMPS
        command strings."""
        return []

    @abstractmethod
    def pair_commands(self, params):
        """The full pair-style installation, as a list of command strings --
        `pair_style ...` followed by its `pair_coeff` line(s)."""

    def live_commands(self, params, changed_name):
        """Commands to re-apply the force field after the live parameter
        `changed_name` moved.

        The default handles the two live tiers generically: a HOT parameter
        re-issues the coefficients, a HOT_RESTYLE parameter re-issues the whole
        pair style first (because it moved the global cutoff, and thus the
        neighbour list). LAMMPS overwrites the stored per-type coefficients in
        place and re-inits the pair style on the next run, so this is safe
        between steps.
        """
        from .params import Tier
        tier = params.tier_of(changed_name)
        if tier is Tier.HOT_RESTYLE:
            return self.pair_commands(params)
        return self.coeff_commands(params)

    def coeff_commands(self, params):
        """Just the coefficient line(s) -- by default the pair_commands minus the
        leading `pair_style`."""
        return [c for c in self.pair_commands(params)
                if not c.startswith("pair_style")]

    def interaction_cutoff(self, params):
        """The largest separation at which this force field does anything, used
        to build the pair list for the energy decomposition and observables."""
        return 0.0

    def energy_terms(self, state, pairs, params):
        """Per-pair energy in each additive term of the potential.

        Returns a dict {label: (M,) array} aligned with `pairs`, or None if this
        force field offers no decomposition. Per-PAIR (rather than pre-summed) so
        one evaluation serves both energy panels: the whole-system total sums
        everything, and a single particle's share sums the pairs touching it.
        """
        return None
