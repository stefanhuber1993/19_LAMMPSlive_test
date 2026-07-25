"""Force fields built from stock LAMMPS pair styles -- no C++ to compile.

These are the "I want to explore an existing potential" case, and they are what
prove the playground layer is not shaped around MesoMem: `metal` units instead of
reduced, real physical parameters, 2D, an external potential table, no
orientations, and a pair style that DOES implement single() (so the interaction
force can come from `compute group/group` rather than being reconstructed).

Both expose live parameters, which the original hand-written systems did not --
epsilon and sigma were module constants there, so you could not feel what argon's
well depth does. That is the point of the refactor: declaring a parameter is now
cheap enough to be worth doing.
"""
from ..playground.forcefield import ForceField, register
from ..playground.params import Param, Tier

# --- Lennard-Jones argon ------------------------------------------------------
# Argon's textbook LJ parameters (e.g. Rahman 1964): sigma = 3.40 A,
# epsilon/kB = 120.7 K, mass = 39.948 amu. So the melting point these reproduce
# is real argon's ~84 K, not a dial mark someone guessed.
AR_EPSILON = 0.0104   # eV
AR_SIGMA = 3.40       # Angstrom
AR_MASS = 39.948      # amu


@register
class LennardJones(ForceField):
    """`lj/cut` -- soft van der Waals bonding.

    Contact forces run ~0.01-0.5 eV/A, roughly two orders of magnitude weaker
    than EAM copper's, which is why the force-feedback profile scaled for
    reduced units would read as permanently numb here.
    """

    name = "lj"
    units = "metal"
    dimension = 2
    atom_style = "atomic"
    has_directors = False
    # lj/cut implements single(), so the pair force on a group can be read
    # directly with compute group/group -- no reconstruction from total force.
    supports_single = True
    energy_terms_labels = ()   # a single-term potential has nothing to decompose

    params = (
        Param("epsilon", AR_EPSILON, "epsilon (well depth)", 0.0, 0.05,
              optimum=AR_EPSILON, fmt="{:.4f}", unit=" eV",
              doc="LJ well depth; argon's value is 0.0104 eV"),
        # Deliberately a narrow span. Unlike epsilon (which only scales the energy),
        # sigma sets the equilibrium spacing, and the crystal's lattice constant is
        # STRUCTURAL -- fixed at build time and calibrated for argon's sigma. Pushed
        # far past that, every atom sits inside its neighbour's repulsive core and
        # the crystal does not merely melt, it detonates. A range whose ends reliably
        # destroy the simulation is not a useful control, so this one stays inside
        # the band where the fixed lattice can still accommodate it.
        Param("sigma", AR_SIGMA, "sigma (atom size)", 3.1, 3.9,
              optimum=AR_SIGMA, fmt="{:.2f}", unit=" A", advanced=True,
              doc="LJ length scale; argon's value is 3.40 A"),
        # Changing the cutoff moves the pair style's global cutoff, so the whole
        # style is re-declared -- the same tier MesoMem's rc uses.
        Param("cutoff", 2.5 * AR_SIGMA, "cutoff", 3.0, 12.0, fmt="{:.2f}",
              unit=" A", advanced=True, tier=Tier.HOT_RESTYLE),
    )

    def __init__(self, mass=AR_MASS):
        self.mass = mass

    def setup_commands(self, params):
        return [f"mass 1 {self.mass}"]

    def pair_commands(self, params):
        return [f"pair_style lj/cut {params['cutoff']}"] + self.coeff_commands(params)

    def coeff_commands(self, params):
        return [f"pair_coeff 1 1 {params['epsilon']} {params['sigma']}"]

    def interaction_cutoff(self, params):
        return float(params["cutoff"])

    def integrator_command(self):
        # The scenario integrates its own groups here (a frozen floor, a mobile
        # crystal, a displacement-limited puller), so there is no single global
        # integrator to install.
        return None


# --- EAM copper ---------------------------------------------------------------

@register
class EAM(ForceField):
    """Embedded-atom-method metallic bonding, read from a tabulated potential.

    EAM's energy is not pairwise-additive (it has a many-body embedding term), so
    there is no per-pair decomposition to offer and no Python reference to check
    -- `energy_terms_labels` is empty and `--verify` correctly skips it. That is
    the honest answer for this potential, and it is worth having the abstraction
    say so rather than pretend.
    """

    name = "eam"
    units = "metal"
    dimension = 2
    atom_style = "atomic"
    has_directors = False
    supports_single = True
    energy_terms_labels = ()

    # The tabulated potential has no coefficients to dial, so the only live
    # parameter is a scale on the pair energy -- which LAMMPS cannot express for
    # eam, so this force field simply declares none. A force field with nothing
    # tunable is a legitimate thing to declare.
    params = ()

    def __init__(self, potential_file=None, mass=63.546, element="Cu"):
        import os
        if potential_file is None:
            here = os.path.dirname(os.path.abspath(__file__))
            potential_file = os.path.join(here, "data", "Cu_u3.eam")
        self.potential_file = potential_file
        self.mass = mass
        self.element = element

    def setup_commands(self, params):
        return [f"mass 1 {self.mass}"]

    def pair_commands(self, params):
        return ["pair_style eam", f"pair_coeff 1 1 {self.potential_file}"]

    def coeff_commands(self, params):
        return [f"pair_coeff 1 1 {self.potential_file}"]

    def interaction_cutoff(self, params):
        # Read from the table by LAMMPS; this is only used to size the analysis
        # pair list, and the EAM tables here cut off well inside 6 A.
        return 6.0

    def integrator_command(self):
        return None
