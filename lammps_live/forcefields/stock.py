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


# --- ionic NaCl ---------------------------------------------------------------
# The rigid-ion model of an alkali halide: long-range Coulomb (Madelung)
# attraction between alternating Na(+) and Cl(-), balanced against a short-range
# Born-Mayer exponential repulsion. Constants below were measured, not guessed:
#
#   rho = 0.32 A   the standard alkali-halide hardness length.
#   A, sigma       set so a pressure sweep of a periodic 2D checkerboard (no free
#                  surface) puts the zero-pressure nearest-neighbour distance at
#                  2.89 A -- close to real NaCl's 2.82 A -- with a cohesive
#                  energy near -3.5 eV/ion, the right ballpark for an alkali
#                  halide. A bare Coulomb plus a point repulsion collapses the
#                  lattice instead (with sigma = 0 the exponential is negligible
#                  at contact), which is why sigma is not simply zero.
#   alpha = 0.25   damping for the shifted-force Coulomb.
NACL_BORN_A = 1.0       # eV
NACL_BORN_RHO = 0.32    # Angstrom
NACL_BORN_SIGMA = 2.4   # Angstrom
NACL_DSF_ALPHA = 0.25   # 1/Angstrom
NACL_COUL_CUTOFF = 12.0  # Angstrom
NA_MASS, CL_MASS = 22.99, 35.45   # amu


@register
class BornDSF(ForceField):
    """`born/coul/dsf` -- Born-Mayer repulsion plus damped-shifted-force Coulomb.

    Why DSF and not Ewald/PPPM: this is a 2D, non-periodic-in-y slab, which is
    exactly the geometry k-space solvers handle worst (they want a periodic third
    dimension and a slab correction). Damped shifted force (Fennell & Gezelter
    2006) gives a smooth, energy-conserving, real-space-only Coulomb instead --
    no reciprocal-space part at all.

    Two atom types, opposite charges. The charge is a live parameter and is the
    most instructive dial in the file: take it to zero and the Madelung bonding
    that holds the checkerboard together simply switches off, leaving a bare
    Born-Mayer repulsion that blows the lattice apart.
    """

    name = "born_dsf"
    units = "metal"
    dimension = 2
    # Ions carry a per-atom electrostatic charge, which the Coulomb term reads.
    atom_style = "charge"
    n_types = 2
    has_directors = False
    # born/coul/dsf offers no single(), so `compute group/group` cannot report the
    # force between two groups and the runtime reconstructs it as "total force on
    # the controlled ion minus the forces we applied to it ourselves".
    supports_single = False
    # No decomposition offered. The pair energy would split cleanly into
    # repulsion and Coulomb, but the DSF Coulomb also carries a PER-ATOM self
    # term (-(e_shift/2 + alpha/sqrt(pi)) q^2), and this interface expresses
    # energies per PAIR -- so a decomposition here would silently disagree with
    # the potential energy LAMMPS reports. An honest empty is better than a
    # breakdown that does not add up.
    energy_terms_labels = ()

    params = (
        # The whole ionic bond, on one slider.
        Param("charge", 1.0, "ion charge |q|", 0.0, 1.5, optimum=1.0,
              fmt="{:.2f}", unit=" e",
              doc="Na(+q) / Cl(-q); 0 switches the Madelung bonding off"),
        Param("born_A", NACL_BORN_A, "Born repulsion A", 0.0, 3.0,
              optimum=NACL_BORN_A, fmt="{:.2f}", unit=" eV", advanced=True,
              doc="prefactor of the exponential core A*exp((sigma-r)/rho)"),
        Param("born_rho", NACL_BORN_RHO, "Born hardness rho", 0.15, 0.60,
              optimum=NACL_BORN_RHO, fmt="{:.2f}", unit=" A", advanced=True,
              doc="decay length of the repulsion; 0.32 A is the alkali-halide value"),
        # Narrow, like LJ's sigma and for the same reason: it sets the
        # equilibrium spacing, while the lattice constant is structural.
        Param("born_sigma", NACL_BORN_SIGMA, "Born onset sigma", 2.0, 2.8,
              optimum=NACL_BORN_SIGMA, fmt="{:.2f}", unit=" A", advanced=True,
              doc="ion contact size: where the exponential repulsion turns on"),
        Param("cutoff", NACL_COUL_CUTOFF, "Coulomb cutoff", 6.0, 12.0,
              fmt="{:.1f}", unit=" A", advanced=True, tier=Tier.HOT_RESTYLE,
              doc="real-space cutoff; DSF needs no reciprocal-space part"),
    )

    def __init__(self, alpha=NACL_DSF_ALPHA, masses=(NA_MASS, CL_MASS)):
        self.alpha = alpha
        self.masses = masses

    def setup_commands(self, params):
        return [f"mass 1 {self.masses[0]}", f"mass 2 {self.masses[1]}"] + \
               self.charge_commands(params)

    def charge_commands(self, params):
        """Type 1 is the cation, type 2 the anion. Equal and opposite, so the
        cell stays exactly neutral -- which the Coulomb sum needs."""
        q = params["charge"]
        return [f"set type 1 charge {q}", f"set type 2 charge {-q}"]

    def pair_commands(self, params):
        return [f"pair_style born/coul/dsf {self.alpha} {params['cutoff']}"] + \
               self.coeff_commands(params)

    def coeff_commands(self, params):
        # C and D (the r^-6 / r^-8 dispersion terms) are zeroed: a pure
        # exponential core added to the Coulomb.
        return [f"pair_coeff * * {params['born_A']} {params['born_rho']} "
                f"{params['born_sigma']} 0.0 0.0"]

    def live_commands(self, params, changed_name):
        """Charge is not a pair coefficient -- it lives on the particles -- so it
        needs `set`, not `pair_coeff`. Everything else takes the generic path."""
        if changed_name == "charge":
            return self.charge_commands(params)
        return super().live_commands(params, changed_name)

    def interaction_cutoff(self, params):
        return float(params["cutoff"])

    def integrator_command(self):
        return None      # the scenario integrates its own groups


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
