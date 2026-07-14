"""Abstract interface every simulated "system" (material + geometry) must
implement, plus the metadata dataclasses the UI and force-feedback layer
read to configure themselves per-system.

All systems use LAMMPS "metal" units (eV, Angstrom, ps, amu) -- real
physical scales, not reduced LJ units -- so numbers on screen mean what a
physicist expects them to mean regardless of which system is active.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SliderSpec:
    """UI-facing description of a live-adjustable parameter (temperature,
    puller damping, ...). Ranges are per-system: a value tuned for copper's
    force scale is meaningless for a soft LJ gas."""
    label: str
    vmin: float
    vmax: float
    default: float
    fmt: str = "{:.3f}"
    unit: str = ""


@dataclass(frozen=True)
class ForceFeedbackProfile:
    """Per-system tuning for translating interaction forces into
    force-feedback / on-screen-arrow signals. These live per-system (not as
    global constants) because they're scaled to each system's characteristic
    force magnitude -- EAM copper contact forces run ~0.1-6 eV/A, a soft LJ
    gas is far weaker, and a single global knee/threshold would make one of
    them feel numb or pegged at max.
    """
    input_force_scale: float   # eV/Angstrom at full joystick/mouse deflection
    ff_exaggeration: float     # amplifies small/medium contact forces before tanh soft-saturation
    ff_knee: float             # raw*exaggeration magnitude at the soft-saturation knee
    ff_max_mag: float = 120.0  # device-unit cap, comfortably inside the SDK's +-127 spring-offset range
    stiffness_threshold: float = 0.05  # eV/A -- below this, spring/arrow reads as fully limp
    stiffness_knee: float = 0.5        # eV/A -- ramp scale above threshold
    damper_min_fraction: float = 0.10  # of DAMPER_COEFFICIENT_MAX, when idle
    damper_max_fraction: float = 0.50  # of DAMPER_COEFFICIENT_MAX, in contact
    vel_damp_max_fraction: float = 0.5  # of CP_OFFSET_MAX, velocity-opposing spring-center bias


@dataclass(frozen=True)
class SystemSpec:
    """Static, UI-facing description of a system -- everything the renderer
    and control loop need without having to introspect the LAMMPS instance."""
    key: str                 # stable id, used on the CLI (--system) and for cycling
    name: str                 # short display name
    description: str          # one-liner shown in the panel header
    element_label: str        # legend text, e.g. "Cu (EAM)"
    lattice_spacing: float    # Angstrom -- equilibrium in-plane spacing, informational
    timestep: float           # ps -- used to convert "sim time per frame" into a step count
    temperature: SliderSpec
    damping: SliderSpec
    melt_temp: float          # K -- approximate dial marker (see each system's docstring)
    force_feedback: ForceFeedbackProfile
    puller_speed_cap: float   # Angstrom/ps -- nve/limit-derived velocity ceiling, for velocity-damping scale
    # Per-species rendering, for systems whose atoms aren't a single
    # indistinguishable species (see get_all_positions' `species`). None ->
    # every crystal atom drawn the same flat color (Cu, Ar). Otherwise an RGB
    # tuple per species index, with an optional matching glyph per species
    # (e.g. "+"/"-" for ions; None entries draw no glyph).
    species_colors: tuple = None   # (RGB, RGB, ...) indexed by species, or None
    species_labels: tuple = None   # (str-or-None, ...) indexed by species, or None
    # Whether to draw the generic "faint line between atoms near their
    # equilibrium spacing" bond overlay. On for crystals; off for systems that
    # supply their own explicit bonds to draw (see get_bond_pairs), e.g. lipids.
    bond_overlay: bool = True


class MDSystem(ABC):
    """A self-contained LAMMPS setup: a thermostatted crystal/fluid plus one
    interactively-controlled "puller" atom. Concrete systems own everything
    LAMMPS-specific (pair style, lattice, region layout); the app loop and
    renderer talk only to this interface.
    """

    spec: SystemSpec

    @abstractmethod
    def set_input_force(self, fx, fy):
        """Joystick/mouse-commanded force on the puller, in eV/Angstrom (0 at center)."""

    @abstractmethod
    def set_target_temp(self, T):
        """Thermostat target, in K."""

    @abstractmethod
    def set_puller_damping(self, gamma):
        """Puller's own velocity-proportional drag, in eV*ps/Angstrom^2."""

    @abstractmethod
    def step(self, n):
        """Advance the simulation by n timesteps."""

    @abstractmethod
    def get_puller_state(self):
        """Returns (pos, vel), each a 2-element array, or (None, None)."""

    @abstractmethod
    def get_puller_energy(self):
        """Returns (ke, pe) of the puller atom alone, in eV."""

    @abstractmethod
    def get_interaction_force(self):
        """Net force on the puller from the rest of the system, in eV/Angstrom."""

    @abstractmethod
    def get_thermo_state(self):
        """Returns (temp[K], press[bar], ke[eV], pe[eV], etotal[eV])."""

    @abstractmethod
    def get_sim_time(self):
        """Elapsed simulated (physical MD) time, in ps, since interactive
        control began -- i.e. since __init__'s silent settle run finished,
        not since the LAMMPS instance itself was created. This is real
        simulated time (nsteps * timestep), not wall-clock render time."""

    @abstractmethod
    def get_rdf(self):
        """Returns (r, g(r)) arrays or None if not yet warmed up."""

    @abstractmethod
    def get_all_positions(self):
        """Returns (ids N, positions Nx2, is_puller boolarray N, species).
        ids are LAMMPS atom ids, stable identities across frames -- needed
        (rather than array index) because LAMMPS is free to reorder its local
        atom arrays between steps (e.g. periodic spatial sorting), which array
        index alone can't survive (see ui/trail.py, which keys per-atom motion
        trails by id for exactly this reason).

        species is a per-atom int array (aligned with positions) indexing into
        the SystemSpec's species_colors/species_labels, or None for
        single-species systems (Cu, Ar) where every crystal atom is drawn the
        same. E.g. NaCl uses 0=cation, 1=anion; lipids use 0=head, 1=tail."""

    def get_bond_pairs(self):
        """Optional: explicit bonds to draw, as an (M, 2) int array of index
        pairs into the CURRENT get_all_positions ordering (so it must be called
        in the same frame, before stepping again). Used to draw molecular
        backbones, e.g. each lipid's head-tail-tail chain. None (default) means
        the system has no explicit bonds to draw."""
        return None

    def steer_orientation(self, rate, dt):
        """Optional: steer the puller's in-plane orientation. rate is a control
        signal in [-1, 1] (joystick yaw / twist axis, or Q/E in mouse mode); dt
        is the frame time in seconds. No-op for systems whose puller has no
        meaningful orientation (a lone atom); lipids integrate it into the
        control lipid's director angle."""

    @abstractmethod
    def get_box_size(self):
        """Returns (width, height) of the simulation box, in Angstrom."""

    @abstractmethod
    def close(self):
        ...
