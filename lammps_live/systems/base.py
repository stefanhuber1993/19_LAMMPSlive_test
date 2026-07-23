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
    # Stable id for extra_sliders (see SystemSpec): the app passes this back to
    # MDSystem.set_extra_param so a system knows which live parameter changed.
    # Empty for the built-in temperature/damping sliders, which have dedicated
    # setters (set_target_temp / set_puller_damping).
    key: str = ""


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
    # Flat draw color for single-species systems (Cu, Ar), so they read as
    # distinct materials -- warm metallic copper vs. cold noble-gas argon --
    # instead of sharing one generic crystal color. None falls back to the
    # theme's CRYSTAL_COLOR. Ignored when species_colors is set.
    crystal_color: tuple = None
    # On-screen atom size, given as a PHYSICAL radius in Angstrom (converted to
    # pixels per-frame at the active box's scale, then clamped) so atoms are
    # drawn at their real relative sizes: argon's atom genuinely dwarfs copper's,
    # a Cl- anion genuinely dwarfs a Na+ cation, and the packing you see is the
    # real packing. atom_radius_A is the single-species value; species_radii_A,
    # if set, gives a per-species radius (indexed like species_colors). Both
    # None -> the theme's fixed-pixel CRYSTAL_RADIUS fallback.
    atom_radius_A: float = None
    species_radii_A: tuple = None
    # Real MD time advanced per rendered frame, in ps (overrides the global
    # config.SIM_TIME_PER_FRAME). Mesoscale systems (the coarse-grained lipid
    # membrane) evolve on a much slower intrinsic time scale, so at the shared
    # default a whole frame barely moves them and the membrane looks frozen /
    # jittery rather than a living, self-healing fluid -- they need a larger
    # per-frame time slice to come alive. None -> use the global default.
    sim_time_per_frame: float = None
    # Whether to draw the generic "faint line between atoms near their
    # equilibrium spacing" bond overlay. On for crystals; off for systems that
    # supply their own explicit bonds to draw (see get_bond_pairs), e.g. lipids.
    bond_overlay: bool = True
    # Whether this system renders as a 3D scene (perspective + depth-cued
    # spheres and directors) rather than the default top-down 2D box. A 3D
    # system additionally implements get_positions_3d / get_dipoles_3d /
    # get_bonds_3d / get_camera_params / get_control_grid, and the app routes
    # those to the renderer's 3D path. The standard 2D readouts (thermo, puller
    # energy, force feedback) are unchanged -- they act on the 2D control-plane
    # projection of the puller the system already returns.
    render_3d: bool = False
    # 3D only: draw the little per-bead director spike. On for small scenes
    # (7-bead patch) where it shows the flip; off for large sheets (hundreds of
    # beads) where hundreds of spikes are clutter and a per-bead draw cost -- the
    # banded pole/equator coloring already shows tilt there.
    director_arrows: bool = True
    # 3D only: extra live-tunable parameters beyond temperature/damping, drawn as
    # additional sliders in the panel. Each SliderSpec's `key` is handed back to
    # set_extra_param(key, value) when the user moves it. Empty -> no extra
    # sliders (every non-MesoMem system).
    extra_sliders: tuple = ()
    # 3D only: for a periodic scene, the crossfade band width as a fraction of
    # the box side. A bead within this fraction of an x/y seam dissolves toward
    # the background while a wrapped ghost fades in at the opposite edge, so it
    # slides across the periodic boundary instead of popping. 0 -> no wrap fade
    # (non-periodic scenes: the 7-bead patch).
    wrap_fade_fraction: float = 0.0


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
        backbones, e.g. each lipid's head-tail-tail chain, or the live covalent
        network of the carbon sheet (which visibly tears as bonds break). None
        (default) means the system has no explicit bonds to draw."""
        return None

    def get_hbond_pairs(self):
        """Optional: hydrogen-bond-like pairs to draw in a distinct, lighter
        style than the solid molecular bonds of get_bond_pairs -- as an (M, 2)
        int array of index pairs into the CURRENT get_all_positions ordering.
        Used by the water model to show the transient hydrogen-bond network
        forming and breaking. None (default) means no such overlay."""
        return None

    def get_hud_lines(self):
        """Optional: a list of short strings drawn as a small live HUD in the
        simulation view (beneath the standard force/time readout), for
        per-system pedagogical state the fixed panel readouts don't cover --
        e.g. the carbon demo's etched-atom / broken-bond tally, or the water
        demo's phase and density-anomaly readout. Empty/None -> nothing drawn."""
        return None

    def get_potential_terms(self):
        """Optional: a live breakdown of the puller's interaction energy into
        the separate additive terms of the force field, for pedagogy -- returned
        as (title, [(label, value), ...], scale) or None. The renderer draws it
        as a compact signed-bar chart (each term's bar, plus their sum) so the
        additive structure of the potential is visible and updates live. Values
        are in the system's own energy units (reduced/LJ for MesoMem); scale is
        the bar half-range. None (default) -> nothing drawn."""
        return None

    def set_extra_param(self, key, value):
        """Optional: update a live-tunable parameter beyond temperature/damping,
        identified by the `key` of one of spec.extra_sliders (e.g. the MesoMem
        systems' k_tilt / k_splay / eta). No-op for systems with no extra
        sliders. Called every frame with the slider's current value, so
        implementations should cheaply no-op when the value is unchanged."""

    def get_bead_brightness(self):
        """Optional (3D): a per-bead albedo brightness multiplier aligned with
        get_positions_3d's ordering (1.0 = normal). Used to spotlight a tagged
        cluster -- e.g. the sheet's diffusion-tracer bead and its neighbours.
        None (default) -> every bead drawn at normal brightness."""
        return None

    def steer_orientation(self, rate, dt):
        """Optional: steer the puller's in-plane orientation. rate is a control
        signal in [-1, 1] (joystick yaw / twist axis, or Q/E in mouse mode); dt
        is the frame time in seconds. No-op for systems whose puller has no
        meaningful orientation (a lone atom); lipids integrate it into the
        control lipid's director angle."""

    def get_scene_fit_points(self):
        """Optional: an (N, 3) array of world-space points the 3D camera should
        frame (zoom to just fit). Used to fill the viewport at any aspect ratio
        instead of a fixed field of view. None (default) -> keep the camera's
        fixed FOV (non-3D systems, or 3D systems that don't want auto-fit)."""
        return None

    def get_torque_signals(self):
        """Optional: (applied, reaction) torques about the control-plane normal
        for the circular torque arrows, each already normalized to [-1, 1]
        (fraction of the display maximum, positive = same screen handedness as a
        positive yaw command). `applied` is the user's steering torque, `reaction`
        is the force field's restoring torque on the puller. Both are the
        component projected onto the 2D plane the scene depicts. None (default)
        -> no torque arrows (systems whose puller has no meaningful orientation)."""
        return None

    def puller_bead_count(self):
        """Number of LAMMPS atoms the puller is made of: 1 for a single-atom
        puller (Cu, Ar, NaCl, the etch O), or the molecule's bead count for a
        molecular puller (lipid = 3, water = 4). set_input_force applies its
        force to each of these atoms, so a caller wanting a specific NET force on
        the puller (e.g. the app's joystick MD-force cancellation) divides by
        this count. Default 1."""
        return 1

    @abstractmethod
    def get_box_size(self):
        """Returns (width, height) of the simulation box, in Angstrom."""

    @abstractmethod
    def close(self):
        ...
