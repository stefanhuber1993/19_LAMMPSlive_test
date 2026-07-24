"""2D coarse-grained lipid membrane -- a solvent-free, self-assembling bilayer,
alongside the metallic (Cu), van-der-Waals (Ar) and ionic (NaCl) crystals.

Inspiration and relation to the source paper. This demo is inspired by the
MesoMem mesoscale membrane model (Sillano, Marrink & Idema, 2026), a
solvent-free one-particle-thick bilayer model in which each bead is a whole
patch of bilayer carrying an orientation vector (director), with additive
repulsion + cosine-squared attraction + orientation-dependent *tilt* and
*splay* energies. Those tilt/splay terms need the authors' custom LAMMPS
pair-style, which isn't in a stock build, so this demo keeps the same physics
in spirit while using only standard LAMMPS styles by dropping down one level
of coarse-graining: instead of one oriented bead per bilayer patch, each lipid
is the minimal *amphiphile* -- a short 3-bead chain, one hydrophilic head + two
hydrophobic tails (the classic implicit-solvent lipid of Cooke, Kremer &
Deserno, 2005). The head->tail vector now plays the role of MesoMem's director
n_i explicitly, as a physical object rather than an abstract vector, and
orientational order emerges from real bonded geometry instead of a tilt/splay
potential. This is arguably *more* pedagogical: you can literally see the
heads, the tails, and why a bilayer forms.

Why this is stable and self-assembling in 2D. Lipids are amphiphilic: the
tails are hydrophobic and the (implicit) solvent is water. With no explicit
solvent, that hydrophobic effect is modeled directly as a short-range
attraction between tail beads only -- heads attract nothing, they only take up
space. Beads that are near their preferred separation are pulled together;
heads, wanting to face the (implicit) water, end up on the outside. In 2D the
energy-minimizing arrangement is the iconic bilayer *cross-section*: two rows
of lipids tail-to-tail, hydrophilic heads on both outer faces, hydrophobic
tails buried in the middle. The demo starts from exactly this pre-formed flat
bilayer (reliable and instantly recognizable) rather than a random gas, so the
membrane is there to interact with from frame 0; the same forces would
self-assemble it from disorder given time.

Interactions (all standard LAMMPS, LJ-scale energies expressed in metal units):
- soft excluded volume + tail attraction via `pair_style cosine/squared` (the
  purpose-built Cooke-Deserno pair style): a WCA repulsive core on every pair,
  plus a smoothly-truncated cosine-squared attractive well on tail-tail pairs
  only. The soft 4-2-like core deliberately avoids crystallization, keeping the
  membrane fluid.
- harmonic backbone bonds (head-tail, tail-tail) and a harmonic angle
  (head-tail-tail, rest 180 degrees) that gives each lipid its rod-like
  stiffness -- the bending rigidity that lets a flat bilayer resist collapse.
- Langevin dynamics (`fix langevin` + `fix nve`): the implicit solvent's
  viscous drag and thermal kicks, exactly as in the paper. This IS the
  thermostat here (there is no separate crystal bath), so the temperature dial
  drives the paper's phase behavior directly -- gel/ordered at low T, a fluid
  membrane at intermediate T, and evaporation (the bilayer boiling apart into a
  gas of lipids) at high T.

The interactive puller is itself a lipid -- a 3-bead control lipid you steer by
position (joystick / mouse) AND by *orientation*: the joystick's yaw (twist)
axis, or the Q/E keys in mouse mode, rotate its in-plane director. The
pedagogical game is to bring your lipid in with the right orientation -- head
out, tails toward the core -- and insert it into a leaflet, feeling the
membrane's resistance through force feedback. Its beads are integrated so its
center of mass responds to your input force, the membrane's reaction, and the
damping dial, while each frame its shape/orientation are re-imposed as a rigid
lipid at the director angle you've dialed in (see _rigidify_puller).

Constants below were tuned by simulation the same way the other systems' were:
the bilayer stays intact and fluid around room temperature, disorders into a
fluid as T rises, and boils apart into a gas at high T (checked by watching the
tail beads' spread and the membrane staying a single connected sheet).
"""
import math
import os
import random
import tempfile

import numpy as np
from lammps import lammps

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec

# LJ-scale interaction constants, in metal units. EPS sets the energy scale so
# that the reduced fluid regime (kT ~ 1) lands near room temperature: with
# EPS = 0.025 eV, reduced T = 1 is ~290 K (see T_MELT).
SIGMA = 6.0        # Angstrom -- bead diameter / bond length
EPS = 0.025        # eV -- LJ well depth / repulsion strength
WC = 1.6 * SIGMA   # Angstrom -- width of the tail-tail attractive branch (fluid at ~1.6 sigma)
RC_TAIL = SIGMA + WC   # tail-tail attraction cutoff
BEAD_MASS = 100.0  # amu -- sets the (arbitrary) mesoscopic time scale

K_BOND = 2.0       # eV/Angstrom^2 -- stiff harmonic backbone bonds
K_ANGLE = 3.0      # eV/rad^2 -- lipid bending stiffness (rest angle 180 deg -> rod-like)

# Membrane geometry (pre-formed flat bilayer).
N_PER_LEAFLET = 26         # lipids per leaflet (box is sized to fit them across, periodic x)
A_LIPID = 1.0 * SIGMA      # lateral spacing between lipids along the membrane
BOND_LEN = SIGMA           # head-tail and tail-tail rest length

SETTLE_STEPS = 800
TIMESTEP = 0.005            # ps -- verified stable through membrane rupture at high T

# Langevin (implicit-solvent) friction and the puller's own drag. The puller's
# default drag is kept small (like the other systems') so it responds
# near-ballistically to the input force -- a large value makes it feel like
# it's stuck in molasses; the dial still lets you add that "heavy" feel.
LANGEVIN_DAMP = 2.0         # ps -- solvent viscous relaxation time
# Puller drag + input scale set the control lipid's top speed = input_scale/gamma.
# At the old 2.0 / 0.02 the lipid's terminal speed was ~100 A/ps -- it shot across
# the whole box in under half a second, uncontrollable. These give ~0.7/0.05 ~=
# 14 A/ps, a steady few-seconds glide across the box that you can actually aim,
# while still out-pushing the membrane's soft (~0.1-0.5 eV/A) contact forces.
PULLER_DAMPING_DEFAULT = 0.05   # eV*ps/Angstrom^2
PULLER_DAMPING_MIN = 0.0
PULLER_DAMPING_MAX = 0.2

# Temperature dial. EPS was chosen so the fluid membrane sits near room
# temperature; below ~200 K it's an ordered gel, above ~450-500 K it boils
# apart into a gas of lipids (the paper's three regimes).
T_MIN = 1.0
T_MAX = 800.0
T_MELT = 320.0     # K -- approximate gel->fluid dial marker (see module docstring)

# Yaw steering: how fast the twist axis / Q-E keys rotate the control lipid.
YAW_RATE = 3.0     # radians per second at full deflection

RDF_NBINS = 100
RDF_CUTOFF = 4.0 * SIGMA
RDF_AVE_EVERY = 5
RDF_AVE_REPEAT = 40
RDF_AVE_FREQ = RDF_AVE_EVERY * RDF_AVE_REPEAT

# Per-species render colors (see SystemSpec.species_colors): 0 = hydrophilic
# head, 1 = hydrophobic tail. Warm head, muted tail.
HEAD_COLOR = (240, 100, 130)
TAIL_COLOR = (200, 190, 120)

# Contact forces here are soft (~0.1-0.5 eV/A, measured), so the force-feedback
# profile is argon-like rather than copper's stiff one -- with enough input
# authority to push a lipid into the membrane.
FORCE_FEEDBACK = ForceFeedbackProfile(
    ff_exaggeration=4.0,
    ff_knee=0.3,
    ff_max_mag=120.0,
    stiffness_threshold=0.03,
    stiffness_knee=0.2,
    damper_min_fraction=0.10,
    damper_max_fraction=0.50,
    vel_damp_max_fraction=0.5,
)

SPEC = SystemSpec(
    key="lipid",
    name="Lipid membrane (coarse-grained)",
    description="A 2D solvent-free lipid bilayer -- steer a lipid (yaw / Q-E rotates it) and insert it into the membrane.",
    element_label="lipid (head/tail)",
    lattice_spacing=SIGMA,
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, 220.0, fmt="{:.0f}", unit=" K"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                        PULLER_DAMPING_DEFAULT, fmt="{:.3f}"),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    max_input_force=0.7,   # eV/A at full deflection, shared by joystick/WASD/mouse
    puller_speed_cap=0.03 * SIGMA / TIMESTEP,   # matches the retuned, calmer top speed
    species_colors=(HEAD_COLOR, TAIL_COLOR),  # 0=head, 1=tail
    species_labels=None,
    # Beads sized a bit under the WCA contact radius (~0.5 sigma) so each lipid's
    # head+two tails read as fat amphiphile blobs while still leaving the drawn
    # backbone sticks visible between them; the head is drawn slightly larger so
    # the hydrophilic end (and the lipid's orientation) stands out.
    species_radii_A=(0.34 * SIGMA, 0.30 * SIGMA),   # 0=head (larger), 1=tail
    # This coarse-grained mesoscale membrane evolves ~20x slower per fs than the
    # atomistic crystals (heavy beads, long Langevin relaxation), so at the
    # shared global time-per-frame it barely moved between frames and looked
    # frozen and jittery rather than a living, self-healing fluid bilayer.
    # Advancing more real time per frame lets its (genuinely soft) interactions
    # actually play out on screen -- the membrane ripples, heals, and resists
    # insertion visibly. Stable: the 0.005 ps timestep is unchanged and was
    # verified through rupture; only the number of steps per frame goes up.
    sim_time_per_frame=0.06,   # ps/frame (12 steps at the 0.005 ps timestep)
    bond_overlay=False,   # lipids draw their own explicit backbones (get_bond_pairs)
)


class LipidMembraneSystem(MDSystem):
    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._input_fx = 0.0
        self._input_fy = 0.0
        # Director angle of the control lipid (head direction), steered by yaw.
        # Start pointing up (+y): head out, tails down toward the membrane below.
        self._target_angle = math.pi / 2
        self._target_temp = None
        self._last_id_to_index = {}
        self._build()
        self.set_input_force(0.0, 0.0)
        self._interactive_ps = 0.0

    # ---- construction -------------------------------------------------------

    def _write_bilayer_data(self, path):
        """Write a LAMMPS data file for a pre-formed flat bilayer: two leaflets
        of 3-bead lipids (head + 2 tails), heads on the outer faces, tails
        meeting at the midplane. Returns (Lx, Ly, n_lipids)."""
        aL = A_LIPID
        Lx = N_PER_LEAFLET * aL      # periodic across the membrane
        Ly = Lx                       # square box; membrane sits across the middle
        yc = Ly / 2
        b = BOND_LEN
        atoms, bonds, angles = [], [], []
        aid = 0
        mol = 0

        def lipid(x, y_head, direction):
            # direction -1: head high, tails descending (upper leaflet)
            # direction +1: head low,  tails ascending  (lower leaflet)
            nonlocal aid, mol
            mol += 1
            h = (aid + 1, mol, 1, x, y_head)
            t1 = (aid + 2, mol, 2, x, y_head + direction * b)
            t2 = (aid + 3, mol, 2, x, y_head + direction * 2 * b)
            atoms.extend([h, t1, t2])
            bonds.append((len(bonds) + 1, 1, aid + 1, aid + 2))
            bonds.append((len(bonds) + 1, 1, aid + 2, aid + 3))
            angles.append((len(angles) + 1, 1, aid + 1, aid + 2, aid + 3))
            aid += 3

        for i in range(N_PER_LEAFLET):
            x = (i + 0.5) * aL
            lipid(x, yc + 2.5 * b, -1)   # upper leaflet
            lipid(x, yc - 2.5 * b, +1)   # lower leaflet

        with open(path, "w") as f:
            f.write("2D coarse-grained lipid bilayer\n\n")
            f.write(f"{len(atoms)} atoms\n{len(bonds)} bonds\n{len(angles)} angles\n\n")
            f.write("2 atom types\n1 bond types\n1 angle types\n\n")
            f.write(f"0 {Lx} xlo xhi\n0 {Ly} ylo yhi\n-0.1 0.1 zlo zhi\n\n")
            f.write(f"Masses\n\n1 {BEAD_MASS}\n2 {BEAD_MASS}\n\n")
            f.write("Atoms # molecular\n\n")
            for a in atoms:
                f.write(f"{a[0]} {a[1]} {a[2]} {a[3]:.4f} {a[4]:.4f} 0.0\n")
            f.write("\nBonds\n\n")
            for bd in bonds:
                f.write(f"{bd[0]} {bd[1]} {bd[2]} {bd[3]}\n")
            f.write("\nAngles\n\n")
            for an in angles:
                f.write(f"{an[0]} {an[1]} {an[2]} {an[3]} {an[4]}\n")
        return Lx, Ly, len(atoms) // 3

    def _build(self):
        lmp = self.lmp
        fd, datafile = tempfile.mkstemp(suffix=".data", prefix="lipid_")
        os.close(fd)
        try:
            Lx, Ly, n_lipids = self._write_bilayer_data(datafile)
            lmp.command("dimension 2")
            lmp.command("units metal")
            lmp.command("atom_style molecular")
            lmp.command("boundary p f p")
            # The control lipid's 3 beads are created after read_data, so leave
            # room for them in the per-type atom arrays.
            lmp.command(f"read_data {datafile} extra/atom/types 0")
        finally:
            os.remove(datafile)

        self.xlo, self.ylo, self.xhi, self.yhi = 0.0, 0.0, Lx, Ly
        self.n_membrane = lmp.get_natoms()

        lmp.command("bond_style harmonic")
        lmp.command(f"bond_coeff 1 {K_BOND} {BOND_LEN}")
        lmp.command("angle_style harmonic")
        lmp.command(f"angle_coeff 1 {K_ANGLE} 180.0")

        # cosine/squared: WCA core on all pairs (the `wca` keyword); attractive
        # cosine-squared well only on tail-tail (2-2), out to RC_TAIL. Head
        # pairs get a bare repulsive core (cutoff = SIGMA -> no attractive tail).
        lmp.command(f"pair_style cosine/squared {RC_TAIL}")
        lmp.command(f"pair_coeff 1 1 {EPS} {SIGMA} {SIGMA} wca")   # head-head, repulsion only
        lmp.command(f"pair_coeff 1 2 {EPS} {SIGMA} {SIGMA} wca")   # head-tail, repulsion only
        lmp.command(f"pair_coeff 2 2 {EPS} {SIGMA} {RC_TAIL} wca")  # tail-tail, repulsion + attraction

        lmp.command("neighbor 3.0 bin")
        lmp.command("neigh_modify every 1 delay 0 check yes")
        # The RDF compute reaches farther than the pair cutoff, so ghost atoms
        # must be communicated out to at least its cutoff + skin.
        lmp.command(f"comm_modify cutoff {RDF_CUTOFF + 4.0}")

        # Control lipid (the puller): 3 unbonded beads (head type 1, two tails
        # type 2). Unbonded because its shape/orientation are imposed each frame
        # by _rigidify_puller, not by bond forces. Placed above the membrane.
        px = (self.xlo + self.xhi) / 2
        py = self.yhi / 2 + 7 * SIGMA
        self.rest_pos = (px, py)
        self.puller_head_id = self.n_membrane + 1
        self.puller_t1_id = self.n_membrane + 2
        self.puller_t2_id = self.n_membrane + 3
        self.puller_ids = (self.puller_head_id, self.puller_t1_id, self.puller_t2_id)
        lmp.command(f"create_atoms 1 single {px} {py + BOND_LEN} 0.0 units box")
        lmp.command(f"create_atoms 2 single {px} {py} 0.0 units box")
        lmp.command(f"create_atoms 2 single {px} {py - BOND_LEN} 0.0 units box")

        idlist = " ".join(str(i) for i in self.puller_ids)
        lmp.command(f"group puller id {idlist}")
        lmp.command("group membrane subtract all puller")

        # Explicit backbone bonds to draw (see get_bond_pairs), as id pairs:
        # every membrane lipid's head-t1-t2 chain, plus the control lipid.
        self._bond_id_pairs = []
        for k in range(n_lipids):
            h, t1, t2 = 3 * k + 1, 3 * k + 2, 3 * k + 3
            self._bond_id_pairs.append((h, t1))
            self._bond_id_pairs.append((t1, t2))
        self._bond_id_pairs.append((self.puller_head_id, self.puller_t1_id))
        self._bond_id_pairs.append((self.puller_t1_id, self.puller_t2_id))

        lmp.command("compute mem_temp membrane temp")
        # Net pair force from the membrane on the control lipid, for the
        # reaction-force arrow and force feedback.
        lmp.command("compute iff puller group/group membrane")

        # Membrane: Langevin (implicit solvent) thermostat on plain nve.
        self._seed = random.randint(1, 900_000_000)
        lmp.command("fix integ_mem membrane nve")
        self._target_temp = SPEC.temperature.default
        lmp.command(
            f"fix lang membrane langevin {self._target_temp} {self._target_temp} "
            f"{LANGEVIN_DAMP} {self._seed}"
        )
        # Puller: its own capped-velocity nve + viscous drag (the damping dial).
        lmp.command(f"fix integ_pul puller nve/limit {0.1 * SIGMA}")
        self._puller_damping = PULLER_DAMPING_DEFAULT
        lmp.command(f"fix damp_pul puller viscous {PULLER_DAMPING_DEFAULT}")
        # Reflecting walls keep everything inside the non-periodic-y box.
        lmp.command(
            f"fix walls all wall/reflect ylo {self.ylo + 0.5 * SIGMA} "
            f"yhi {self.yhi - 0.5 * SIGMA} units box"
        )
        lmp.command("fix twod all enforce2d")

        lmp.command(f"timestep {TIMESTEP}")
        lmp.command("thermo 100000")

        self._rigidify_puller()
        lmp.command(f"run {SETTLE_STEPS}")
        self._rigidify_puller()

        lmp.command("compute ke_atom all ke/atom")
        lmp.command("compute pe_atom all pe/atom")
        lmp.command(f"compute rdf_raw membrane rdf {RDF_NBINS} cutoff {RDF_CUTOFF}")
        lmp.command(
            f"fix rdf_avg membrane ave/time {RDF_AVE_EVERY} {RDF_AVE_REPEAT} {RDF_AVE_FREQ} "
            f"c_rdf_raw[*] mode vector"
        )
        self._rdf_bins_ready = False
        self._rdf_r = None
        self._rdf_ready_step = lmp.extract_global("ntimestep") + RDF_AVE_FREQ + RDF_AVE_EVERY

    # ---- puller orientation -------------------------------------------------

    def _puller_local_indices(self):
        """Current local array indices of the 3 control-lipid beads (LAMMPS may
        reorder atoms between steps), as (head, t1, t2), plus nlocal."""
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        idx = {int(i): k for k, i in enumerate(ids)}
        return (idx.get(self.puller_head_id), idx.get(self.puller_t1_id),
                idx.get(self.puller_t2_id)), nlocal

    def _rigidify_puller(self):
        """Impose the control lipid's shape and orientation: keep its beads a
        rigid 3-in-a-line lipid, centered on its (dynamic) center of mass and
        aligned with the user-steered director angle. The center of mass is left
        untouched (so it still responds to the input force, the membrane's
        reaction and the damping dial), only the internal shape/orientation are
        overwritten. Bead velocities are set to the common COM velocity so no
        spurious internal motion is injected."""
        (ih, i1, i2), nlocal = self._puller_local_indices()
        if ih is None or i1 is None or i2 is None:
            return
        x = self.lmp.numpy.extract_atom("x")
        v = self.lmp.numpy.extract_atom("v")
        com = (x[ih][:2] + x[i1][:2] + x[i2][:2]) / 3.0
        comv = (v[ih][:2] + v[i1][:2] + v[i2][:2]) / 3.0
        d = np.array([math.cos(self._target_angle), math.sin(self._target_angle)])
        head = com + BOND_LEN * d      # head points along the director
        tail2 = com - BOND_LEN * d     # far tail opposite
        x[ih][0], x[ih][1] = head[0], head[1]
        x[i1][0], x[i1][1] = com[0], com[1]
        x[i2][0], x[i2][1] = tail2[0], tail2[1]
        for i in (ih, i1, i2):
            v[i][0], v[i][1] = comv[0], comv[1]

    def steer_orientation(self, rate, dt):
        self._target_angle += rate * YAW_RATE * dt

    def puller_bead_count(self):
        return 3   # head + 2 tails

    # ---- controls -----------------------------------------------------------

    def set_input_force(self, fx, fy):
        # Applied to each of the 3 puller beads; net force on the lipid is
        # 3*(fx,fy) over mass 3*BEAD_MASS -> the COM accelerates as a single
        # bead of mass BEAD_MASS would, so the feel matches the other systems.
        self._input_fx = fx
        self._input_fy = fy
        self.lmp.command(f"fix input_force puller addforce {fx} {fy} 0.0")

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        self.lmp.command(
            f"fix lang membrane langevin {T} {T} {LANGEVIN_DAMP} {self._seed}"
        )

    def set_puller_damping(self, gamma):
        gamma = max(PULLER_DAMPING_MIN, min(PULLER_DAMPING_MAX, gamma))
        if gamma == self._puller_damping:
            return
        self._puller_damping = gamma
        self.lmp.command(f"fix damp_pul puller viscous {gamma}")

    def step(self, n=4):
        self.lmp.command(f"run {n}")
        self._interactive_ps += n * TIMESTEP
        self._rigidify_puller()

    # ---- readouts -----------------------------------------------------------

    def get_sim_time(self):
        return self._interactive_ps

    def _puller_com(self):
        (ih, i1, i2), nlocal = self._puller_local_indices()
        if ih is None or i1 is None or i2 is None:
            return None, None, None
        x = self.lmp.numpy.extract_atom("x")[:nlocal]
        v = self.lmp.numpy.extract_atom("v")[:nlocal]
        com = (x[ih][:2] + x[i1][:2] + x[i2][:2]) / 3.0
        comv = (v[ih][:2] + v[i1][:2] + v[i2][:2]) / 3.0
        return com.copy(), comv.copy(), (ih, i1, i2)

    def get_puller_state(self):
        com, comv, _ = self._puller_com()
        if com is None:
            return None, None
        return com, comv

    def get_puller_energy(self):
        com, comv, idxs = self._puller_com()
        if com is None:
            return None, None
        nlocal = self.lmp.get_natoms()
        ke = self.lmp.numpy.extract_compute("ke_atom", 1, 1)[:nlocal]
        pe = self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:nlocal]
        ke_tot = float(sum(ke[i] for i in idxs))
        pe_tot = float(sum(pe[i] for i in idxs))
        return ke_tot, pe_tot

    def get_interaction_force(self):
        vec = self.lmp.extract_compute("iff", 0, 1)
        return np.array([vec[0], vec[1]])

    def get_thermo_state(self):
        temp = self.lmp.extract_compute("mem_temp", 0, 0)
        press = self.lmp.get_thermo("press")
        ke = self.lmp.get_thermo("ke")
        pe = self.lmp.get_thermo("pe")
        etotal = self.lmp.get_thermo("etotal")
        return temp, press, ke, pe, etotal

    def get_rdf(self):
        step = self.lmp.extract_global("ntimestep")
        if step < self._rdf_ready_step:
            return None
        if not self._rdf_bins_ready:
            self._rdf_r = np.array(
                [self.lmp.extract_fix("rdf_avg", 0, 2, nrow=i, ncol=0) for i in range(RDF_NBINS)]
            )
            self._rdf_bins_ready = True
        g = np.array(
            [self.lmp.extract_fix("rdf_avg", 0, 2, nrow=i, ncol=1) for i in range(RDF_NBINS)]
        )
        return self._rdf_r, g

    def get_all_positions(self):
        nlocal = self.lmp.get_natoms()
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        types = self.lmp.numpy.extract_atom("type")[:nlocal]
        species = (types - 1).astype(int)   # 0 = head (type 1), 1 = tail (type 2)
        is_puller = np.isin(ids, self.puller_ids)
        self._last_id_to_index = {int(i): k for k, i in enumerate(ids)}
        return ids.copy(), xs[:, :2].copy(), is_puller, species

    def get_bond_pairs(self):
        idx = self._last_id_to_index
        pairs = []
        for a, b in self._bond_id_pairs:
            ia, ib = idx.get(a), idx.get(b)
            if ia is not None and ib is not None:
                pairs.append((ia, ib))
        return np.array(pairs, dtype=int) if pairs else None

    def get_box_size(self):
        return self.xhi - self.xlo, self.yhi - self.ylo

    def close(self):
        self.lmp.close()
