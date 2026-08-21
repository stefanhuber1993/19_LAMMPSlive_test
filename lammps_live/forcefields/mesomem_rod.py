"""MesoMem membrane plus one rigid rod -- the wrapping force field.

A second species is added to the membrane of `mesomem.py`: a single rigid rod
(the "bacterium"), interacting with the membrane beads through Pietro Sillano's
`rod_lj` pair style. That style is a generalised Lennard-Jones between a POINT
and a LINE SEGMENT: for each bead it finds the closest point on the rod's axis
and applies a 12-6 LJ at that separation, with

    sigma_eff = sigma_pair + r_rod

so the contact distance is measured surface-to-surface, and with the force
applied at the contact point -- which is what gives the rod a TORQUE and lets it
be wrapped rather than merely pushed.

The rod's geometry travels on the particle itself, which is the convention the
pair style fixes and the one reason this force field has per-particle setup at
all:

    q       the rod's length L. A particle with q == 0 is a point; the style
            resolves which of a pair is the rod by the sign of q, and skips
            point-point and rod-rod pairs entirely.
    radius  the rod's cylindrical half-thickness, added to sigma as above.
    mu      the rod's long axis (a unit vector).

So the membrane keeps `pair_style mesomem` among itself, the rod feels `rod_lj`
against every bead, and LAMMPS routes the two with `pair_style hybrid`. The
membrane's own physics is untouched -- this class inherits every parameter,
command and energy expression from MesoMem and adds one term.

The collaborator's original deck is kept beside the sources as
`mesomem_ff/planar_wrapping_rod.lmp`; it is the reference this reproduces
interactively (its four-phase tension ramp becomes the barostat settle the
HexSheet scenario already does, and its `velocity rod set 0 0 -1` becomes a
joystick).
"""
import numpy as np

from ..playground.forcefield import register
from ..playground.params import Param, Tier
from ..playground.state import segment_distance
from .mesomem import SIGMA, MesoMem

# Membrane-bead radius as the pair style wants it: `sigma_pair = r_mem`, and the
# style adds the rod's own radius to it. This is the paper's bead radius
# (sigma/2), NOT whatever `bead_diameter` the playground draws the beads at --
# the contact distance is a property of the potential, not of the sphere LAMMPS
# happens to integrate.
R_MEM = 0.5 * SIGMA

ADHESION = "adhesion  (rod-membrane)"


@register
class MesoMemRod(MesoMem):
    """The membrane force field with one rigid rod added.

    Three dials on top of the membrane's own. Together they are the wrapping
    phase diagram: adhesion strong enough beats the bending cost and the membrane
    engulfs the rod, and how strong "enough" is depends on the rod's radius
    against the membrane's stiffness.

      - eps_rod: rod-membrane well depth. The collaborator's deck runs 3.0, which
        is where wrapping happens at the paper's standard moduli; below ~1 the
        rod just rests on the surface, and far above it the membrane tears itself
        onto the rod.
      - rod_length / rod_radius: the rod's shape, live rather than structural
        because they are per-particle attributes and a `set` command changes them
        in place -- no rebuild, so the membrane you have already deformed stays
        deformed while the thing deforming it changes size. Both move the pair
        style's cutoff (rod_radius through sigma_eff, rod_length through the
        neighbour-list extension the style asks for), so both re-issue it.
    """

    name = "mesomem_rod"
    n_types = 2
    energy_terms_labels = MesoMem.energy_terms_labels + (ADHESION,)
    # `energy_scale_per_particle` is deliberately inherited unchanged: it sets the
    # WHOLE-SYSTEM panel's range, and there the membrane's own cohesion still
    # dominates. The ROD's panel is the one that needs a different range -- it is
    # one particle in contact with a hundred beads, where a bead touches a dozen
    # -- and that is the playground's `pulled_energy_scale`, not this.

    params = MesoMem.params + (
        Param("eps_rod", 3.0, "eps_rod (rod-membrane adhesion)", 0.0, 8.0,
              optimum=3.0, fmt="{:.2f}",
              doc="well depth of the rod-membrane LJ -- the dial that decides "
                  "whether the membrane wraps the rod or just dents"),
        Param("rod_length", 5.0, "rod length L", 1.0, 12.0, optimum=5.0,
              fmt="{:.1f}", tier=Tier.HOT_RESTYLE,
              doc="length of the rod's axis, in sigma (the pair style reads it "
                  "off the particle's charge)"),
        Param("rod_radius", 1.5, "rod radius", 0.5, 4.0, optimum=1.5,
              fmt="{:.2f}", tier=Tier.HOT_RESTYLE,
              doc="cylindrical half-thickness of the rod, added to sigma to set "
                  "the surface-to-surface contact distance. Dragged, it inflates "
                  "the rod in place; JUMPED to the top of the range while the rod "
                  "is buried in the membrane it overlaps it with dozens of beads "
                  "at once, which can destroy the simulation (recoverable with R)"),
    )

    def __init__(self, bead_diameter=1.0, mass=1.0, rod_mass=6.0):
        super().__init__(bead_diameter=bead_diameter, mass=mass)
        # WHY THIS IS A NUMBER AND NOT A DENSITY. The reference deck matches the
        # rod's density to the membrane's, which at the paper's bead volume makes
        # the rod ~130 times a bead's mass -- a bacterium no joystick can steer.
        # (The deck gets away with it by inflating the membrane beads to diameter
        # 4, which is what makes ITS rod come out near 1.) Nothing about wrapping
        # depends on the rod's inertia: the transition is set by eps_rod against
        # the membrane's moduli. So the mass is chosen for the hand instead --
        # around a dozen beads' worth, which feels substantial without being
        # immovable.
        #
        # Note also that this must be set as a per-atom mass. Under `atom_style
        # hybrid sphere dipole` the dynamics use per-atom `rmass`, and the
        # `mass <type> <m>` command does not touch it -- so `set type 2 mass` is
        # the only way to say what the rod weighs.
        self.rod_mass = rod_mass

    # ---- LAMMPS side --------------------------------------------------------

    def rod_cutoff(self, params):
        """Rod-membrane cutoff, measured from the rod's AXIS.

        The reference deck's `rc_mix`: 1.12 sigma_mix (just past the LJ minimum
        at 2^(1/6) sigma) plus two membrane radii of attractive tail.
        """
        sigma_mix = float(params["rod_radius"]) + R_MEM
        return 1.12 * sigma_mix + 2.0 * R_MEM

    def rod_attribute_commands(self, params):
        """The rod's geometry, as the per-particle attributes `rod_lj` reads.

        Re-issued whenever the shape changes, which is why it is its own method
        rather than inline in setup_commands.
        """
        return [
            f"set type 2 diameter {2.0 * float(params['rod_radius'])}",
            # q IS the length -- and a non-zero q is also what marks this particle
            # as the rod rather than a point (see the module docstring).
            f"set type 2 charge {float(params['rod_length'])}",
            f"set type 2 mass {self.rod_mass}",
        ]

    def setup_commands(self, params):
        """Per-type rather than per-group: MesoMem sets the bead diameter on
        `all`, which here would also size the rod. Stated by type so neither
        species depends on the order the commands happen to be issued in."""
        return [
            f"mass 1 {self.mass}",
            f"set type 1 diameter {self.bead_diameter}",
            # Membrane beads must read as points. create_atoms leaves q at zero
            # already; saying so is cheap, and the pair style's rod/point
            # resolution turns on it entirely.
            "set type 1 charge 0.0",
            f"mass 2 {self.rod_mass}",
        ] + self.rod_attribute_commands(params)

    def pair_commands(self, params):
        """`hybrid` over the two styles: membrane among itself, rod against it.

        2-2 has to be declared even though there is only one rod -- pair hybrid
        requires every type pair to be assigned -- and is inert either way, since
        rod_lj skips rod-rod pairs by construction.
        """
        return ([f"pair_style hybrid mesomem {params['rc']} "
                 f"rod_lj {self.rod_cutoff(params)}"]
                + self.coeff_commands(params))

    def coeff_commands(self, params):
        rc_rod = self.rod_cutoff(params)
        eps_rod = float(params["eps_rod"])
        membrane = super().coeff_commands(params)
        # The inherited line is `pair_coeff 1 1 <args>`; under hybrid it needs the
        # sub-style name after the type pair.
        membrane = [c.replace("pair_coeff 1 1 ", "pair_coeff 1 1 mesomem ", 1)
                    for c in membrane]
        return membrane + [
            f"pair_coeff 1 2 rod_lj {R_MEM} {eps_rod} {rc_rod}",
            f"pair_coeff 2 2 rod_lj {R_MEM} {eps_rod} {rc_rod}",
        ]

    def live_commands(self, params, changed_name):
        """A shape change is a per-particle change first, a pair-style change
        second: the style's neighbour-list extension is derived from the largest
        q it can find, so the new length has to be on the particle before the
        style is re-declared."""
        cmds = []
        if changed_name in ("rod_length", "rod_radius"):
            cmds += self.rod_attribute_commands(params)
        return cmds + super().live_commands(params, changed_name)

    def rod_reach(self, params):
        """How far from the rod's CENTRE a bead can still be in contact: its
        cutoff is measured from the axis, and the axis runs half a length either
        way."""
        return self.rod_cutoff(params) + 0.5 * float(params["rod_length"])

    def extended_pairs(self, state, pairs, params):
        """The rod's own pairs, found separately.

        The rod reaches more than twice as far as the membrane does (5.7 against
        2.5 at the default geometry), and `interaction_cutoff` is GLOBAL -- so
        widening it to cover the rod finds every membrane pair at the long range
        too. Measured on this playground's 3600 beads that is 335k pairs instead
        of 55k, a 37 ms lump on the frame the energy panels land on, all to find
        the ~120 pairs one rod is having.

        One rod against every bead is an O(N) numpy pass, so it is done directly
        here. That leaves the membrane's list meaning exactly what it says, which
        is also what makes `coordination` usable on this playground.
        """
        is_rod = self._rod_mask(state)
        if is_rod is None:
            return None
        from ..playground.state import PairData
        rods = np.flatnonzero(is_rod)
        reach = self.rod_reach(params)
        # Only what the base list could NOT have found. Inside the membrane's own
        # cutoff the rod's pairs are already in `pairs`, and adding them again
        # would double the adhesion energy.
        base = super().interaction_cutoff(params)
        a, b, d = [], [], []
        for i in rods:
            offsets = state.positions - state.positions[i]
            if state.box is not None:
                offsets = state.box.minimum_image(offsets)
            r = np.linalg.norm(offsets, axis=1)
            near = np.flatnonzero((r >= base) & (r < reach) & ~is_rod)
            if not len(near):
                continue
            # (a, b, d) with d = r_a - r_b, matching build_pairs' convention.
            a.append(np.full(len(near), i))
            b.append(near)
            d.append(-offsets[near])
        if not a:
            return None
        d = np.vstack(d)
        return PairData(np.concatenate(a), np.concatenate(b), d,
                        np.linalg.norm(d, axis=1))

    # ---- the Python reference expression ------------------------------------

    def energy_terms(self, state, pairs, params):
        """The membrane's three terms plus the rod's adhesion, per pair.

        The split is by TYPE, exactly as `pair_style hybrid` routes the forces:
        the membrane terms are defined only between two beads, and the adhesion
        term only between a bead and the rod. Zeroing the membrane terms on
        rod pairs is not cosmetic -- evaluated there they would read a rod axis
        as a membrane director and report a tilt energy for it.
        """
        terms = super().energy_terms(state, pairs, params)
        if not len(pairs):
            terms[ADHESION] = np.zeros(0)
            return terms

        is_rod = self._rod_mask(state)
        if is_rod is None:
            # Single-species state (the verifier's synthetic membrane patches,
            # or a frame that arrived without types): no rod, no adhesion.
            terms[ADHESION] = np.zeros(len(pairs))
            return terms

        rod_a, rod_b = is_rod[pairs.a], is_rod[pairs.b]
        membrane_pair = ~rod_a & ~rod_b
        for label in MesoMem.energy_terms_labels:
            if label in terms:
                terms[label] = np.where(membrane_pair, terms[label], 0.0)
        terms[ADHESION] = self._adhesion(state, pairs, params, rod_a, rod_b)
        return terms

    @staticmethod
    def _rod_mask(state):
        types = state.types
        if types is None or len(types) == 0:
            return None
        mask = np.asarray(types) == 2
        return mask if mask.any() else None

    def _adhesion(self, state, pairs, params, rod_a, rod_b):
        """Per-pair segment-to-point 12-6 LJ, vectorized.

        The same closest-point-on-segment arithmetic as the pair style's
        compute(): with dv the bead's position relative to the rod's centre and n
        the rod's axis, the closest point on the axis is at s = clamp(dv.n,
        +-L/2), so the contact separation is |dv - s n|. Beyond that it is a plain
        12-6 LJ at sigma_eff = r_mem + r_rod, cut (like the style) on the
        SEGMENT separation rather than the centre-to-centre one -- which is why
        the pair list has to be built wider than this cutoff.
        """
        out = np.zeros(len(pairs))
        one_rod = rod_a ^ rod_b
        if not one_rod.any() or state.directors is None:
            return out
        idx = np.nonzero(one_rod)[0]
        # d is r_a - r_b, minimum-imaged. Orient it bead-relative-to-rod.
        rod_first = rod_a[idx]
        rod_index = np.where(rod_first, pairs.a[idx], pairs.b[idx])
        dv = np.where(rod_first[:, None], -pairs.d[idx], pairs.d[idx])

        r = segment_distance(dv, state.directors[rod_index],
                             float(params["rod_length"]))

        sigma_eff = R_MEM + float(params["rod_radius"])
        eps = float(params["eps_rod"])
        inside = (r < self.rod_cutoff(params)) & (r > 1e-9)
        sr6 = (sigma_eff / r[inside]) ** 6
        out[idx[inside]] = 4.0 * eps * (sr6 * sr6 - sr6)
        return out

    # ---- rendering ----------------------------------------------------------

    def glyph_spheres(self, state, params):
        """The rod's body, as impostor spheres along its axis.

        LAMMPS integrates the rod as ONE particle, and drawing it as one sphere
        would misrepresent the only thing about it that matters -- that it is
        long, and that which way it points is what the user steers. So the
        renderer is handed a capsule's worth of extra spheres to draw: they carry
        no state and take part in no physics, they are the shape of the particle
        that is already there. The rod's own bead sits inside them.

        Note that the drawn body is LONGER than L by one diameter: L is the
        length of the axis the pair style measures from, and the surface it
        interacts through is a radius outside that in every direction -- so a
        capsule of end-to-end length L + 2 r_rod is the honest picture of what the
        membrane is touching.

        Returns (centers, radii, directors, owners) or None; `owners` are indices
        into `state` so the renderer can give each sphere its particle's colour.
        """
        is_rod = self._rod_mask(state)
        if is_rod is None or state.directors is None:
            return None
        L = float(params["rod_length"])
        r_rod = float(params["rod_radius"])
        # Spaced well inside a radius of each other, so the silhouette is smooth
        # rather than beaded: the impostors are opaque spheres and overlapping
        # them IS the capsule.
        n_seg = max(2, int(np.ceil(2.0 * L / r_rod)) + 1)
        s = np.linspace(-0.5 * L, 0.5 * L, n_seg)
        rods = np.flatnonzero(is_rod)
        axes = state.directors[rods]
        centers = (state.positions[rods][:, None, :]
                   + s[None, :, None] * axes[:, None, :]).reshape(-1, 3)
        # THE DRAWN DIRECTOR IS ACROSS THE BODY, NOT ALONG IT. The renderer bands
        # each sphere about its own director, so handing every sphere the rod's
        # AXIS puts a band around each one perpendicular to the body -- and the
        # capsule comes out looking like a stack of coins. Handing them an axis
        # ACROSS the body instead lines every sphere's band up with its
        # neighbours', and they merge into one stripe running the length of the
        # rod: a single object, which is what it is.
        across = np.column_stack([-axes[:, 2], np.zeros(len(axes)), axes[:, 0]])
        norm = np.linalg.norm(across, axis=1, keepdims=True)
        # Degenerate only if the axis is along y, which the control plane's
        # constraint does not allow; fall back to +z rather than dividing by zero.
        across = np.where(norm > 1e-9, across / np.maximum(norm, 1e-9),
                          np.array([0.0, 0.0, 1.0]))
        return (centers,
                np.full(len(centers), r_rod),
                np.repeat(across, n_seg, axis=0),
                np.repeat(rods, n_seg))
