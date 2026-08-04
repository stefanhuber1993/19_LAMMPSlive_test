"""The 2D deposition scenario: a cold crystal slab with one free atom above it.

The classic teaching setup -- pull an atom down onto a crystal and feel it stick.
Structurally quite unlike the membrane scenarios, which is why porting it was the
real test of this layer: it is 2D, in physical units, and it builds its lattice
with LAMMPS' own `lattice`/`create_atoms region` rather than a numpy array,
because the whole point is a perfect crystallographic slab.

It also needs three things the membrane scenarios never did, all now pluggable:
a CSVR thermostat (see thermostat.py), a free displacement-limited puller instead
of a leashed one (see modes.FreeConfinement), and LAMMPS' native time-averaged
`compute rdf` instead of the Python one.
"""
import math

import numpy as np

from .params import structural
from .scenario import Scenario, ScenarioBuild
from .state import Box
from .thermostat import CSVR


class Deposition2D(Scenario):
    """A 2D hexagonal crystal slab, periodic in x, with a single puller atom
    released in the gap above it.

    The layout, bottom to top: an optionally frozen floor (rows pinned with
    setforce, so the slab cannot sink), the mobile crystal that the thermostat
    acts on, then vacuum, then the puller. Reflecting walls half a lattice
    spacing inside the y faces keep an evaporated atom in the box.
    """

    name = "deposition_2d"
    thermostat = CSVR()

    params = (
        structural("a", 3.784884, "nearest-neighbour spacing, Angstrom"),
        structural("n_cols", 16, "lattice cells across the box"),
        structural("crystal_rows", 7, "rows of crystal"),
        structural("floor_rows", 0, "bottom rows held rigid (0 = fully mobile)"),
        structural("puller_gap", 3.0, "puller's start height above the slab, in spacings"),
        structural("settle_steps", 600, "silent relaxation before control begins"),
        structural("thermostat_damp", 0.5, "ps -- relaxation time for total KE"),
        # Weak drag on bulk translation only. The thermostat is COM-blind, so
        # without this the slab can acquire a free drift; a small fraction of the
        # COM velocity removed per frame bleeds that off while leaving thermal
        # motion untouched and without pinning a rising gas.
        structural("drift_damp_per_frame", 0.03,
                   "fraction of COM velocity removed each frame"),
    )

    def __init__(self, **overrides):
        super().__init__(**overrides)

    # --- geometry ------------------------------------------------------------

    def _geometry(self, params):
        a = params["a"]
        row_h = a * math.sqrt(3.0) / 2.0
        row_eps = 0.1 * row_h
        box_size = params["n_cols"] * a
        crystal_top = params["crystal_rows"] * row_h + row_eps
        return a, row_h, row_eps, box_size, crystal_top

    def build(self, params, rng):
        """LAMMPS places the atoms (see atom_creation_commands), so this only
        establishes the cell. Periodic in x, free in y -- the slab sits on the
        bottom and the puller comes down from above."""
        a, _row_h, _eps, box_size, _top = self._geometry(params)
        box = Box((0.0, 0.0, -0.25 * a), (box_size, box_size, 0.25 * a),
                  periodic=(True, False, True))
        return ScenarioBuild(positions=np.zeros((0, 3)), directors=None, box=box)

    def atom_creation_commands(self, params, seed):
        a, _row_h, _eps, box_size, crystal_top = self._geometry(params)
        return [
            f"lattice hex {a}",
            f"region crystal block 0 {box_size} 0 {crystal_top} -0.25 0.25 units box",
            "create_atoms 1 region crystal",
            # The puller is created last, so it holds the highest atom id and the
            # control selector can find it as "last".
            f"create_atoms 1 single {box_size / 2.0} "
            f"{crystal_top + params['puller_gap'] * a} 0.0 units box",
        ]

    # --- groups, integrators, walls ------------------------------------------

    def group_commands(self, params, controlled_id):
        """The slab's group structure. `crystal_mobile` is what the thermostat
        acts on; `floor` is pinned; `mobile` is everything that integrates.

        In sim mode there is no controlled particle and therefore no `controlled`
        group, so the free atom is simply part of the crystal -- it becomes a stray
        adatom that lands under its own dynamics, which is a perfectly reasonable
        thing to watch.
        """
        _a, row_h, row_eps, _box, _top = self._geometry(params)
        floor_top = params["floor_rows"] * row_h + row_eps
        cmds = [
            f"region floor_region block INF INF 0.0 {floor_top} INF INF units box",
            "group floor region floor_region",
        ]
        if controlled_id is None:
            cmds.append("group crystal union all")
        else:
            cmds.append("group crystal subtract all controlled")
        cmds.append("group crystal_mobile subtract crystal floor")
        cmds.append("group mobile union crystal_mobile"
                    + ("" if controlled_id is None else " controlled"))
        return cmds

    def thermostat_group(self):
        return "crystal_mobile"

    def integrator_commands(self, params):
        return [
            "fix freeze floor setforce 0.0 0.0 0.0",
            "fix integ_crystal crystal_mobile nve",
        ]

    def wall_commands(self, box):
        """Reflecting walls half a spacing inside the y faces, on the mobile group
        only (the frozen floor needs none)."""
        a = self._wall_inset
        return [f"fix walls mobile wall/reflect ylo {box.lo[1] + a} "
                f"yhi {box.hi[1] - a} units box"]

    _wall_inset = 0.5 * 3.784884   # overwritten per-build from the spacing

    def post_control_settle(self, params):
        """Relax, then bring everything to rest so control starts from a still,
        cold lattice rather than whatever the relaxation left moving."""
        return [f"run {int(params['settle_steps'])}",
                "velocity mobile set 0.0 0.0 0.0"]

    # --- per-frame ------------------------------------------------------------

    def frame_commands(self, params, lmp):
        """Weak bulk-drift drag, issued per frame.

        Not expressible as a fix: it needs this frame's measured COM velocity,
        which is why it is a hook rather than part of the deck.

        Skipped if that velocity is not finite. A user is free to drag a parameter
        into a regime that destroys the crystal (epsilon to zero, or sigma far past
        the lattice spacing so every atom sits inside its neighbour's core), and
        the measured COM velocity then goes NaN. Feeding that to `velocity set`
        raises out of LAMMPS and takes the whole app down, which is a much worse
        outcome than a visibly exploded simulation the user can reset.
        """
        f = params["drift_damp_per_frame"]
        if f <= 0.0:
            return []
        vx = lmp.extract_variable("vcmx")
        vy = lmp.extract_variable("vcmy")
        if not (math.isfinite(vx) and math.isfinite(vy)):
            return []
        return [f"velocity crystal_mobile set {-f * vx} {-f * vy} 0.0 "
                f"sum yes units box"]

    def extra_setup_commands(self, params):
        """COM-velocity variables read each frame by frame_commands, plus the
        comm cutoff the native RDF's reach needs."""
        return [
            "variable vcmx equal vcm(crystal_mobile,x)",
            "variable vcmy equal vcm(crystal_mobile,y)",
            f"comm_modify cutoff {self._rdf_cutoff(params) + 2.0}",
        ]

    def _rdf_cutoff(self, params):
        return 4.0 * params["a"]

    def make_rdf(self, params, lmp, box):
        """LAMMPS' native time-averaged g(r) over the crystal group. The Python
        in-plane RDF would normalize against a bounding box that is mostly the
        vacuum above the slab, putting a spurious offset on g(r)."""
        from .rdf import NativeRDF
        return NativeRDF(lmp, group="crystal", nbins=100,
                         cutoff=self._rdf_cutoff(params))

    # --- rendering ------------------------------------------------------------

    def camera(self, box):
        return None      # 2D scenario: the renderer's top-down path is used

    def fit_points(self, params, box):
        return None


class IonicSlab2D(Deposition2D):
    """The deposition slab, on a bipartite CHECKERBOARD lattice of two species.

    Everything structural about the deposition setup carries over -- the frozen
    floor, the mobile crystal the thermostat acts on, the vacuum above it, the
    reflecting walls, the free puller. Two things do not, and both follow from
    the bonding being ionic rather than neutral:

    WHY A SQUARE LATTICE, NOT HEXAGONAL. The neutral crystals sit on a 2D
    close-packed triangular lattice because their bonding just maximises
    neighbour count. Ionic bonding does the opposite: every ion wants its
    NEAREST neighbours to be the opposite charge and its like-charge neighbours
    pushed out to the next shell, which needs a lattice you can two-colour --
    a bipartite one. The square lattice is bipartite: colour it like a
    checkerboard and every ion has 4 nearest neighbours of opposite sign
    (attraction) with the 4 like-charge ones held further out on the diagonal.
    That is the 2D analogue of rock-salt and a genuine Madelung minimum. A
    triangular lattice is NOT bipartite -- its 3-membered rings are
    geometrically frustrated, so no alternating assignment exists and any
    arrangement leaves like charges in contact. The alternation is built into
    the lattice itself (a 4-site custom cell, two sublattices per species)
    rather than painted on afterwards, so it -- and the exact charge neutrality
    the Coulomb sum needs -- holds from frame 0.

    WHY THE VACUUM GAP BELOW. A bare ionic (001) surface relaxes outward
    strongly. With the bottom row sitting exactly on the non-periodic y = 0
    boundary it relaxes straight out of the box (observed: an immediate "Lost
    atoms"). A couple of empty rows give it room to relax in place.
    """

    name = "ionic_slab_2d"

    params = Deposition2D.params + (
        structural("gap_rows", 2, "empty rows between the box floor and the crystal"),
        # Must clear the pair cutoff plus the neighbour skin at the WIDEST the
        # Coulomb cutoff slider can reach, or LAMMPS refuses the run mid-drag.
        structural("comm_cutoff", 16.0, "ghost-atom communication cutoff, Angstrom"),
    )

    def _geometry(self, params):
        """Rows are one nearest-neighbour distance apart here, not the
        sqrt(3)/2 of a close-packed row, and the crystal starts above the gap."""
        a = params["a"]
        row_h = a
        row_eps = 0.1 * row_h
        box_size = params["n_cols"] * a
        crystal_bot = params["gap_rows"] * row_h + row_eps
        crystal_top = crystal_bot + params["crystal_rows"] * row_h + row_eps
        return a, row_h, row_eps, box_size, crystal_top

    def atom_creation_commands(self, params, seed):
        a, row_h, row_eps, box_size, crystal_top = self._geometry(params)
        crystal_bot = params["gap_rows"] * row_h + row_eps
        return [
            # A 2x2-spacing cell whose two even-parity sites become type 1 and
            # whose two odd-parity sites become type 2. Coordinates are in units
            # of the nearest-neighbour distance.
            f"lattice custom {a} a1 2 0 0 a2 0 2 0 "
            f"basis 0 0 0 basis 0.5 0.5 0 basis 0.5 0 0 basis 0 0.5 0",
            f"region crystal block 0 {box_size} {crystal_bot} {crystal_top} "
            f"-0.25 0.25 units box",
            "create_atoms 1 region crystal basis 1 1 basis 2 1 basis 3 2 basis 4 2",
            # The puller last, so it holds the highest id and "last" finds it. A
            # cation, like one sublattice, so the lattice pulls it onto an anion
            # site electrostatically -- the ionic analogue of Cu-on-Cu.
            f"create_atoms 1 single {box_size / 2.0} "
            f"{crystal_top + params['puller_gap'] * a} 0.0 units box",
        ]

    def extra_setup_commands(self, params):
        """As Deposition2D, but the ghost cutoff has to clear the COULOMB reach,
        which is much longer than the RDF's and grows with the cutoff slider."""
        return [
            "variable vcmx equal vcm(crystal_mobile,x)",
            "variable vcmy equal vcm(crystal_mobile,y)",
            f"comm_modify cutoff {params['comm_cutoff']}",
        ]


def deposition_2d(at=None, **overrides):
    scenario = Deposition2D(**overrides)
    # The wall inset tracks the actual spacing rather than the class default.
    params = scenario.new_params()
    scenario._wall_inset = 0.5 * params["a"]
    return (scenario, at) if at is not None else scenario


def ionic_slab_2d(at=None, **overrides):
    scenario = IonicSlab2D(**overrides)
    params = scenario.new_params()
    scenario._wall_inset = 0.5 * params["a"]
    return (scenario, at) if at is not None else scenario
