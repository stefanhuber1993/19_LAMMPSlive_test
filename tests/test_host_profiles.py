"""What a HostProfile does to a force field, and what happens when a deck fails.

Both of these exist because of the third real Snellius run, where the server
reached the node, said it was listening, and then died on the FIRST command its
force field emitted:

    ERROR: Cannot set per-type atom mass for atom style dipole_sphere_angle/kk

-- taking the allocation with it, via a SIGABRT and a core dump that buried the
message under sixty lines of Kokkos backtrace. One bug in what we sent, one in how
we failed. See remote/hosts.py and playground/system.py.
"""
import pytest

from lammps_live.forcefields.mesomem import MesoMem
from lammps_live.remote import hosts

pytest.importorskip("lammps")

PARAMS = {p.name: p.default for p in MesoMem.params}


def test_a_per_atom_mass_host_gets_the_other_spelling():
    """`mass I M` and `set type I mass M` say the same thing to different builds."""
    local = hosts.get("local").adapt(MesoMem()).setup_commands(PARAMS)
    assert "mass 1 1.0" in local

    for name in ("cluster-gpu", "cluster-cpu"):
        cluster = hosts.get(name).adapt(MesoMem()).setup_commands(PARAMS)
        assert "set type 1 mass 1.0" in cluster, name
        assert not any(c.startswith("mass ") for c in cluster), name
        # Everything else in the list is untouched -- the rewrite is one command,
        # not a rebuild of the deck.
        assert "set group all diameter 1.0" in cluster, name
        assert len(cluster) == len(local), name


def test_the_mass_rewrite_composes_with_the_coeff_truncation():
    """Both wrap methods on the same instance; the second must not lose the first.

    The 8-value cluster build is the configuration that actually failed, so it is
    the one worth pinning: `with_coeff_values(8)` then `adapt` has to leave BOTH
    adaptations in place.
    """
    ff = hosts.get("cluster-gpu").with_coeff_values(8).adapt(MesoMem())
    assert "set type 1 mass 1.0" in ff.setup_commands(PARAMS)
    coeff = ff.coeff_commands(PARAMS)[0]
    assert len(coeff.split()) == 3 + 8


def test_a_deck_that_fails_does_not_leave_a_live_lammps_behind():
    """The instance is closed on the way out, so Kokkos finalises while CUDA is
    still up rather than from a destructor during interpreter shutdown."""
    from lammps_live.playground import Playground, random_fill
    from lammps_live.playground.system import PlaygroundSystem

    caught = {}

    class Boom(PlaygroundSystem):
        def _pick_controlled(self, build):        # after the deck, before the fixes
            caught["system"] = self
            raise RuntimeError("boom")

    playground = Playground(
        name="failing deck", force_field="mesomem",
        scenario=random_fill(n=60, box=6.0), mode="sim", seed=3,
    )
    with pytest.raises(RuntimeError, match="boom"):
        Boom(playground, mode_name="sim", analysis=False)
    assert caught["system"].lmp is None, "the failed build kept its LAMMPS instance"
