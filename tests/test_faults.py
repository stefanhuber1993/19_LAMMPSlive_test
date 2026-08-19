"""A simulation destroyed by its own sliders is an event, not a crash.

The failure this file exists for: `zeta` below 1 is fine to the CPU pair style and
rejected by the Kokkos one, so a remote run streamed happily -- the per-chunk
`run ... pre no` never re-validates coefficients -- until Reset rebuilt, `run 0`
validated them, and the exception took out the server AND its A100 allocation, via
the server's own `scancel`.

So: a rebuild falls back to values that are known to build, the event is reported
once with a sentence anyone can read and the raw error underneath, and the sliders
follow whatever the simulation settled on.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from lammps_live.playground.faults import Fault, first_line, summarise
from lammps_live.ui.alert import Alert, wrap_text


# ---- the summaries ---------------------------------------------------------

@pytest.mark.parametrize("error, expect", [
    ("ERROR: Lost atoms: original 900 current 41 (src/thermo.cpp:483)",
     "flew out of the box"),
    ("ERROR on proc 0: Non-numeric atom coords - simulation unstable", "blew up"),
    ("ERROR: mesomem/kk requires zeta >= 1 (src/KOKKOS/pair_mesomem_kokkos.cpp:585)",
     "will not accept that parameter value"),
    ("ERROR: Neighbor list overflow, boost neigh_modify one", "too many neighbours"),
    ("ERROR: Unknown pair style mesomem", "does not have a style"),
    ("ERROR: Incorrect args for pair coefficients", "was not valid for this build"),
])
def test_a_raw_lammps_error_gets_a_sentence(error, expect):
    assert expect in summarise(error)


def test_an_unrecognised_error_says_so_rather_than_guessing():
    assert summarise("ERROR: something nobody has met yet") == \
        "LAMMPS stopped the simulation."
    assert summarise("") == "LAMMPS stopped the simulation."


def test_the_detail_keeps_the_input_line_and_drops_the_traceback():
    raw = ("ERROR: mesomem/kk requires zeta >= 1 (src/pair.cpp:585)\n"
           "Last input line: run 0\n"
           "  File \"server.py\", line 274, in serve_client\n")
    detail = first_line(raw)
    assert "requires zeta >= 1" in detail
    assert "Last input line: run 0" in detail
    assert "serve_client" not in detail


def test_a_fault_survives_the_wire():
    fault = Fault.from_error("ERROR: Lost atoms: original 900 current 41",
                            reverted={"zeta": 5.0}, fatal=False)
    back = Fault.from_message(fault.as_message())
    assert back.summary == fault.summary
    assert back.detail == fault.detail
    assert back.reverted == {"zeta": 5.0}
    assert back.fatal is False
    assert "reverted zeta to 5" in fault.line()
    assert Fault.from_message(None) is None


# ---- the card --------------------------------------------------------------

def _fonts():
    pygame.display.init()
    pygame.font.init()

    class FakeRenderer:
        screen = pygame.Surface((1200, 800))
        header_font = pygame.font.Font(None, 26)
        small_font = pygame.font.Font(None, 14)
        font = pygame.font.Font(None, 18)
        sim_width = 900
        window_size = (1200, 800)

    return FakeRenderer()


def test_the_card_shows_for_three_seconds_then_stops():
    alert = Alert(show_seconds=0.15)
    assert not alert.visible
    alert.show("The simulation blew up.", "ERROR: Lost atoms")
    assert alert.visible
    renderer = _fonts()
    alert.draw(renderer)                      # must not raise, headless
    import time
    time.sleep(0.2)
    assert not alert.visible
    alert.draw(renderer)                      # nor when it is gone


def test_a_second_fault_replaces_the_first_rather_than_queueing():
    alert = Alert(show_seconds=5.0)
    alert.show_fault(Fault.from_error("ERROR: Lost atoms"))
    alert.show_fault(Fault.from_error("ERROR: Neighbor list overflow"))
    assert "neighbours" in alert.summary
    assert len(alert.shown) == 2               # both recorded, one on screen


def test_wrapping_never_loses_a_long_path():
    pygame.font.init()
    font = pygame.font.Font(None, 14)
    path = "/gpfs/home3/stefanh/Projects/MesoMemLive/mesomem_gpu/pair_mesomem.cpp:585"
    lines = wrap_text(f"ERROR: at {path}", font, 120)
    assert any(path in line for line in lines), lines


# ---- the rebuild ladder ----------------------------------------------------

def test_a_rebuild_falls_back_off_a_value_lammps_refuses():
    """The zeta failure, reproduced with a build-time check the local style lacks."""
    pytest.importorskip("lammps")
    from lammps_live.playground import Playground, random_fill
    from lammps_live.playground.system import PlaygroundSystem

    playground = Playground(
        name="fault ladder", force_field="mesomem",
        scenario=random_fill(n=120, box=8.0), mode="sim", seed=11,
    )
    system = PlaygroundSystem(playground, mode_name="sim", analysis=False)
    try:
        # Stand in for `mesomem/kk requires zeta >= 1`: a coefficient this build
        # accepts until a rebuild validates it.
        real = system.force_field.pair_commands
        system.force_field.pair_commands = (
            lambda params: (real(params) + ["pair_coeff 1 1 nonsense"]
                            if float(params["zeta"]) < 1.0 else real(params)))
        good = system.params["zeta"]

        system.set_extra_param("zeta", 0.4)
        system.reset()                          # must NOT raise

        fault = system.take_fault()
        assert fault is not None
        assert not fault.fatal, "there is a running simulation, so not fatal"
        assert fault.reverted == {"zeta": good}
        assert "Restarted with the values it last built with." in fault.summary
        assert "pair_coeff 1 1 nonsense" in fault.detail
        # And it is really running, with the value that works.
        assert system.lmp is not None
        assert system.params["zeta"] == good
        assert system.live_param_values()["zeta"] == good
        system.step(5)
        assert system.unstable is None
        # Popped, not latched: the card is shown once.
        assert system.take_fault() is None
    finally:
        system.close()
