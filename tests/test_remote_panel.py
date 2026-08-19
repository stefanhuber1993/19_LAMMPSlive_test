"""The connect panel's wiring, with no SSH and no cluster.

Small, but it covers the one bug class that would look like a working demo showing
nothing: a session that reaches READY whose link is never handed to the system. And
the modal-keystroke rule, which if it broke would mean a one-time code starting with
a digit silently switching playgrounds mid-login.
"""
import os
import threading

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from lammps_live.playground import registry
from lammps_live.remote import session as session_mod
from lammps_live.ui import clipboard
from lammps_live.ui.remote_panel import RemotePanel

PLAYGROUND_SOURCE = '''
from lammps_live.playground import Playground, random_fill
from lammps_live.remote import RemoteTarget

PLAYGROUND = Playground(
    name="panel test", force_field="mesomem",
    scenario=random_fill(n=120, box=8.0), mode="sim", seed=7,
    remote=RemoteTarget(host="nowhere", profile="local"),
)
'''


class StubLink:
    """Enough of a FrameLink for RemoteSystem.attach to adopt it."""

    def __init__(self, natoms=120):
        self.welcome = {"natoms": natoms, "host": "stub", "profile": "local",
                        "slurm_job": "1", "sim_time": 0.0,
                        "box": {"lo": [-4, -4, -4], "hi": [4, 4, 4],
                                "periodic": [True, True, True]}}
        self.closed = threading.Event()
        self.sent = []
        self.error = None
        self.rtt_ms = None
        self.dropped = 0

    def send(self, message):
        self.sent.append(message)
        return True

    def rates(self):
        return 0.0, 0.0

    def take_frame(self, timeout=0.0):
        return None

    def close(self, say_goodbye=True):
        self.closed.set()


def _renderer():
    """The handful of attributes RemotePanel.draw actually reaches for."""
    pygame.display.init()
    pygame.font.init()
    surface = pygame.Surface((1200, 800))

    class FakeRenderer:
        screen = surface
        font = pygame.font.Font(None, 18)
        small_font = pygame.font.Font(None, 14)
        header_font = pygame.font.Font(None, 22)
        sim_width = 900
        window_size = (1200, 800)

    return FakeRenderer()


class StubSession:
    """A RemoteSession that does nothing, driven by the test instead of by ssh."""

    def __init__(self, target, playground_ref="", on_log=None, log_lines=None):
        self.target = target.resolved()
        self.playground_ref = playground_ref
        self.state = session_mod.DOWN
        self.detail = "not connected"
        self.error = None
        self.prompt = None
        self.link = None
        self.log = []
        self.job_id = "4242"
        self.node = "gcn12"
        self.started = 0
        self.shutdowns = 0
        self.cancels = 0
        self.lost = []

    @property
    def busy(self):
        return self.state not in (session_mod.DOWN, session_mod.READY,
                                  session_mod.FAILED)

    def progress(self):
        return (0, 7)

    def start(self):
        self.started += 1
        self.state = session_mod.LOGIN

    def cancel(self):
        self.cancels += 1

    def shutdown(self):
        self.shutdowns += 1
        self.state = session_mod.DOWN

    def answer(self, text):
        self.answered = text
        self.prompt = None

    def diagnostics(self):
        return f"report for {self.playground_ref}\nstate {self.state}"

    def save_report(self, path=None):
        return "/tmp/report.txt"

    def note_link_lost(self, reason):
        self.lost.append(reason)
        self.state = session_mod.DOWN

    # What the real session does over a tunnel that is still open. `alive` is the
    # test's stand-in for "the job is still there" -- clear it and the reconnect
    # fails the way an expired allocation does.
    alive = True
    reopened = 0

    def reopen_link(self):
        if self.state != session_mod.READY:
            return None
        if not self.alive:
            self.note_link_lost("the job ended")
            return None
        self.reopened += 1
        self.link = StubLink()
        return self.link


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "RemoteSession", StubSession)
    path = tmp_path / "panel_playground.py"
    path.write_text(PLAYGROUND_SOURCE)
    system = registry.build(str(path))
    p = RemotePanel()
    p.attach_system(system, str(path))
    p.playground_path = str(path)       # for the tests that rebuild it
    yield p, system, p.session
    p.release()


def _connect(panel):
    """Bring a fixture's session up to a streaming link."""
    p, _system, session = panel
    session.state = session_mod.READY
    session.link = StubLink()
    p.update()
    return session


def test_it_comes_up_visible_and_disconnected(panel):
    p, system, session = panel
    assert p.visible and p.active
    assert not system.connected
    assert session.playground_ref.endswith("panel_playground.py")


def test_a_ready_session_hands_its_link_over_and_gets_out_of_the_way(panel):
    p, system, session = panel
    link = StubLink()
    session.state = session_mod.READY
    session.link = link
    p.update()
    assert system.connected
    assert not p.visible
    # Everything the panel is showing was pushed to the far side on attach: the
    # frame config, the temperature, every slider, and the play state.
    kinds = [m["t"] for m in link.sent]
    assert kinds[0] == "config" and "temp" in kinds and "pause" in kinds
    assert {m["key"] for m in link.sent if m["t"] == "set"} >= {"k_tilt", "k_splay"}


def test_a_link_that_dies_reopens_the_panel_with_the_reason(panel):
    p, system, session = panel
    session.state = session_mod.READY
    session.link = StubLink()
    p.update()
    assert not p.visible

    system.link.error = "the job hit its time limit"
    system.link.closed.set()
    p.update()
    assert p.visible
    assert session.lost == ["the job hit its time limit"]


def test_a_failed_session_shows_itself(panel):
    p, session_ = panel[0], panel[2]
    p.visible = False
    session_.state = session_mod.FAILED
    session_.error = "no usable lammps Python module"
    p.update()
    assert p.visible


def test_a_pending_prompt_swallows_digits(panel):
    """A one-time code is digits, and digits are the playground shortcuts."""
    p, _system, session = panel
    session.prompt = "Verification code:"
    consumed = p.handle_event(pygame.event.Event(
        pygame.KEYDOWN, {"key": pygame.K_4, "unicode": "4", "mod": 0}))
    assert consumed
    assert p.field.value == "4"

    p.handle_event(pygame.event.Event(
        pygame.KEYDOWN, {"key": pygame.K_2, "unicode": "2", "mod": 0}))
    p.handle_event(pygame.event.Event(
        pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": "\\r", "mod": 0}))
    assert session.answered == "42"
    assert p.field.value == ""          # not left on screen after sending


def test_no_prompt_means_keys_fall_through_to_the_app(panel):
    p, _system, session = panel
    assert session.prompt is None
    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_4, "unicode": "4",
                                                "mod": 0})
    assert not p.handle_event(event)


def test_c_copies_the_report_without_swallowing_other_keys(panel, monkeypatch):
    """The failure this exists for is read somewhere else than on the card."""
    p, _system, session = panel
    session.state = session_mod.FAILED
    session.error = "the tunnel is open but the server did not answer"
    taken = []
    monkeypatch.setattr(clipboard, "copy", lambda text: taken.append(text) or True)

    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c, "unicode": "c",
                                                "mod": 0})
    assert p.handle_event(event)
    assert taken and "state failed" in taken[0]
    assert "/tmp/report.txt" in p._notice

    # Still only 'c': the digits stay the app's playground shortcuts.
    assert not p.handle_event(pygame.event.Event(
        pygame.KEYDOWN, {"key": pygame.K_4, "unicode": "4", "mod": 0}))


def test_a_prompt_keeps_its_c(panel, monkeypatch):
    """A one-time code can contain a 'c', and it must reach the field."""
    p, _system, session = panel
    session.prompt = "Password:"
    monkeypatch.setattr(clipboard, "copy", lambda text: pytest.fail("copied"))
    assert p.handle_event(pygame.event.Event(
        pygame.KEYDOWN, {"key": pygame.K_c, "unicode": "c", "mod": 0}))
    assert p.field.value == "c"


def test_the_copy_button_is_offered_in_every_state(panel):
    p, _system, session = panel
    renderer = _renderer()
    for state in (session_mod.DOWN, session_mod.LOGIN, session_mod.READY,
                  session_mod.FAILED):
        session.state = state
        session.link = StubLink() if state == session_mod.READY else None
        p.visible = True
        p.draw(renderer)
        assert "copy" in p._shown, state


def test_releasing_shuts_the_session_down(panel):
    p, system, session = panel
    session.state = session_mod.READY
    session.link = StubLink()
    p.update()
    p.release()
    assert session.shutdowns >= 1
    assert not p.active and not p.visible


def test_toggle_only_works_when_there_is_a_session():
    bare = RemotePanel()
    bare.toggle()
    assert not bare.visible          # nothing to show, and nothing raised


# ---- switching playground with a GPU allocated -----------------------------
# The demo goes off to show something else and comes back. The allocation has to
# survive that: another queue wait mid-talk is the failure this whole panel exists
# to avoid, and the server holds its simulation between clients on purpose.

def test_switching_away_keeps_the_session_and_hides_the_card(panel):
    p, system, session = panel
    _connect(panel)

    p.detach_system()

    assert session.shutdowns == 0, "the job is still ours"
    assert p.session is session
    assert not p.visible
    assert not p.active, "N must not raise the card over another playground"
    assert "still held" in p.standby_note()


def test_coming_back_reconnects_instead_of_reallocating(panel):
    p, system, session = panel
    _connect(panel)
    p.detach_system()
    # What the app does on the way out and back: the old system closes its link,
    # and a fresh RemoteSystem is built for the same playground.
    system.close()
    rebuilt = registry.build(p.playground_path)

    p.attach_system(rebuilt, p.playground_path)

    assert p.session is session, "same session -- no new allocation"
    assert session.started == 0 and session.shutdowns == 0
    assert session.reopened == 1, "one fresh socket through the same tunnel"
    assert rebuilt.connected
    assert not p.visible, "straight back to the scene, no card"
    assert p.standby_note() is None
    rebuilt.close()


def test_a_link_that_landed_while_away_is_adopted_not_reconnected(panel):
    """The server serves one client at a time, so a second socket would queue
    behind the link this session already holds."""
    p, system, session = panel
    p.detach_system()
    session.state = session_mod.READY
    session.link = StubLink()
    p.update()                            # nothing to hand it to, and it must not crash
    rebuilt = registry.build(p.playground_path)

    p.attach_system(rebuilt, p.playground_path)

    assert session.reopened == 0
    assert rebuilt.link is session.link
    rebuilt.close()


def test_an_allocation_that_is_gone_asks_to_connect_again(panel):
    p, system, session = panel
    _connect(panel)
    p.detach_system()
    system.close()
    session.alive = False                 # the job ended while we were away
    rebuilt = registry.build(p.playground_path)

    p.attach_system(rebuilt, p.playground_path)

    assert p.session is not session, "a dead session is replaced, not resumed"
    assert session.shutdowns == 1
    assert p.visible and p.active, "the card is back, asking to connect"
    assert not rebuilt.connected
    rebuilt.close()


def test_switching_away_mid_connect_does_not_cancel_the_queue_wait(panel):
    p, _system, session = panel
    session.state = session_mod.ALLOCATE
    p.detach_system()
    assert session.cancels == 0
    assert "still connecting" in p.standby_note()

    rebuilt = registry.build(p.playground_path)
    p.attach_system(rebuilt, p.playground_path)
    assert p.session is session and session.started == 0
    # ... and the link still lands on the system that is now on screen.
    session.state = session_mod.READY
    session.link = StubLink()
    p.update()
    assert rebuilt.connected
    rebuilt.close()


def test_a_different_playground_does_not_inherit_the_session(panel, tmp_path):
    """Resuming is per playground: both ends have to agree on which one, or they
    would disagree about how many beads there are."""
    p, _system, session = panel
    _connect(panel)
    p.detach_system()
    other = tmp_path / "other_playground.py"
    other.write_text(PLAYGROUND_SOURCE)
    system2 = registry.build(str(other))

    p.attach_system(system2, str(other))

    assert p.session is not session
    assert session.shutdowns == 1
    assert p.visible
    system2.close()
