"""The connect panel: get a GPU on the cluster, from inside the running app.

A card over the sim view, shown whenever the remote playground is selected and not
streaming. It runs `remote.session.RemoteSession` -- allocate, deploy, launch,
tunnel -- and shows what it is doing, one line per step, with the far side's own
output underneath. The only thing it asks for is the answer to whatever the login
prompts for, because that is the one thing no amount of automation can supply.

WHY IT IS IN THE APP AND NOT A SCRIPT. The demo is a thing you stand in front of.
Allocating the GPU from a terminal first, then starting the app, then reconnecting
by hand when the tunnel drops, is three ways for a demo to fail in public. And the
teardown matters more than the setup: an A100 held by a forgotten allocation is
expensive, so the thing that ends the session has to be the thing you close --
which means the app has to own it.

NOTHING HERE BLOCKS. Every step runs on the session's worker thread; this reads its
state once per frame. The app keeps drawing at 60 fps through an SSH login, a queue
wait and a LAMMPS build on the far side.
"""
import time

import pygame

from . import clipboard
from .scale import UI
from .theme import (
    BUTTON_BORDER, DIM_TEXT_COLOR, HEADER_TEXT_COLOR, PANEL_BG, PANEL_DIVIDER,
    SLIDER_HANDLE_ACTIVE, TEXT_COLOR,
)
from .widgets import Button, TextField

# The palette this panel adds to the theme: one colour per resting state, so the
# card reads at a glance from across a room.
OK_COLOR = (120, 230, 170)
BUSY_COLOR = (255, 210, 90)
FAIL_COLOR = (255, 110, 90)

CARD_WIDTH = 620
LOG_LINES = 14


class RemotePanel:
    """Owns the session, the buttons and the prompt field for one remote system."""

    def __init__(self):
        self.visible = False
        self.system = None
        self.session = None
        # Which playground the session belongs to, so coming back to that one
        # resumes it and coming back to a different one does not (see _resume).
        self.playground_key = None
        self.field = TextField(masked=True, placeholder="type the answer, then Enter")
        self.buttons = {
            "connect": Button("connect", "Connect"),
            "cancel": Button("cancel", "Cancel"),
            "disconnect": Button("disconnect", "Disconnect"),
            "copy": Button("copy", "Copy report (C)"),
            "close": Button("close", "Close (N)"),
        }
        self._shown = ()               # which buttons are on screen right now
        self._last_state = None
        # What the copy button just did, and until when to say so. A copy that
        # produces no visible change is a copy the user does again, and again.
        self._notice = None
        self._notice_until = 0.0

    # ---- lifecycle ----------------------------------------------------------

    def attach_system(self, system, playground_key):
        """Point the panel at a newly built RemoteSystem.

        A session that is still up is RESUMED rather than replaced: switching to
        another playground and back must not cost another allocation and another
        queue wait. The job, the tunnel and the server all stay as they were, the
        server keeps the simulation it was holding, and coming back is one fresh
        socket through the same tunnel. Only when there is nothing live to go back
        to does this open a session and put the card up.
        """
        if self._resume(system, playground_key):
            return
        self.release()
        self.system = system
        self.playground_key = playground_key
        from ..remote.session import RemoteSession
        self.session = RemoteSession(system.target, playground_ref=playground_key)
        self.visible = True            # nothing is connected yet, so say so
        self.field.clear()

    def _resume(self, system, playground_key):
        """Adopt an existing session for a freshly rebuilt system. True if it took.

        Three cases, and only the first two are worth keeping:
          * READY -- reconnect (or adopt a link that landed while we were away) and
            carry straight on from the state the server is holding;
          * still working -- leave it working, and `update` hands the link over when
            it lands. Switching playground mid-connect should not cancel a queue
            wait that is nearly done;
          * anything else (never started, link lost, failed) -- there is nothing to
            come back to, so the caller starts a new session and shows the card.
        """
        session = self.session
        if session is None or playground_key != self.playground_key:
            return False
        from ..remote.session import READY
        if session.state == READY:
            link = session.link
            # A link the app closed on the way out (system.close -> detach) has to
            # be reopened; one that arrived while this playground was not on screen
            # is still good, and MUST be reused rather than reconnected -- the
            # server serves one client at a time, so a second socket would sit in
            # the backlog behind our own.
            if link is None or link.closed.is_set():
                link = session.reopen_link()
            if link is None:
                return False
            self.system = system
            system.attach(link)
            self.visible = False
            return True
        if session.busy:
            self.system = system
            self.visible = True
            return True
        return False

    def detach_system(self):
        """Switch away from the remote playground WITHOUT giving the GPU back.

        The allocation, the tunnel and the server survive, holding the simulation
        where it was, so the demo can go and show something else and come back to
        the same coarsened box (see attach_system). What ends the session is closing
        the window, pressing Disconnect, or the server's own idle timeout -- so a
        forgotten allocation still gives itself back, it just is not this method's
        job.
        """
        self.system = None
        self.visible = False
        self.field.clear()

    def release(self):
        """Drop everything: cancels the job, closes the tunnel and the login."""
        if self.session is not None:
            self.session.shutdown()
            self.session = None
        self.system = None
        self.visible = False
        self.field.clear()

    @property
    def active(self):
        """Is there a session attached to the playground currently on screen?

        Both halves matter: N must not raise this card over an unrelated
        playground just because a session is still held in the background.
        """
        return self.session is not None and self.system is not None

    def standby_note(self):
        """One line for the panel when the GPU is held but its playground is not on
        screen -- an allocation nobody can see is the one thing about this that
        would be expensive to forget. None when there is nothing being held."""
        session = self.session
        if session is None or self.system is not None:
            return None
        from ..remote.session import READY
        if session.state == READY:
            return (f"remote GPU still held: job {session.job_id or '-'} on "
                    f"{session.node or session.target.label} -- switch back to "
                    f"{self.playground_key} to use it")
        if session.busy:
            return (f"remote session still connecting ({session.detail}) -- switch "
                    f"back to {self.playground_key} to watch it")
        return None

    def toggle(self):
        if self.active:
            self.visible = not self.visible

    # ---- per frame ----------------------------------------------------------

    def update(self):
        """Called once per frame. Hands a finished session's link to the system."""
        if self.session is None:
            return
        from ..remote.session import FAILED, READY
        session = self.session
        if session.state != self._last_state:
            self._last_state = session.state
            if self.system is None:
                # Another playground is on screen: there is nothing to hand a link
                # to and nothing to show a failure over. The state is remembered
                # here, and `_resume` acts on it when this playground comes back.
                pass
            elif session.state == READY and session.link is not None:
                self.system.attach(session.link)
                # Out of the way: the scene is what matters now, and the link's own
                # readouts are on the HUD. N brings it back.
                self.visible = False
            elif session.state == FAILED:
                self.visible = True
        # A link that dies on its own (the job hit its time limit, the network
        # dropped, the node failed) reopens the panel -- with the reason in the log
        # rather than a frozen picture and no explanation.
        if (self.system is not None and not self.system.connected
                and session.state == READY):
            session.note_link_lost(self.system.link_error())
            self.visible = True

    # ---- input --------------------------------------------------------------

    def handle_event(self, event):
        """True if the panel consumed the event.

        While a prompt is pending the panel takes every keystroke: the answer may
        be all digits, and those are the app's system-switch shortcuts.
        """
        if not self.visible or self.session is None:
            return False
        # C copies whatever the card is showing, in every state -- including while a
        # step is still running, which is when a session that is going to fail is
        # most worth capturing. Not while a prompt is pending: those keystrokes
        # belong to the answer, and a one-time code can contain a 'c'.
        if (event.type == pygame.KEYDOWN and not self.session.prompt
                and event.key == pygame.K_c):
            self._copy_report()
            return True
        if event.type == pygame.KEYDOWN and self.session.prompt:
            action = self.field.handle_event(event)
            if action == "submit":
                self.session.answer(self.field.value)
                self.field.clear()
            return action is not None or event.key not in (pygame.K_ESCAPE,)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name in self._shown:
                if self.buttons[name].hit(event.pos):
                    self._act(name)
                    return True
        return False

    def _act(self, name):
        session = self.session
        if name == "connect":
            session.start()
        elif name == "cancel":
            session.cancel()
        elif name == "disconnect":
            if self.system is not None:
                self.system.detach()
            session.shutdown()
        elif name == "copy":
            self._copy_report()
        elif name == "close":
            self.visible = False

    def _copy_report(self):
        """Put the whole session report on the clipboard, and on disk as well.

        Both, always: the clipboard is where it is wanted (a chat window, a ticket)
        and the file is what survives the next copy -- and on a machine with no
        clipboard command at all, the file is the only way the report gets out.
        """
        report = self.session.diagnostics()
        copied = clipboard.copy(report)
        path = self.session.save_report()
        lines = len(report.splitlines())
        if copied:
            note = f"copied {lines} lines to the clipboard"
        else:
            note = "could not reach the clipboard"
        if path:
            note += f"  --  saved to {path}"
        self._notice = note
        self._notice_until = time.monotonic() + 8.0

    # ---- drawing ------------------------------------------------------------

    def draw(self, renderer):
        """Draw the card over the sim view. Called by the renderer, before it
        composites and flips, so it lands on top of the 3D scene."""
        if not self.visible or self.session is None:
            return
        session = self.session
        screen = renderer.screen
        font, small = renderer.font, renderer.small_font
        header = renderer.header_font

        width = min(UI(CARD_WIDTH), max(UI(320), renderer.sim_width - UI(40)))
        # Height from the content, not a constant: the card carries a prompt block
        # only while something is being asked, and the log grows into it as the
        # session proceeds. Fixed, it is a third empty at the start and clips the
        # log at the end.
        lines = list(session.log)[-LOG_LINES:]
        error_lines = ([] if session.prompt or not session.error
                       else _wrap(session.error, small, width - UI(36)))
        height = (UI(58)                               # title and subtitle
                  + UI(26) + UI(16)                    # state line, progress bar
                  + (UI(64) if session.prompt else 0)
                  + UI(15) * len(error_lines) + (UI(6) if error_lines else 0)
                  + UI(14) * max(len(lines), 3)
                  + UI(78))                            # hint line + button row
        height = min(height, renderer.window_size[1] - UI(40))
        rect = pygame.Rect(0, 0, width, height)
        rect.center = (renderer.sim_width // 2, renderer.window_size[1] // 2)

        # A dim wash over the scene behind it: this is modal, and it should look it.
        wash = pygame.Surface((renderer.sim_width, renderer.window_size[1]),
                              pygame.SRCALPHA)
        wash.fill((0, 0, 0, 150))
        screen.blit(wash, (0, 0))
        pygame.draw.rect(screen, PANEL_BG, rect, border_radius=UI(10))
        pygame.draw.rect(screen, PANEL_DIVIDER, rect, width=UI.w(1),
                         border_radius=UI(10))

        x = rect.x + UI(18)
        w = rect.width - UI(36)
        y = rect.y + UI(14)

        target = session.target
        title = header.render(f"Remote simulation -- {target.label}", True,
                              HEADER_TEXT_COLOR)
        screen.blit(title, (x, y))
        y += title.get_height() + UI(2)
        sub = (f"{target.destination}  |  {target.partition}, {target.gpus} GPU, "
               f"{target.cpus_per_task} cores, {target.time}")
        screen.blit(small.render(sub, True, DIM_TEXT_COLOR), (x, y))
        y += UI(20)

        # State line: the step it is on, out of how many, and what it is doing.
        step, total = session.progress()
        from ..remote.session import DOWN, FAILED, READY
        color = {READY: OK_COLOR, FAILED: FAIL_COLOR,
                 DOWN: DIM_TEXT_COLOR}.get(session.state, BUSY_COLOR)
        label = session.state.upper()
        if session.busy:
            label = f"{label} ({step}/{total})"
        screen.blit(font.render(label, True, color), (x, y))
        detail = font.render(session.detail, True, TEXT_COLOR)
        screen.blit(detail, (x + UI(130), y))
        y += detail.get_height() + UI(6)

        # A progress bar, because the two slow steps (the queue, and LAMMPS
        # starting) can each take minutes and a static line looks like a hang.
        bar = pygame.Rect(x, y, w, UI(4))
        pygame.draw.rect(screen, PANEL_DIVIDER, bar, border_radius=UI(2))
        if step:
            filled = pygame.Rect(x, y, int(w * step / total), UI(4))
            pygame.draw.rect(screen, color, filled, border_radius=UI(2))
        y += UI(16)

        if session.prompt:
            prompt_box = pygame.Rect(x - UI(6), y - UI(4), w + UI(12), UI(60))
            pygame.draw.rect(screen, (32, 30, 22), prompt_box,
                             border_radius=UI(6))
            pygame.draw.rect(screen, SLIDER_HANDLE_ACTIVE, prompt_box,
                             width=UI.w(1), border_radius=UI(6))
            # The prompt verbatim: it may be a password, a one-time code, or a host
            # key to confirm, and only the thing that asked knows which.
            screen.blit(font.render(session.prompt, True, SLIDER_HANDLE_ACTIVE), (x, y))
            y += UI(22)
            self.field.rect = pygame.Rect(x, y, w, UI(26))
            self.field.draw(screen, font, blink=(time.monotonic() % 1.0) < 0.6)
            y += UI(40)
        elif error_lines:
            for line in error_lines:
                screen.blit(small.render(line, True, FAIL_COLOR), (x, y))
                y += UI(15)
            y += UI(6)

        # The log: this end's steps and the far side's own output, interleaved, which
        # is what makes a failure diagnosable without going to look for a log file.
        for line in lines:
            screen.blit(small.render(line[:96], True, DIM_TEXT_COLOR), (x, y))
            y += UI(14)

        # Buttons, contextual: only the ones that mean something in this state.
        if session.busy:
            names = ("copy", "cancel")
        elif session.state == READY:
            names = ("copy", "disconnect", "close")
        else:
            names = ("copy", "connect", "close")
        self._shown = names
        bw, bh, gap = UI(130), UI(30), UI(10)
        bx = rect.right - UI(18) - (len(names) * bw + (len(names) - 1) * gap)
        by = rect.bottom - UI(18) - bh
        for i, name in enumerate(names):
            button = self.buttons[name]
            button.rect = pygame.Rect(bx + i * (bw + gap), by, bw, bh)
            button.draw(screen, font, active=(name == "connect"))

        # On its own line above the buttons, not beside them: with three buttons
        # there is no room left on that row, and a hint that runs under a button is
        # worse than no hint.
        if self._notice and time.monotonic() < self._notice_until:
            hint, tint = self._notice, OK_COLOR
        elif session.prompt:
            hint, tint = "Enter sends the answer. C copies the report.", BUTTON_BORDER
        else:
            hint = ("N hides this panel, C copies the whole report. Closing the "
                    "window cancels the job.")
            tint = BUTTON_BORDER
        screen.blit(small.render(hint, True, tint), (x, by - UI(18)))


def _wrap(text, font, width):
    """Greedy word wrap to a pixel width."""
    words, lines, current = str(text).split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]
