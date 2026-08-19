"""The red card that says the simulation just died and what is being done about it.

WHY A POPUP AND NOT A HUD LINE. A destroyed simulation is an EVENT: it happens at
one moment, because of the slider that was moved a second earlier, and then it is
over -- the app has already rebuilt and is running again. A HUD line describes a
state, so it either lingers after the thing it describes is fixed or it is missed
entirely. This appears, is impossible to miss for three seconds, and leaves.

TWO SIZES, ON PURPOSE. The big line is for whoever is standing in front of the
screen: what happened, in a sentence, and that the run restarted. The small line is
the LAMMPS error verbatim -- the only text that can actually be acted on, and the
thing to copy into a bug report. Neither is a substitute for the other, so both are
here rather than one being chosen for the reader (see playground/faults.py).

It draws through the renderer's `overlay` hook, like the connect panel, because in
GL mode every 2D surface has to be on `self.screen` before the compositor runs.
"""
import time

import pygame

from .scale import UI

# A warning, so: red. The fill is dark enough for white text and the border is the
# part that catches the eye from across a room.
ALERT_BG = (46, 16, 16)
ALERT_BORDER = (255, 96, 80)
ALERT_TEXT = (255, 235, 230)
ALERT_ACCENT = (255, 140, 120)

SHOW_SECONDS = 3.0
FADE_SECONDS = 0.5          # of those three, the last half second fades out


class Alert:
    """One transient warning card. Showing a second one replaces the first.

    Replaces rather than queues: these arrive in bursts (one blow-up can fault on
    the step, the rebuild and the step after it), and a queue would make the user
    read the same thing three times while the app has long since recovered.
    """

    def __init__(self, show_seconds=SHOW_SECONDS):
        self.show_seconds = show_seconds
        self.summary = ""
        self.detail = ""
        self._until = 0.0
        # Kept for the tests and for --debug: what was shown, most recent last.
        self.shown = []

    def show(self, summary, detail=""):
        self.summary = str(summary or "")
        self.detail = str(detail or "")
        self._until = time.monotonic() + self.show_seconds
        self.shown.append((self.summary, self.detail))

    def show_fault(self, fault):
        """Show a `playground.faults.Fault`."""
        self.show(fault.summary, fault.detail)

    def dismiss(self):
        self._until = 0.0

    @property
    def visible(self):
        return time.monotonic() < self._until

    def _alpha(self):
        left = self._until - time.monotonic()
        if left <= 0:
            return 0
        return int(255 * min(1.0, left / FADE_SECONDS))

    def draw(self, renderer):
        """Draw over the sim view. Safe to call every frame."""
        if not self.visible:
            return
        screen = renderer.screen
        big, small = renderer.header_font, renderer.small_font
        width = min(UI(680), max(UI(320), renderer.sim_width - UI(40)))
        pad = UI(16)

        summary = wrap_text(self.summary, big, width - 2 * pad)
        detail = wrap_text(self.detail, small, width - 2 * pad, limit=3)
        line_h = big.get_height() + UI(2)
        small_h = small.get_height() + UI(2)
        height = (pad + line_h * len(summary)
                  + (UI(8) + small_h * len(detail) if detail else 0) + pad)

        # Top of the sim view, not the middle: the middle is where the simulation
        # the user just broke is, and covering it hides the thing they need to see
        # come back.
        rect = pygame.Rect(0, 0, width, height)
        rect.centerx = renderer.sim_width // 2
        rect.y = UI(24)

        card = pygame.Surface(rect.size, pygame.SRCALPHA)
        card.fill((*ALERT_BG, 235))
        pygame.draw.rect(card, ALERT_BORDER, card.get_rect(), width=UI.w(2),
                         border_radius=UI(8))
        y = pad
        for line in summary:
            card.blit(big.render(line, True, ALERT_TEXT), (pad, y))
            y += line_h
        if detail:
            y += UI(8)
            for line in detail:
                card.blit(small.render(line, True, ALERT_ACCENT), (pad, y))
                y += small_h
        card.set_alpha(self._alpha())
        screen.blit(card, rect.topleft)


def wrap_text(text, font, width, limit=None):
    """Greedy word wrap to a pixel width, at most `limit` lines.

    A long word (a file path in a LAMMPS error, which is most of them) is not
    broken: it overhangs rather than being cut in a way that makes it unsearchable.
    """
    words = str(text or "").split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:limit] if limit else lines
