"""Small reusable pygame UI widgets."""
import math

import pygame

from .theme import (
    MELT_MARK_COLOR, OPTIMUM_MARK_COLOR, SLIDER_HANDLE, SLIDER_HANDLE_ACTIVE,
    SLIDER_TRACK, TEXT_COLOR,
)


class Slider:
    """Horizontal, mouse-draggable value slider. Geometry-only + drag state
    -- callers own the semantics of what the value drives."""

    def __init__(self, rect, vmin, vmax, value, label, fmt="{:.3f}", unit="",
                 optimum=None):
        self.rect = pygame.Rect(rect)
        self.vmin, self.vmax = vmin, vmax
        self.value = max(vmin, min(vmax, value))
        self.label = label
        self.fmt = fmt
        self.unit = unit
        self.optimum = optimum
        self.dragging = False

    @classmethod
    def from_spec(cls, rect, spec):
        """Build from a systems.base.SliderSpec."""
        return cls(rect, spec.vmin, spec.vmax, spec.default, spec.label,
                   fmt=spec.fmt, unit=spec.unit, optimum=spec.optimum)

    def reset(self, spec):
        """Re-apply a (possibly different, on system switch) SliderSpec's
        range/default/label in place, so callers don't have to juggle new
        Slider instances or re-wire event handling."""
        self.vmin, self.vmax = spec.vmin, spec.vmax
        self.value = spec.default
        self.label = spec.label
        self.fmt = spec.fmt
        self.unit = spec.unit
        self.optimum = spec.optimum
        self.dragging = False

    def _frac(self):
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def _handle_pos(self):
        return (self.rect.x + self._frac() * self.rect.width, self.rect.centery)

    def _set_from_x(self, x):
        frac = max(0.0, min(1.0, (x - self.rect.x) / self.rect.width))
        self.value = self.vmin + frac * (self.vmax - self.vmin)

    def handle_event(self, event):
        """Returns True if this slider consumed the event."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hx, hy = self._handle_pos()
            near_handle = math.hypot(event.pos[0] - hx, event.pos[1] - hy) < 12
            on_track = self.rect.collidepoint(event.pos)
            if near_handle or on_track:
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False

    def nudge(self, delta):
        self.value = max(self.vmin, min(self.vmax, self.value + delta))

    def _mark_x(self, value):
        return self.rect.x + (value - self.vmin) / (self.vmax - self.vmin) * self.rect.width

    def draw(self, screen, font, mark_value=None, mark_label=None):
        pygame.draw.line(screen, SLIDER_TRACK, (self.rect.x, self.rect.centery),
                          (self.rect.right, self.rect.centery), 4)
        # Recommended-value ("optimum") marker: a distinct-colored tick with an
        # "opt" caption below, so the paper's sweet spot is easy to find again.
        if self.optimum is not None and self.vmax > self.vmin:
            ox = self._mark_x(self.optimum)
            pygame.draw.line(screen, OPTIMUM_MARK_COLOR, (ox, self.rect.top - 3),
                             (ox, self.rect.bottom + 3), 2)
            cap = font.render("opt", True, OPTIMUM_MARK_COLOR)
            screen.blit(cap, (ox - cap.get_width() / 2, self.rect.bottom + 4))
        if mark_value is not None:
            mx = self.rect.x + (mark_value - self.vmin) / (self.vmax - self.vmin) * self.rect.width
            pygame.draw.line(screen, MELT_MARK_COLOR, (mx, self.rect.top - 3), (mx, self.rect.bottom + 3), 2)
            if mark_label:
                lbl = font.render(mark_label, True, MELT_MARK_COLOR)
                screen.blit(lbl, (mx - lbl.get_width() / 2, self.rect.bottom + 4))
        hx, hy = self._handle_pos()
        color = SLIDER_HANDLE_ACTIVE if self.dragging else SLIDER_HANDLE
        pygame.draw.circle(screen, color, (int(hx), int(hy)), 8)
        text = font.render(f"{self.label}: {self.fmt.format(self.value)}{self.unit}", True, TEXT_COLOR)
        screen.blit(text, (self.rect.x, self.rect.y - 18))
