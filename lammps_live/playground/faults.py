"""When LAMMPS stops the simulation, and how to say so to somebody watching.

Exploring a parameter space means being able to reach settings that destroy the
simulation. That is not misuse -- it is the point of having sliders -- so a
destroyed simulation has to be an event the app handles, not an exception that
takes the process down. On the cluster it is worse than a crash: the server dying
cancels its own allocation, so one bad slider costs the GPU and a queue wait.

TWO AUDIENCES, ONE EVENT. The raw LAMMPS error is the only thing that can be acted
on -- it names the style, the file and the line -- and it is also unreadable if you
are standing in front of the demo rather than writing it. So a `Fault` carries both:
a `summary` in plain language, big, and the `detail` verbatim, small. Neither
replaces the other.

WHAT `summarise` IS AND IS NOT. It is a lookup table from the handful of LAMMPS
errors this app can actually provoke to a sentence about what the user just did. It
is deliberately not clever: an unrecognised error gets a generic sentence and its
own text, which is exactly what an unrecognised error deserves. Adding a row is
cheap and is the right response to meeting a new one.
"""
import re
from dataclasses import dataclass, field

# (pattern, summary). First match wins, so put the specific ones first. The
# patterns are matched case-insensitively against the whole error text.
_SUMMARIES = (
    (r"lost atoms",
     "The simulation blew up -- particles flew out of the box."),
    (r"simulation unstable|non-numeric|nan|inf\b",
     "The simulation blew up -- the forces went to infinity."),
    (r"requires\s+\w+\s*[<>=]",
     "This build will not accept that parameter value."),
    (r"neighbor list overflow|too many neighbor|neighbor list",
     "The interaction cutoff is too large for this box -- too many neighbours."),
    (r"out of range atoms|domain too small|bins",
     "The box no longer matches the particles in it."),
    (r"cannot (open|find)|unknown (pair|atom|fix|compute) style|invalid (pair|atom) style",
     "This LAMMPS build does not have a style this playground needs."),
    (r"illegal|incorrect args|unknown (keyword|identifier)",
     "A command this playground issued was not valid for this build."),
    (r"out of memory|cannot allocate",
     "The simulation ran out of memory."),
)

_GENERIC = "LAMMPS stopped the simulation."


def summarise(error):
    """One plain sentence for a raw LAMMPS error. Never empty."""
    text = str(error or "").strip()
    if not text:
        return _GENERIC
    for pattern, summary in _SUMMARIES:
        if re.search(pattern, text, re.IGNORECASE):
            return summary
    return _GENERIC


def first_line(error):
    """The raw error, trimmed to the one line that says what happened.

    LAMMPS' Python exceptions arrive with the message first and a `Last input line:`
    note after it; both are worth keeping, and nothing below them is.
    """
    lines = [l.strip() for l in str(error or "").strip().splitlines() if l.strip()]
    keep = lines[:1]
    for line in lines[1:3]:
        if line.lower().startswith("last input line"):
            keep.append(line)
    return "  ".join(keep) or str(error)


@dataclass
class Fault:
    """One simulation-destroying event, ready to be shown and to be logged.

    `reverted` maps a parameter name to the value the rebuild fell back to, when it
    had to fall back -- so the caller can put the sliders back where the simulation
    actually is rather than leaving them pointing at the value that killed it.
    """

    summary: str
    detail: str
    reverted: dict = field(default_factory=dict)
    # True while the simulation is not running and needs a rebuild to come back.
    fatal: bool = True

    @classmethod
    def from_error(cls, error, reverted=None, fatal=True):
        return cls(summary=summarise(error), detail=first_line(error),
                   reverted=dict(reverted or {}), fatal=fatal)

    def line(self):
        """One line, for a log or the HUD."""
        note = ""
        if self.reverted:
            note = " -- reverted " + ", ".join(
                f"{k} to {v:g}" for k, v in sorted(self.reverted.items()))
        return f"{self.summary}{note}  [{self.detail}]"

    def as_message(self):
        """The wire form, for the remote server's frame header."""
        return {"summary": self.summary, "detail": self.detail,
                "reverted": {k: float(v) for k, v in self.reverted.items()},
                "fatal": bool(self.fatal)}

    @classmethod
    def from_message(cls, msg):
        if not msg:
            return None
        return cls(summary=str(msg.get("summary") or _GENERIC),
                   detail=str(msg.get("detail") or ""),
                   reverted={k: float(v)
                             for k, v in (msg.get("reverted") or {}).items()},
                   fatal=bool(msg.get("fatal", True)))
