"""Getting text out of the app and into wherever it is going to be read.

Exists for exactly one reason: a remote session that fails does so on a card in a
pygame window, and the thing that explains WHY is a two-hundred-line interleaved
log of this end's steps and the cluster's own output. That log gets read somewhere
else -- in a message to whoever runs the cluster, next to `sacct` output, in a
ticket -- and retyping it off a screenshot is not debugging.

WHY NOT `pygame.scrap`. It is the obvious answer and it is the fallback here, not
the first choice: it needs an initialised display, its behaviour differs across
pygame versions and SDL video drivers, and on a headless run (the tests, a
`--no-display` smoke check) it raises. The platform's own clipboard command is a
pipe, works whether or not a window exists, and cannot be broken by whatever the
renderer is doing to the video subsystem at the time.

Everything here returns a bool rather than raising: a copy that failed is worth a
line on screen, never worth taking the app down with it.
"""
import platform
import subprocess

# Per platform, in order of preference. Each takes the text on stdin.
_COMMANDS = {
    "Darwin": [["pbcopy"]],
    # Wayland first: `xclip` under XWayland works but silently loses the selection
    # when the process that owns it exits, which is immediately.
    "Linux": [["wl-copy"], ["xclip", "-selection", "clipboard"],
              ["xsel", "--clipboard", "--input"]],
    "Windows": [["clip"]],
}


def copy(text):
    """Put `text` on the system clipboard. True if something took it."""
    text = str(text)
    for argv in _COMMANDS.get(platform.system(), []):
        try:
            proc = subprocess.run(argv, input=text.encode("utf-8", "replace"),
                                  capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            return True
    return _copy_via_pygame(text)


def _copy_via_pygame(text):
    """SDL's clipboard, for a platform with no command worth trying."""
    try:
        import pygame

        if not pygame.display.get_init():
            return False
        pygame.scrap.init()
        pygame.scrap.put_text(text)
        return True
    except Exception:                      # noqa: BLE001 -- best effort by design
        return False
