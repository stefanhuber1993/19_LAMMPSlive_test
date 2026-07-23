#!/usr/bin/env python3
"""Interactive real-time MD demos built on LAMMPS: a puller atom under
continuous mouse/joystick control interacts with a live, thermostatted 2D
crystal or fluid. Multiple systems (materials/potentials) are registered in
lammps_live.systems and can be switched between at runtime (1-9 or Tab) or
picked up front with --system.

    lammps-live --input mouse            # pointer: position moves, L/R buttons rotate
    lammps-live --input keyboard         # WASD moves, Q/E rotate
    lammps-live --input joystick         # Sidewinder FF2 (via hidapi -- no sudo needed)
    lammps-live --list-systems

See README.md for setup, controls, and how to add a new system.
"""
import argparse
import sys

from .systems import list_systems


def build_parser():
    parser = argparse.ArgumentParser(prog="lammps-live", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", choices=["mouse", "keyboard", "joystick"], default="mouse",
                         help="control source for the puller atom: mouse (pointer "
                              "position + L/R buttons), keyboard (WASD + Q/E), or "
                              "joystick (default: mouse)")
    parser.add_argument("--system", default=None,
                         help="system to start with (default: first registered; see --list-systems)")
    parser.add_argument("--fullscreen", action="store_true",
                         help="start in fullscreen (toggle any time with F11)")
    parser.add_argument("--debug", action="store_true",
                         help="show a per-frame timing breakdown (simulation vs. "
                              "rendering vs. other) in the GUI header")
    parser.add_argument("--list-systems", action="store_true",
                         help="print available systems and exit")
    parser.add_argument("--calibrate", action="store_true",
                         help="print live joystick state for a few seconds and exit")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    systems = list_systems()

    if args.list_systems:
        for key, spec in systems:
            print(f"{key:12s} {spec.name} -- {spec.description}")
        return 0

    if args.calibrate:
        from .input import JoystickInput
        js = JoystickInput()
        try:
            js.calibrate()
        finally:
            js.close()
        return 0

    initial_key = args.system or systems[0][0]
    if initial_key not in dict(systems):
        available = ", ".join(key for key, _ in systems)
        parser.error(f"unknown --system {initial_key!r}. Available: {available}")

    from .app import App
    app = App(input_mode=args.input, initial_system_key=initial_key,
              fullscreen=args.fullscreen, debug=args.debug)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
