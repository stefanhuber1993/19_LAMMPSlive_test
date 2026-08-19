#!/usr/bin/env python3
"""Interactive real-time MD built on LAMMPS: explore a force field by feel.

A PLAYGROUND names a force field, a scenario and a mode; every live force-field
parameter becomes a slider you can drag while the simulation runs. Add a new one
by writing a ~30-line file in lammps_live/playgrounds/ (or anywhere, and pass its
path).

    lammps-live --list                          # everything runnable
    lammps-live --playground mesomem_sheet      # drag one bead out of a membrane
    lammps-live --playground mesomem_assembly   # watch 1500 beads self-assemble
    lammps-live --playground mesomem_assembly --mode game   # ... then poke it
    lammps-live --playground mesomem_sheet --preset buckled --input joystick
    lammps-live --verify                        # check the force fields' energy
    lammps-live --playground ./my_idea.py       # your own file, anywhere

See README.md for setup, controls, and how to write a playground.
"""
import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(prog="lammps-live", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", choices=["mouse", "keyboard", "joystick"],
                        default="mouse",
                        help="control source: mouse (pointer position + L/R "
                             "buttons), keyboard (WASD + Q/E), or joystick "
                             "(default: mouse)")
    parser.add_argument("--playground", "--system", dest="target", default=None,
                        metavar="KEY_OR_PATH",
                        help="what to run: a bundled playground key, or a path "
                             "to your own playground .py "
                             "(default: the first playground; see --list)")
    parser.add_argument("--mode", choices=["game", "sim"], default=None,
                        help="override a playground's mode: game (drive a "
                             "particle with the input device) or sim (Play / "
                             "Pause / Reset). Both work on any playground")
    parser.add_argument("--preset", default=None, metavar="NAME",
                        help="named parameter set from the playground's presets "
                             "(see --list-presets)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="start in fullscreen (toggle any time with F11)")
    parser.add_argument("--ui-scale", type=float, default=None, metavar="FACTOR",
                        help="how big to draw the 2D UI (text, panel, plots): 1 = "
                             "the classic pixel sizes, 2 = double, which is what a "
                             "4K screen at its NATIVE resolution wants -- the fonts "
                             "are rasterized at that size, so the UI is sharp "
                             "rather than magnified. Default: 2 on a screen taller "
                             "than 1800 px, else 1")
    parser.add_argument("--debug", action="store_true",
                        help="show a per-frame timing breakdown (sim vs. analysis "
                             "vs. render vs. device I/O) in the GUI header")
    parser.add_argument("--list", "--list-systems", "--list-playgrounds",
                        dest="list_all", action="store_true",
                        help="print everything runnable and exit")
    parser.add_argument("--list-presets", action="store_true",
                        help="print each playground's named presets and exit")
    parser.add_argument("--verify", action="store_true",
                        help="check every force field's Python energy expression "
                             "against the potential energy LAMMPS computes from "
                             "the compiled pair style, then exit")
    parser.add_argument("--calibrate", action="store_true",
                        help="print live joystick state for a few seconds and exit")
    return parser


def _print_listing():
    from .playground import registry
    entries = registry.list_playgrounds()
    width = max((len(k) for k, _s in entries), default=12)
    print("\nPlaygrounds:")
    for key, spec in entries:
        mode = "sim " if spec.playback_controls else "game"
        print(f"  {key:<{width}}  [{mode}]  {spec.name} -- {spec.description}")
    print()
    return 0


def _print_presets():
    from .playground import registry
    for key, pg in registry.all_playgrounds():
        print(f"\n{key}  ({pg.name})")
        if not pg.presets:
            print("  (no presets defined)")
            continue
        for name, overrides in pg.presets.items():
            detail = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "declared defaults"
            print(f"  {name:<18} {detail}")
    print()
    return 0


def _run_verify(target):
    """Cross-check the Python energy expressions against LAMMPS. This is the
    force-field regression check -- see playground/verify.py."""
    from .playground import registry
    from .playground.verify import verify_all
    refs = [target] if target else registry.bundled_keys()
    ok, results = verify_all(refs)
    for label, res in results:
        if isinstance(res, Exception):
            print(f"[verify] ERROR {label}: {res}")
            continue
        print(res.report(label))
    print()
    print("[verify] all force fields agree with LAMMPS" if ok
          else "[verify] MISMATCH -- the Python expression and the pair style disagree")
    return 0 if ok else 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_all:
        return _print_listing()
    if args.list_presets:
        return _print_presets()
    if args.verify:
        return _run_verify(args.target)

    if args.calibrate:
        from .input import JoystickInput
        # Synchronous (no I/O worker): calibrate reads the device directly on this
        # thread, so a background reader would race it for reports.
        js = JoystickInput(background=False)
        try:
            js.calibrate()
        finally:
            js.close()
        return 0

    from .playground import registry
    if args.target:
        initial_key = registry.resolve(args.target)
    else:
        keys = registry.bundled_keys()
        if not keys:
            parser.error("no playgrounds found")
        initial_key = keys[0]

    from .app import App
    app = App(input_mode=args.input, initial_system_key=initial_key,
              fullscreen=args.fullscreen, debug=args.debug,
              mode=args.mode, preset=args.preset)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
