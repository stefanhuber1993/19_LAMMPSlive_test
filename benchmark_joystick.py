#!/usr/bin/env python3
"""Micro-benchmark the Sidewinder FF2 HID operations in isolation, to see exactly
where per-frame joystick time goes (read vs. each force-feedback write).

Run on the machine with the stick plugged in:

    ./venv/bin/python benchmark_joystick.py

It times each raw device operation over many iterations and prints
min/median/mean/p95/max in milliseconds. Values are varied each iteration where
relevant so the driver-side "skip unchanged" cache can't hide the real cost.

Interpreting it against the in-app `--debug` line (fields `read` and `ff`):
  * `read_input(timeout_ms=0)`  -> the `read` field's cost per frame.
  * one `set_condition` (1 HID write) x how many the frame does -> the `ff` field.
    Per frame the app issues up to: 2 spring writes (send_force, X+Y) + 2 damper
    writes (set_damper_coefficient, X+Y) + 2 sine writes (update_jitter) = 6.
"""
import statistics
import sys
import time

sys.path.insert(0, ".")

from lammps_live.input.joystick import (
    JoystickInput, SPRING_STIFFNESS_MAX, SPRING_SATURATION,
    DAMPER_SATURATION, JITTER_PERIOD_MS,
)


def bench(label, fn, n=300, warmup=20):
    for _ in range(warmup):
        fn(-1)
    samples = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p95 = samples[int(0.95 * len(samples)) - 1]
    print(f"{label:<34s} min {samples[0]:6.2f}  med {statistics.median(samples):6.2f}  "
          f"mean {statistics.mean(samples):6.2f}  p95 {p95:6.2f}  max {samples[-1]:6.2f}  ms")


def main():
    print("Opening FF2 device...")
    # background=False: this benchmark drives the device directly from this
    # thread, so it must not also run the async I/O worker (that would race it).
    src = JoystickInput(background=False)
    ff = src.ff
    print("Nudge the stick a moment so a first input report lands...\n")
    time.sleep(0.5)

    # --- reads at different timeouts ---
    bench("read_input(timeout_ms=0)", lambda i: ff.read_input(timeout_ms=0))
    bench("read_input(timeout_ms=8)", lambda i: ff.read_input(timeout_ms=8), n=100)
    bench("read_input(timeout_ms=20) [old]", lambda i: ff.read_input(timeout_ms=20), n=100)

    print()

    # --- single HID write (one Set Condition report) ---
    def one_write(i):
        src.spring.set_condition(axis=0, cp_offset=(i % 50) - 25,
                                 pos_coeff=SPRING_STIFFNESS_MAX, neg_coeff=SPRING_STIFFNESS_MAX,
                                 pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)
    bench("1x set_condition (1 HID write)", one_write)

    # --- what the app does each frame (values varied to defeat the cache) ---
    def send_force(i):
        # bypass the cache by writing both axes directly with changing offsets
        for ax in (0, 1):
            src.spring.set_condition(axis=ax, cp_offset=(i % 40) - 20,
                                     pos_coeff=SPRING_STIFFNESS_MAX, neg_coeff=SPRING_STIFFNESS_MAX,
                                     pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)
    bench("send_force equiv (2 HID writes)", send_force)

    def set_damper(i):
        k = 13 + (i % 100)
        for ax in (0, 1):
            src.damper.set_condition(axis=ax, pos_coeff=k, neg_coeff=k,
                                     pos_sat=DAMPER_SATURATION, neg_sat=DAMPER_SATURATION)
    bench("set_damper equiv (2 HID writes)", set_damper)

    def update_jitter(i):
        mag = 10 + (i % 20)
        src.jitter.set_base(direction_deg=(i * 13) % 360, axis_x=True, axis_y=True,
                            direction_enable=True)
        src.jitter.set_periodic(magnitude=mag, period_ms=JITTER_PERIOD_MS)
    bench("update_jitter (2 HID writes)", update_jitter)

    print()

    # --- a full frame's worth of FF writes back-to-back ---
    def full_frame_ff(i):
        send_force(i)
        set_damper(i)
        update_jitter(i)
    bench("full-frame ff (6 HID writes)", full_frame_ff, n=200)

    src.close()
    print("\nDone. The `ff` field in the app's --debug line should match roughly the "
          "sum of the writes the frame actually issues (often fewer than 6, since "
          "unchanged values are cached/skipped).")


if __name__ == "__main__":
    main()
