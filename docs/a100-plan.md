# MesoMem on an A100, rendered locally

Plan for running a large MesoMem assembly real-time on the cluster A100 while
drawing it on the laptop. Every number below is measured on this repo (M1 Pro,
8 cores = 6P + 2E; the pip LAMMPS wheel, 20250722) unless flagged as an estimate.

**Realistic target: 25-50k beads at 60 fps**, against 1500 today. 100k is
reachable only if the Python analysis stops running per frame.

## 1. Feasibility gate -- do this FIRST (half a day, decides everything)

The pair kernel is the easy part. The risk is the *fixes*: if
`fix nve/sphere update dipole` or `fix langevin ... omega yes` have no Kokkos
variant, LAMMPS copies the whole state host<->device **every step** and the entire
gain evaporates. Kokkos coverage of the DIPOLE package is thin.

- Build LAMMPS with `-DPKG_KOKKOS=yes -DKokkos_ENABLE_CUDA=yes
  -DKokkos_ARCH_AMPERE80=yes`, plus DIPOLE, ASPHERE, MOLECULE, EXTRA-FIX.
- Run the existing deck with `-k on g 1 -sf kk`. A dumped deck is 24 self-contained
  lines (`plugin load` ... `run`); regenerate it by recording `lmp.command` during
  `PlaygroundSystem._setup`.
- Read the timing breakdown for host/device sync. **No sync = green light.**
- Kokkos means a real in-tree build, not the runtime-plugin path used now.

## 2. Split sim from render -- DONE

Built as `lammps_live/remote/` and the `mesomem_remote` playground:
[remote-gpu.md](remote-gpu.md) is the full write-up of how and why, and
docs/snellius/README.md is how to run it. Both ends build the same `Playground`
file, so there is one definition of the demo and no deck to keep in sync. The
original reasoning, which held up:

- Server = headless LAMMPS; client = this app, rendering received frames.
- **Already in our favour:** every readout hands back copies (the `stepper.py`
  contract) and everything is id-ordered, so the wire format is just the
  `frame_state` arrays.
- Control channel back: the sliders already produce plain LAMMPS command *strings*
  (`ForceField.live_commands`, `set_target_temp`) -- send the text.
- Sim mode only. Do not attempt the joystick/haptic *loop* over the network.
  The stick still drives the remote demo -- camera, Play/Pause/Reset, every
  slider (see the joystick section of the README) -- because none of that is a
  haptic loop: it is local UI on one side and the same command strings on the
  other. What stays local is force feedback, which needs the contact force back
  within a frame.

## 3. The Python analysis is the real wall

Measured **2.5 us/bead** for a full `Analysis.update`, **1.5 us/bead/chunk** as
throttled live, linear to 100k (a full update at 100k = 275 ms). This caps N at
**~10k beads even with an infinitely fast GPU**. Pick one:

| analysis | max N @ 60 fps |
|---|---|
| as-is | ~10,600 |
| 10x cheaper (cadence / subsample) | ~60,000 |
| off, or from GPU-side computes | ~125,000 |

The middle row is the one that happened -- by subsampling the pair list, measured
at 10x on the nose. See "The wall came down by subsampling" below.

### Re-measured at 10k, on the real thing (2026-08-18)

Against a *coarsened* 10k configuration -- 22.8 neighbours per bead, not the 11.5
of the initial gas the extrapolation above was based on:

| | measured |
|---|---|
| `build_pairs` (cKDTree, rc = 2.5) | **21.7 ms** |
| `energy_terms` over 113k pairs | 9.7 ms |
| all three observables together | 0.8 ms |
| one full `Analysis.update` | 31.7 ms |
| throttled average per frame, as shipped | **6.8 ms** |

So the estimate was right about the size of the wall (1.74 us/bead/chunk against
1.5 predicted) and the throttled average is now well inside a 60 fps frame -- but
only after **the scheduler was fixed**, which was worth 2.6x on its own and cost
nothing:

- The observables' phases were deliberately *staggered* so they would not land on
  the same frame. With a shared pair list that is backwards: the list is the
  expensive part, and three every-4-frames observables spread over three frames
  built it three times instead of once. They are now aligned.
- Two of the three never look at the pair list at all (`nematic_S`, `thickness`).
  They now declare that (`needs_pairs=False`) and cannot trigger a build.

What remains is a **31.7 ms peak** on the one frame in eight where the pair build
and the energy panel coincide -- a visible dip to ~31 fps at 7.5 Hz. The next
things to do about it, cheapest first: build the pair list with
`query_ball_point(..., workers=-1)` (multithreaded, unlike `query_pairs`);
subsample `coordination`, which only needs a mean; or move the energy decomposition
to a GPU-side compute and off this machine entirely.

### The wall came down by subsampling (2026-08-20)

The middle option, generalized: **the whole pair list** is now built over a
bounded uniform random sample of the beads rather than `coordination` alone
(`Analysis.MAX_PAIR_BEADS = 6000`), because the list is the cost and every
consumer of it tolerates sampling. `PairData.dilution` carries the fraction, and
each consumer divides it back out -- once for a per-particle mean, twice for a
pair sum, since a pair survives only if both its beads were drawn.

Measured at **N = 50,000** (the size `mesomem_remote` now runs), against a
condensed configuration:

| | full list | sampled |
|---|---|---|
| `build_pairs` | 66-95 ms (313k-438k pairs) | 5.8 ms |
| one `Analysis.update` on a due frame | **113 ms** | **11 ms** |
| per second of a 20 fps wire | 564 ms (56% of the thread) | 57 ms (5.7%) |

564 ms of work per second is why the demo stalled several times a second: each
113 ms lump is six times the 17 ms render it was supposed to hide under, so
`stepper.wait()` blocked for the remainder and the window stopped. It is now
five 11 ms lumps, all of which fit.

Accuracy, over 12 redrawn samples at a 5x dilution on a physical (non-overlapping)
20k configuration -- see `tests/test_analysis_budget.py`, which asserts these:

| | exact | sampled | error |
|---|---|---|---|
| coordination | 47.997 | 47.835 | +0.3% (0.33 spread per frame) |
| isotropic energy | -80578 | -80763 | +0.2% |
| tilt energy | 81493 | 82330 | +1.0% |
| splay energy | 13551 | 13569 | +0.1% |

The sample is redrawn every analysis frame, so the per-frame noise averages away
over a second rather than sitting there as a fixed wrong answer. Two limits are
deliberate: naming a particle (`keep_index`, used by every puller playground for
its single-bead panel) switches sampling off entirely, because one bead's fraction
of its own neighbours is a handful of pairs and no scale factor repairs that; and
every local playground is under the budget, so none of them sample at all.

Two smaller things moved off the drawing thread at the same time, both O(N) work
that was running in front of a vsync'd `flip()`:

- the **cluster labelling** (39-52 ms at 50k, once every ~1.6 s when the colouring
  is on) now runs in `RemoteSystem._ingest`, on the stepper thread, which is what
  `playground/clustering.py` already named as the fix;
- the **RDF sample** (5.9 ms per sample) likewise, with `sample_every=1` on the
  remote path so the number of samples per second, and the window the rolling
  average covers, is unchanged.

The remaining per-frame gather on the drawing thread is **0.8 ms** at 50k, down
from 3.1 ms. Note for reading the `--debug` line: `flip()` is inside the render
timer and swap interval is never set, so on macOS `render` includes the wait for
the next refresh -- it is frame pacing, not GPU work, and total work above 16.7 ms
doubles the frame to 33 ms rather than degrading smoothly.

## 4. Budget table

Per-bead cost of a 20-step chunk against the 16.7 ms frame:

| | sim | analysis | render | max N @ 60 fps |
|---|---|---|---|---|
| today, 1 core | 9.0 | 1.5 | ~0.03 | 1,850 (measured 1500 @ 16.5 ms) |
| 6 MPI ranks | 2.2 | 1.5 | ~0.03 | ~4,500 |
| A100, rest unchanged | 0.07 | 1.5 | ~0.05 | ~10,600 |
| A100 + analysis 10x cheaper | 0.07 | 0.15 | 0.06 | ~60,000 |
| A100 + analysis off | 0.07 | -- | 0.06 | ~125,000 |

(us/bead/chunk.) A100 sim figure is an **estimate**: 0.34 us/bead/step in pair on
one M1 core (76% of 0.45 total, ~15 ns per pair interaction over ~11.5 half-list
neighbours) divided by 60-150x for a double-precision Kokkos port. FP64 is 1:2 on
A100, so double precision is fine there (it would not be on a consumer card).
The conclusion is insensitive to this being wrong by 2-3x, because the GPU stops
being the bottleneck.

**Local render ceiling** (measured, GL pass only): 2.6 ms @ 6k, 4.0 @ 50k,
6.4 @ 100k, 10.3 @ 200k. Not yet measured: the CPU-side scene assembly (wrap
ghosts, periodic images, depth sort) -- measure early, it is the likely surprise.

## 5. Network transfer

Measured on a real assembly trajectory, scaled to 100k beads. zlib-1 stands in for
zstd-1 (not installed here; zstd would be modestly smaller and several times
faster).

| scheme | B/bead | 100k @ 60 fps | @ 30 fps |
|---|---|---|---|
| float32 positions + directors (naive) | 24.0 | 1152 Mb/s | 576 |
| uint16 positions + octahedral-8 directors | 8.0 | 384 | 192 |
| + zlib-1 on byte planes | 8.0 | 385 | 192 |
| + temporal delta, raw trajectory | 6.5 | 313 | 157 |
| + temporal delta, **smoothed** trajectory | 5.5 | 264 | 132 |
| 12-bit quantisation, delta, smoothed | **4.3** | **207** | **104** |

Total achievable: **~5.5x**, and nearly all of it is quantisation, not entropy
coding.

- **Entropy coding alone buys nothing** (8.01 vs 8.00 B/bead): quantised bead
  positions are high-entropy. Only *deltas* compress, and only somewhat.
- **Quantisation is free visually.** At the 100k box (81 sigma): 16 bits/axis =
  0.0012 sigma = 0.015 px; 12 bits = 0.020 sigma = 0.23 px; 10 bits = 0.079 sigma
  = 0.94 px. Bead radius is 0.5 sigma. Octahedral-8 directors cost 0.34 deg mean,
  0.87 deg max -- invisible in the band shading, and 2 bytes instead of 12.
- **Smoothing helps the wire, not just the eye.** Turning on trajectory smoothing
  (already shipped) shrinks the mean per-frame step from 0.194 to 0.075 sigma, so
  deltas are smaller: 6.5 -> 5.5 B/bead. Smooth server-side and send the smoothed
  stream.
- **Sparse "unchanged bead" updates are a dead end** -- measured 0-3% of beads
  unchanged frame to frame even at heavy smoothing. Do not build it.
- **Compression cost:** ~5.5 ms/frame at 100k for zlib-1, on the server. Budget it,
  or use zstd.

Ways to actually fit a thin link, in order of payoff:

1. **Send fewer frames.** SHIPPED (2026-08-20), at 20 fps by default -- and with
   `q12` underneath that is 45 Mb/s at 100k rather than this row's 69. Two
   corrections to what this line originally said, both measured:
   *the interpolation is not what shipped.* It is the most accurate filler (1 px
   from the 60 Hz truth with smoothing on, against 10 px for holding the frame)
   but it plays a whole wire frame behind, which is 50 ms on every slider.
   *Extrapolation, the zero-latency alternative, is worse than freezing the
   picture* -- between frames the motion is uncorrelated thermal noise, so a
   velocity fitted to two samples is a random number. What ships instead
   synthesises the rattle (playground/jitter.py). Note also that "smoothed
   positions are band-limited" is not load-bearing: the demo runs with the
   Smoothing slider at zero.
2. **Server-side camera-aware culling.** The client already knows the camera; send
   it up and drop occluded beads. A dense 81 sigma box only ever shows its shell:
   a 2 sigma outer shell is ~14% of beads, so up to ~7x. Caveat: lamellae are
   visible through gaps, so a naive shell cull deletes real structure -- use a
   coarse depth test, not a radius test.
3. **Drop or downsample directors** -- 2 of the 4.3 B/bead. They change slowly under
   smoothing, so a lower update rate than positions is fine.

Check the actual link first: 100k @ 60 fps needs ~200 Mb/s sustained. Trivial on a
wired research network, not happening over a home VPN.

### What shipped (2026-08-18)

`lammps_live/remote/protocol.py` implements the second row of that table, and then
went *past* it. It first shipped with **octahedral-16 directors, not
octahedral-8**, so 10 B/bead rather than 8, on the argument that the client
MEASURES from these directors as well as drawing them (`nematic_S` is the number
the k_tilt transition shows up in) and an order parameter should not carry a
codec's error.

**That was measured and reversed (2026-08-20.)** Coding directors at 8 bits moves
`nematic_S` by 8e-5 -- the angular error is random and zero-mean and S is a second
moment over the whole population, so it averages out rather than accumulating.
Positions went to 12 bits per axis at the same time (0.2 px of quantisation in the
windowed viewport, 0.33 fullscreen), packed two codes to three bytes. The shipped
`q12` codec is **6.5 B/bead**: 65 kB/frame, 3.9 MB/s at 10k and 60 fps, 31 Mb/s.
`q16` is still selectable. The reasoning, and the pixel table both bit counts were
chosen from, is remote-networking.md §4.

Everything below the second row is deliberately NOT built. Delta coding and
entropy coding both need a stateful decoder that a dropped frame invalidates, and
the client drops frames on purpose (see remote/client.py). At 10k the link is not
the bottleneck; when 100k is, the order to add them in is the order above.

## 6. Validation

The port must reproduce the CPU reference: total energy at t=0, then a short NVE
run. Template exists -- the plugin pair style is already **bit-identical across
1/2/4/8 MPI ranks** under NVE (`-0.12278115`), which is how we know its ghost and
dipole exchange are decomposition-exact. Under Langevin, trajectories diverge with
rank count (per-rank RNG streams); that is expected, not a bug. Check
`Dangerous builds = 0` after scaling.

## Notes

- Physics rate is unchanged by any of this: same frame rate = same **12 tau/s**.
  The win is an 81 sigma box at 100k (phi = 0.1) holding many independent lamellae,
  not faster coarsening. Paper timeline: patches ~500 tau, large lamellae ~2000 tau.
- One A100 is not saturated below ~200k beads -- skip multi-GPU initially.
- No GPU or KOKKOS package in the pip wheel (`-pk gpu` errors out), and the wheel
  was built without `-fopenmp` (`OpenMP support not enabled during compilation`),
  so `-sf omp` and `-sf opt` both measure exactly 1.00x here.
- **Fallbacks if the port stalls.** 6 MPI ranks on the laptop -> ~4,500 beads;
  MPI scaling measured 4.1x at 6 ranks and *collapses* at 7+ (ranks landing on the
  E-cores, Comm 46%, CPU use 60%). Or compile the plugin dylib itself with
  `-fopenmp` and thread its `compute()` -- ~2.7x ceiling (Amdahl on the 76% pair
  fraction), but zero changes to the app's data model, which MPI cannot say.
