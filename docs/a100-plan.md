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

## 2. Split sim from render (do on the laptop, no GPU needed)

De-risks the whole pipeline before any kernel work, and is what makes the demo
possible at all.

- Server = headless LAMMPS; client = this app, rendering received frames.
- **Already in our favour:** every readout hands back copies (the `stepper.py`
  contract) and everything is id-ordered, so the wire format is just the
  `frame_state` arrays.
- Control channel back: the sliders already produce plain LAMMPS command *strings*
  (`ForceField.live_commands`, `set_target_temp`) -- send the text.
- Sim mode only. Do not attempt the joystick/haptic loop over the network.

## 3. The Python analysis is the real wall

Measured **2.5 us/bead** for a full `Analysis.update`, **1.5 us/bead/chunk** as
throttled live, linear to 100k (a full update at 100k = 275 ms). This caps N at
**~10k beads even with an infinitely fast GPU**. Pick one:

| analysis | max N @ 60 fps |
|---|---|
| as-is | ~10,600 |
| 10x cheaper (cadence / subsample) | ~60,000 |
| off, or from GPU-side computes | ~125,000 |

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

1. **Send fewer frames and interpolate locally.** 30 fps stream -> 104 Mb/s, 20 fps
   -> 69 Mb/s. Smoothed positions are band-limited, so interpolation is nearly
   exact -- this is the cheapest big win.
2. **Server-side camera-aware culling.** The client already knows the camera; send
   it up and drop occluded beads. A dense 81 sigma box only ever shows its shell:
   a 2 sigma outer shell is ~14% of beads, so up to ~7x. Caveat: lamellae are
   visible through gaps, so a naive shell cull deletes real structure -- use a
   coarse depth test, not a radius test.
3. **Drop or downsample directors** -- 2 of the 4.3 B/bead. They change slowly under
   smoothing, so a lower update rate than positions is fine.

Check the actual link first: 100k @ 60 fps needs ~200 Mb/s sustained. Trivial on a
wired research network, not happening over a home VPN.

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
