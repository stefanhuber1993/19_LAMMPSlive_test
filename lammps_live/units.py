"""Conversions from LAMMPS "metal" units (eV, Angstrom, ps, amu) to
everyday SI, for on-screen readouts -- the internal simulation and all
other modules stay in metal units throughout; this is display-only."""

ANGSTROM_PER_PS_TO_M_PER_S = 100.0
# 1 A/ps = 1e-10 m / 1e-12 s = 1e2 m/s. Sanity check: typical thermal atomic
# speeds are hundreds of m/s at room temperature, which is also the range
# the puller atom's speed lands in during normal play -- not a coincidence,
# it's the same physics.


def speed_to_m_per_s(angstrom_per_ps):
    return angstrom_per_ps * ANGSTROM_PER_PS_TO_M_PER_S


def format_sim_time(ps):
    """Elapsed simulated time, auto-scaled to a readable unit -- MD sim
    times in this demo run from sub-picosecond up to a few hundred ps over
    a play session, rarely a clean fit for any single fixed unit."""
    if ps < 1.0:
        return f"{ps * 1000.0:.1f} fs"
    if ps < 1000.0:
        return f"{ps:.2f} ps"
    return f"{ps / 1000.0:.3f} ns"


def format_energy(ev):
    """A single per-atom energy (eV), formatted to stay readable across the
    orders of magnitude it spans between systems: a bound EAM/ionic atom sits
    at several eV (cohesive energies ~ -3 eV), while a weakly-bound LJ argon
    atom or a coarse-grained lipid bead can be a thousandth of that. The number
    of shown decimals shrinks as the magnitude grows so it stays ~3-4
    significant figures wide, and the sign is always shown -- a negative energy
    (bound, below the free-atom reference) is the whole point of the readout.
    Very small/large magnitudes fall back to scientific notation rather than
    printing a screen-full of zeros or digits."""
    mag = abs(ev)
    if mag < 5e-4:
        return "0 eV"
    if mag < 0.1:
        return f"{ev:+.4f} eV"
    if mag < 10.0:
        return f"{ev:+.3f} eV"
    if mag < 1000.0:
        return f"{ev:+.1f} eV"
    return f"{ev:+.2e} eV"
