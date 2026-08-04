"""Display formatting for the on-screen readouts.

Two unit systems are in play, and which one a system uses is its own property
(`SystemSpec.reduced_units`), not a global:

  * LAMMPS **metal** units -- eV, Angstrom, ps, amu -- for the systems built on
    real, measured potentials (copper EAM, argon LJ, ionic NaCl). Numbers there
    mean what a physicist expects: a force of `1.2 eV/A` is literally -dE/dx.
  * **Reduced LJ** units -- sigma = epsilon = m = 1, time in tau -- for the
    MesoMem membrane systems, which are the paper's coarse-grained model and have
    no Angstrom or Kelvin to be converted to. Printing "K" or "eV" there would be
    inventing a physical scale the model does not have.

Everything in this module is display-only; the simulation stays in whichever
units its force field declared.
"""

ANGSTROM_PER_PS_TO_M_PER_S = 100.0
# 1 A/ps = 1e-10 m / 1e-12 s = 1e2 m/s. Sanity check: typical thermal atomic
# speeds are hundreds of m/s at room temperature, which is also the range
# the puller atom's speed lands in during normal play -- not a coincidence,
# it's the same physics.


def speed_to_m_per_s(angstrom_per_ps):
    return angstrom_per_ps * ANGSTROM_PER_PS_TO_M_PER_S


def format_sim_time(t, reduced=False):
    """Elapsed simulated time.

    In metal units it is picoseconds, auto-scaled to a readable unit -- MD sim
    times in this demo run from sub-picosecond up to a few hundred ps over a play
    session, rarely a clean fit for any single fixed one.

    In reduced units it is the model's own tau, and there is nothing to scale it
    to: tau is not a second, and writing "fs" against a coarse-grained membrane
    would claim a physical time scale the model never fixed.
    """
    if reduced:
        return f"{t:,.1f} tau"
    if t < 1.0:
        return f"{t * 1000.0:.1f} fs"
    if t < 1000.0:
        return f"{t:.2f} ps"
    return f"{t / 1000.0:.3f} ns"


def energy_unit(reduced=False):
    """Suffix for an energy readout: reduced energies are in units of the pair
    potential's own well depth epsilon."""
    return " eps" if reduced else " eV"


def force_unit(reduced=False):
    """Suffix for a force readout: eps/sigma in reduced units."""
    return " eps/sigma" if reduced else " eV/A"


def format_energy(e, reduced=False):
    """A single per-particle energy, formatted to stay readable across the orders
    of magnitude it spans between systems: a bound EAM/ionic atom sits at several
    eV (cohesive energies ~ -3 eV), while a weakly-bound LJ argon atom or a
    coarse-grained membrane bead can be a thousandth of that. The number of shown
    decimals shrinks as the magnitude grows so it stays ~3-4 significant figures
    wide, and the sign is always shown -- a negative energy (bound, below the
    free-particle reference) is the whole point of the readout. Very small/large
    magnitudes fall back to scientific notation rather than printing a screen-full
    of zeros or digits."""
    u = energy_unit(reduced)
    mag = abs(e)
    if mag < 5e-4:
        return f"0{u}"
    if mag < 0.1:
        return f"{e:+.4f}{u}"
    if mag < 10.0:
        return f"{e:+.3f}{u}"
    if mag < 1000.0:
        return f"{e:+.1f}{u}"
    return f"{e:+.2e}{u}"
