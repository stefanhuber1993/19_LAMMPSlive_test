"""Thermostats.

Pulled out as its own pluggable piece because the two shipped ones are not
variations on a theme -- they differ in what they do to the physics, and the
choice belongs to the scenario:

  Langevin  per-atom friction plus random forcing. Stands in for an implicit
            solvent, which is exactly right for a coarse-grained membrane in
            water.
  CSVR      global velocity rescaling toward a target (Bussi et al. 2007), with
            NO per-atom random forcing: atoms move under real pair forces only,
            so on-screen motion is genuine lattice dynamics, the RDF stays
            canonically correct, and a quench reaches a true 0 K instead of the
            noise floor a Langevin bath leaves behind.

CSVR also carries real behaviour beyond issuing a fix, which is why it is an
object and not a format string: a velocity rescaling cannot warm a lattice that
is at rest, so heating from near-frozen has to seed a Maxwell-Boltzmann
distribution, and a sharp quench has to zero the net momentum once or a hot
upward-billowing cloud sails off as it solidifies.
"""


class Thermostat:
    """Base: issues a fix on a group and re-issues it when the target moves."""

    fix_name = "bath"

    def initial_commands(self, group, temperature, damp, seed):
        return self.set_commands(group, temperature, damp, seed)

    def set_commands(self, group, temperature, damp, seed):
        raise NotImplementedError

    def pre_change_commands(self, group, current, target, seed):
        """Commands to issue BEFORE the new setpoint, given the measured current
        temperature. Empty by default."""
        return []

    def temperature_compute(self, group):
        """(compute name, command) for the temperature this thermostat drives and
        the panel displays, or None to use LAMMPS' global `temp`."""
        return None


class Langevin(Thermostat):
    """Implicit-solvent bath: per-atom friction + random forcing, with the
    rotational degrees of freedom thermostatted too (`omega yes`) for particles
    that carry an orientation."""

    def __init__(self, rotational=True):
        self.rotational = rotational

    def set_commands(self, group, temperature, damp, seed):
        omega = " omega yes" if self.rotational else ""
        return [f"fix {self.fix_name} {group} langevin {temperature} {temperature} "
                f"{damp} {seed}{omega}"]


class CSVR(Thermostat):
    """Canonical-sampling velocity rescaling, bound to a COM-subtracted
    temperature compute.

    `temp/com` subtracts the group's bulk translation so only thermal motion
    counts -- which is what lets a quench reach 0 K and a hot gas rise to fill the
    box instead of being pinned. Because the fix is bound to that exact compute
    via fix_modify, the displayed temperature and the thermostat's setpoint are
    the same number, with no measured-vs-target fudge factor.
    """

    compute_name = "bath_temp_com"

    def __init__(self, cold_seed_temp=5.0, quench_zero_drop_frac=0.3):
        # Below cold_seed_temp the lattice is effectively at rest, so heating is
        # seeded rather than rescaled.
        self.cold_seed_temp = cold_seed_temp
        self.quench_zero_drop_frac = quench_zero_drop_frac

    def temperature_compute(self, group):
        return (self.compute_name,
                f"compute {self.compute_name} {group} temp/com")

    def set_commands(self, group, temperature, damp, seed):
        return [
            f"fix {self.fix_name} {group} temp/csvr {temperature} {temperature} "
            f"{damp} {seed}",
            f"fix_modify {self.fix_name} temp {self.compute_name}",
        ]

    def pre_change_commands(self, group, current, target, seed):
        if target > current and current < self.cold_seed_temp:
            # A velocity rescaling cannot warm ~zero motion into existence.
            return [f"velocity {group} create {target} {seed} "
                    f"mom yes rot yes dist gaussian"]
        if (current > self.cold_seed_temp
                and target < current - self.quench_zero_drop_frac * current):
            # Sharp quench: zero the net linear momentum once so a hot, rising
            # cloud decelerates in place rather than sailing off as it freezes.
            return [f"velocity {group} zero linear"]
        return []
