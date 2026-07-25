"""Bundled force fields.

Importing this package registers every bundled ForceField by name (see
playground.forcefield.register), which is how a playground file's
`force_field="mesomem"` resolves. The import is lazy -- triggered by
forcefield.get() -- so listing playgrounds does not pull in LAMMPS or scipy.
"""
from .mesomem import MesoMem
from .stock import EAM, LennardJones

__all__ = ["MesoMem", "LennardJones", "EAM"]
