/* MesoMem pair style, packaged as a runtime-loadable LAMMPS plugin.
   Wraps PairMesoMem (Sillano, Marrink & Idema 2026) so it can be pulled
   into a stock LAMMPS build via `plugin load mesomem.dylib` -- no full
   rebuild needed. */

#include "lammpsplugin.h"
#include "version.h"

#include <cstring>

#include "pair_membrane_sillano_v2.h"

using namespace LAMMPS_NS;

static Pair *mesomemcreator(LAMMPS *lmp)
{
  return new PairMesoMem(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "mesomem";
  plugin.info = "MesoMem mesoscale membrane pair style (Sillano, Marrink & Idema 2026)";
  plugin.author = "Pietro Sillano (TU Delft)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &mesomemcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
