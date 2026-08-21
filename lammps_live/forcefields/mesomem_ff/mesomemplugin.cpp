/* MesoMem pair styles, packaged as a runtime-loadable LAMMPS plugin.
   Wraps PairMesoMem (Sillano, Marrink & Idema 2026) and PairRodLJ so they
   can be pulled into a stock LAMMPS build via `plugin load mesomem.dylib`
   -- no full rebuild needed.

   Both styles ship in the one library because they are used together: the
   wrapping-rod playground runs `pair_style hybrid mesomem ... rod_lj ...`,
   and a hybrid style needs every sub-style present in the same instance.
   The playgrounds that use only the membrane simply never name the second. */

#include "lammpsplugin.h"
#include "version.h"

#include <cstring>

#include "pair_membrane_sillano_v2.h"
#include "pair_rod_lj.h"

using namespace LAMMPS_NS;

static Pair *mesomemcreator(LAMMPS *lmp)
{
  return new PairMesoMem(lmp);
}

static Pair *rodljcreator(LAMMPS *lmp)
{
  return new PairRodLJ(lmp);
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

  plugin.name = "rod_lj";
  plugin.info = "Rigid-rod / point-particle generalised LJ (segment-to-point)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &rodljcreator;
  (*register_plugin)(&plugin, lmp);
}
