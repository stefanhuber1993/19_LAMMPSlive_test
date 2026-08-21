/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Pietro Sillano (TU Delft), 2026

#ifdef PAIR_CLASS
// clang-format off
PairStyle(rod_lj,PairRodLJ);
// clang-format on
#else

#ifndef LMP_PAIR_ROD_LJ_H
#define LMP_PAIR_ROD_LJ_H

#include "pair.h"

namespace LAMMPS_NS {

class PairRodLJ : public Pair {
 public:
  PairRodLJ(LAMMPS *lmp);
  ~PairRodLJ() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;

 protected:
  double **cut;
  double **sigma, **eps;
  double cut_global;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
