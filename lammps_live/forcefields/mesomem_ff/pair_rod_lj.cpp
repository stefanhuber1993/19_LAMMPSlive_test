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
//
// Generalised LJ interaction between a point-like particle and a rigid rod,
// ported from new_line_style/freedts.cpp (RodWrapping::EvolveOneStep).
//
// Atom-style requirement: hybrid sphere dipole.
//   - mu[0..2] : rod long-axis direction (normalised through mu[3])
//   - q        : rod length L (point particles must have q == 0)
//   - radius   : rod cylindrical half-thickness (added to sigma)
//
// Rod / point assignment is resolved per pair at runtime by the sign of q.

#include "pair_rod_lj.h"

#include <cmath>
#include <cstring>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairRodLJ::PairRodLJ(LAMMPS *lmp) :
    Pair(lmp), cut(nullptr), sigma(nullptr), eps(nullptr)
{
  writedata = 1;
  single_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairRodLJ::~PairRodLJ()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(sigma);
    memory->destroy(eps);
  }
}

/* ---------------------------------------------------------------------- */

void PairRodLJ::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(sigma, np1, np1, "pair:sigma");
  memory->create(eps, np1, np1, "pair:eps");
}

/* ---------------------------------------------------------------------- */

void PairRodLJ::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style rod_lj command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   pair_coeff I J sigma eps cut
   (LJ exponent is hard-coded to m = 6, i.e. standard 12-6 LJ)
------------------------------------------------------------------------- */

void PairRodLJ::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR, "Incorrect args for pair_coeff rod_lj");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double sigma_one = utils::numeric(FLERR, arg[2], false, lmp);
  double eps_one = utils::numeric(FLERR, arg[3], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[4], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      sigma[i][j] = sigma_one;
      eps[i][j] = eps_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair_coeff rod_lj");
}

/* ---------------------------------------------------------------------- */

void PairRodLJ::init_style()
{
  if (!atom->q_flag || !atom->mu_flag || !atom->torque_flag || !atom->radius_flag)
    error->all(FLERR,
               "Pair style rod_lj requires atom attributes q, mu, torque, radius "
               "(use atom_style hybrid sphere dipole)");

  neighbor->request(this, instance_me);
}

/* ---------------------------------------------------------------------- */

double PairRodLJ::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, "All pair coeffs must be set manually for pair_style rod_lj");

  eps[j][i] = eps[i][j];
  sigma[j][i] = sigma[i][j];
  cut[j][i] = cut[i][j];

  // Extend the communication / neighbour cutoff by half the longest rod so
  // that pairs where the rod tip sticks out past the centre-to-centre cutoff
  // are not missed. The rod's length lives in atom->q, so reduce over all
  // local atoms.
  double L_local = 0.0;
  if (atom->q) {
    double *q = atom->q;
    int nlocal = atom->nlocal;
    for (int k = 0; k < nlocal; k++)
      if (q[k] > L_local) L_local = q[k];
  }
  double L_max = 0.0;
  MPI_Allreduce(&L_local, &L_max, 1, MPI_DOUBLE, MPI_MAX, world);

  return cut[i][j] + 0.5 * L_max;
}

/* ---------------------------------------------------------------------- */

void PairRodLJ::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  int *ilist, *jlist, *numneigh, **firstneigh;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  double *q = atom->q;
  double *radius = atom->radius;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    double xi[3] = {x[i][0], x[i][1], x[i][2]};
    double qi = q[i];
    double ri_sphere = radius[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      double factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];

      double qj = q[j];

      // Identify rod and point. Skip if neither is a rod or both are rods.
      int rod_idx, pt_idx;
      double L, r_rod;
      double n[3];

      if (qi > 0.0 && qj == 0.0) {
        rod_idx = i;
        pt_idx = j;
        L = qi;
        r_rod = ri_sphere;
        double inv_mag = 1.0 / mu[i][3];
        n[0] = mu[i][0] * inv_mag;
        n[1] = mu[i][1] * inv_mag;
        n[2] = mu[i][2] * inv_mag;
      } else if (qj > 0.0 && qi == 0.0) {
        rod_idx = j;
        pt_idx = i;
        L = qj;
        r_rod = radius[j];
        double inv_mag = 1.0 / mu[j][3];
        n[0] = mu[j][0] * inv_mag;
        n[1] = mu[j][1] * inv_mag;
        n[2] = mu[j][2] * inv_mag;
      } else {
        continue;    // both point or both rod
      }

      // Quick reject on centre-centre distance vs per-type cutoff + rod arm.
      double dxc = xi[0] - x[j][0];
      double dyc = xi[1] - x[j][1];
      double dzc = xi[2] - x[j][2];
      double rcc2 = dxc * dxc + dyc * dyc + dzc * dzc;
      double half_arm = 0.5 * L;
      double r_outer = cut[itype][jtype] + half_arm;
      if (rcc2 > r_outer * r_outer) continue;

      // Closest point on segment [A,B] to the point particle.
      double xr[3] = {x[rod_idx][0], x[rod_idx][1], x[rod_idx][2]};
      double xp[3] = {x[pt_idx][0], x[pt_idx][1], x[pt_idx][2]};
      double AB[3] = {L * n[0], L * n[1], L * n[2]};
      double A[3] = {xr[0] - 0.5 * AB[0], xr[1] - 0.5 * AB[1], xr[2] - 0.5 * AB[2]};
      double AP[3] = {xp[0] - A[0], xp[1] - A[1], xp[2] - A[2]};

      double AB2 = AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2];
      double t = (AP[0] * AB[0] + AP[1] * AB[1] + AP[2] * AB[2]) / (AB2 + 1.0e-30);

      double C[3];
      if (t <= 0.0) {
        C[0] = A[0];
        C[1] = A[1];
        C[2] = A[2];
      } else if (t >= 1.0) {
        C[0] = A[0] + AB[0];
        C[1] = A[1] + AB[1];
        C[2] = A[2] + AB[2];
      } else {
        C[0] = A[0] + t * AB[0];
        C[1] = A[1] + t * AB[1];
        C[2] = A[2] + t * AB[2];
      }

      double delta[3] = {C[0] - xp[0], C[1] - xp[1], C[2] - xp[2]};
      double rsq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
      if (rsq < 1.0e-30) continue;

      // Effective sigma; cutoff is the user-supplied per-pair value.
      double sigma_eff = sigma[itype][jtype] + r_rod;
      if (rsq >= cut[itype][jtype] * cut[itype][jtype]) continue;

      double r = sqrt(rsq);

      // Standard 12-6 LJ: U = 4 eps [s^12 - s^6], s = sigma_eff / r.
      // F_rod = (24 eps / r^2) * (2 s^12 - s^6) * (C - x_p)
      double eps_val = eps[itype][jtype];
      double sr2 = (sigma_eff * sigma_eff) / rsq;
      double sr6 = sr2 * sr2 * sr2;
      double sr12 = sr6 * sr6;

      double U = 4.0 * eps_val * (sr12 - sr6);
      double fmag_over_r2 = 24.0 * eps_val * (2.0 * sr12 - sr6) / rsq;

      double F[3] = {fmag_over_r2 * delta[0], fmag_over_r2 * delta[1],
                     fmag_over_r2 * delta[2]};

      // Apply factor_lj uniformly.
      F[0] *= factor_lj;
      F[1] *= factor_lj;
      F[2] *= factor_lj;
      double evdwl = factor_lj * U;

      // Lever arm from rod centre to contact point.
      double lever[3] = {C[0] - xr[0], C[1] - xr[1], C[2] - xr[2]};
      double tau[3] = {lever[1] * F[2] - lever[2] * F[1],
                       lever[2] * F[0] - lever[0] * F[2],
                       lever[0] * F[1] - lever[1] * F[0]};

      // Force on rod: +F ; force on point: -F.
      if (newton_pair || rod_idx < nlocal) {
        f[rod_idx][0] += F[0];
        f[rod_idx][1] += F[1];
        f[rod_idx][2] += F[2];
        torque[rod_idx][0] += tau[0];
        torque[rod_idx][1] += tau[1];
        torque[rod_idx][2] += tau[2];
      }
      if (newton_pair || pt_idx < nlocal) {
        f[pt_idx][0] -= F[0];
        f[pt_idx][1] -= F[1];
        f[pt_idx][2] -= F[2];
      }

      // Virial: use (delx,dely,delz) = xi - xj and the force on i.
      // F_on_i = +F when i is the rod, -F when i is the point.
      if (evflag) {
        double fx_i = (rod_idx == i) ? F[0] : -F[0];
        double fy_i = (rod_idx == i) ? F[1] : -F[1];
        double fz_i = (rod_idx == i) ? F[2] : -F[2];
        ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx_i, fy_i, fz_i,
                     dxc, dyc, dzc);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   restart / data file boilerplate
------------------------------------------------------------------------- */

void PairRodLJ::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&eps[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

void PairRodLJ::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (comm->me == 0)
        utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (comm->me == 0) {
          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &eps[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&eps[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}

void PairRodLJ::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

void PairRodLJ::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

void PairRodLJ::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g\n", i, sigma[i][i], eps[i][i], cut[i][i]);
}

void PairRodLJ::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g\n", i, j, sigma[i][j], eps[i][j], cut[i][j]);
}
