"""
Generate a vesicle data file for atom_style dipole_sphere_angle (monolithic).

Membrane atoms are placed at icosphere-triangle centers; their dipoles point
radially outward (= face normals). The data file declares 2 atom types so a
LAMMPS input script can later add type-2 solvent/polymer atoms via create_atoms.

Atoms section column order (OVITO angle+sphere+dipole convention):
    id type x y z mol-ID diameter density q mux muy muz

Usage:
    python spherical_vesicle_ds.py --path . --N 10000 [--target_dist 0.85] [--box_factor 2.5]

--N is interpreted as a *target* membrane count; the closest icosphere
subdivision (N = 20*nu^2) is selected automatically.
"""
import os
import argparse
import numpy as np
from icosphere import icosphere
from sklearn.neighbors import BallTree
from scipy.spatial.distance import pdist


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=".")
    p.add_argument("--N", type=int, required=True,
                   help="target membrane atom count (rounded to nearest 20*nu^2)")
    p.add_argument("--target_dist", type=float, default=0.85,
                   help="target nearest-neighbor distance between membrane beads")
    p.add_argument("--box_factor", type=float, default=1.2,
                   help="box half-size = box_factor * vesicle_radius")
    p.add_argument("--diameter", type=float, default=1.0,
                   help="membrane bead diameter (per-atom; reset by LAMMPS later)")
    p.add_argument("--density", type=float, default=1.0)
    p.add_argument("--charge", type=float, default=1.0)
    return p.parse_args()


def nu_for_target(N_target):
    """Pick the icosphere subdivision nu so that 20*nu^2 ~= N_target."""
    nu = max(2, int(round(np.sqrt(N_target / 20.0))))
    return nu


def build_vesicle(nu, target_dist):
    vertices, faces = icosphere(nu)
    triangles = vertices[faces]

    # outward-pointing face normals (will be the dipole direction)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]])
    face_normals /= np.sqrt(np.sum(face_normals**2, axis=1, keepdims=True))

    # center-of-mass of each triangle = bead position
    cm = (triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]) / 3.0

    # rescale so mean nearest-neighbor distance matches target
    tree = BallTree(cm)
    dist_neighbours, _ = tree.query(cm, k=6)
    mean_dist = np.mean(dist_neighbours)
    scale = target_dist / mean_dist
    positions = cm * scale

    radius = np.max(pdist(positions, metric='euclidean')) / 2.0
    return positions, face_normals, radius


def write_data(outfile, positions, mu, radius, box_half,
               diameter, density, charge, n_types=2):
    n = positions.shape[0]
    with open(outfile, "w") as f:
        f.write(f"LAMMPS data file for atom_style dipole_sphere_angle "
                f"(vesicle radius {radius:.3f})\n\n")
        f.write(f"{n} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"{-box_half:.6f} {box_half:.6f} xlo xhi\n")
        f.write(f"{-box_half:.6f} {box_half:.6f} ylo yhi\n")
        f.write(f"{-box_half:.6f} {box_half:.6f} zlo zhi\n\n")
        # OVITO angle+sphere+dipole: id type x y z mol-ID diameter density q mux muy muz
        # mol-ID = 0 for membrane atoms (no molecule topology)
        f.write("Atoms # hybrid angle sphere dipole\n\n")
        for i in range(n):
            f.write(
                f"{i+1} 1 "
                f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f} "
                f"0 "
                f"{diameter:.6f} {density:.6f} "
                f"{charge:.6f} "
                f"{mu[i,0]:.6f} {mu[i,1]:.6f} {mu[i,2]:.6f}\n"
            )
    print(f"Wrote {n} membrane atoms to {outfile}")


def main():
    args = parse_arguments()

    nu = nu_for_target(args.N)
    n_actual = 20 * nu * nu
    print(f"Target N={args.N} -> nu={nu} (actual N={n_actual})")

    positions, face_normals, radius = build_vesicle(nu, args.target_dist)
    print(f"Vesicle radius: {radius:.3f}")

    box_half = args.box_factor * radius
    print(f"Box half-size: {box_half:.3f}  (box volume = {(2*box_half)**3:.0f})")

    outfile = os.path.join(
        args.path, f"vesicle_ds_N{n_actual}_d{args.target_dist:.2f}.data")
    write_data(outfile, positions, face_normals, radius, box_half,
               args.diameter, args.density, args.charge)

    # also emit the metadata the LAMMPS script needs
    info = os.path.join(args.path, f"vesicle_ds_N{n_actual}_d{args.target_dist:.2f}.info")
    with open(info, "w") as f:
        f.write(f"N_membrane {n_actual}\n")
        f.write(f"radius {radius:.6f}\n")
        f.write(f"box_half {box_half:.6f}\n")
    print(f"Wrote metadata to {info}")


if __name__ == "__main__":
    main()
