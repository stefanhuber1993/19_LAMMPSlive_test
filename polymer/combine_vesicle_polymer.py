"""
combine_vesicle_polymer.py

Combines a dipole_sphere_angle vesicle data file with an angle-style polymer data file
into a single LAMMPS data file using atom_style dipole_sphere_angle.

- Streams the vesicle file line-by-line (file is ~80 MB, not loaded fully).
- Loads the polymer file in memory (5.4 MB).
- Strips solvent/extra types from the vesicle; membrane becomes type 1.
- Polymer atoms become type 2, renumbered IDs after membrane.

Column order (OVITO angle+sphere+dipole convention):
    atom-ID type x y z mol-ID diameter density q mux muy muz

Usage:
    python combine_vesicle_polymer.py \
        --vesicle vesicle_ds_N24500_d0.85.data \
        --polymer poly_relaxed_64000.data \
        --out     input/combined_N50000_poly64000.data \
        --viz_out input/datafile_viz.data
"""

import argparse
import os
import re
import sys


def parse_header(path):
    """Read the header section of a LAMMPS data file.

    Returns a dict with integer/float fields and the byte offset where
    the header ends (i.e. where section keywords begin).
    """
    info = {}
    offset = 0
    with open(path, "r") as f:
        for line in f:
            offset += len(line.encode())
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Box bounds: "lo hi xlo xhi" etc.
            m = re.match(r"^\s*([-\d.e+]+)\s+([-\d.e+]+)\s+xlo\s+xhi", stripped)
            if m:
                info["xlo"], info["xhi"] = float(m.group(1)), float(m.group(2))
                continue
            m = re.match(r"^\s*([-\d.e+]+)\s+([-\d.e+]+)\s+ylo\s+yhi", stripped)
            if m:
                info["ylo"], info["yhi"] = float(m.group(1)), float(m.group(2))
                continue
            m = re.match(r"^\s*([-\d.e+]+)\s+([-\d.e+]+)\s+zlo\s+zhi", stripped)
            if m:
                info["zlo"], info["zhi"] = float(m.group(1)), float(m.group(2))
                continue
            # Counts: "N atoms", "N bonds", etc.
            m = re.match(r"^\s*(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)$", stripped)
            if m:
                info[m.group(2)] = int(m.group(1))
                continue
            # Type counts
            m = re.match(r"^\s*(\d+)\s+(atom|bond|angle|dihedral|improper)\s+types$", stripped)
            if m:
                info[m.group(2) + "_types"] = int(m.group(1))
                continue
            # Section keyword — header is over
            if re.match(r"^(Atoms|Velocities|Bonds|Angles|Masses)\b", stripped):
                break
    return info


def stream_membrane_atoms(vesicle_path):
    """Yield (new_id, x, y, z, mol, diameter, density, q, mux, muy, muz) for type-1
    atoms in the vesicle file, streaming line by line.

    dipole_sphere_angle Atoms format (OVITO angle+sphere+dipole convention):
        atom-ID type x y z mol-ID diameter density q mux muy muz [ix iy iz]
    """
    in_atoms = False
    new_id = 0
    with open(vesicle_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"^Atoms\b", stripped):
                in_atoms = True
                continue
            # Any new section keyword ends the Atoms block
            if in_atoms and re.match(r"^[A-Z][a-zA-Z]+\b", stripped) and not stripped[0].isdigit() and stripped != "":
                break
            if not in_atoms:
                continue
            parts = stripped.split()
            if len(parts) < 12:
                continue
            atom_type = int(parts[1])
            if atom_type != 1:
                continue  # skip non-membrane types
            # id type x y z mol-ID diameter density q mux muy muz [ix iy iz]
            x, y, z       = parts[2], parts[3], parts[4]
            mol           = parts[5]
            diameter      = parts[6]
            density       = parts[7]
            q             = parts[8]
            mux, muy, muz = parts[9], parts[10], parts[11]
            new_id += 1
            yield (new_id, x, y, z, mol, diameter, density, q, mux, muy, muz)


def load_polymer(polymer_path):
    """Load the full polymer data file (angle style) into memory.

    Returns:
        atoms:  list of (orig_id, mol, type, x, y, z)
        bonds:  list of (bond_id, bond_type, i, j)   -- orig atom IDs
        angles: list of (angle_id, angle_type, i, j, k) -- orig atom IDs
    """
    atoms, bonds, angles = [], [], []
    section = None
    with open(polymer_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"^Atoms\b", stripped):
                section = "atoms"; continue
            if re.match(r"^Bonds\b", stripped):
                section = "bonds"; continue
            if re.match(r"^Angles\b", stripped):
                section = "angles"; continue
            if re.match(r"^(Masses|Velocities|Dihedrals|Impropers)\b", stripped):
                section = None; continue

            parts = stripped.split()
            if section == "atoms" and len(parts) >= 6:
                # atom_style angle: id mol type x y z [ix iy iz]
                # No charge field in angle style.
                orig_id = int(parts[0])
                mol     = int(parts[1])
                atype   = int(parts[2])
                x, y, z = parts[3], parts[4], parts[5]
                atoms.append((orig_id, mol, atype, x, y, z))
            elif section == "bonds" and len(parts) >= 4:
                bonds.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
            elif section == "angles" and len(parts) >= 5:
                angles.append((int(parts[0]), int(parts[1]),
                               int(parts[2]), int(parts[3]), int(parts[4])))
    return atoms, bonds, angles


def _write_header(out, N_total, N_bonds, N_ang, v_info):
    out.write("LAMMPS combined vesicle+polymer data file\n\n")
    out.write(f"{N_total}  atoms\n")
    out.write(f"{N_bonds}  bonds\n")
    out.write(f"{N_ang}  angles\n")
    out.write(f"2  atom types\n")
    out.write(f"1  bond types\n")
    out.write(f"1  angle types\n")
    out.write("\n")
    out.write(f"{v_info['xlo']:.6f}  {v_info['xhi']:.6f}  xlo xhi\n")
    out.write(f"{v_info['ylo']:.6f}  {v_info['yhi']:.6f}  ylo yhi\n")
    out.write(f"{v_info['zlo']:.6f}  {v_info['zhi']:.6f}  zlo zhi\n")
    out.write("\n")
    # No Masses section: atom_style dipole_sphere_angle uses mass_type=PER_ATOM,
    # so mass is stored per-atom in rmass (computed from density*volume in data_atom_post).


def _write_bonds_angles(out, poly_bonds, poly_angles, orig_to_local, offset):
    out.write("Bonds\n\n")
    for (bid, btype, i, j) in poly_bonds:
        ni = offset + orig_to_local[i] + 1
        nj = offset + orig_to_local[j] + 1
        out.write(f"{bid} {btype} {ni} {nj}\n")
    out.write("\n")

    out.write("Angles\n\n")
    for (aid, atype, i, j, k) in poly_angles:
        ni = offset + orig_to_local[i] + 1
        nj = offset + orig_to_local[j] + 1
        nk = offset + orig_to_local[k] + 1
        out.write(f"{aid} {atype} {ni} {nj} {nk}\n")


def combine(vesicle_path, polymer_path, out_path, viz_path=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if viz_path:
        os.makedirs(os.path.dirname(viz_path) or ".", exist_ok=True)

    print(f"Loading polymer file: {polymer_path}")
    poly_atoms, poly_bonds, poly_angles = load_polymer(polymer_path)
    N_pol   = len(poly_atoms)
    N_bonds = len(poly_bonds)
    N_ang   = len(poly_angles)
    print(f"  Polymer: {N_pol} atoms, {N_bonds} bonds, {N_ang} angles")

    poly_orig_ids = [a[0] for a in poly_atoms]
    orig_to_local = {oid: i for i, oid in enumerate(poly_orig_ids)}

    print(f"Parsing vesicle header: {vesicle_path}")
    v_info = parse_header(vesicle_path)
    print(f"  Box: x=[{v_info['xlo']:.4f}, {v_info['xhi']:.4f}]  "
          f"y=[{v_info['ylo']:.4f}, {v_info['yhi']:.4f}]  "
          f"z=[{v_info['zlo']:.4f}, {v_info['zhi']:.4f}]")

    # Single pass over vesicle: collect membrane atoms into memory (~50k tuples, ~5 MB).
    print(f"Reading membrane atoms (type 1) from vesicle file...")
    mem_atoms = list(stream_membrane_atoms(vesicle_path))
    N_mem   = len(mem_atoms)
    N_total = N_mem + N_pol
    offset  = N_mem  # polymer IDs start at N_mem+1
    print(f"  Membrane atoms: {N_mem}  |  total: {N_total}")

    # ---- LAMMPS data file ----
    # Field merge order for atom_style hybrid dipole_sphere angle
    # (from atom_vec_hybrid: base{id,type,x} + dipole_sphere{q,radius,rmass,mu3} + angle{mol}):
    #   atom-ID type x y z charge diameter density mux muy muz molecule-ID
    
    
    print(f"Writing LAMMPS file: {out_path}")
    with open(out_path, "w") as out:
        _write_header(out, N_total, N_bonds, N_ang, v_info)
        # Column order (OVITO angle+sphere+dipole): id type x y z mol-ID diameter density q mux muy muz
        # fields_data_atom = {"id", "type", "x", "molecule", "radius", "rmass", "q", "mu3"}
        out.write("Atoms  # hybrid angle sphere dipole\n\n")
        for (new_id, x, y, z, mol, diameter, density, q, mux, muy, muz) in mem_atoms:
            out.write(f"{new_id} 1 {x} {y} {z} {mol} {diameter} {density} {q} {mux} {muy} {muz}\n")
        for i, (orig_id, mol, atype, x, y, z) in enumerate(poly_atoms):
            new_id = offset + i + 1
            out.write(f"{new_id} 2 {x} {y} {z} {mol} 1.0 1.0 0.0 0.0 0.0 0.0\n")
        out.write("\n")
        _write_bonds_angles(out, poly_bonds, poly_angles, orig_to_local, offset)

    # The output file already follows the OVITO angle+sphere+dipole column order,
    # so OVITO can open it directly with atom_style dipole_sphere_angle.

    print(f"Done.")
    print(f"  {N_mem} membrane + {N_pol} polymer = {N_total} atoms, {N_bonds} bonds, {N_ang} angles")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vesicle", default="vesicle_ds_N24500_d0.85.data",
                        help="Path to vesicle dipole_sphere data file")
    parser.add_argument("--polymer", default="input/poly_relaxed_64000.data",
                        help="Path to polymer angle-style data file")
    parser.add_argument("--out", default="input/combined_N50000_poly64000.data",
                        help="Output path for LAMMPS combined data file")
    parser.add_argument("--viz_out", default="input/datafile_viz.data",
                        help="Output path for OVITO visualization data file "
                             "(atom-ID type x y z diameter density q mux muy muz mol)")
    args = parser.parse_args()

    if not os.path.exists(args.vesicle):
        sys.exit(f"Error: vesicle file not found: {args.vesicle}")
    if not os.path.exists(args.polymer):
        sys.exit(f"Error: polymer file not found: {args.polymer}")

    combine(args.vesicle, args.polymer, args.out, viz_path=args.viz_out)


if __name__ == "__main__":
    main()
