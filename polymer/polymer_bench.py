"""polymer_bench.py — Benchmark driver for vesicle-membrane + ring-polymer system.

Typical usage (GPU sweep):
    python polymer_bench.py --steps 500 --mpi_list 1 --omp_list 1 4 16 --test_gpu --hwthread --neigh full

CPU MPI scaling:
    python polymer_bench.py --steps 100 \
        --mpi_list 1 2 4 8 --omp_list 1 --hwthread
"""

import subprocess
import os
import argparse
import time
import re
import platform
import pandas as pd


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_hw_info():
    cpu = "Unknown"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu = re.sub(".*model name.*:", "", line, 1).strip()
                        break
    except Exception:
        pass

    gpu = {"name": "None", "cap": "N/A"}
    try:
        cmd = "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode().strip().split(",")
        gpu = {"name": out[0].strip(), "cap": out[1].strip()}
    except Exception:
        pass

    return cpu, gpu


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_performance(log_path):
    tps = 0.0
    if not os.path.exists(log_path):
        return 0.0
    with open(log_path, "r") as f:
        for line in f:
            if "Performance:" in line and "timesteps/s" in line:
                for p in line.split(","):
                    if "timesteps/s" in p:
                        try:
                            tps = float(p.strip().split()[0])
                        except Exception:
                            pass
    return tps


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_bench(args, n_atoms, data_file, mpi, omp, use_gpu, hw_info):
    cpu_model, gpu_info = hw_info
    sim_id  = f"N{n_atoms}_M{mpi}_O{omp}_G{int(use_gpu)}"
    sim_dir = os.path.join("bench_runs", sim_id)
    os.makedirs(sim_dir, exist_ok=True)

    if use_gpu:
        cmd = [args.lmp_bin, "-in", os.path.abspath("polymer.lmp")]
    else:
        cmd = ["mpirun", "-np", str(mpi), "--map-by", f"socket:PE={omp}"]
        if args.hwthread:
            cmd += ["--use-hwthread-cpus"]
        cmd += [args.lmp_bin, "-in", os.path.abspath("polymer.lmp")]
    cmd += ["-v", "t_run", str(args.steps)]
    cmd += ["-v", "data_file", data_file]

    if use_gpu:
        cmd += ["-k", "on", "g", "1", "t", str(omp), "-sf", "kk"]
        cmd += ["-pk", "kokkos", "newton", args.newton, "neigh", args.neigh, "comm", "device"]
    else:
        if omp > 1:
            cmd += ["-k", "on", "t", str(omp), "-sf", "kk"]

    print(f"--> Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    env["OMP_PROC_BIND"]   = "spread"
    env["OMP_PLACES"]      = "cores"

    start = time.time()
    try:
        subprocess.run(cmd, cwd=sim_dir, check=True, env=env)
        wall = time.time() - start
        tps  = parse_performance(os.path.join(sim_dir, "log.lammps"))
        return {
            "N": n_atoms, "MPI": mpi, "OMP": omp, "GPU": use_gpu,
            "TPS": tps, "Wall_Time": wall,
            "CPU_Info": cpu_model, "GPU_Name": gpu_info["name"], "GPU_Cap": gpu_info["cap"],
            "Neigh": args.neigh,
        }
    except Exception as e:
        print(f"Error in {sim_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lmp_bin",  default="lmp_kokkos_89",
                        help="LAMMPS binary name")
    parser.add_argument("--steps",    type=int, default=1000,
                        help="Number of MD timesteps per benchmark run")
    parser.add_argument("--mpi_list", type=int, nargs="+", default=[1],
                        help="List of MPI rank counts to sweep")
    parser.add_argument("--omp_list", type=int, nargs="+", default=[1],
                        help="List of OpenMP thread counts to sweep")
    parser.add_argument("--test_gpu", action="store_true",
                        help="Enable GPU (Kokkos) benchmark")
    parser.add_argument("--hwthread", action="store_true",
                        help="Pass --use-hwthread-cpus to mpirun")
    # Kokkos flags
    parser.add_argument("--neigh",    default="half")
    # Input data file
    parser.add_argument("--data_file", default="input/combined_N30420_poly64000.data",
                        help="Combined LAMMPS data file (relative to this script)")

    args = parser.parse_args()
    args.newton = "on" if args.neigh == "half" else "off"
    hw_info = get_hw_info()

    # Ensure we're running from the polymer/ directory so relative paths work.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    data_file = args.data_file
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Count total atoms from filename for labelling.
    m = re.search(r"combined_N(\d+)_poly(\d+)", data_file)
    n_total = int(m.group(1)) + int(m.group(2)) if m else 0

    print(f"\nBenchmarking: {data_file}  (~{n_total} atoms)")

    gpu_options = [True] if args.test_gpu else [False]
    results = []

    for mpi in args.mpi_list:
        for omp in args.omp_list:
            for use_gpu in gpu_options:
                res = run_bench(args, n_total, data_file, mpi, omp, use_gpu, hw_info)
                if res:
                    results.append(res)

    df = pd.DataFrame(results)
    fname = f"bench_polymer_{time.strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=False)
    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()


# python polymer_bench.py --steps 1000 --mpi_list 1 --omp_list 4 --test_gpu --hwthread --newton off --neigh full --lmp_bin lmp_kokkos_89