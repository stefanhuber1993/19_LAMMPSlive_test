#!/bin/bash
# Three-way comparison of the same 100k MesoMem deck: 1 CPU core, all CPU cores,
# and one MIG slice of the A100. The point is not any single number -- it is
# whether the GPU column beats the CPU columns at all, which for a pair style
# with no Kokkos variant it may well not.
#
#   ./run_snellius.sh            # 100k beads
#   N=25000 ./run_snellius.sh    # smaller, faster sanity pass
#
# Assumes `lmp` is the Kokkos+CUDA build on PATH and that you are on the node
# (interactive or inside a batch script), not the login node.
set -u

N=${N:-100000}
DECK=${DECK:-in.mesomem_100k}
LMP=${LMP:-lmp}
NCORE=${NCORE:-$(nproc)}
OUT=${OUT:-results}
mkdir -p "$OUT"

echo "=== N = $N, deck = $DECK, cores = $NCORE ==="

# --- 1. one CPU core: the baseline every other number is a speedup over ------
# Compare against 0.45 us/bead/step measured on one M1 P-core with this deck.
echo "--- 1 core ---"
"$LMP" -in "$DECK" -var nbeads "$N" -log "$OUT/cpu1.log" -screen none

# --- 2. all CPU cores on the node -------------------------------------------
# The path that needs no kernel work at all. A 128- or 192-core Snellius node at
# even 60% parallel efficiency is ~55x over one core, which is the whole factor
# the 60 fps target needs -- so measure this before writing any CUDA.
echo "--- $NCORE cores (MPI) ---"
srun -n "$NCORE" "$LMP" -in "$DECK" -var nbeads "$N" \
     -log "$OUT/cpu$NCORE.log" -screen none

# --- 3. one MIG slice -------------------------------------------------------
# A MIG slice is a fraction of the A100 (a 1g profile is ~1/7 of the SMs), and
# MIG instances cannot be combined, so this measures a seventh of a GPU. Multiply
# by the slice fraction before comparing to the plan's full-A100 estimate -- and
# note the plan's 0.07 us/bead/chunk assumes a Kokkos pair kernel that does not
# exist yet.
mapfile -t mig < <(nvidia-smi -L | sed -nr "s|^.*UUID:\s*(MIG-[^)]+)\)|\1|p")
if [ ${#mig[@]} -eq 0 ]; then
  echo "no MIG instances found; using the whole visible GPU"
else
  echo "${#mig[@]} MIG instances; using ${mig[0]}"
  export CUDA_VISIBLE_DEVICES=${mig[0]}
fi

echo "--- 1 MIG slice (Kokkos) ---"
# No -pk override: mesomem/kk requests `full, newton off`, which is the
# Kokkos/CUDA default and the right list for a GPU pair kernel. Forcing
# `newton on neigh half` here would be fighting the port.
#
# `timer sync` fences before each section stamp so the Pair/Comm/Neigh
# breakdown is actually attributable -- without it, asynchronous kernel launches
# make the Pair line meaningless (see README point 1). It costs a little wall
# clock, so the honest headline number is the un-synced `Loop time`; run both.
"$LMP" -k on g 1 -sf kk -in "$DECK" -var nbeads "$N" \
       -log "$OUT/gpu.log" -screen none
"$LMP" -k on g 1 -sf kk -in "$DECK" -var nbeads "$N" -var timer_mode sync \
       -log "$OUT/gpu.sync.log" -screen none

# --- summary ----------------------------------------------------------------
echo
echo "=== summary ==="
for f in "$OUT"/*.log; do
  echo "--- $f"
  # Loop time / steps-per-second, the percentage breakdown, and the two lines
  # that decide whether the GPU run is real: which pair style actually got used,
  # and whether anything is being copied host<->device.
  grep -E "^Performance|Dangerous builds" "$f" | sort -u
  grep -iE "not supported by Kokkos|host.*device|device.*host" "$f" | head -5
  awk '/^Section \| min time/{p=1} p&&/^(Pair|Neigh|Comm|Modify|Other)/{print} /^Nlocal:/{p=0}' "$f"
  # Step 3 of the deck emits one `Loop time` per 20-step chunk. The last 100 are
  # the frame cadence: mean ms/chunk is the per-frame sim budget, and it has to
  # come in under 16.7 for 60 fps.
  grep -E "^Loop time" "$f" | tail -100 \
    | awk '{s+=$4; n++} END{if(n) printf "  frame cadence: %.2f ms/chunk over %d chunks -> %.0f fps sim-only\n", 1000*s/n, n, n/s}'
done
