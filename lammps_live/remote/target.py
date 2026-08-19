"""Where a remote playground runs: one machine, one queue, one tunnel.

Written as a declaration on the playground (`remote=RemoteTarget(...)`) rather
than as flags on the command line, because it is a property of the demo -- "this
one runs on the A100" -- and because the connect flow has to be driveable from a
button, with nothing to type but the login prompt's answer.

Every field can be overridden from the environment (LAMMPS_LIVE_REMOTE_USER,
_HOST, _TIME, _PARTITION, _PORT, ...), which is what makes the same playground
file work for a second person with a different account and a different scratch
path without editing it.
"""
import os
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class RemoteTarget:
    """An SSH login, a Slurm allocation, and where the code lives on the far side."""

    # --- the login ------------------------------------------------------------
    host: str = "snellius.surf.nl"
    user: str = ""                     # "" -> whatever ~/.ssh/config resolves
    label: str = "remote GPU"
    # --- the allocation -------------------------------------------------------
    partition: str = "gpu_a100"
    gpus: int = 1
    ntasks: int = 1
    cpus_per_task: int = 18
    # Wall clock for the allocation. This is the backstop that releases the GPU if
    # everything else fails to -- a crashed app, a lost network, a closed lid --
    # so it is deliberately not generous.
    time: str = "01:00:00"
    account: str = ""
    job_name: str = "mesomem-live"
    extra_salloc: tuple = ()
    # --- the far side ---------------------------------------------------------
    # Where the cluster's own LAMMPS build lives, and the script that puts it on
    # PATH. Sourced before the server starts, in a login shell.
    remote_dir: str = "~/Projects/MesoMemLive/mesomem_gpu"
    env_script: str = "_build/hpc/env.sh"
    # Where the connect flow unpacks this package. Not on PATH, not installed:
    # PYTHONPATH points at it, so nothing on the cluster is modified.
    deploy_dir: str = "~/.lammps_live_remote"
    python: str = "python3"
    profile: str = "cluster-gpu"       # a hosts.HostProfile name
    # --- the link -------------------------------------------------------------
    port: int = 5723                   # the server's listening port, on the node
    local_port: int = 5723             # this end of the tunnel
    # HOW THE LOCAL PORT REACHES THE SERVER. Two ways, and the difference is where
    # the SSH session ENDS:
    #
    #   "jump"     (default) a second SSH whose session terminates ON THE COMPUTE
    #              NODE, carried through the login node as an opaque stream. The
    #              login node relays bytes it cannot read, and the forward's far end
    #              is loopback on the node -- so the server binds 127.0.0.1 and the
    #              port is unreachable from the rest of the cluster.
    #   "forward"  one hop: `ssh -O forward -L <local>:<node>:<port>` added to the
    #              login-node connection. The login node terminates the session,
    #              decrypts, and opens a separate PLAIN TCP connection onward to the
    #              node -- so the server has to bind 0.0.0.0 and the port is
    #              reachable (token-protected) from the cluster's internal network.
    #
    # "jump" is better on both counts and is the default. "forward" is kept because
    # ssh-to-a-compute-node is a site policy: it needs sshd running on the node and
    # a PAM stack that lets the owner of a running job in (`pam_slurm_adopt`). Where
    # that is not allowed, one hop is the only way through.
    tunnel: str = "jump"               # "jump" | "forward"
    codec: str = "q16"
    fps: float = 60.0
    free_run: bool = False
    # Idle timeout handed to the server, so an abandoned allocation ends itself
    # even if the teardown never runs. Slurm's --time is the outer backstop; this
    # is the one that gives the GPU back in minutes rather than an hour.
    exit_when_idle: float = 900.0

    def resolved(self):
        """A copy with LAMMPS_LIVE_REMOTE_* environment overrides applied."""
        overrides = {}
        for f in fields(self):
            raw = os.environ.get(f"LAMMPS_LIVE_REMOTE_{f.name.upper()}")
            if raw is None:
                continue
            if f.type is int or isinstance(getattr(self, f.name), int):
                overrides[f.name] = int(raw)
            elif isinstance(getattr(self, f.name), float):
                overrides[f.name] = float(raw)
            elif isinstance(getattr(self, f.name), tuple):
                overrides[f.name] = tuple(raw.split())
            else:
                overrides[f.name] = raw
        return replace(self, **overrides) if overrides else self

    @property
    def destination(self):
        return f"{self.user}@{self.host}" if self.user else self.host

    def node_destination(self, node):
        """The compute node as an SSH destination, for the two-hop tunnel."""
        return f"{self.user}@{node}" if self.user else node

    @property
    def server_bind(self):
        """Which interface the server listens on -- a consequence of the tunnel
        mode, never a separate choice. Two hops end on the node, so loopback is
        reachable and nothing else needs to be; one hop arrives from the login node,
        so loopback would be unreachable."""
        return "127.0.0.1" if self.tunnel == "jump" else "0.0.0.0"

    def salloc_args(self):
        """The allocation request, as argv. `--no-shell` is the whole trick: the
        allocation is created and the command returns, so it does not have to be
        held open by an interactive shell on a pipe -- which is what made the
        obvious `ssh host salloc ... bash` approach so fragile."""
        args = ["salloc", "--no-shell",
                f"--partition={self.partition}",
                f"--gpus={self.gpus}",
                f"--ntasks={self.ntasks}",
                f"--cpus-per-task={self.cpus_per_task}",
                f"--time={self.time}",
                f"--job-name={self.job_name}"]
        if self.account:
            args.append(f"--account={self.account}")
        args.extend(self.extra_salloc)
        return args
