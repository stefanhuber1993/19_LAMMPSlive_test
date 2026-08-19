"""Running the simulation on a cluster GPU and drawing it here.

Four pieces, each usable on its own:

    target.py     where it runs: the SSH login, the Slurm request, the tunnel
    protocol.py   the wire: message framing and the frame codec
    hosts.py      what differs about a LAMMPS that is not the local pip wheel
    probe.py      one command that answers "will the server run on this node?"
    server.py     headless: builds the playground, integrates, sends frames
    client.py     RemoteSystem -- an MDSystem3D whose frames arrive by socket
    session.py    SSH + Slurm: allocate a GPU, tunnel to it, tear it down

The split the whole thing rests on is described in docs/a100-plan.md section 2:
the server integrates and the client measures and draws. Nothing about the app
above `MDSystem3D` knows which of the two it is talking to.
"""

from .target import RemoteTarget

__all__ = ["RemoteTarget"]
