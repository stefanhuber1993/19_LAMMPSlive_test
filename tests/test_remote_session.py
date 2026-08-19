"""The connect flow, against a cluster that fits on one machine.

Everything session.py does except crossing a machine boundary: the login prompts
(including a one-time code as a second prompt), the tar deploy, the probe, the
allocation and its queue wait, the token going down stdin, waiting for the server
to say it is listening, the port forward, the frames, and the teardown that has to
give the GPU back. See tests/fake_cluster/.

The point of the test is the ORDER and the FAILURE MODES: those are what cost a
one-time code and a queue slot to get wrong against the real cluster.
"""
import os
import shlex
import socket
import sys
import time

import pytest

from dataclasses import replace

from lammps_live.remote import RemoteTarget, session as session_mod
from lammps_live.remote.session import FAILED, READY, RemoteSession

# tests/ is on sys.path (pytest's default import mode), and there is no
# tests/__init__.py, so this is a plain import rather than a relative one.
from fake_cluster.build import build

pytest.importorskip("lammps")

PASSWORD = "hunter2"
OTP = "424242"

PLAYGROUND_SOURCE = '''
from lammps_live.playground import Playground, random_fill
from lammps_live.remote import RemoteTarget

PLAYGROUND = Playground(
    name="fake cluster assembly",
    description="600 beads on a pretend GPU",
    force_field="mesomem",
    scenario=random_fill(n=600, box=14.0),
    mode="sim",
    observables=["nematic_S"],
    seed=99,
    remote=RemoteTarget(host="fake-cluster", profile="local"),
)
'''


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """A fake cluster on PATH, plus a target pointing at it."""
    bindir = build(str(tmp_path / "bin"))
    monkeypatch.setenv("PATH", bindir + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_SSH_LOG", str(tmp_path / "ssh.log"))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(tmp_path / "slurm.log"))
    monkeypatch.setenv("FAKE_SSH_ANSWERS", f"{PASSWORD}|{OTP}")
    # No LAMMPS_LIVE_REMOTE_* override from the developer's own shell may leak in.
    for name in list(os.environ):
        if name.startswith("LAMMPS_LIVE_REMOTE_"):
            monkeypatch.delenv(name)

    playground = tmp_path / "fake_playground.py"
    playground.write_text(PLAYGROUND_SOURCE)
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(session_mod.__file__))))
    target = RemoteTarget(
        host="fake-cluster", user="tester", profile="local",
        # `cd .` and no env script: the command the session builds then runs
        # verbatim on this machine, which is the point -- the string under test is
        # the real one.
        remote_dir=".", env_script="",
        deploy_dir=str(tmp_path / "deployed"),
        python=sys.executable,
        port=_free_port(), local_port=_free_port(),
        exit_when_idle=60.0, fps=0.0,
    )
    return {"target": target, "playground": str(playground), "tmp": tmp_path,
            "ssh_log": tmp_path / "ssh.log", "slurm_log": tmp_path / "slurm.log",
            "repo_root": repo_root}


def _drive(session, answers=(PASSWORD, OTP), timeout=180.0):
    """Run a session to completion, answering prompts as they appear."""
    pending = list(answers)
    asked = []
    session.start()
    deadline = time.monotonic() + timeout
    while session.busy and time.monotonic() < deadline:
        if session.prompt:
            asked.append(session.prompt)
            session.answer(pending.pop(0) if pending else "")
        time.sleep(0.05)
    assert not session.busy, f"session stuck in {session.state}: {session.detail}"
    return asked


@pytest.fixture
def session(cluster):
    sess = RemoteSession(cluster["target"], playground_ref=cluster["playground"])
    yield sess
    sess.shutdown()


def test_the_whole_flow_ends_in_frames(session, cluster):
    asked = _drive(session)
    assert session.state == READY, f"{session.error}\n" + "\n".join(session.log)

    # Both prompts were shown, verbatim, rather than assumed.
    assert asked == ["Password:", "Verification code:"]

    # The steps happened, in order.
    log = "\n".join(session.log)
    for phrase in ("logged in", "package in place", "python and numpy are fine",
                   "job 4242 allocated", "got localhost",
                   "build checks out on the node", "server listening",
                   "streaming from localhost"):
        assert phrase in log, f"{phrase!r} missing from:\n{log}"
    assert log.index("python and numpy are fine") < log.index("job 4242 allocated")
    assert log.index("job 4242 allocated") < log.index("build checks out on the node")
    # The queue wait is reported rather than silently spun on.
    assert "queued: PENDING" in log

    assert session.job_id == "4242"
    assert session.node == "localhost"
    assert session.local_port is not None

    # And there is a live link at the end of it.
    link = session.link
    assert link is not None and not link.closed.is_set()
    assert link.welcome["natoms"] == 600
    assert link.welcome["profile"] == "local"
    link.send({"t": "play"})
    frame = None
    deadline = time.monotonic() + 60
    while frame is None and time.monotonic() < deadline:
        frame = link.take_frame(timeout=1.0)
    assert frame is not None, "no frame came through the forward"
    header, payload = frame
    assert header["n"] == 600
    assert len(payload) == 600 * 10          # q16: 6 bytes position, 4 director


def test_the_build_is_checked_in_two_places(session, cluster):
    """The login node can answer the interpreter questions for free; only the GPU
    node can answer whether the library OPENS, because a Kokkos/CUDA liblammps links
    against the NVIDIA driver. So the probe runs twice, at two levels."""
    _drive(session)
    assert session.state == READY, session.error
    ran = cluster["ssh_log"].read_text()

    light = [l for l in ran.splitlines() if "--level light" in l]
    full = [l for l in ran.splitlines() if "--level full" in l]
    assert len(light) == 1 and len(full) == 1, ran
    # The light one is a plain login-node command; the full one runs inside the
    # allocation, which is the only place the driver exists.
    assert "srun" not in light[0]
    assert "srun --jobid=4242" in full[0]

    # And the arity the server is told comes from the node, not from the login node.
    assert session.coeff_values == 9
    assert "--coeff-values 9" in ran


def test_the_package_really_arrives(session, cluster):
    _drive(session)
    deployed = cluster["tmp"] / "deployed" / "lammps_live"
    assert (deployed / "remote" / "server.py").is_file()
    assert (deployed / "playgrounds" / "mesomem_remote.py").is_file()
    # No compiled artifacts: a macOS .dylib is useless there and the cluster has
    # the pair style in its own build.
    assert not list(deployed.rglob("*.dylib"))
    assert not list(deployed.rglob("*.so"))
    assert not list(deployed.rglob("__pycache__"))


def test_the_token_never_appears_in_a_command_line(session, cluster):
    _drive(session)
    # Everything the session ran, as the fake cluster saw it.
    seen = cluster["ssh_log"].read_text() + cluster["slurm_log"].read_text()
    assert session._token not in seen
    assert "--token-stdin" in seen
    # And it did authenticate the link, so it did arrive.
    assert session.link is not None


def test_teardown_gives_the_gpu_back(session, cluster):
    _drive(session)
    port = session.local_port
    session.shutdown()

    assert "scancel 4242" in cluster["slurm_log"].read_text()
    assert session.link is None
    assert session.job_id is None
    assert session.local_port is None
    # The forward is gone with it.
    with socket.socket() as probe:
        probe.settimeout(0.5)
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    # The master connection is closed and its private directory removed.
    assert session._master is None
    assert session._bridge is None
    assert not cluster["ssh_log"].read_text().count("unhandled")


def test_the_server_cancels_its_own_allocation_when_it_exits(session, cluster):
    """The backstop for a hard-killed app: the remote command ends with a scancel of
    its own job, so an abandoned allocation costs minutes, not the time limit."""
    command = None
    _drive(session)
    for line in cluster["ssh_log"].read_text().splitlines():
        if "remote.server" in line:
            command = line
    assert command is not None
    assert "scancel $SLURM_JOB_ID" in command
    assert "--exit-when-idle 60.0" in command


def test_a_wrong_answer_fails_with_something_readable(cluster):
    sess = RemoteSession(cluster["target"], playground_ref=cluster["playground"])
    try:
        _drive(sess, answers=("wrong", "wrong"), timeout=90)
        assert sess.state == FAILED
        assert "SSH connection closed" in sess.error
        assert "Permission denied" in "\n".join(sess.log)
        # Nothing was allocated, so there is nothing to give back.
        assert not cluster["slurm_log"].exists() or "salloc" not in \
            cluster["slurm_log"].read_text()
    finally:
        sess.shutdown()


def test_cancelling_a_prompt_does_not_hang(cluster):
    sess = RemoteSession(cluster["target"], playground_ref=cluster["playground"])
    try:
        sess.start()
        deadline = time.monotonic() + 60
        while not sess.prompt and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sess.prompt, "no prompt appeared"
        sess.cancel()
        deadline = time.monotonic() + 60
        while sess.busy and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sess.state == FAILED
    finally:
        sess.shutdown()


def test_a_busy_local_port_moves_to_the_next_one(cluster):
    """Two sessions, or a leftover forward, must not fail the connect -- it takes
    the next port instead."""
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", cluster["target"].local_port))
    blocker.listen(1)
    sess = RemoteSession(cluster["target"], playground_ref=cluster["playground"])
    try:
        _drive(sess)
        assert sess.state == READY, sess.error
        assert sess.local_port == cluster["target"].local_port + 1
        assert "was busy" in "\n".join(sess.log)
    finally:
        sess.shutdown()
        blocker.close()


def test_a_remote_command_survives_ssh_joining_its_arguments(cluster):
    """The bug the first real Snellius run found.

    ssh does not pass an argv array to the far side: it joins everything after the
    destination with single spaces and hands the one string to the remote shell.
    So a command containing `&&` has to arrive as a SINGLE word, or `bash -c` (which
    takes exactly one argument) runs only the first fragment -- which is how
    `mkdir -p <dir> && tar xzf -` became a bare `mkdir` reporting "missing operand".
    """
    sess = RemoteSession(cluster["target"])
    command = "mkdir -p ~/x && tar xzf - -C ~/x"
    wrapped = sess._login_shell(command)

    # What the remote shell does with the joined string:
    parts = shlex.split(wrapped)
    assert parts[0] == "bash" and parts[1] == "-lc"
    assert len(parts) == 3, f"the command must be one argument, got {parts}"
    assert parts[2] == command


# --- the tunnel ---------------------------------------------------------------

def test_the_default_tunnel_ends_on_the_node(session, cluster):
    """Two hops: the SSH session terminates on the compute node, so the server binds
    loopback there and the login node only ever relays bytes it cannot read."""
    _drive(session)
    assert session.state == READY, session.error
    assert session.target.tunnel == "jump"

    ssh_log = cluster["ssh_log"].read_text()
    # The forward's far end is loopback ON THE NODE, and the jump hop rides the
    # already-authenticated master rather than logging in again.
    tunnel_lines = [l for l in ssh_log.splitlines() if l.startswith("tunnel ")]
    assert len(tunnel_lines) == 1, ssh_log
    spec, _, options = tunnel_lines[0].partition(" | ")
    assert spec.endswith(f":127.0.0.1:{session.target.port}")
    assert "ProxyCommand=ssh -S" in options and "-W %h:%p" in options
    # And it refuses to be multiplexed. A `~/.ssh/config` with `ControlMaster auto`
    # + `ControlPersist` for `gcn*` turns `ssh -N -L` into a process that binds the
    # port, puts the master in the background and exits 0 -- so the forward looks
    # like it came up and is gone a moment later. This hop must not inherit that.
    assert "ControlPath=none" in options and "ControlMaster=no" in options
    # And the server was told to listen on loopback only.
    assert "--bind 127.0.0.1" in ssh_log

    # It is a process of its own, and the frames come through it.
    assert session._tunnel_proc is not None and session._tunnel_proc.poll() is None
    assert session._forwarded is None          # no -O forward was used
    session.link.send({"t": "play"})
    frame = None
    deadline = time.monotonic() + 60
    while frame is None and time.monotonic() < deadline:
        frame = session.link.take_frame(timeout=1.0)
    assert frame is not None, "no frame came through the two-hop tunnel"


def test_teardown_kills_the_tunnel_process(session):
    _drive(session)
    proc, port = session._tunnel_proc, session.local_port
    session.shutdown()
    assert proc.poll() is not None, "the tunnel ssh is still running"
    with socket.socket() as probe:
        probe.settimeout(0.5)
        assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_the_one_hop_fallback_still_works(cluster):
    """For a site that does not allow ssh to a compute node. The login node then
    terminates the session, so the server has to listen on all interfaces."""
    target = replace(cluster["target"], tunnel="forward")
    sess = RemoteSession(target, playground_ref=cluster["playground"])
    try:
        _drive(sess)
        assert sess.state == READY, sess.error
        ssh_log = cluster["ssh_log"].read_text()
        assert "control forward" in ssh_log
        assert "--bind 0.0.0.0" in ssh_log
        assert sess._tunnel_proc is None
        assert sess._forwarded is not None
        assert sess.link is not None
    finally:
        sess.shutdown()


def test_progress_reports_where_it_got_to(session):
    assert session.progress() == (0, 7)
    _drive(session)
    step, total = session.progress()
    assert (step, total) == (7, 7)


# ---- the report, and failing fast ------------------------------------------
#
# Both exist because of one real failure: "the tunnel is open but the server did
# not answer", on a card, with the explanation off the bottom of it.


def _bare_session(monkeypatch):
    for name in list(os.environ):
        if name.startswith("LAMMPS_LIVE_REMOTE_"):
            monkeypatch.delenv(name)
    return RemoteSession(RemoteTarget(host="nowhere", user="tester"),
                         playground_ref="somewhere/thing.py")


def test_the_report_carries_the_whole_log_and_never_the_token(monkeypatch):
    sess = _bare_session(monkeypatch)
    sess.state = FAILED
    sess.detail = "FAILED: the server did not answer"
    sess.error = "the tunnel is open but the server did not answer"
    sess.job_id, sess.node = "12345", "gcn12"
    sess.local_port, sess._remote_port = 5723, 5723
    for i in range(60):
        sess.log.append(f"line {i}")

    report = sess.diagnostics()
    assert sess._token not in report, "the report is meant to be pasted in public"
    # The questions that were asked about the real failure, all answerable from it.
    assert "tester@nowhere" in report
    assert "12345" in report and "gcn12" in report
    assert "jump" in report and "127.0.0.1:5723" in report
    assert "somewhere/thing.py" in report
    assert sess.error in report
    # The whole log, not the fourteen lines the card had room for.
    assert "line 0" in report and "line 59" in report


def test_a_dead_server_is_not_retried_ten_times(monkeypatch):
    """The commonest shape of this failure: the server goes, and its own scancel
    takes the allocation -- and with it the ssh session the tunnel is made of."""

    class Dead:
        returncode = 137

        def poll(self):
            return 137

    sess = _bare_session(monkeypatch)
    sess.local_port = 1              # nothing is listening there, and nothing tries
    sess._server_proc = Dead()
    with pytest.raises(session_mod.SessionError) as caught:
        sess._connect()
    assert "exited (status 137)" in str(caught.value)
    assert "scancel" in str(caught.value)


def test_a_dead_tunnel_says_so_rather_than_blaming_the_server(monkeypatch):
    class Dead:
        returncode = 255

        def poll(self):
            return 255

    sess = _bare_session(monkeypatch)
    sess.local_port = 1
    sess._tunnel_proc = Dead()
    with pytest.raises(session_mod.SessionError) as caught:
        sess._connect()
    assert "tunnel ssh exited (status 255)" in str(caught.value)


def test_the_postmortem_splits_the_two_failures_that_look_the_same(monkeypatch):
    sess = _bare_session(monkeypatch)
    sess.local_port = 1
    text = sess._connect_postmortem()
    assert "NOT listening" in text

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        sess.local_port = listener.getsockname()[1]
        assert "accepts connections" in sess._connect_postmortem()


def test_a_forward_whose_ssh_backgrounded_itself_is_not_a_tunnel(monkeypatch):
    """The ControlPersist trap, as the session sees it: for a moment the port
    answers AND the process that was asked to hold it has exited, status 0.

    Reported as a working tunnel -- which it was, in 2017 -- it becomes "the tunnel
    is open but the server did not answer" half a second later.
    """
    sess = _bare_session(monkeypatch)

    class Backgrounded:
        returncode = 0

        def poll(self):
            return 0

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert not sess._await_forward(Backgrounded(), port, timeout=2.0)


def test_a_name_is_not_a_different_machine():
    """squeue says `gcn12`, the node says `gcn12.int.snellius.surf.nl`, and
    rerouting the tunnel on that difference would break a route that works."""
    assert session_mod._same_host("gcn12", "gcn12.int.snellius.surf.nl")
    assert session_mod._same_host("GCN12", "gcn12")
    assert session_mod._same_host("gcn12", None)
    assert not session_mod._same_host("gcn12", "gcn13")
