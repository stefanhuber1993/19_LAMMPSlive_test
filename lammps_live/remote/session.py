"""Getting a GPU, putting the server on it, and giving it back afterwards.

Six steps, run on a worker thread so the app keeps drawing at 60 fps throughout,
each one reported to whatever is watching (the connect panel, or a terminal):

    1. LOGIN      one SSH master connection, authenticated once
    2. DEPLOY     ship this package to the cluster
    3. PROBE      ask that machine whether it can run the server at all
    4. ALLOCATE   salloc --no-shell, then wait for the node
    5. LAUNCH     srun the server inside the allocation
    6. TUNNEL     a second SSH, ending ON the node, carrying a local port to it

WHY ONE MASTER CONNECTION. Every step above needs the cluster, and each `ssh`
would authenticate again -- which with a one-time code means typing a fresh one
five times. So the first connection is a ControlMaster and every later command
rides on it over the same socket, instantly and without a prompt.

WHY THE TUNNEL IS TWO HOPS. The obvious one-hop form,
`ssh -L <local>:<node>:<port> <login>`, ends its session on the LOGIN node: sshd
there decrypts the stream and opens a separate, plain TCP connection onward to the
compute node. So the frames cross the cluster's internal network in clear text, and
the server has to listen on the node's external interface where every other user on
that network can reach it.

Two hops instead: an SSH session that terminates ON THE COMPUTE NODE, carried
through the login node as an opaque stream it cannot read. The forward's far end is
then `localhost` as seen from the node, so the server binds 127.0.0.1 and the port
simply does not exist for the rest of the cluster. The jump hop rides the master
connection (`ProxyCommand ssh -S <ctl> -W %h:%p`), so it costs no extra
authentication -- verified on Snellius: `gcn12` does not ask for a second one-time
code. `RemoteTarget.tunnel = "forward"` keeps the one-hop form for a site where
ssh to a compute node is not permitted.

AND EVERY HOP SAYS WHAT IT NEEDS, rather than inheriting it. Both ssh commands here
disable connection multiplexing explicitly (`ControlPath=none` on the tunnel,
`ControlPersist=no` on the master), because a `~/.ssh/config` that turns it on for
the cluster -- a sensible thing for a human to have -- changes what `ssh -N -L`
does: it backgrounds the master and exits 0, and the forward goes with it. That cost
a debugging session; the long version is at the tunnel command below.

WHY `salloc --no-shell`. The obvious `ssh host salloc ... bash` holds the
allocation inside an interactive shell on a pipe, and then everything -- the node
name, the job's lifetime, the teardown -- hangs off that pipe staying alive. With
`--no-shell` the allocation is created, the job id is printed, and the command
returns: the allocation exists on its own, `squeue` says where it is, `srun
--jobid=` puts work on it and `scancel` ends it. Nothing depends on a pipe.

HOW THE PASSWORD AND THE ONE-TIME CODE GET IN. ssh will not read a secret from a
pipe -- deliberately, and no amount of arranging changes that. What it will do is
run an SSH_ASKPASS helper and use what it prints. So this writes a small helper
into a private temporary directory, points ssh at it with
SSH_ASKPASS_REQUIRE=force, and the helper asks THIS process over a unix socket.
Every prompt ssh raises therefore arrives here as text -- "Password:",
"Verification code:", a host-key fingerprint question, a key passphrase -- is shown
verbatim, and the answer is passed back. Nothing is stored, nothing is logged, and
nothing appears in a command line or an environment variable. It also means this
code does not have to know what your cluster asks for or in what order.

GIVING THE GPU BACK. Three independent backstops, because an allocated A100 that
nobody is using is the one failure with a bill attached:

    the app       cancels the job when the window closes (`shutdown`)
    the server    cancels its OWN allocation when it exits, including after
                  --exit-when-idle, so a hard-killed app costs minutes not hours
    Slurm         --time ends it regardless

Run this file directly to test the whole flow without the GUI, which is the way to
debug the SSH leg:

    python -m lammps_live.remote.session --playground mesomem_remote
"""
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import textwrap
import threading
import time
from collections import deque

from .client import FrameLink, LinkClosed

# The states, in order. `FAILED` and `DOWN` are the two resting places.
DOWN = "down"
LOGIN = "login"
DEPLOY = "deploy"
PROBE = "probe"
ALLOCATE = "allocate"
LAUNCH = "launch"
TUNNEL = "tunnel"
READY = "ready"
FAILED = "failed"
CLOSING = "closing"

_STEP_ORDER = (LOGIN, DEPLOY, PROBE, ALLOCATE, LAUNCH, TUNNEL, READY)

# What the helper script asks this process over its unix socket. Written to a file
# with a shebang because that is what SSH_ASKPASS needs: an executable.
_ASKPASS_SOURCE = '''#!{python}
"""Bridge between ssh's password prompt and the app showing it. Written at run
time by lammps_live/remote/session.py; safe to delete."""
import os, socket, sys

prompt = sys.argv[1] if len(sys.argv) > 1 else "Password:"
path = os.environ["LAMMPS_LIVE_ASKPASS_SOCK"]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(600)
try:
    s.connect(path)
    s.sendall(prompt.encode("utf-8", "replace") + b"\\n")
    answer = b""
    while not answer.endswith(b"\\n"):
        chunk = s.recv(4096)
        if not chunk:
            break
        answer += chunk
except Exception as exc:
    sys.stderr.write("askpass bridge failed: %s\\n" % exc)
    sys.exit(1)
if not answer.strip():
    sys.exit(1)          # tells ssh the prompt was cancelled
sys.stdout.write(answer.decode("utf-8", "replace").rstrip("\\n"))
'''


class SessionError(Exception):
    """A step failed, with a message worth showing the user."""


def _indented(text, width):
    """`text` wrapped to fit the report's second column, continuation lines aligned.

    The one-line-per-fact shape is what makes the report skimmable, and the error is
    the one fact that can be three hundred characters long.
    """
    lines = textwrap.wrap(str(text), 88 - width) or [""]
    return ("\n" + " " * width).join(lines)


def _same_host(a, b):
    """Whether two names denote the same machine.

    `squeue` prints bare node names, `socket.gethostname()` on the node may print
    an FQDN, and neither is wrong -- so compare what is in front of the first dot,
    case-insensitively. A missing name compares equal to anything, because "we do
    not know yet" must not read as "a different machine".
    """
    if not a or not b:
        return True
    return a.split(".")[0].lower() == b.split(".")[0].lower()


class PromptBridge:
    """The unix-socket end of the askpass helper.

    One connection per prompt: read the question, publish it, wait for the answer,
    write it back. The socket lives in a 0700 directory that only this user can
    traverse, and the answer exists only in memory.
    """

    def __init__(self, on_prompt):
        self.on_prompt = on_prompt
        self.dir = tempfile.mkdtemp(prefix="lammps-live-ask-")
        os.chmod(self.dir, stat.S_IRWXU)
        self.socket_path = os.path.join(self.dir, "ask.sock")
        self.script_path = os.path.join(self.dir, "askpass.py")
        import sys
        with open(self.script_path, "w") as fh:
            fh.write(_ASKPASS_SOURCE.format(python=sys.executable))
        os.chmod(self.script_path, stat.S_IRWXU)
        self._answer = None
        self._answered = threading.Event()
        self._cancelled = False
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.socket_path)
        self._listener.listen(4)
        self._listener.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="askpass-bridge",
                                        daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(600)
                    prompt = conn.makefile("rb").readline().decode(
                        "utf-8", "replace").strip()
                    answer = self._ask(prompt)
                    if answer is None:
                        continue          # closing without an answer cancels it
                    conn.sendall(answer.encode("utf-8") + b"\n")
                except OSError:
                    continue

    def _ask(self, prompt):
        self._answer = None
        self._answered.clear()
        self.on_prompt(prompt)
        while not self._answered.wait(0.25):
            if self._stop.is_set() or self._cancelled:
                return None
        return self._answer

    def answer(self, text):
        self._answer = text
        self._answered.set()

    def cancel(self):
        self._cancelled = True
        self._answered.set()

    def env(self):
        """The environment ssh needs to use the helper.

        SSH_ASKPASS_REQUIRE=force is the important one: without it ssh prefers a
        terminal, and the app has none it can prompt on. DISPLAY is set because
        older OpenSSH refuses to run an askpass helper without one, and is
        harmless on a version that no longer cares.
        """
        env = dict(os.environ)
        env["SSH_ASKPASS"] = self.script_path
        env["SSH_ASKPASS_REQUIRE"] = "force"
        env["LAMMPS_LIVE_ASKPASS_SOCK"] = self.socket_path
        env.setdefault("DISPLAY", ":0")
        return env

    def close(self):
        self._stop.set()
        self.cancel()
        try:
            self._listener.close()
        except OSError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)


class RemoteSession:
    """One remote demo session: allocate, launch, connect, tear down.

    Drive it from anywhere: `start()` runs the whole flow on a worker thread,
    `answer()` feeds it a login prompt's answer, `link` is the connected
    `FrameLink` once it reaches READY, and `shutdown()` gives the GPU back.
    """

    LOG_LINES = 200

    def __init__(self, target, playground_ref="mesomem_remote", on_log=None,
                 log_lines=None):
        self.target = target.resolved()
        # Which playground the server is told to build. Both ends must name the
        # same one or they would disagree about how many beads there are.
        self.playground_ref = playground_ref
        self.state = DOWN
        self.reached = DOWN                 # the furthest step this session got to
        self.detail = "not connected"
        self.error = None
        self.prompt = None                  # the question ssh is waiting on
        self.job_id = None
        self.node = None
        self.local_port = None
        self.link = None
        self.probe_report = None
        self.coeff_values = None
        self.log = deque(maxlen=log_lines or self.LOG_LINES)
        self._on_log = on_log
        self._token = secrets.token_hex(24)
        self._bridge = None
        self._master = None                 # the ControlMaster ssh process
        self._server_proc = None            # the srun holding the server
        self._control_path = None
        self._tmpdir = None
        self._thread = None
        self._cancel = threading.Event()
        self._forwarded = None               # one-hop: the -L spec to cancel
        self._tunnel_proc = None             # two-hop: the ssh holding the tunnel
        self._remote_port = None             # what the server said it is listening on
        self._server_host = None             # and which host it said it from
        self._multi_node = False             # does the allocation span more than one?
        # Teardown can be reached from two threads at once -- the worker giving up
        # on a failed step, and the app closing the window -- so it takes the lock
        # and empties each field before acting on it. Without that, the second
        # caller finds a half-cleared session and trips over a None.
        self._teardown_lock = threading.Lock()

    # ---- reporting ----------------------------------------------------------

    def _say(self, message, state=None):
        if state:
            self.state = state
            # The furthest step reached, kept separately: `state` becomes FAILED and
            # the one thing the report most needs -- which step it died on -- would
            # be gone with it.
            if state in _STEP_ORDER:
                self.reached = state
        self.detail = message
        stamped = f"{time.strftime('%H:%M:%S')} {message}"
        self.log.append(stamped)
        if self._on_log:
            self._on_log(stamped)

    def _log_line(self, line):
        """A line of output from the far side."""
        line = line.rstrip()
        if not line:
            return
        self.log.append(line)
        if self._on_log:
            self._on_log(line)

    @property
    def busy(self):
        return self.state not in (DOWN, READY, FAILED)

    @property
    def connected(self):
        return self.state == READY and self.link is not None

    def progress(self):
        """(step index, total) for a progress readout."""
        if self.state in _STEP_ORDER:
            return _STEP_ORDER.index(self.state) + 1, len(_STEP_ORDER)
        return (len(_STEP_ORDER), len(_STEP_ORDER)) if self.state == READY else (0, len(_STEP_ORDER))

    def diagnostics(self):
        """Everything known about this session, as text somebody else can read.

        This is what the panel's Copy button puts on the clipboard, and it is
        deliberately more than the panel can show: the whole log rather than the
        last fourteen lines, untruncated rather than clipped to the card's width,
        and the settings that produced it -- because the first question anyone
        asked about a failure was "which node, which port, which tunnel mode?" and
        the answer was off the bottom of the card.

        THE TOKEN IS NOT IN HERE. It is the one secret the session holds, this text
        is meant to be pasted into a chat window, and a live token plus a node name
        is enough for anyone on the cluster to take over the stream.
        """
        target = self.target
        lines = [
            f"lammps-live remote session report  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"state       {self.state}",
            # The detail line is the FAILED message repeated verbatim once a step
            # has thrown, so it is only worth a line while it says something else.
            *([] if str(self.detail).startswith("FAILED:")
              else ["doing       " + _indented(self.detail, 12)]),
            f"step        {self.reached} "
            f"({_STEP_ORDER.index(self.reached) + 1 if self.reached in _STEP_ORDER else 0}"
            f"/{len(_STEP_ORDER)})",
            "error       " + _indented(self.error or "-", 12),
            "",
            f"playground  {self.playground_ref}",
            f"login       {target.destination}",
            f"allocation  {target.partition}, {target.gpus} gpu, "
            f"{target.cpus_per_task} cores, {target.time}"
            + (f", account {target.account}" if target.account else ""),
            f"job         {self.job_id or '-'} on node {self.node or '-'}",
            f"tunnel      {target.tunnel}: 127.0.0.1:{self.local_port or target.local_port}"
            f" -> {self.node or '<node>'}:{self._remote_port or target.port}"
            f" (server binds {target.server_bind})",
            f"far side    {target.remote_dir}, env {target.env_script or '<none>'}, "
            f"python {target.python}, profile {target.profile}",
            f"deploy      {target.deploy_dir}",
            f"stream      {target.codec} at {target.fps} fps"
            + (", free-run" if target.free_run else "")
            + f", idle timeout {target.exit_when_idle:.0f}s",
            "",
            "processes   " + ", ".join(
                f"{name}={self._proc_state(proc)}" for name, proc in (
                    ("master", self._master), ("server", self._server_proc),
                    ("tunnel", self._tunnel_proc))),
            f"local port  {self._describe_local_port()}",
        ]
        if self.link is not None:
            frames, mbs = self.link.rates()
            lines += ["", f"link        {self.link.received} frames in, "
                          f"{self.link.dropped} dropped, {frames:.0f} fps, "
                          f"{mbs:.2f} MB/s, error {self.link.error or '-'}"]
        if self.probe_report:
            lines += ["", "probe report"]
            lines += ["  " + line for line in
                      (self.probe_report.get("summary") or ["<empty>"])]
        lines += ["", f"log ({len(self.log)} lines)", ""]
        lines += list(self.log)
        return "\n".join(lines)

    def save_report(self, path=None):
        """Write `diagnostics()` to a file and return its path, or None.

        Next to the clipboard rather than instead of it: a file can be attached to
        a mail, it survives the next thing that gets copied, and on a machine with
        no clipboard command at all it is the only way the report gets out.
        """
        path = path or os.path.join(
            tempfile.gettempdir(),
            f"lammps-live-remote-{time.strftime('%Y%m%d-%H%M%S')}.txt")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.diagnostics().rstrip() + "\n")
        except OSError:
            return None
        return path

    @staticmethod
    def _proc_state(proc):
        """How a child process is doing, in one word, for the report."""
        if proc is None:
            return "none"
        code = proc.poll()
        return "running" if code is None else f"exited({code})"

    def _describe_local_port(self):
        """Whether this end of the tunnel is still listening, right now.

        The one fact that splits the two failures that look identical from the
        client's side: "the tunnel died" and "the tunnel is fine, the far end is
        not answering".
        """
        port = self.local_port or self.target.local_port
        return (f"127.0.0.1:{port} accepts connections" if self._port_in_use(port)
                else f"127.0.0.1:{port} is NOT listening")

    # ---- the flow -----------------------------------------------------------

    def start(self):
        """Begin connecting. Returns immediately."""
        if self.busy:
            return
        self._cancel.clear()
        self.error = None
        self.prompt = None
        # Set here, not in the worker: a caller that checks `busy` on the next line
        # (the GUI's own frame, a test's wait loop) must see that something is
        # happening rather than race the thread's first step.
        self.state = LOGIN
        self.detail = "starting"
        self._thread = threading.Thread(target=self._run, name="remote-session",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._tmpdir = tempfile.mkdtemp(prefix="lammps-live-ssh-")
            self._control_path = os.path.join(self._tmpdir, "ctl")
            self._open_master()
            self._deploy()
            self._probe()
            self._allocate()
            self._probe_node()
            self._launch()
            self._tunnel()
            self._connect()
            self._say(f"streaming from {self.node} (job {self.job_id})", READY)
        except SessionError as exc:
            self.error = str(exc)
            self._say(f"FAILED: {exc}", FAILED)
            self._teardown()
        except Exception as exc:                      # noqa: BLE001 -- reported
            self.error = f"{type(exc).__name__}: {exc}"
            self._say(f"FAILED: {self.error}", FAILED)
            self._teardown()
        finally:
            self.prompt = None

    def answer(self, text):
        """Answer whatever ssh is currently asking."""
        if self._bridge is not None:
            self.prompt = None
            self._bridge.answer(text)

    def reopen_link(self):
        """Open a fresh link over the tunnel this session already holds.

        This is what makes switching to another playground and back cheap. The
        server keeps its simulation between clients -- it holds the run where it
        was and waits for the next one to connect (see server.serve_forever) -- so
        coming back is one socket through a tunnel that never closed, not another
        allocation and another queue wait. The GPU is still ours the whole time;
        the server's own `--exit-when-idle` is what eventually gives it back if
        nobody returns.

        Returns the new link, or None if there is nothing to come back to (the job
        ended, the tunnel died, this session never got there) -- in which case the
        session is marked DOWN with the reason, and the caller should start a new
        one.
        """
        if self.state != READY or self.local_port is None:
            return None
        try:
            self._check_still_alive()
            self.link = FrameLink.connect("127.0.0.1", self.local_port,
                                          self._token, timeout=15.0,
                                          on_notice=self._say)
        except (SessionError, LinkClosed, OSError) as exc:
            self.link = None
            self.note_link_lost(str(exc))
            return None
        self._say(f"reconnected to {self.node} (job {self.job_id})", READY)
        return self.link

    def note_link_lost(self, reason):
        """Record that the link died on its own -- the job hit its time limit, the
        node failed, the network dropped. Called by whatever notices (the connect
        panel), because the session's own steps have all completed by then and it
        has nothing left watching."""
        if self.state != READY:
            return
        self.state = DOWN
        self.detail = reason or "the link closed"
        self._say(f"link lost: {self.detail}")

    def cancel(self):
        """Give up on a connection attempt in progress."""
        self._cancel.set()
        if self._bridge is not None:
            self._bridge.cancel()

    # ---- 1. the master connection -------------------------------------------

    def _ssh_base(self):
        return ["ssh", "-S", self._control_path,
                "-o", "BatchMode=no",
                "-o", "StrictHostKeyChecking=accept-new"]

    def _open_master(self):
        self._say(f"connecting to {self.target.destination}", LOGIN)
        self._bridge = PromptBridge(self._on_prompt)
        cmd = ["ssh", "-M", "-S", self._control_path, "-N", "-T",
               # ControlPersist=no: the master lives exactly as long as this
               # process holds it, so a crashed app cannot leave an
               # authenticated connection to the cluster lying around.
               "-o", "ControlPersist=no",
               "-o", "ControlMaster=yes",
               "-o", "BatchMode=no",
               "-o", "NumberOfPasswordPrompts=3",
               "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=4",
               self.target.destination]
        self._master = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, env=self._bridge.env(),
            start_new_session=True)
        threading.Thread(target=self._drain, args=(self._master.stdout, "ssh: "),
                         daemon=True).start()
        # The master is up when it will answer a check. Nothing else tells us: the
        # process starts before it has authenticated and stays running after.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise SessionError("cancelled")
            if self._master.poll() is not None:
                raise SessionError(
                    "the SSH connection closed before it was ready -- see the log "
                    "above (a wrong password or code is the usual reason)")
            check = subprocess.run(self._ssh_base() + ["-O", "check",
                                                       self.target.destination],
                                   capture_output=True, text=True)
            if check.returncode == 0:
                self._say("logged in")
                return
            time.sleep(0.5)
        raise SessionError("timed out waiting for the SSH login")

    def _on_prompt(self, prompt):
        self.prompt = prompt
        self._say(f"waiting for: {prompt}")

    def _drain(self, stream, prefix=""):
        for line in iter(stream.readline, ""):
            self._log_line(prefix + line.rstrip())

    # ---- running things on the far side --------------------------------------

    def _login_shell(self, command):
        """`command`, wrapped so it survives the trip to the far side's shell.

        THE TRAP THAT COST A REAL DEBUG ROUND: `ssh host bash -lc 'a && b'` does NOT
        pass an argv array. ssh joins all of its arguments with spaces and hands the
        single resulting string to the remote login shell, which then parses it
        itself. So that invocation arrives as

            bash -lc a && b

        and since `-c` takes exactly ONE argument, the far side runs `a` with `b` as
        a positional parameter -- or, in the case that found this, ran a bare `mkdir`
        with its `-p` as $0 and reported "missing operand". Quoting the command into
        a single shell word is the fix; there is no argv-preserving mode to switch
        on. A login shell (`-l`) is still wanted, because that is what puts the
        cluster's module system and `env.sh`'s dependencies on PATH.
        """
        return "bash -lc " + shlex.quote(command)

    def _remote(self, command, timeout=120, check=True, login_shell=True):
        """Run one command over the master connection and return it completed."""
        if self._cancel.is_set():
            raise SessionError("cancelled")
        argv = self._ssh_base() + [self.target.destination]
        argv += [self._login_shell(command)] if login_shell else [command]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SessionError(f"remote command timed out after {timeout}s: "
                               f"{command[:80]}") from None
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise SessionError(f"remote command failed ({proc.returncode}): "
                               + (" / ".join(detail[-3:]) or "no output")
                               + f"  [{command[:70]}]")
        return proc

    def _env_prefix(self):
        """`cd` into the build and source its environment, in one string.

        Sourced in a login shell so the cluster's module system is available, which
        is what `env.sh` almost certainly needs.
        """
        parts = [f"cd {self.target.remote_dir}"]
        if self.target.env_script:
            parts.append(f"source {self.target.env_script}")
        return " && ".join(parts)

    # ---- 2. deploy -----------------------------------------------------------

    def _deploy(self):
        """Ship this package to the cluster.

        Sent as a tar over the master connection rather than installed: no pip, no
        virtualenv, nothing on the far side modified. The server runs with
        PYTHONPATH pointing at the unpacked directory. Compiled artifacts are
        excluded -- a macOS .dylib is no use there, and the cluster's pair style is
        in its own LAMMPS build anyway.
        """
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent = os.path.dirname(package_dir)
        name = os.path.basename(package_dir)
        self._say(f"shipping {name} to {self.target.deploy_dir}", DEPLOY)
        tar = subprocess.Popen(
            ["tar", "czf", "-", "-C", parent,
             "--exclude", "__pycache__", "--exclude", "*.pyc",
             "--exclude", "*.so", "--exclude", "*.dylib", "--exclude", "*.o",
             name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        unpack = (f"mkdir -p {self.target.deploy_dir} && "
                  f"tar xzf - -C {self.target.deploy_dir}")
        remote = subprocess.Popen(
            self._ssh_base() + [self.target.destination,
                                self._login_shell(unpack)],
            stdin=tar.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        tar.stdout.close()
        out, _ = remote.communicate(timeout=300)
        tar.wait(timeout=10)
        if remote.returncode != 0:
            raise SessionError("could not unpack the package on the cluster: "
                               + ((out or "").strip()[-300:] or "no output")
                               + f"  [{unpack}]")
        self._deploy_playground_file()
        self._say("package in place")

    def _deploy_playground_file(self):
        """Ship a playground given as a PATH, and point the server at the copy.

        A bundled key needs nothing -- it came over with the package. But
        `--playground ./my_idea.py` is the whole point of the registry accepting a
        path, and it would otherwise fail on the far side with a file-not-found for
        a directory that only exists on this laptop.
        """
        ref = self.playground_ref
        if not (os.sep in ref or ref.endswith(".py")):
            return
        local = os.path.abspath(os.path.expanduser(ref))
        if not os.path.isfile(local):
            raise SessionError(f"no playground file at {local}")
        remote_path = f"{self.target.deploy_dir}/{os.path.basename(local)}"
        with open(local, "rb") as fh:
            proc = subprocess.run(
                self._ssh_base() + [self.target.destination,
                                    self._login_shell(f"cat > {remote_path}")],
                stdin=fh, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise SessionError("could not copy the playground file: "
                               + (proc.stderr or "").strip()[-200:])
        self.playground_ref = remote_path
        self._log_line(f"playground file -> {remote_path}")

    # ---- 3. probe ------------------------------------------------------------

    def _probe_command(self, level):
        return (f"{self._env_prefix()} && PYTHONPATH={self.target.deploy_dir} "
                f"{self.target.python} -m lammps_live.remote.probe --json "
                f"--profile {self.target.profile} --level {level}")

    def _probe(self):
        """Ask the LOGIN node what a login node can answer.

        Which is less than it first appears, and the difference cost a debugging
        round. The interpreter, numpy, and whether the `lammps` module and its
        shared library are where they should be: all of that is checkable here, for
        free, before any queue time is spent. But OPENING that library is not. A
        Kokkos/CUDA liblammps links against `libcuda.so.1` -- the NVIDIA driver --
        which is installed on GPU nodes and nowhere else, so the load fails on a
        login node with an error that looks exactly like a broken build and is not
        one. The real build check therefore runs on the node, inside the allocation
        (see `_probe_node`).
        """
        self._say("checking python and numpy on the login node", PROBE)
        proc = self._remote(self._probe_command("light"), timeout=600, check=False)
        report = None
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                try:
                    report = json.loads(line)
                except ValueError:
                    pass
        if report is None:
            raise SessionError(
                "the probe produced no report -- is python and the LAMMPS module "
                "reachable after sourcing " + str(self.target.env_script) + "? "
                + ((proc.stderr or proc.stdout).strip().splitlines() or [""])[-1])
        self.probe_report = report
        for line in report.get("summary") or []:
            self._log_line("probe: " + line)
        if not report.get("ok"):
            raise SessionError("the far side cannot run the server -- see the probe "
                               "lines above")
        self._say("python and numpy are fine")

    def _probe_node(self):
        """The real build check, on the GPU node, inside the allocation.

        This is where the library is actually opened, which is the only place it
        can be. It also answers the one question the server NEEDS rather than
        merely likes: how many values this build's `pair_coeff` takes, which
        decides the commands every slider will issue.

        A failure here is fatal but cheap in the only sense that matters: the
        allocation is already ours, so it is released on the way out.
        """
        self._say(f"checking the build on {self.node}", LAUNCH)
        # login_shell=False because the srun flags must reach the remote shell as
        # plain words; the probe command inside them is quoted by _login_shell.
        proc = self._remote(
            f"srun --jobid={self.job_id} --ntasks=1 --gpus={self.target.gpus} "
            f"--cpus-per-task={self.target.cpus_per_task} "
            + self._login_shell(self._probe_command("full")),
            timeout=900, check=False, login_shell=False)
        report = None
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.startswith("{"):
                try:
                    report = json.loads(line)
                except ValueError:
                    pass
        if report is None:
            raise SessionError(
                "the build check on the node produced no report: "
                + ((proc.stderr or proc.stdout).strip().splitlines() or [""])[-1])
        self.probe_report = report
        for line in report.get("summary") or []:
            self._log_line("node: " + line)
        if not report.get("ok"):
            raise SessionError("this build cannot run the server on the node -- see "
                               "the lines above")
        self.coeff_values = (report.get("coeff") or {}).get("values")
        self._say("build checks out on the node")

    # ---- 4. allocate ---------------------------------------------------------

    def _allocate(self):
        target = self.target
        self._say(f"asking for {target.gpus} GPU on {target.partition} "
                  f"for {target.time}", ALLOCATE)
        proc = self._remote(" ".join(target.salloc_args()), timeout=120,
                            check=False)
        text = proc.stdout + proc.stderr
        for line in text.splitlines():
            if line.strip():
                self._log_line("salloc: " + line.strip())
        match = re.search(r"Granted job allocation (\d+)", text)
        if not match:
            raise SessionError("salloc did not grant an allocation: "
                               + (text.strip().splitlines() or ["no output"])[-1])
        self.job_id = match.group(1)
        self._say(f"job {self.job_id} allocated; waiting for the node")
        # A queued job is normal and can take a while; report the reason so the
        # wait is legible rather than a spinner.
        deadline = time.monotonic() + 1800
        last = None
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise SessionError("cancelled")
            # Explicitly separated fields: the padded --Format output is
            # ambiguous when a pending job has no node list to print.
            proc = self._remote(f"squeue -h -j {self.job_id} -o '%T|%N|%r'",
                                timeout=60, check=False)
            line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            if not line:
                raise SessionError(f"job {self.job_id} is no longer in the queue")
            fields = (line.split("|") + ["", "", ""])[:3]
            state, node, reason = (f.strip() for f in fields)
            if state == "RUNNING" and node and not node.startswith("("):
                self.node = self._single_hostname(node)
                self._say(f"got {self.node}")
                return
            status = f"queued: {state} {reason}".strip()
            if status != last:
                self._say(status)
                last = status
            time.sleep(3.0)
        raise SessionError("the allocation never started -- the queue is busy; "
                           "the job has been cancelled")

    def _single_hostname(self, nodelist):
        """One real hostname from what squeue printed.

        A one-node allocation prints a plain name, which is the whole story 99% of
        the time. But Slurm's `%N` is a nodelist, and a nodelist can be a bracketed
        range -- which is not something `ssh` can connect to, and would fail with a
        name-resolution error that says nothing about why. `scontrol show hostnames`
        is the canonical expansion, so ask it rather than guess.
        """
        if "[" not in nodelist:
            return nodelist
        proc = self._remote(f"scontrol show hostnames {nodelist}", timeout=60,
                            check=False)
        names = [n.strip() for n in proc.stdout.split() if n.strip()]
        if not names:
            raise SessionError(f"could not expand the node list {nodelist!r}")
        if len(names) > 1:
            self._multi_node = True
            self._log_line(f"allocation spans {len(names)} nodes; the server runs "
                           f"on whichever one srun picks")
        return names[0]

    # ---- 5. launch -----------------------------------------------------------

    def _server_command(self):
        target = self.target
        server = (
            f"{target.python} -u -m lammps_live.remote.server "
            f"--playground {self.playground_ref} "
            f"--profile {target.profile} "
            f"--port {target.port} --bind {target.server_bind} --token-stdin "
            f"--fps {target.fps} --codec {target.codec} "
            f"--exit-when-idle {target.exit_when_idle}"
        )
        if target.free_run:
            server += " --free-run"
        if self.coeff_values:
            server += f" --coeff-values {self.coeff_values}"
        # The server cancels its own allocation on the way out, whatever took it
        # out: a finished session, --exit-when-idle after an abandoned one, or a
        # crash. That is the backstop that does not depend on this laptop still
        # being alive to run scancel.
        return (f"{self._env_prefix()} && "
                f"PYTHONPATH={target.deploy_dir} {server}; "
                f"rc=$?; echo \"[server] exited with $rc\"; "
                f"scancel $SLURM_JOB_ID; exit $rc")

    def _launch(self):
        target = self.target
        self._say(f"starting the server on {self.node}", LAUNCH)
        argv = self._ssh_base() + [
            target.destination,
            "srun", f"--jobid={self.job_id}", "--unbuffered",
            f"--ntasks={target.ntasks}", f"--gpus={target.gpus}",
            f"--cpus-per-task={target.cpus_per_task}",
            # The srun flags are plain words and survive being joined; the command
            # itself has to be one quoted word (see _login_shell).
            self._login_shell(self._server_command())]
        self._server_proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        # The token goes down stdin, which srun forwards to the task, so it never
        # appears in a command line that the rest of the cluster can read.
        self._server_proc.stdin.write(self._token + "\n")
        self._server_proc.stdin.flush()

        listening = threading.Event()
        port_seen = {}

        def watch():
            for line in iter(self._server_proc.stdout.readline, ""):
                self._log_line(line.rstrip())
                match = re.search(r"LISTENING host=(\S+) port=(\d+)", line)
                if match:
                    port_seen["host"] = match.group(1)
                    port_seen["port"] = int(match.group(2))
                    listening.set()

        threading.Thread(target=watch, name="server-log", daemon=True).start()
        # Ten minutes is generous on purpose: it covers LAMMPS starting, Kokkos
        # initialising the GPU and `create_atoms random` placing 10,000 particles
        # with overlap rejection. But the srun is watched while we wait, because the
        # common failures here (a step that cannot be created, a module that is not
        # loaded, a syntax error in env.sh) exit in seconds and there is no reason to
        # sit out the full timeout for them.
        deadline = time.monotonic() + 600
        while not listening.wait(1.0):
            if self._cancel.is_set():
                raise SessionError("cancelled")
            if self._server_proc.poll() is not None:
                raise SessionError(
                    f"the server exited (status {self._server_proc.returncode}) "
                    f"before it was listening -- see the log above")
            if time.monotonic() > deadline:
                raise SessionError("the server did not come up within 10 minutes")
        self._remote_port = port_seen["port"]
        self._server_host = port_seen["host"]
        self._say(f"server listening on {self._server_host}:{self._remote_port}")
        # WHERE THE SERVER ACTUALLY IS, versus where we are about to tunnel to.
        # `squeue` gave us the allocation's node list and we took the first name;
        # `srun --ntasks=1` is free to place that one task on ANY node in the
        # allocation. When those disagree the tunnel goes to a machine where nothing
        # is listening, and in "jump" mode nothing else would ever notice, because
        # the server binds loopback on ITS node and the forward is simply refused at
        # the far end -- which reaches the user as "the server did not answer" and
        # says nothing about why.
        #
        # Only a multi-node allocation gets its route changed. A single node that
        # reports a different name is reporting a NAME, not a different machine:
        # `socket.gethostname()` there may be an FQDN, or an interface alias that
        # `ssh` cannot resolve from the login node, and preferring it over the
        # canonical Slurm name would break a route that works. So say so and keep
        # going; the mismatch is in the log and in the report either way.
        if not _same_host(self._server_host, self.node):
            if self._multi_node:
                self._say(f"the server came up on {self._server_host}, not "
                          f"{self.node}; the tunnel will go there instead")
                self.node = self._server_host
            else:
                self._log_line(f"note: the node calls itself {self._server_host}, "
                               f"Slurm calls it {self.node}; tunnelling to "
                               f"{self.node}")

    # ---- 6. tunnel and connect ----------------------------------------------

    def _tunnel(self):
        """Bring a local port to the server, by whichever route the target asks for.

        Both routes end with `self.local_port` being a port on this machine that
        reaches the server, and that is all the client knows about it.
        """
        if self.target.tunnel == "jump":
            self._tunnel_two_hop()
        else:
            self._tunnel_one_hop()

    def _free_local_ports(self):
        """Candidate local ports, starting at the requested one, skipping any that
        are already listening -- a second app, or a forward left over from a session
        that did not shut down cleanly."""
        wanted = self.target.local_port
        return [p for p in range(wanted, wanted + 20) if not self._port_in_use(p)]

    def _tunnel_two_hop(self):
        """A second SSH whose session ENDS ON THE COMPUTE NODE (the default).

        `-W %h:%p` makes the jump hop a stdio pipe over the connection that is
        already authenticated, so nothing new is typed. Everything after that is an
        ordinary SSH session to the node, and `-L <local>:127.0.0.1:<port>` asks it
        to carry the port -- `127.0.0.1` here means loopback ON THE NODE, which is
        the whole point.

        Host keys go in a session-local file rather than the user's `known_hosts`.
        Compute nodes are reimaged and their keys change, so pinning them in the
        real file turns a routine node reallocation into a hard failure and a scary
        warning weeks later; and a per-session file still pins the key for the
        lifetime of the session. The MITM this gives up on is the login node itself,
        which already has the user's home directory and Slurm credentials.
        """
        self._say(f"opening the tunnel to {self.node}", TUNNEL)
        # The ProxyCommand is handed to /bin/sh by ssh, so the control path is
        # quoted: TMPDIR is not guaranteed to be free of spaces. `%h:%p` must stay
        # unquoted-looking, and does -- ssh substitutes those tokens itself before
        # the shell ever sees the string.
        proxy = (f"ssh -S {shlex.quote(self._control_path)} "
                 f"-o ControlMaster=no -W %h:%p "
                 f"{self.target.destination}")
        known_hosts = os.path.join(self._tmpdir, "node_known_hosts")
        destination = self.target.node_destination(self.node)
        candidates = self._free_local_ports()
        if not candidates:
            raise SessionError(
                f"no free local port near {self.target.local_port} -- another "
                f"session may still be running")
        last = ""
        for candidate in candidates:
            spec = f"{candidate}:127.0.0.1:{self._remote_port}"
            cmd = ["ssh", "-N", "-T",
                   "-o", f"ProxyCommand={proxy}",
                   # NO MULTIPLEXING ON THIS HOP. The cost of getting this wrong was
                   # a whole debugging session, so at length:
                   #
                   # A `~/.ssh/config` that sets `ControlMaster auto` and
                   # `ControlPersist` for the cluster's compute nodes -- which is a
                   # sensible thing to have, and the reason `gcn*` is in there is to
                   # make ssh-ing to your own job's node cheap -- changes what
                   # `ssh -N -L ...` MEANS. With ControlPersist and no remote
                   # command, ssh puts the master in the BACKGROUND and the process
                   # we started exits immediately, status 0. So:
                   #
                   #   - the local port is bound for a moment, so the forward looks
                   #     like it came up, and
                   #   - the process holding it is gone, and the forward with it, so
                   #     the connect a fraction of a second later is refused --
                   #     reported as "the tunnel is open but the server did not
                   #     answer", which is true and useless.
                   #
                   # `ControlPath=none` disables multiplexing for this connection
                   # outright: no master to join, no ControlPersist, nothing
                   # backgrounds itself, and `-N` means what this code assumes it
                   # means -- a process that stays in the foreground for exactly as
                   # long as the forward exists. The login master does the same
                   # thing for the same reason (see `_open_master`); this hop was
                   # missing it.
                   "-o", "ControlMaster=no",
                   "-o", "ControlPath=none",
                   # Fail loudly if the port cannot be bound, rather than sitting
                   # there with a connection that carries nothing.
                   "-o", "ExitOnForwardFailure=yes",
                   "-o", f"UserKnownHostsFile={known_hosts}",
                   "-o", "StrictHostKeyChecking=accept-new",
                   "-o", "ServerAliveInterval=30",
                   "-o", "ServerAliveCountMax=4",
                   "-L", spec, destination]
            # LAMMPS_LIVE_SSH_VERBOSE=1 puts ssh's own trace in the session log.
            # Worth having as a switch rather than always on: `-v` explains a
            # channel that could not be opened and a session the node closed --
            # the two failures that otherwise reach the user as nothing but
            # "the server did not answer" -- at the cost of fifty lines of noise
            # in front of every successful connect.
            if os.environ.get("LAMMPS_LIVE_SSH_VERBOSE"):
                cmd.insert(1, "-v")
            # The askpass bridge is passed along even though Snellius does not ask
            # again: a site where the node DOES prompt then shows its question in
            # the panel instead of hanging on a terminal that is not there.
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=self._bridge.env(),
                start_new_session=True)
            threading.Thread(target=self._drain, args=(proc.stdout, "tunnel: "),
                             daemon=True).start()
            if self._await_forward(proc, candidate):
                self._tunnel_proc = proc
                self.local_port = candidate
                if candidate != self.target.local_port:
                    self._say(f"local port {self.target.local_port} was busy; "
                              f"using {candidate}")
                self._say(f"tunnel up: 127.0.0.1:{candidate} -> "
                          f"{self.node}:{self._remote_port}")
                return
            last = f"the tunnel to {destination} exited (status {proc.returncode})"
            if proc.returncode == 0:
                # The signature of an ssh that backgrounded itself rather than one
                # that failed: a clean exit from a process that was asked to stay.
                last += (" without failing -- something is putting that connection "
                         "in the background. Check for ControlMaster / "
                         "ControlPersist settings this hop is not already "
                         "overriding")
            try:
                proc.kill()
            except OSError:
                pass
        raise SessionError(last or "could not open the tunnel -- see the log above")

    def _await_forward(self, proc, port, timeout=60.0):
        """True once `port` is listening AND `proc` is still the one holding it.

        Polling the port is the only honest signal that the forward exists: `ssh -N`
        reports success by continuing to run, and it is already running before the
        forward is set up.

        But a listening port on its own is not enough, which is the whole lesson of
        the ControlPersist trap above: an ssh that backgrounds itself binds the port
        and exits, and for a moment both "the port answers" and "the process is
        gone" are true. So the process is checked at the same time as the port, and
        an exited one fails this candidate however healthy the port looks.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise SessionError("cancelled")
            exited = proc.poll() is not None
            if self._port_in_use(port) and not exited:
                return True
            if exited:
                return False
            time.sleep(0.2)
        return False

    def _tunnel_one_hop(self):
        """`-O forward` on the login-node master: the fallback.

        Adds the forward to the connection that is already authenticated, so there
        is nothing new to type -- but the login node terminates the session and
        makes its own plain-TCP connection onward, which is why this mode also makes
        the server bind 0.0.0.0 (see RemoteTarget.server_bind) and why it is not the
        default.
        """
        self._say("opening the tunnel (one hop, via the login node)", TUNNEL)
        candidates = self._free_local_ports()
        if not candidates:
            raise SessionError(
                f"no free local port near {self.target.local_port} -- another "
                f"session may still be running")
        last_error = ""
        for candidate in candidates:
            spec = f"{candidate}:{self.node}:{self._remote_port}"
            proc = subprocess.run(
                self._ssh_base() + ["-O", "forward", "-L", spec,
                                    self.target.destination],
                capture_output=True, text=True)
            if proc.returncode == 0:
                self.local_port = candidate
                self._forwarded = spec
                if candidate != self.target.local_port:
                    self._say(f"local port {self.target.local_port} was busy; "
                              f"using {candidate}")
                return
            last_error = (proc.stderr or proc.stdout).strip()
        raise SessionError(f"could not open a local forward: {last_error}")

    @staticmethod
    def _port_in_use(port):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(("127.0.0.1", port)) == 0

    def _connect(self):
        self._say(f"connecting to 127.0.0.1:{self.local_port}")
        # The first attempt can beat the forward into place by a few milliseconds.
        last = None
        for attempt in range(10):
            if self._cancel.is_set():
                raise SessionError("cancelled")
            # Both ends of the thing we are connecting through, checked every time
            # round. Retrying nine more times against a process that has already
            # exited only delays the real message -- and one of these dying is the
            # commonest way this step fails: the server's own `scancel` on the way
            # out ends the allocation, and Slurm then kills the ssh session on the
            # node that the tunnel is made of, so the local port stops listening a
            # moment after the server goes.
            self._check_still_alive()
            try:
                self.link = FrameLink.connect("127.0.0.1", self.local_port,
                                              self._token, timeout=15.0,
                                              on_notice=self._say)
                return
            except LinkClosed as exc:
                # Each DISTINCT reason, once: ten identical lines say nothing, but a
                # refusal that turns into a handshake failure on the fourth try is
                # the whole story and used to be invisible.
                if str(exc) != str(last):
                    self._log_line(f"connect attempt {attempt + 1}: {exc}")
                last = exc
                time.sleep(0.5)
        raise SessionError(f"the tunnel is open but the server did not answer: "
                           f"{last} -- {self._connect_postmortem()}")

    def _check_still_alive(self):
        """Raise if the server or the tunnel has exited under us."""
        server = self._server_proc
        if server is not None and server.poll() is not None:
            raise SessionError(
                f"the server exited (status {server.returncode}) after saying it "
                f"was listening -- see the [server] lines in the log. Its own "
                f"`scancel` then ends the allocation, which is why the tunnel goes "
                f"down with it")
        tunnel = self._tunnel_proc
        if tunnel is not None and tunnel.poll() is not None:
            raise SessionError(
                f"the tunnel ssh exited (status {tunnel.returncode}) after opening "
                f"the forward -- see the `tunnel:` lines in the log. A job that has "
                f"ended takes the session on the node with it; "
                f"LAMMPS_LIVE_SSH_VERBOSE=1 adds ssh's own trace")

    def _connect_postmortem(self):
        """Which end broke, established while the evidence is warm.

        "The server did not answer" is three different failures wearing one
        message, and a client socket cannot tell them apart:

            nothing listening on this end   the ssh holding the forward has exited,
                                            so the connect is refused locally and
                                            the far side was never involved
            listening, refused at the far   the forward is up but the server is not
                                            on that node or not on that port
            connected, handshake failed     both ends are alive and disagree about
                                            the protocol version or the token

        The first two are indistinguishable from the outside -- both arrive as a
        failed `connect()` -- which is why this asks directly rather than guessing.
        """
        facts = [self._describe_local_port()]
        for name, proc in (("the tunnel ssh", self._tunnel_proc),
                           ("the srun holding the server", self._server_proc),
                           ("the login connection", self._master)):
            if proc is None:
                continue
            code = proc.poll()
            facts.append(f"{name} is still running" if code is None
                         else f"{name} has exited (status {code})")
        if self._server_host and self.node and not _same_host(self._server_host,
                                                              self.node):
            facts.append(f"the server reported host {self._server_host} but the "
                         f"tunnel goes to {self.node}")
        return ("; ".join(facts) + ". The log has both ends' own output; "
                "LAMMPS_LIVE_SSH_VERBOSE=1 adds ssh's trace to it.")

    # ---- teardown ------------------------------------------------------------

    def shutdown(self):
        """Give everything back. Safe to call from anywhere, more than once."""
        if self.state in (DOWN, CLOSING):
            self._teardown()
            return
        self._say("closing down", CLOSING)
        self._cancel.set()
        self._teardown()
        self.state = DOWN
        self.detail = "not connected"

    def _teardown(self):
        """Undo everything, in the order that leaves nothing stranded.

        Stop drawing, stop the job, stop the forward, close the login: cancelling
        the job first would leave the client blocked on a socket that will never
        answer again. Each field is taken and cleared under the lock before it is
        used, so two threads arriving here at once (the worker abandoning a failed
        step, the window closing) cannot trip over each other's cleanup.
        """
        with self._teardown_lock:
            link, self.link = self.link, None
            server, self._server_proc = self._server_proc, None
            tunnel, self._tunnel_proc = self._tunnel_proc, None
            job_id, self.job_id = self.job_id, None
            forwarded, self._forwarded = self._forwarded, None
            master, self._master = self._master, None
            bridge, self._bridge = self._bridge, None
            tmpdir, self._tmpdir = self._tmpdir, None
            control = self._control_path
            self.node = None
            self.local_port = None

        if link is not None:
            try:
                link.close()
            except Exception:                          # noqa: BLE001 -- best effort
                pass
        if server is not None:
            # Killing the srun ends the step; the remote shell's own `scancel`
            # then ends the allocation. The explicit scancel below is what
            # actually guarantees it -- this is just the quick way.
            try:
                server.terminate()
            except Exception:                          # noqa: BLE001
                pass
        if tunnel is not None:
            # The two-hop tunnel is a process of its own; killing it takes the
            # forward and the session on the node with it.
            try:
                tunnel.terminate()
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
            except OSError:
                pass
        if job_id and control and os.path.exists(control):
            try:
                self._remote_over(control, f"scancel {job_id}")
                self._log_line(f"scancel {job_id}")
            except Exception:                          # noqa: BLE001
                pass
        if forwarded and control and os.path.exists(control):
            self._control_op(control, ["-O", "cancel", "-L", forwarded])
        if master is not None:
            self._control_op(control, ["-O", "exit"])
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
        if bridge is not None:
            bridge.close()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
            self._control_path = None

    def _control_op(self, control, args):
        """One `ssh -O ...` control operation, ignoring its outcome -- during
        teardown there is nothing useful to do about a failure."""
        if not control:
            return
        subprocess.run(["ssh", "-S", control] + args + [self.target.destination],
                       capture_output=True, text=True)

    def _remote_over(self, control, command, timeout=60):
        """Like `_remote`, but with the control path passed in: teardown has
        already cleared the instance's copy."""
        subprocess.run(["ssh", "-S", control, self.target.destination,
                        self._login_shell(command)],
                       capture_output=True, text=True, timeout=timeout)


# --- driving it from a terminal -----------------------------------------------

def main(argv=None):
    """Run the whole flow with the prompts on the terminal.

    This is how to debug the SSH and Slurm half without the GUI in the way: it
    shows every step, asks for the password / one-time code on the terminal, and
    then streams frames and prints what it is receiving until interrupted.
    """
    import argparse
    import getpass

    from ..playground import registry

    parser = argparse.ArgumentParser(
        prog="python -m lammps_live.remote.session",
        description=main.__doc__.splitlines()[0])
    parser.add_argument("--playground", default="mesomem_remote")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to stream before tearing down")
    parser.add_argument("--play", action="store_true",
                        help="run the simulation, rather than only connecting")
    args = parser.parse_args(argv)

    playground = registry.load(args.playground)
    if playground.remote is None:
        raise SystemExit(f"{args.playground} declares no remote target")
    session = RemoteSession(playground.remote, playground_ref=args.playground,
                            on_log=lambda line: print(line, flush=True))
    session.start()
    try:
        while session.busy:
            if session.prompt:
                prompt = session.prompt
                session.prompt = None
                session.answer(getpass.getpass(prompt.rstrip() + " "))
            time.sleep(0.2)
        if session.state != READY:
            # The whole report, on disk, before the teardown in `finally` clears the
            # facts it is made of. The terminal has the log already; the file is
            # what gets attached to the mail to whoever runs the cluster.
            path = session.save_report()
            raise SystemExit(f"\nfailed: {session.error}"
                             + (f"\nreport: {path}" if path else ""))
        link = session.link
        print(f"\nconnected: {link.welcome}\n")
        if args.play:
            link.send({"t": "play"})
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline and not link.closed.is_set():
            frame = link.take_frame(timeout=1.0)
            if frame is None:
                continue
            header, payload = frame
            fps, mbs = link.rates()
            print(f"frame {header['seq']:5d}  n={header['n']}  "
                  f"t={header['sim_time']:8.2f} tau  "
                  f"{len(payload) / 1024:6.1f} kB  {fps:4.0f} f/s  "
                  f"{mbs:5.2f} MB/s  rtt {link.rtt_ms or 0:.0f} ms", flush=True)
            link.ping()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\ntearing down (this cancels the job)...")
        session.shutdown()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
