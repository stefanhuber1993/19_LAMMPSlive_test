"""A cluster that fits on one machine, for testing remote/session.py.

The SSH and Slurm half of the remote demo is the half that cannot be exercised
against the real thing casually -- every attempt costs a one-time code and a queue
wait. So this stands in for it: a fake `ssh` that understands the handful of
invocations session.py actually makes, plus fake `salloc`, `squeue` and `scancel`.

It is a stand-in, not a mock. The commands the session builds are RUN, not
inspected: the tar is really unpacked, the probe really runs, the server really
starts, the token really arrives on stdin, and the forwarded port really carries
frames. What is faked is only the machine boundary.
"""
