"""The wire between a LAMMPS running on a cluster GPU and the app drawing it.

ONE SOCKET, TWO DIRECTIONS. Server to client: one message per rendered frame,
carrying the particle state. Client to server: the same slider changes the local
app applies to its own LAMMPS, as short JSON commands. That asymmetry is the whole
design -- control is a handful of bytes a second, state is megabytes a second --
so all the care here goes into the frame encoding and none into the control path.

WHY IT IS CHEAP TO SPLIT AT ALL: every readout in `PlaygroundSystem` already hands
back an id-ordered copy of the arrays (see stepper.py for why), so a frame IS the
`FrameState` this file encodes, and the control channel already exists as plain
LAMMPS command strings. Nothing had to be invented; this file is a codec.

THE FRAME BUDGET (docs/a100-plan.md section 5). Naive float32 positions plus
directors is 24 B/bead: 14 MB/s at 10k beads and 60 fps, which is 115 Mbit/s over
an SSH tunnel -- borderline on a good link and hopeless on a hotel one. The
default codec here is `q16`:

    positions   3 x uint16 over the cell's own extent      6 B
    directors   2 x uint16, octahedral                     4 B
    energies    1 x uint8 over the render style's range    1 B (only when the
                                                               energy colouring
                                                               is switched on)

so 10 B/bead, 600 kB/s at 10k, 6 MB/s at 100k. The quantisation error is
L/65535 = 6e-4 sigma on a 37-sigma cell and 0.005 degrees on a director -- three
orders of magnitude below the thermal rattle, and far below the 1/2000 of a screen
width that one pixel is. `raw32` is kept for the loopback tests, where being able
to assert exact equality is worth more than the bandwidth.

WHAT IS DELIBERATELY NOT HERE: no compression and no temporal delta. Both are in
the plan's table (they take 10 B/bead down to ~4.3) and both need a stateful,
sequence-tracking decoder that a dropped or reordered frame invalidates. At 10k
beads the link is not the wall, so this stays stateless: every frame decodes on
its own, and a client that missed one just draws the next.
"""
import json
import socket
import struct

import numpy as np

# Bumped when a change would make an old client misread a new server's frames.
# Sent in the handshake, and refused rather than guessed at.
VERSION = 1

DEFAULT_PORT = 5723

# Message header: the JSON part's length, then the binary part's length. Both are
# needed up front so the reader can do exactly two reads and never has to scan for
# a delimiter inside binary data.
_HEADER = struct.Struct("!II")

# Padding added to the cell's extent before positions are quantised, in sigma.
# LAMMPS remaps atoms into the periodic cell on a neighbour rebuild, not on every
# step, so a coordinate can legitimately sit a fraction of a sigma outside the box
# it belongs to. Clamping those to the wall would make beads pile up on it
# visibly; widening the quantisation range instead costs 3% of the resolution.
QUANT_PAD = 1.0


class ProtocolError(Exception):
    """A malformed or truncated message, or a version mismatch."""


# --- framing ------------------------------------------------------------------

def pack(header, payload=b""):
    """One message: a JSON header, optionally followed by raw array bytes."""
    js = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(js), len(payload)) + js + bytes(payload)


def _recv_exactly(sock, n):
    """`n` bytes from a stream socket, or None at a clean end of stream.

    recv() on a stream may return fewer bytes than asked for at any point -- a
    truth that only shows up under load, which is exactly when the frames are
    biggest. Assembling into a memoryview rather than concatenating avoids a copy
    of the whole 100 kB frame per chunk received.
    """
    if n == 0:
        return b""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv_into(view[got:], n - got)
        if not chunk:
            if got == 0:
                return None
            raise ProtocolError(f"stream ended {n - got} bytes into a {n}-byte read")
        got += chunk
    return buf


def recv_message(sock):
    """(header, payload) for the next message, or (None, None) at end of stream."""
    head = _recv_exactly(sock, _HEADER.size)
    if head is None:
        return None, None
    js_len, payload_len = _HEADER.unpack(bytes(head))
    if js_len > 1 << 20 or payload_len > 1 << 30:
        raise ProtocolError(f"implausible message lengths {js_len}/{payload_len}")
    js = _recv_exactly(sock, js_len)
    if js is None:
        raise ProtocolError("stream ended inside a message header")
    payload = _recv_exactly(sock, payload_len) if payload_len else b""
    if payload is None:
        raise ProtocolError("stream ended inside a message payload")
    try:
        header = json.loads(bytes(js).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"unreadable message header: {exc}") from exc
    return header, bytes(payload)


def set_socket_options(sock):
    """Latency options every socket on this link wants.

    TCP_NODELAY because a control message is one small write and Nagle would sit
    on it for a round trip, which on a slider drag is felt directly. Keepalives
    because the far end is a batch job that can vanish (time limit, node failure,
    scancel) without ever closing the socket, and a blocked read would otherwise
    hang until the OS gives up minutes later.
    """
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


# --- quantisation -------------------------------------------------------------

def quantise(values, lo, hi, dtype):
    """`values` mapped linearly onto the full range of an unsigned integer type.

    Out-of-range input is clamped rather than wrapped: a wrap would put a bead on
    the opposite side of the cell, which is the one error that is impossible to
    mistake for noise.
    """
    info = np.iinfo(dtype)
    span = float(hi) - float(lo)
    if span <= 0.0:
        return np.zeros(np.shape(values), dtype=dtype)
    scaled = (np.asarray(values, dtype=np.float64) - float(lo)) * (info.max / span)
    return np.clip(np.rint(scaled), 0, info.max).astype(dtype)


def dequantise(codes, lo, hi):
    """Inverse of `quantise`, as float32 -- the precision the renderer uses."""
    info = np.iinfo(codes.dtype)
    span = (float(hi) - float(lo)) / info.max
    return (codes.astype(np.float32) * np.float32(span)) + np.float32(lo)


def quant_bounds(box):
    """(lo, hi) arrays the position codec spans: the cell, padded (see QUANT_PAD)."""
    lo = np.asarray(box.lo, dtype=float) - QUANT_PAD
    hi = np.asarray(box.hi, dtype=float) + QUANT_PAD
    return lo, hi


def encode_positions_q16(positions, box):
    lo, hi = quant_bounds(box)
    out = np.empty(np.shape(positions), dtype=np.uint16)
    for axis in range(3):
        out[:, axis] = quantise(positions[:, axis], lo[axis], hi[axis], np.uint16)
    return out


def decode_positions_q16(codes, box):
    lo, hi = quant_bounds(box)
    out = np.empty(codes.shape, dtype=np.float32)
    for axis in range(3):
        out[:, axis] = dequantise(codes[:, axis], lo[axis], hi[axis])
    return out


# --- directors: octahedral ----------------------------------------------------
# A unit vector has two degrees of freedom, so storing three components wastes a
# third of the bytes. The octahedral map is the standard way to spend the other
# two well: project onto the unit octahedron |x|+|y|+|z| = 1, then unfold the
# lower half outward into the unit square. It is continuous and nearly
# area-preserving, so the worst-case angular error is within a factor of ~1.2 of
# the theoretical best for the bit budget, and it is a handful of vector ops in
# either direction. At 16 bits per component that is 0.005 degrees -- kept that
# fine deliberately, because the client measures the nematic order parameter from
# these directors as well as drawing them.

def oct_encode(directors):
    """(N, 3) unit vectors -> (N, 2) uint16 octahedral codes."""
    n = np.asarray(directors, dtype=np.float64)
    denom = np.abs(n).sum(axis=1, keepdims=True)
    # A director LAMMPS never had set comes back as exactly zero; it has no
    # direction to preserve, so send it as the centre of the square.
    denom = np.where(denom < 1e-12, 1.0, denom)
    p = n / denom
    lower = p[:, 2] < 0.0
    px, py = p[:, 0].copy(), p[:, 1].copy()
    if lower.any():
        sx = np.where(px[lower] >= 0.0, 1.0, -1.0)
        sy = np.where(py[lower] >= 0.0, 1.0, -1.0)
        fx = (1.0 - np.abs(py[lower])) * sx
        fy = (1.0 - np.abs(px[lower])) * sy
        px[lower], py[lower] = fx, fy
    out = np.empty((len(n), 2), dtype=np.uint16)
    out[:, 0] = quantise(px, -1.0, 1.0, np.uint16)
    out[:, 1] = quantise(py, -1.0, 1.0, np.uint16)
    return out


def oct_decode(codes):
    """(N, 2) uint16 octahedral codes -> (N, 3) unit vectors, float32."""
    px = dequantise(codes[:, 0], -1.0, 1.0).astype(np.float64)
    py = dequantise(codes[:, 1], -1.0, 1.0).astype(np.float64)
    pz = 1.0 - np.abs(px) - np.abs(py)
    lower = pz < 0.0
    if lower.any():
        sx = np.where(px[lower] >= 0.0, 1.0, -1.0)
        sy = np.where(py[lower] >= 0.0, 1.0, -1.0)
        fx = (1.0 - np.abs(py[lower])) * sx
        fy = (1.0 - np.abs(px[lower])) * sy
        px[lower], py[lower] = fx, fy
    v = np.column_stack([px, py, pz])
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return (v / norm).astype(np.float32)


# --- the frame codecs ---------------------------------------------------------
# Each codec is a pair of functions over the same array manifest: the header
# lists (name, dtype, shape) for every array in the payload, in order, so the
# decoder needs no knowledge of which fields a particular frame chose to carry.
# Adding the plan's delta/zlib scheme later means adding a third name here and
# nothing else.

CODECS = ("q16", "raw32")


def encode_frame(state, box, energies=None, energy_range=(-6.0, 0.0), codec="q16"):
    """(manifest, payload) for one frame's arrays.

    `state` is a FrameState; `energies` is the optional per-bead potential energy
    the bead colouring paints, quantised over the render style's own colour range
    because that range is exactly the information the client can display.
    """
    arrays = []
    if codec == "q16":
        arrays.append(("pos", encode_positions_q16(state.positions, box)))
        if state.directors is not None:
            arrays.append(("dir", oct_encode(state.directors)))
        if energies is not None:
            lo, hi = energy_range
            arrays.append(("pe", quantise(energies, lo, hi, np.uint8)))
    elif codec == "raw32":
        arrays.append(("pos", np.asarray(state.positions, dtype=np.float32)))
        if state.directors is not None:
            arrays.append(("dir", np.asarray(state.directors, dtype=np.float32)))
        if energies is not None:
            arrays.append(("pe", np.asarray(energies, dtype=np.float32)))
    else:
        raise ProtocolError(f"unknown codec {codec!r} (have {', '.join(CODECS)})")

    manifest = [[name, arr.dtype.str, list(arr.shape)] for name, arr in arrays]
    payload = b"".join(np.ascontiguousarray(arr).tobytes() for _name, arr in arrays)
    return manifest, payload


def decode_frame(manifest, payload, box, energy_range=(-6.0, 0.0), codec="q16"):
    """{name: array} with positions/directors/energies back in physical units."""
    raw = {}
    offset = 0
    for name, dtype_str, shape in manifest:
        dtype = np.dtype(dtype_str)
        count = int(np.prod(shape)) if shape else 0
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(payload):
            raise ProtocolError(f"payload too short for array {name!r}")
        raw[name] = np.frombuffer(payload, dtype=dtype, count=count,
                                  offset=offset).reshape(shape)
        offset += nbytes

    out = {}
    if codec == "q16":
        if "pos" in raw:
            out["positions"] = decode_positions_q16(raw["pos"], box)
        if "dir" in raw:
            out["directors"] = oct_decode(raw["dir"])
        if "pe" in raw:
            out["energies"] = dequantise(raw["pe"], *energy_range)
    elif codec == "raw32":
        if "pos" in raw:
            out["positions"] = np.array(raw["pos"], dtype=np.float32)
        if "dir" in raw:
            out["directors"] = np.array(raw["dir"], dtype=np.float32)
        if "pe" in raw:
            out["energies"] = np.array(raw["pe"], dtype=np.float32)
    else:
        raise ProtocolError(f"unknown codec {codec!r} (have {', '.join(CODECS)})")
    return out


def bytes_per_bead(codec, with_directors=True, with_energies=False):
    """What a codec costs per bead -- for the HUD, and for the tests that keep the
    numbers in this module's docstring honest."""
    if codec == "q16":
        return 6 + (4 if with_directors else 0) + (1 if with_energies else 0)
    if codec == "raw32":
        return 12 + (12 if with_directors else 0) + (4 if with_energies else 0)
    raise ProtocolError(f"unknown codec {codec!r}")
