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
default codec here is `q12`:

    positions   3 x 12 bits over the cell's own extent     4.5 B
    directors   2 x uint8, octahedral                      2 B
    energies    1 x uint8 over the render style's range    1 B (only when the
                                                               energy colouring
                                                               is switched on)

so 6.5 B/bead, 3.9 MB/s at 10k and 19.5 MB/s at 50k -- against 10 B/bead for the
older `q16`, which is still selectable and spends 16 bits per position axis and
per director component.

WHY THOSE BIT COUNTS, AND HOW FAR THEY WERE PUSHED. Both were chosen against the
picture, not against a round number, on the 50k playground's 64-sigma cell:

    positions   12 bits/axis = 0.016 sigma per step, which is 0.20 px in the
                windowed viewport and 0.33 px fullscreen. Invisible. 10 bits
                would be tidier -- three of them pack into a uint32 with no bit
                fiddling at all -- but the step is then 0.8 px windowed and 1.3
                fullscreen, and a bead that is briefly almost still visibly
                snaps between grid points. So: 12, packed in pairs (see pack12).
    directors    8 bits/component = 0.94 degrees worst case, 0.34 mean. The
                previous 16 bits were kept on the argument that the client
                MEASURES nematic_S from these and an order parameter should not
                carry a codec's error. That argument was three orders of
                magnitude too cautious: measured over 200k directors, 8-bit
                coding moves S by 8e-5 -- the fourth decimal of a number that is
                read off a plot. The angular error is random and zero-mean, and
                S is a second moment over the whole population, so the error
                averages out of it rather than accumulating.

`raw32` is kept for the loopback tests, where being able to assert exact equality
is worth more than the bandwidth.

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
#
# 2: the `q12` codec, and with it a manifest entry that may carry a fourth field
#    (the logical shape a packed array decodes back to). A version-1 client walks
#    the manifest with `for name, dtype, shape in ...` and would raise on the
#    longer entry rather than misread it -- but it would also silently accept a
#    q12 frame's bytes as q16 positions if the server were asked for q16-only
#    fields, so this is a refusal, not a negotiation.
VERSION = 2

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

def quantise(values, lo, hi, dtype, top=None):
    """`values` mapped linearly onto the codes 0..`top` of an unsigned int type.

    Out-of-range input is clamped rather than wrapped: a wrap would put a bead on
    the opposite side of the cell, which is the one error that is impossible to
    mistake for noise.

    `top` overrides the type's own maximum code, and is how a field narrower than
    its container is written: 12-bit position codes are produced as uint16 here
    and only become 12 bits wide in `pack12`. Both sides must be told the same
    `top`, so it is never inferred -- see `POS_TOP`.
    """
    top = int(np.iinfo(dtype).max if top is None else top)
    span = float(hi) - float(lo)
    if span <= 0.0:
        return np.zeros(np.shape(values), dtype=dtype)
    scaled = (np.asarray(values, dtype=np.float64) - float(lo)) * (top / span)
    return np.clip(np.rint(scaled), 0, top).astype(dtype)


def dequantise(codes, lo, hi, top=None):
    """Inverse of `quantise`, as float32 -- the precision the renderer uses."""
    top = int(np.iinfo(codes.dtype).max if top is None else top)
    span = (float(hi) - float(lo)) / top
    return (codes.astype(np.float32) * np.float32(span)) + np.float32(lo)


# --- 12-bit fields ------------------------------------------------------------
# Three 12-bit numbers do not fit a machine word and 4.5 bytes is not an address,
# so the packing is done over the FLAT stream of values rather than per bead: two
# consecutive 12-bit codes, whichever bead or axis they came from, share three
# bytes. That keeps both directions to a handful of vectorised shifts, and means
# the only ragged case is a stream of odd length -- one padding code, dropped on
# the way back out.
#
#     byte 0        byte 1                    byte 2
#     a[7:0]        b[3:0] | a[11:8]          b[11:4]
#
# The low nibble of the middle byte is a's top, not b's bottom, purely so the
# little-endian reading of the first two bytes IS a: it makes a hexdump legible
# when this goes wrong.
POS_BITS = 12
POS_TOP = (1 << POS_BITS) - 1


def pack12(codes):
    """(M,) uint16 codes in 0..4095 -> (ceil(M/2) * 3,) uint8, 1.5 B each."""
    v = np.ascontiguousarray(codes, dtype=np.uint16).ravel()
    if len(v) % 2:
        v = np.append(v, np.uint16(0))
    a, b = v[0::2].astype(np.uint16), v[1::2].astype(np.uint16)
    out = np.empty((len(a), 3), dtype=np.uint8)
    out[:, 0] = (a & 0xFF).astype(np.uint8)
    out[:, 1] = (((a >> 8) & 0x0F) | ((b & 0x0F) << 4)).astype(np.uint8)
    out[:, 2] = ((b >> 4) & 0xFF).astype(np.uint8)
    return out.reshape(-1)


def unpack12(raw, count):
    """Inverse of `pack12`: the first `count` codes back, as uint16."""
    packed = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) \
        else np.ascontiguousarray(raw, dtype=np.uint8)
    if len(packed) % 3:
        raise ProtocolError(f"12-bit field is {len(packed)} bytes, not a multiple of 3")
    triples = packed.reshape(-1, 3).astype(np.uint16)
    v = np.empty(2 * len(triples), dtype=np.uint16)
    v[0::2] = triples[:, 0] | ((triples[:, 1] & 0x0F) << 8)
    v[1::2] = (triples[:, 1] >> 4) | (triples[:, 2] << 4)
    if count > len(v):
        raise ProtocolError(f"12-bit field holds {len(v)} codes, not the {count} claimed")
    return v[:count]


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


def encode_positions_q12(positions, box):
    """(N, 3) positions -> a flat uint8 buffer of 3 x 12-bit codes per bead.

    The codes are laid out bead-major (x, y, z, x, y, z, ...) rather than by
    axis, because the packer pairs ADJACENT values: bead-major means the pair
    that shares three bytes is almost always two axes of one bead, so a truncated
    payload loses whole beads off the end rather than one coordinate of every
    bead in the second half of the frame.
    """
    lo, hi = quant_bounds(box)
    codes = np.empty(np.shape(positions), dtype=np.uint16)
    for axis in range(3):
        codes[:, axis] = quantise(positions[:, axis], lo[axis], hi[axis],
                                  np.uint16, top=POS_TOP)
    return pack12(codes.reshape(-1))


def decode_positions_q12(raw, box, n):
    lo, hi = quant_bounds(box)
    codes = unpack12(raw, 3 * int(n)).reshape(int(n), 3)
    out = np.empty(codes.shape, dtype=np.float32)
    for axis in range(3):
        out[:, axis] = dequantise(codes[:, axis], lo[axis], hi[axis], top=POS_TOP)
    return out


# --- directors: octahedral ----------------------------------------------------
# A unit vector has two degrees of freedom, so storing three components wastes a
# third of the bytes. The octahedral map is the standard way to spend the other
# two well: project onto the unit octahedron |x|+|y|+|z| = 1, then unfold the
# lower half outward into the unit square. It is continuous and nearly
# area-preserving, so the worst-case angular error is within a factor of ~1.2 of
# the theoretical best for the bit budget, and it is a handful of vector ops in
# either direction. `q12` spends 8 bits per component (0.94 degrees worst case,
# 0.34 mean); `q16` spends 16 (0.018 worst case). See the module docstring for
# why 8 is enough even though nematic_S is measured from these.
#
# Only the ENCODER needs telling which: the codes carry their own width in the
# manifest's dtype string, and `oct_decode` reads the scale off `codes.dtype`.

def oct_encode(directors, dtype=np.uint16):
    """(N, 3) unit vectors -> (N, 2) octahedral codes of the given uint type."""
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
    out = np.empty((len(n), 2), dtype=dtype)
    out[:, 0] = quantise(px, -1.0, 1.0, dtype)
    out[:, 1] = quantise(py, -1.0, 1.0, dtype)
    return out


def oct_decode(codes):
    """(N, 2) octahedral codes of any uint width -> (N, 3) unit vectors, f32."""
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
# Adding the plan's delta/zlib scheme later means adding a fourth name here and
# nothing else.
#
# `q12` adds one thing to the manifest entry: an optional FOURTH field, the
# logical shape the array decodes back to. A packed array's own shape no longer
# says how many beads it holds -- 12-bit codes are paired into bytes, so a buffer
# of 3M bytes could be 2M codes or 2M-1 -- and the alternative, teaching the
# decoder to read `n` off the frame header, would have made the manifest stop
# being the whole schema. It is written only where it differs, so every entry a
# q16 frame emits is exactly the three fields it always was.

CODECS = ("q12", "q16", "raw32")
DEFAULT_CODEC = "q12"


def encode_frame(state, box, energies=None, energy_range=(-6.0, 0.0),
                 codec=DEFAULT_CODEC):
    """(manifest, payload) for one frame's arrays.

    `state` is a FrameState; `energies` is the optional per-bead potential energy
    the bead colouring paints, quantised over the render style's own colour range
    because that range is exactly the information the client can display.
    """
    arrays = []
    if codec == "q12":
        n = len(state.positions)
        arrays.append(("pos", encode_positions_q12(state.positions, box), [n, 3]))
        if state.directors is not None:
            arrays.append(("dir", oct_encode(state.directors, np.uint8), None))
        if energies is not None:
            lo, hi = energy_range
            arrays.append(("pe", quantise(energies, lo, hi, np.uint8), None))
    elif codec == "q16":
        arrays.append(("pos", encode_positions_q16(state.positions, box), None))
        if state.directors is not None:
            arrays.append(("dir", oct_encode(state.directors, np.uint16), None))
        if energies is not None:
            lo, hi = energy_range
            arrays.append(("pe", quantise(energies, lo, hi, np.uint8), None))
    elif codec == "raw32":
        arrays.append(("pos", np.asarray(state.positions, dtype=np.float32), None))
        if state.directors is not None:
            arrays.append(("dir", np.asarray(state.directors, dtype=np.float32), None))
        if energies is not None:
            arrays.append(("pe", np.asarray(energies, dtype=np.float32), None))
    else:
        raise ProtocolError(f"unknown codec {codec!r} (have {', '.join(CODECS)})")

    manifest = [[name, arr.dtype.str, list(arr.shape)] + ([list(logical)] if logical
                                                          else [])
                for name, arr, logical in arrays]
    payload = b"".join(np.ascontiguousarray(arr).tobytes()
                       for _name, arr, _logical in arrays)
    return manifest, payload


def decode_frame(manifest, payload, box, energy_range=(-6.0, 0.0),
                 codec=DEFAULT_CODEC):
    """{name: array} with positions/directors/energies back in physical units."""
    raw = {}
    logical = {}
    offset = 0
    for entry in manifest:
        if not 3 <= len(entry) <= 4:
            raise ProtocolError(f"manifest entry has {len(entry)} fields, want 3 or 4")
        name, dtype_str, shape = entry[0], entry[1], entry[2]
        dtype = np.dtype(dtype_str)
        count = int(np.prod(shape)) if shape else 0
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(payload):
            raise ProtocolError(f"payload too short for array {name!r}")
        raw[name] = np.frombuffer(payload, dtype=dtype, count=count,
                                  offset=offset).reshape(shape)
        logical[name] = entry[3] if len(entry) == 4 else None
        offset += nbytes

    out = {}
    if codec == "q12":
        if "pos" in raw:
            shape = logical.get("pos")
            if not shape or len(shape) != 2 or int(shape[1]) != 3:
                raise ProtocolError(
                    f"q12 positions need an (N, 3) logical shape, got {shape!r}")
            out["positions"] = decode_positions_q12(raw["pos"], box, int(shape[0]))
        if "dir" in raw:
            out["directors"] = oct_decode(raw["dir"])
        if "pe" in raw:
            out["energies"] = dequantise(raw["pe"], *energy_range)
    elif codec == "q16":
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
    numbers in this module's docstring honest.

    Fractional for `q12`, and honestly so: its positions are 4.5 B and the odd
    half really is shared with the next bead rather than rounded away.
    """
    if codec == "q12":
        return 4.5 + (2 if with_directors else 0) + (1 if with_energies else 0)
    if codec == "q16":
        return 6 + (4 if with_directors else 0) + (1 if with_energies else 0)
    if codec == "raw32":
        return 12 + (12 if with_directors else 0) + (4 if with_energies else 0)
    raise ProtocolError(f"unknown codec {codec!r}")
