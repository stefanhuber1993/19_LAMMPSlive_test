"""The wire codec: does a frame survive the round trip, and to what precision.

The precision assertions here are the ones the protocol docstring claims. They
are what makes it safe to quantise at all -- if the error ever grows past the
thermal rattle, the demo would be showing a smoother simulation than it is
running, which is the one failure mode that would not look like a bug.
"""
import numpy as np
import pytest

from lammps_live.playground.state import Box, FrameState, normalize_rows
from lammps_live.remote import protocol as proto


def _angle_error_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    cos = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _state(n=500, seed=3):
    rng = np.random.default_rng(seed)
    box = Box.cube(37.44, (True, True, True))
    pos = rng.uniform(low=box.lo, high=box.hi, size=(n, 3))
    dirs = normalize_rows(rng.normal(size=(n, 3)))
    return FrameState(positions=pos, directors=dirs,
                      ids=np.arange(1, n + 1), box=box), box


def test_message_framing_round_trip():
    header = {"t": "frame", "seq": 7}
    payload = np.arange(64, dtype=np.uint16).tobytes()
    blob = proto.pack(header, payload)

    class FakeSock:
        def __init__(self, data):
            self.data, self.pos = data, 0

        def recv_into(self, view, n):
            chunk = self.data[self.pos:self.pos + min(n, 7)]   # short reads
            view[:len(chunk)] = chunk
            self.pos += len(chunk)
            return len(chunk)

    got_header, got_payload = proto.recv_message(FakeSock(blob))
    assert got_header == header
    assert got_payload == payload


def test_clean_end_of_stream_reads_as_none():
    class Closed:
        def recv_into(self, view, n):
            return 0

    assert proto.recv_message(Closed()) == (None, None)


@pytest.mark.parametrize("codec", proto.CODECS)
def test_frame_round_trip(codec):
    state, box = _state()
    energies = np.linspace(-6.0, 0.0, len(state.positions))
    manifest, payload = proto.encode_frame(state, box, energies=energies,
                                           codec=codec)
    # q12 pairs its 12-bit codes into bytes, so an odd number of them rounds the
    # payload up by half a byte -- which is why the claim is only exact to within
    # one byte, and why bytes_per_bead is allowed to be fractional.
    assert abs(len(payload) - len(state.positions) * proto.bytes_per_bead(
        codec, with_directors=True, with_energies=True)) <= 1

    out = proto.decode_frame(manifest, payload, box, codec=codec)
    assert out["positions"].shape == state.positions.shape
    # Positions: L/65535 with the padding for q16 (under 1e-3 sigma on a 37-sigma
    # cell), L/4095 for q12 (under 1e-2). raw32 is exact to float32. Both are
    # quoted against the same thing -- the sub-pixel budget in the module
    # docstring -- rather than against each other.
    tol_pos = {"q16": 1e-3, "q12": 1e-2}.get(codec, 1e-5)
    assert np.abs(out["positions"] - state.positions).max() < tol_pos
    # Directors: the angular error, which is what the picture and the nematic
    # order parameter both actually depend on. Measured in float64 -- an arccos
    # taken near 1 amplifies its argument's error by sqrt, so doing the dot
    # product in the decoded float32 reports ~0.02 degrees of float32 arithmetic
    # noise and tells you nothing about the codec.
    tol_deg = {"q16": 0.02, "q12": 1.0}.get(codec, 1e-4)
    assert _angle_error_deg(out["directors"], state.directors).max() < tol_deg
    tol_pe = {"q16": 0.03, "q12": 0.03}.get(codec, 1e-5)
    assert np.abs(out["energies"] - energies).max() < tol_pe


def test_octahedral_covers_the_poles_and_the_seam():
    """The map's edge cases: the two poles, the six axis directions, and vectors
    exactly on the |x|+|y|+|z| = 1 fold where the lower hemisphere unfolds."""
    awkward = np.array([
        [0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [1, 1, 1], [1, -1, -1], [-1, -1, 1], [1, 1, 0], [0, 1, -1],
    ], dtype=float)
    awkward = normalize_rows(awkward)
    back = proto.oct_decode(proto.oct_encode(awkward))
    assert _angle_error_deg(back, awkward).max() < 0.005
    assert np.allclose(np.linalg.norm(back, axis=1), 1.0, atol=1e-6)


def test_unset_director_survives_as_a_unit_vector():
    """LAMMPS hands back mu = 0 for a particle whose dipole was never set. It has
    no direction to preserve, but it must not decode to a NaN that then poisons
    the renderer's normal."""
    back = proto.oct_decode(proto.oct_encode(np.zeros((1, 3))))
    assert np.isfinite(back).all()
    assert np.isclose(np.linalg.norm(back[0]), 1.0)


def test_positions_outside_the_cell_are_clamped_not_wrapped():
    """LAMMPS remaps atoms on a neighbour rebuild, not every step, so coordinates
    a little outside the cell are normal. They must stay on the side they left
    from -- a wrap would teleport a bead across the box."""
    box = Box.cube(20.0, (True, True, True))
    pos = np.array([[10.5, 0.0, 0.0], [-10.5, 0.0, 0.0], [0.0, 40.0, 0.0]])
    state = FrameState(positions=pos, box=box)
    for codec in ("q16", "q12"):
        manifest, payload = proto.encode_frame(state, box, codec=codec)
        out = proto.decode_frame(manifest, payload, box, codec=codec)["positions"]
        assert out[0, 0] > 10.0 and out[1, 0] < -10.0
        # clamped to the padded wall (q12's step on this cell is 0.005 sigma)
        assert out[2, 1] == pytest.approx(11.0, abs=1e-2)


def test_bandwidth_claims():
    """The numbers the module docstring and docs/a100-plan.md quote."""
    assert proto.bytes_per_bead("q12") == 6.5
    assert proto.bytes_per_bead("q16") == 10
    assert proto.bytes_per_bead("raw32") == 24
    assert 10 * 10_000 * 60 / 1e6 == pytest.approx(6.0)
    # ...and what the default codec costs on the playground it was sized for.
    assert 6.5 * 50_000 * 60 / 1e6 == pytest.approx(19.5)


# ---- q12: the bit packing, and the claim that 8-bit directors are enough -----

@pytest.mark.parametrize("m", [0, 1, 2, 3, 7, 4095, 10_001])
def test_twelve_bit_packing_is_exact_at_any_length(m):
    """The packer pairs adjacent codes into three bytes, so an odd-length stream
    carries one padding code. Every real code must come back untouched, and the
    padding must not add a bead."""
    rng = np.random.default_rng(m)
    codes = rng.integers(0, 1 << proto.POS_BITS, size=m).astype(np.uint16)
    packed = proto.pack12(codes)
    assert len(packed) == 3 * ((m + 1) // 2)
    assert np.array_equal(proto.unpack12(packed, m), codes)


def test_q12_carries_the_bead_count_that_its_bytes_cannot():
    """A packed buffer's own length is one bead ambiguous, so the manifest entry
    carries the logical shape as a fourth field. Without it the decoder must
    refuse rather than guess -- half a frame of positions silently reinterpreted
    would draw a plausible, wrong picture."""
    state, box = _state(n=501)                       # odd: 1503 codes, one padded
    manifest, payload = proto.encode_frame(state, box, codec="q12")
    pos_entry = next(e for e in manifest if e[0] == "pos")
    assert len(pos_entry) == 4 and pos_entry[3] == [501, 3]
    assert len(payload) == 3 * ((1503 + 1) // 2) + 501 * 2

    out = proto.decode_frame(manifest, payload, box, codec="q12")
    assert out["positions"].shape == (501, 3)

    stripped = [e[:3] if e[0] == "pos" else e for e in manifest]
    with pytest.raises(proto.ProtocolError, match="logical shape"):
        proto.decode_frame(stripped, payload, box, codec="q12")


def test_eight_bit_directors_leave_the_order_parameter_alone():
    """The reason 16 bits were spent here was that the client MEASURES nematic_S
    from these directors. It moves the fourth decimal: the angular error is
    random and zero-mean, and S is a second moment over the whole population, so
    it averages out rather than accumulating. This is the assertion that keeps
    the docstring's claim honest."""
    rng = np.random.default_rng(11)
    # A partially aligned population -- the regime k_tilt actually moves through.
    z = rng.uniform(0.0, 1.0, 200_000)
    kappa = 3.0
    ct = np.log(np.exp(kappa) * z + (1.0 - z) * np.exp(-kappa)) / kappa
    st = np.sqrt(np.clip(1.0 - ct ** 2, 0.0, 1.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, len(ct))
    v = np.column_stack([st * np.cos(ph), st * np.sin(ph), ct])

    def nematic_S(u):
        Q = 1.5 * np.einsum("ni,nj->ij", u, u) / len(u) - 0.5 * np.eye(3)
        return float(np.linalg.eigvalsh(Q).max())

    exact = nematic_S(v)
    back = proto.oct_decode(proto.oct_encode(v, np.uint8)).astype(np.float64)
    assert _angle_error_deg(back, v).max() < 1.0        # the per-director claim
    assert abs(nematic_S(back) - exact) < 5e-4          # the one that matters
