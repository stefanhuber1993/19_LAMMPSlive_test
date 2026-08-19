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
    assert len(payload) == len(state.positions) * proto.bytes_per_bead(
        codec, with_directors=True, with_energies=True)

    out = proto.decode_frame(manifest, payload, box, codec=codec)
    # Positions: the claim is L/65535 with the padding, i.e. under 1e-3 sigma on a
    # 37-sigma cell. raw32 is exact to float32.
    tol_pos = 1e-3 if codec == "q16" else 1e-5
    assert np.abs(out["positions"] - state.positions).max() < tol_pos
    # Directors: the angular error, which is what the picture and the nematic
    # order parameter both actually depend on. Measured in float64 -- an arccos
    # taken near 1 amplifies its argument's error by sqrt, so doing the dot
    # product in the decoded float32 reports ~0.02 degrees of float32 arithmetic
    # noise and tells you nothing about the codec.
    tol_deg = 0.005 if codec == "q16" else 1e-4
    assert _angle_error_deg(out["directors"], state.directors).max() < tol_deg
    assert np.abs(out["energies"] - energies).max() < (0.03 if codec == "q16" else 1e-5)


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
    manifest, payload = proto.encode_frame(state, box, codec="q16")
    out = proto.decode_frame(manifest, payload, box, codec="q16")["positions"]
    assert out[0, 0] > 10.0 and out[1, 0] < -10.0
    assert out[2, 1] == pytest.approx(11.0, abs=1e-2)   # clamped to the padded wall


def test_bandwidth_claims():
    """The numbers the module docstring and docs/a100-plan.md quote."""
    assert proto.bytes_per_bead("q16") == 10
    assert proto.bytes_per_bead("raw32") == 24
    assert 10 * 10_000 * 60 / 1e6 == pytest.approx(6.0)
