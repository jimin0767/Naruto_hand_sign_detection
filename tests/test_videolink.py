"""Tests for the face-cam wire format.

The header layout is a contract with Unity's WebcamFeedReceiver. If a field moves, the
feeds go black with no error anywhere, so the byte offsets are pinned here deliberately.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from handsign.videolink import (
    HEADER_SIZE,
    MAGIC,
    MAX_JPEG_BYTES,
    SLOT_UNKNOWN,
    VERSION,
    FrameEncoder,
    VideoLinkError,
    pack_frame,
    unpack_frame,
)


# --------------------------------------------------------------------------- framing


def test_header_is_exactly_ten_bytes():
    # Unity reads fixed offsets. Changing this silently breaks both feeds.
    assert HEADER_SIZE == 10


def test_round_trip_preserves_everything():
    packet = pack_frame(b"\xff\xd8jpegbytes", frame_id=7, slot=1)
    frame = unpack_frame(packet)

    assert frame is not None
    assert frame.version == VERSION
    assert frame.slot == 1
    assert frame.frame_id == 7
    assert frame.jpeg == b"\xff\xd8jpegbytes"


def test_header_field_order_matches_the_documented_layout():
    packet = pack_frame(b"x", frame_id=0x01020304, slot=3)
    magic, version, slot, frame_id = struct.unpack("<4sBBI", packet[:HEADER_SIZE])

    assert magic == MAGIC == b"NJVF"
    assert version == VERSION
    assert slot == 3
    assert frame_id == 0x01020304


def test_slot_defaults_to_unknown_because_identity_comes_from_the_port():
    frame = unpack_frame(pack_frame(b"x", frame_id=1))
    assert frame.slot == SLOT_UNKNOWN


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"short",
        b"NJVF" + b"\x00" * 6,          # header only, no jpeg
        b"XXXX" + b"\x00" * 6 + b"body",  # wrong magic
    ],
)
def test_unusable_packets_are_dropped_not_raised(payload):
    # Anything can send a datagram to an open UDP port; junk must not crash the game.
    assert unpack_frame(payload) is None


def test_oversize_frame_is_refused_with_an_actionable_message():
    with pytest.raises(VideoLinkError) as excinfo:
        pack_frame(b"x" * (MAX_JPEG_BYTES + 1), frame_id=1)

    assert "--video-quality" in str(excinfo.value)


def test_a_frame_at_exactly_the_limit_still_fits():
    packet = pack_frame(b"x" * MAX_JPEG_BYTES, frame_id=1)
    assert len(packet) <= 65507


def test_frame_id_wraps_instead_of_overflowing():
    frame = unpack_frame(pack_frame(b"x", frame_id=0x1_0000_0000 + 5))
    assert frame.frame_id == 5


# --------------------------------------------------------------------------- encoder


def blank_frame(width=640, height=480):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, : width // 2] = 200          # give the encoder something to compress
    return frame


def test_encoder_rejects_nonsense_settings():
    with pytest.raises(VideoLinkError):
        FrameEncoder(width=0)
    with pytest.raises(VideoLinkError):
        FrameEncoder(quality=0)
    with pytest.raises(VideoLinkError):
        FrameEncoder(quality=101)


def test_encoder_produces_a_decodable_packet():
    encoder = FrameEncoder(width=320, quality=55, fps=0)
    packet = encoder.encode(blank_frame(), now=0.0)

    assert packet is not None
    frame = unpack_frame(packet)
    assert frame is not None
    assert frame.jpeg[:2] == b"\xff\xd8", "JPEG magic"
    assert frame.frame_id == 1


def test_encoder_rate_limits_to_the_requested_fps():
    encoder = FrameEncoder(width=320, fps=20.0)     # one frame per 0.05s

    assert encoder.encode(blank_frame(), now=100.0) is not None
    assert encoder.encode(blank_frame(), now=100.02) is None, "too soon"
    assert encoder.encode(blank_frame(), now=100.06) is not None


def test_encoder_increments_frame_id_only_when_it_actually_sends():
    encoder = FrameEncoder(width=320, fps=20.0)

    encoder.encode(blank_frame(), now=0.0)
    assert encoder.frame_id == 1

    encoder.encode(blank_frame(), now=0.01)          # rate limited
    assert encoder.frame_id == 1, "a skipped frame must not consume an id"

    encoder.encode(blank_frame(), now=0.06)
    assert encoder.frame_id == 2


def test_encoder_scales_down_to_the_requested_width():
    small = FrameEncoder(width=160, quality=55, fps=0)
    large = FrameEncoder(width=640, quality=55, fps=0)

    small_packet = small.encode(blank_frame(), now=0.0)
    large_packet = large.encode(blank_frame(), now=0.0)

    assert len(small_packet) < len(large_packet)


def test_the_first_frame_is_always_sent_immediately():
    # _last_sent_at starting at 0.0 would withhold the first frame whenever the clock
    # starts near zero -- which is exactly what a test or a replay does.
    encoder = FrameEncoder(width=320, fps=20.0)
    assert encoder.encode(blank_frame(), now=0.0) is not None


def test_a_noisy_frame_degrades_in_quality_rather_than_being_dropped():
    """A dim room is the realistic worst case, and must not black out the feed.

    Measured at 640x480: a normal webcam picture is ~30KB at quality 55, but sensor noise
    is over 130KB -- twice what one datagram holds. Random data is the pessimistic bound.
    """
    encoder = FrameEncoder(width=640, quality=55, fps=0)
    rng = np.random.default_rng(seed=1)
    noisy = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    packet = encoder.encode(noisy, now=0.0)

    assert packet is not None, "a noisy frame must still be sent, just uglier"
    assert len(packet) <= 65507
    assert encoder.degraded == 1, "and it should record that it had to degrade"
    assert encoder.dropped_oversize == 0


def test_an_easy_frame_keeps_the_requested_quality():
    encoder = FrameEncoder(width=640, quality=55, fps=0)

    encoder.encode(blank_frame(), now=0.0)

    assert encoder.degraded == 0, "a well-lit frame must not be degraded needlessly"


def test_quality_ladder_descends_and_stops_at_the_floor():
    ladder = FrameEncoder(width=320, quality=55)._quality_ladder()

    assert ladder[0] == 55, "the requested quality is tried first"
    assert ladder == sorted(ladder, reverse=True), "must only ever step down"
    assert ladder[-1] == FrameEncoder.QUALITY_FLOOR
    assert len(set(ladder)) == len(ladder), "no wasted duplicate attempts"


def test_quality_ladder_is_a_single_step_when_already_at_the_floor():
    ladder = FrameEncoder(width=320, quality=FrameEncoder.QUALITY_FLOOR)._quality_ladder()
    assert ladder == [FrameEncoder.QUALITY_FLOOR]
