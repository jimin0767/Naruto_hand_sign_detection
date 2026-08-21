"""Webcam frames on the wire, for the face cams in the Unity duel screen.

Framing only. No sockets and no camera here, so the format can be tested exactly, and so
the same packing works from a live loop, a recorded clip, or a test.

The recogniser already has a decoded frame in hand for YOLO, so sending the face cam costs
one JPEG encode and one `sendto` -- not a second capture, and not a second consumer of a
camera device that may refuse to open twice.

Frames are *latest-wins*, the opposite of casts. Losing a cast costs the player the
seconds they spent forming the sequence, so casts are repeated and de-duplicated. Losing
one frame at 20fps is invisible, so frames carry a counter used only to discard packets
that arrive out of order.

One frame is one datagram. A 640x480 JPEG at quality 55 lands around 25-35KB, well inside
the 65507-byte ceiling, so there is no chunking to get wrong. Anything that does not fit
is dropped rather than split -- see `MAX_JPEG_BYTES`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"NJVF"
VERSION = 1

#: `<4sBBI` -- little-endian, no padding: magic, version, slot, frame id.
HEADER_FORMAT = "<4sBBI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

#: Largest payload a single UDP datagram can carry.
MAX_DATAGRAM_BYTES = 65507
MAX_JPEG_BYTES = MAX_DATAGRAM_BYTES - HEADER_SIZE

#: Sent when this process does not know its slot. Slot is for debugging only -- identity
#: is decided by which port the packet arrived on, never by this field.
SLOT_UNKNOWN = 255

#: Unity's WebcamFeedReceiver binds these.
DEFAULT_LOCAL_PORT = 5012   # our own recogniser -> our own Unity
DEFAULT_PEER_PORT = 5013    # the opponent's Unity -> our Unity


class VideoLinkError(RuntimeError):
    """Raised on a frame that cannot be represented on the wire."""


@dataclass(frozen=True)
class VideoFrame:
    """One decoded packet."""

    version: int
    slot: int
    frame_id: int
    jpeg: bytes


def pack_frame(jpeg: bytes, frame_id: int, slot: int = SLOT_UNKNOWN) -> bytes:
    """Wrap encoded JPEG bytes in the wire header.

    Raises `VideoLinkError` if the frame cannot fit in one datagram. The caller should
    lower quality or resolution rather than splitting -- see the module docstring.
    """
    if len(jpeg) > MAX_JPEG_BYTES:
        raise VideoLinkError(
            f"frame is {len(jpeg)} bytes, over the {MAX_JPEG_BYTES} that fit in one "
            f"datagram; lower --video-quality or --video-width"
        )

    header = struct.pack(HEADER_FORMAT, MAGIC, VERSION, slot & 0xFF, frame_id & 0xFFFFFFFF)
    return header + jpeg


def unpack_frame(payload: bytes) -> VideoFrame | None:
    """Decode one packet, or return ``None`` if it is not one of ours.

    Junk on the port is an expected condition, not a programming error: anything may send
    a datagram to an open UDP socket. Unusable input is dropped silently.
    """
    if len(payload) <= HEADER_SIZE:
        return None

    magic, version, slot, frame_id = struct.unpack(
        HEADER_FORMAT, payload[:HEADER_SIZE]
    )
    if magic != MAGIC:
        return None

    return VideoFrame(
        version=version,
        slot=slot,
        frame_id=frame_id,
        jpeg=payload[HEADER_SIZE:],
    )


class FrameEncoder:
    """Turn BGR frames into wire packets at a bounded rate.

    Owns the frame counter and the send-rate decision so the recognition loop does not
    have to. `cv2` is imported lazily so that importing this module -- and testing the
    framing -- does not require OpenCV.
    """

    def __init__(
        self,
        width: int = 640,
        quality: int = 55,
        fps: float = 20.0,
        slot: int = SLOT_UNKNOWN,
    ):
        if width <= 0:
            raise VideoLinkError("width must be positive")
        if not 1 <= quality <= 100:
            raise VideoLinkError("quality must be between 1 and 100")

        self.width = width
        self.quality = quality
        self.min_interval = 1.0 / fps if fps > 0 else 0.0
        self.slot = slot

        self._frame_id = 0
        self._last_sent_at: float | None = None

        #: Frames that had to be re-encoded below the requested quality to fit.
        self.degraded = 0
        #: Frames abandoned even at the quality floor. Should stay 0 in practice.
        self.dropped_oversize = 0

    #: Quality floor for the adaptive retry. Below this a face is not worth sending.
    QUALITY_FLOOR = 18

    @property
    def frame_id(self) -> int:
        """Counter of the most recently packed frame (0 before the first)."""
        return self._frame_id

    def due(self, now: float) -> bool:
        """Whether enough time has passed to send another frame.

        The first frame is always due -- otherwise a clock starting near zero would
        withhold it.
        """
        if self._last_sent_at is None:
            return True
        return now - self._last_sent_at >= self.min_interval

    def encode(self, frame, now: float) -> bytes | None:
        """Scale, JPEG-encode and wrap one BGR frame.

        Returns ``None`` when it is not yet time to send, or when even the quality floor
        will not fit in a datagram.

        **Quality degrades rather than the frame being dropped.** A noisy image -- which
        in practice means a dimly lit room -- compresses far worse than a well-lit one:
        measured at 640x480, quality 55 gives ~30KB on a normal webcam picture but over
        130KB on sensor noise, twice what a datagram holds. Dropping those frames would
        black out the feed exactly when the room is already hard to see in, so the encoder
        steps quality down until the frame fits instead.
        """
        if not self.due(now):
            return None

        import cv2

        height, width = frame.shape[:2]
        if width != self.width:
            scale = self.width / float(width)
            frame = cv2.resize(
                frame, (self.width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        jpeg = self._encode_to_fit(cv2, frame)
        if jpeg is None:
            self.dropped_oversize += 1
            return None

        self._frame_id += 1
        self._last_sent_at = now
        return pack_frame(jpeg, self._frame_id, self.slot)

    def _encode_to_fit(self, cv2, frame) -> bytes | None:
        """Encode at the requested quality, stepping down until it fits a datagram."""
        for attempt, quality in enumerate(self._quality_ladder()):
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ok:
                return None

            if buffer.size <= MAX_JPEG_BYTES:
                if attempt:
                    self.degraded += 1
                return buffer.tobytes()

        return None

    def _quality_ladder(self) -> list[int]:
        """Requested quality first, then progressively cheaper fallbacks."""
        ladder = [self.quality]
        for factor in (0.7, 0.45, 0.3):
            step = max(self.QUALITY_FLOOR, int(self.quality * factor))
            if step < ladder[-1]:
                ladder.append(step)
        if ladder[-1] > self.QUALITY_FLOOR:
            ladder.append(self.QUALITY_FLOOR)
        return ladder
