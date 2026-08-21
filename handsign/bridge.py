"""Wire protocol between the recogniser and the Unity game.

Pure serialisation and state: no sockets, no camera, no model. That keeps the whole
protocol testable from a unit test, and lets the same encoder drive a replay harness or a
fake-cast console session as easily as a live webcam.

Two channels, deliberately separate from the body-tracking team's ports:

    uplink   5010   this process -> Unity   casts, confirmed signs, progress, heartbeat
    downlink 5011   Unity -> this process   recognition on/off, cast rejections

Ports 5005/5008/5009 belong to ``run_tracker`` (pose, hands, seal animation). Writing to
5009 from here would clobber its ``blend``/``clip`` fields, because Unity's StateReceiver
keeps only the most recent packet on that port.

Why casts are sent more than once: UDP may drop a datagram, and a lost cast costs the
player the entire 2.5--3.5s they spent forming the sequence. Signs and heartbeats are
continuous enough that a loss is invisible, so only casts carry a sequence number and get
repeated; Unity de-duplicates on that number.

Why the downlink exists at all: the recognition lock in the battle design turns *off*
recognition while an attack is in flight. Merely ignoring casts in Unity is not the same
thing -- the sign buffer here would keep filling, and the moment the lock lifted a stale
sequence would complete instantly. Unity therefore tells us to stop, and we reset the
smoother and the sequence buffer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

PROTOCOL_VERSION = 1

#: Unity's JutsuReceiver listens here.
DEFAULT_UPLINK_PORT = 5010

#: Unity's JutsuControlSender sends here.
DEFAULT_DOWNLINK_PORT = 5011

#: How many copies of each cast datagram to send. See module docstring.
DEFAULT_CAST_REPEAT = 3


# --------------------------------------------------------------------------- uplink


class UplinkEncoder:
    """Builds the JSON payloads Unity's ``JutsuReceiver`` expects.

    Unity parses these with ``JsonUtility``, which is strict about types but ignores
    fields it does not know, so extra keys are safe to include for debugging.
    """

    def __init__(
        self, version: int = PROTOCOL_VERSION, cast_repeat: int = DEFAULT_CAST_REPEAT
    ):
        if cast_repeat < 1:
            raise ValueError("cast_repeat must be at least 1")
        self.version = version
        self.cast_repeat = cast_repeat
        self._seq = 0

    @property
    def last_seq(self) -> int:
        """Sequence number of the most recently encoded cast (0 before the first)."""
        return self._seq

    def cast(self, jutsu: str, signs: Sequence[str] = ()) -> list[bytes]:
        """Encode a completed jutsu.

        Returns *several identical datagrams*. Send all of them; Unity keeps the first and
        discards the rest by sequence number.
        """
        self._seq += 1
        payload = _encode(
            {
                "v": self.version,
                "seq": self._seq,
                "type": "cast",
                "jutsu": jutsu,
                "signs": list(signs),
            }
        )
        return [payload] * self.cast_repeat

    def sign(self, name: str, confidence: float) -> bytes:
        """Encode a single confirmed sign, for the on-screen sequence readout."""
        return _encode(
            {
                "v": self.version,
                "type": "sign",
                "sign": name,
                "conf": round(float(confidence), 4),
            }
        )

    def progress(self, jutsu: str, matched: int, total: int) -> bytes:
        """Encode partial progress towards a jutsu, for the HUD progress bar."""
        return _encode(
            {
                "v": self.version,
                "type": "progress",
                "jutsu": jutsu,
                "matched": int(matched),
                "total": int(total),
            }
        )

    def heartbeat(self, fps: float) -> bytes:
        """Encode a liveness ping. Unity shows a warning when these stop arriving."""
        return _encode(
            {"v": self.version, "type": "heartbeat", "fps": round(float(fps), 2)}
        )


def _encode(obj: dict) -> bytes:
    # separators: Unity does not care, but smaller datagrams are less likely to fragment.
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------------------- downlink


@dataclass(frozen=True)
class ControlMessage:
    """One decoded message from Unity."""

    type: str
    enabled: bool = True
    reason: str = ""
    seconds: float = 0.0
    jutsu: str = ""
    state: str = ""


def decode_control(payload: bytes) -> ControlMessage | None:
    """Decode one downlink datagram, or return ``None`` if it is unusable.

    Malformed input is dropped rather than raised: a corrupt packet from the network is
    an expected condition, not a programming error, and must not take down recognition.
    """
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(raw, dict):
        return None

    kind = raw.get("type")
    if not isinstance(kind, str) or not kind:
        return None

    return ControlMessage(
        type=kind,
        enabled=bool(raw.get("enabled", True)),
        reason=str(raw.get("reason", "")),
        seconds=float(raw.get("seconds", 0.0) or 0.0),
        jutsu=str(raw.get("jutsu", "")),
        state=str(raw.get("state", "")),
    )


class RecognitionGate:
    """Whether Unity currently wants us recognising, and why.

    Holds no smoother or tracker of its own. ``apply`` reports whether the gate just
    closed so the caller can reset its own state -- keeping this class free of any
    dependency on the recognition pipeline.
    """

    def __init__(self) -> None:
        self.enabled = True
        self.reason = ""
        self.resume_at: float | None = None
        self.match_running = True
        self.last_rejection: ControlMessage | None = None

    def apply(self, msg: ControlMessage, now: float) -> bool:
        """Fold one control message in. Returns ``True`` if recognition just turned off.

        A ``True`` return is the caller's cue to clear the smoother window and the
        sequence buffer, so no half-formed sequence survives the lock.
        """
        if msg.type == "recognition":
            just_closed = self.enabled and not msg.enabled
            self.enabled = msg.enabled
            self.reason = msg.reason
            self.resume_at = now + msg.seconds if msg.seconds > 0 else None
            return just_closed

        if msg.type == "match":
            self.match_running = msg.state != "over"
            if not self.match_running:
                just_closed = self.enabled
                self.enabled = False
                self.reason = "match_over"
                return just_closed
            # A new match re-opens the gate; the server sends an explicit
            # recognition message too, but do not rely on ordering.
            self.enabled = True
            self.reason = ""
            self.resume_at = None
            return False

        if msg.type == "rejected":
            self.last_rejection = msg
            return False

        return False

    def seconds_remaining(self, now: float) -> float:
        """How long until recognition is expected back, for console feedback."""
        if self.enabled or self.resume_at is None:
            return 0.0
        return max(0.0, self.resume_at - now)

    def describe(self, now: float) -> str:
        if self.enabled:
            return "recognising"
        left = self.seconds_remaining(now)
        reason = self.reason or "locked"
        return f"paused ({reason}{f', {left:.1f}s' if left else ''})"
