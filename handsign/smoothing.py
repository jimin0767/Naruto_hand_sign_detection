"""Temporal smoothing and jutsu sequence matching.

Framework-agnostic and dependency-light: this module takes per-frame class names and
returns confirmed signs and matched sequences. It never touches a camera or a model, so
it can be driven from a game loop, a test, or a recorded log equally well.

Why smoothing is not optional: measured per-frame top-1 on an unseen subject is 92%, so
roughly one frame in twelve names the wrong sign. Displayed raw, that reads as flicker.
The errors are also structured rather than random -- `hare` is misread as `monkey` in
about 20% of cases, because both are "one hand stacked over the other" and differ only in
finger detail that motion blur erases.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

from .classes import CANONICAL


class HandSignError(RuntimeError):
    """Raised on unusable configuration."""


class SignSmoother:
    """Turn noisy per-frame predictions into stable, de-duplicated sign events.

    Two problems, one mechanism. *Flicker*: a held sign occasionally misreads for a frame
    or two. *Repetition*: a sign held for a second produces ~30 identical frames, which
    must register as one sign rather than thirty.

    Both fall out of tracking a "held" sign. A new sign is emitted only when the window
    agrees on something different from what is currently held; the held sign clears when
    the window agrees the hands are down, which is what allows the same sign twice in a
    row.

    Size the window in *frames*, against your actual frame rate. At 78 FPS a 9-frame
    window is only 0.115s -- short enough for a spurious burst to satisfy it. The default
    25 is roughly 0.3s at that rate.
    """

    def __init__(
        self, window: int = 25, min_votes: int = 18, clear_votes: int | None = None
    ):
        # Derived, not a fixed default: a constant here would exceed the window as soon
        # as a caller shrinks it, turning a tuning change into a crash.
        if clear_votes is None:
            clear_votes = max(1, window // 2)
        if min_votes > window or clear_votes > window:
            raise HandSignError(
                f"min_votes ({min_votes}) and clear_votes ({clear_votes}) cannot exceed "
                f"window ({window}); the threshold would be unreachable"
            )
        self.window: deque[str | None] = deque(maxlen=window)
        self.min_votes = min_votes
        self.clear_votes = clear_votes
        self.held: str | None = None

    def update(self, prediction: str | None) -> str | None:
        """Feed one frame's top-1 class (or None). Returns a newly confirmed sign, if any."""
        self.window.append(prediction)
        counts = Counter(p for p in self.window if p is not None)

        if counts:
            candidate, votes = counts.most_common(1)[0]
            if votes >= self.min_votes:
                if candidate != self.held:
                    self.held = candidate
                    return candidate
                return None

        if sum(1 for p in self.window if p is None) >= self.clear_votes:
            self.held = None
        return None

    @property
    def stability(self) -> float:
        """Fraction of the window agreeing with the held sign, in [0, 1]."""
        if not self.window or self.held is None:
            return 0.0
        return sum(1 for p in self.window if p == self.held) / self.window.maxlen

    def reset(self) -> None:
        self.window.clear()
        self.held = None


@dataclass
class Jutsu:
    name: str
    english: str
    signs: list[str]


@dataclass
class SequenceTracker:
    """Accumulate confirmed signs and report when the tail matches a known jutsu."""

    jutsu: list[Jutsu]
    timeout_s: float = 4.0
    max_length: int = 16
    buffer: list[str] = field(default_factory=list)
    last_sign_at: float = 0.0
    matched: Jutsu | None = None
    matched_at: float = 0.0

    def __post_init__(self) -> None:
        # Longest first, so a 6-sign jutsu wins over a 3-sign one sharing its ending.
        self.jutsu = sorted(self.jutsu, key=lambda j: len(j.signs), reverse=True)

    def add(self, sign: str, now: float) -> Jutsu | None:
        self.buffer.append(sign)
        del self.buffer[: max(0, len(self.buffer) - self.max_length)]
        self.last_sign_at = now

        for candidate in self.jutsu:
            n = len(candidate.signs)
            if n and self.buffer[-n:] == candidate.signs:
                self.matched = candidate
                self.matched_at = now
                self.buffer.clear()
                return candidate
        return None

    def tick(self, now: float) -> None:
        """Expire a stale partial sequence. Call once per frame."""
        if self.buffer and now - self.last_sign_at > self.timeout_s:
            self.buffer.clear()

    def banner(self, now: float, hold_s: float = 3.0) -> Jutsu | None:
        """The jutsu to display right now, or None once its moment has passed."""
        if self.matched and now - self.matched_at <= hold_s:
            return self.matched
        return None

    def progress(self) -> list[tuple[Jutsu, int]]:
        """Jutsu whose opening signs match the buffer, with how far along each is."""
        out = []
        for candidate in self.jutsu:
            for depth in range(len(self.buffer), 0, -1):
                if candidate.signs[:depth] == self.buffer[-depth:]:
                    out.append((candidate, depth))
                    break
        return sorted(out, key=lambda pair: -pair[1])


def load_jutsu(path: str | Path) -> list[Jutsu]:
    """Read and validate a jutsu YAML file, rejecting signs the model cannot produce."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise HandSignError(f"{path} not found")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("jutsu") or []
    if not entries:
        raise HandSignError(f"{path} defines no jutsu")

    out = []
    for entry in entries:
        signs = [str(s).strip().lower() for s in entry.get("signs", [])]
        if not signs:
            raise HandSignError(f"jutsu {entry.get('name')!r} has no signs")
        unknown = sorted(set(signs) - set(CANONICAL))
        if unknown:
            # Gassho and Mizunoe land here: real signs the reference project detects but
            # this model was never trained on. Failing loudly beats a jutsu that can
            # silently never fire.
            raise HandSignError(
                f"jutsu {entry.get('name')!r} uses sign(s) {unknown} that this model "
                f"cannot detect. Valid signs: {list(CANONICAL)}"
            )
        out.append(Jutsu(str(entry.get("name", "?")), str(entry.get("english", "")), signs))
    return out
