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
    timeout_s: float = 3.0
    max_length: int = 16
    buffer: list[str] = field(default_factory=list)
    last_sign_at: float = 0.0
    matched: Jutsu | None = None
    matched_at: float = 0.0
    idle_since: float | None = None

    def __post_init__(self) -> None:
        # Longest first, so a 6-sign jutsu wins over a 3-sign one sharing its ending.
        self.jutsu = sorted(self.jutsu, key=lambda j: len(j.signs), reverse=True)

    def add(self, sign: str, now: float) -> Jutsu | None:
        self.buffer.append(sign)
        del self.buffer[: max(0, len(self.buffer) - self.max_length)]
        self.last_sign_at = now
        self.idle_since = None

        for candidate in self.jutsu:
            n = len(candidate.signs)
            if n and self.buffer[-n:] == candidate.signs:
                self.matched = candidate
                self.matched_at = now
                self.buffer.clear()
                return candidate
        return None

    def tick(self, now: float, holding: bool = False) -> None:
        """Expire a stale partial sequence. Call once per frame.

        The countdown runs only while **no sign is held**. Holding one sign steady for
        longer than the timeout must not wipe the sequence -- the clock is measuring
        "have you stopped casting", not "how long since the last sign".
        """
        if holding or not self.buffer:
            self.idle_since = None
            return
        if self.idle_since is None:
            self.idle_since = now
        elif now - self.idle_since > self.timeout_s:
            self.buffer.clear()
            self.idle_since = None

    def time_left(self, now: float) -> float | None:
        """Seconds until the sequence expires, or None when the clock is not running."""
        if self.idle_since is None or not self.buffer:
            return None
        return max(0.0, self.timeout_s - (now - self.idle_since))

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


def find_unreachable(jutsu: list[Jutsu]) -> list[tuple[str, str, int]]:
    """Find jutsu that can never fire because a shorter one triggers partway through.

    `SequenceTracker` matches after every sign and **clears the buffer** on a match. So if
    a shorter jutsu completes midway through a longer one, it fires, wipes the buffer, and
    the longer jutsu becomes permanently uncastable.

    Note this is a *prefix-position* problem, not a suffix one. A shorter jutsu that is
    merely the tail of a longer one is harmless, because candidates are tried longest
    first and the longer match wins.

    Returns ``(unreachable_name, blocking_name, sign_index)``.
    """
    blocked = []
    for outer in jutsu:
        for inner in jutsu:
            if inner is outer or len(inner.signs) >= len(outer.signs):
                continue
            n = len(inner.signs)
            for k in range(n, len(outer.signs)):
                if outer.signs[k - n:k] == inner.signs:
                    blocked.append((outer.name, inner.name, k))
                    break
            else:
                continue
            break
    return blocked


def _validate(name: str, signs: list[str]) -> list[str]:
    if not signs:
        raise HandSignError(f"jutsu {name!r} has no signs")
    unknown = sorted(set(signs) - set(CANONICAL))
    if unknown:
        # Gassho, Mizunoe and the Clone Seal land here: real seals the source material
        # uses but this model was never trained on. Failing loudly beats a jutsu that can
        # silently never fire.
        raise HandSignError(
            f"jutsu {name!r} uses sign(s) {unknown} that this model cannot detect. "
            f"Valid signs: {list(CANONICAL)}"
        )
    return signs


def load_jutsu(path: str | Path, warn: bool = True) -> list[Jutsu]:
    """Read and validate a jutsu table. Accepts .csv (preferred) or .yaml.

    CSV layout, matching the reference project's `setting/jutsu.csv` but in English:

        element,jutsu,sign1,sign2,...
        Fire Style,Fireball Jutsu,snake,ram,monkey,boar,horse,tiger,

    Trailing empty sign columns are ignored, so every row can be the same width.
    """
    path = Path(path)
    if not path.exists():
        raise HandSignError(f"{path} not found")

    out: list[Jutsu] = []
    if path.suffix.lower() == ".csv":
        import csv

        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
        rows = [r for r in rows if any(cell.strip() for cell in r)]
        if rows and rows[0][:2] == ["element", "jutsu"]:
            rows = rows[1:]                      # header is optional
        for row in rows:
            if len(row) < 3:
                raise HandSignError(f"{path}: row {row!r} has no signs")
            element, name = row[0].strip(), row[1].strip()
            signs = [c.strip().lower() for c in row[2:] if c.strip()]
            out.append(Jutsu(name, element, _validate(name, signs)))
    else:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in raw.get("jutsu") or []:
            name = str(entry.get("name", "?"))
            signs = [str(s).strip().lower() for s in entry.get("signs", [])]
            out.append(Jutsu(name, str(entry.get("english", "")), _validate(name, signs)))

    if not out:
        raise HandSignError(f"{path} defines no jutsu")

    if warn:
        for outer, inner, k in find_unreachable(out):
            # A warning rather than an error: the rest of the table still works, and
            # aborting mid-demo over one shadowed entry would be worse than the bug.
            print(f"  WARNING: {outer!r} can never fire -- {inner!r} completes at "
                  f"sign {k} and clears the buffer")
    return out
