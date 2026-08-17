"""The canonical class vocabulary. Single source of truth for the whole project.

The order below IS the model's output index order. Every stage imports it from here
rather than declaring its own copy, because the original twelve Roboflow sources used
three mutually incompatible index spaces and silently mislabelled ~18% of boxes when
naively merged. A second copy of this list anywhere is how that bug comes back.

Anything consuming the model -- including code in another language or another engine --
must use exactly this order.
"""

from __future__ import annotations

CANONICAL: tuple[str, ...] = (
    "bird",     # 0
    "boar",     # 1
    "dog",      # 2
    "dragon",   # 3
    "hare",     # 4
    "horse",    # 5
    "monkey",   # 6
    "ox",       # 7
    "ram",      # 8
    "rat",      # 9
    "snake",    # 10
    "tiger",    # 11
)

CANONICAL_INDEX: dict[str, int] = {name: i for i, name in enumerate(CANONICAL)}

# Japanese romaji, for display. Not used for indexing -- `otani` and `chayawat` order
# their classes by these names, which is precisely why the remap table exists.
ROMAJI: dict[str, str] = {
    "bird": "Tori", "boar": "I", "dog": "Inu", "dragon": "Tatsu",
    "hare": "U", "horse": "Uma", "monkey": "Saru", "ox": "Ushi",
    "ram": "Hitsuji", "rat": "Ne", "snake": "Mi", "tiger": "Tora",
}

# Zodiac kanji, for display. Rendering these needs a CJK font -- OpenCV's putText
# cannot draw them, so the UI pre-renders each glyph once with PIL.
KANJI: dict[str, str] = {
    "bird": "酉",    # tori
    "boar": "亥",    # i
    "dog": "戌",     # inu
    "dragon": "辰",  # tatsu
    "hare": "卯",    # u
    "horse": "午",   # uma
    "monkey": "申",  # saru
    "ox": "丑",      # ushi
    "ram": "未",     # hitsuji
    "rat": "子",     # ne
    "snake": "巳",   # mi
    "tiger": "寅",   # tora
}

assert len(CANONICAL) == len(set(CANONICAL)) == 12
assert set(KANJI) == set(ROMAJI) == set(CANONICAL)
