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

assert len(CANONICAL) == len(set(CANONICAL)) == 12
