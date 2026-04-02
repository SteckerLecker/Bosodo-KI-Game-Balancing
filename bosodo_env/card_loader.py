"""
Karten-Datenmodell und Loader für BOSODO.

Lädt Monster- und Wissenskarten aus JSON-Dateien und stellt sie
als strukturierte Datenklassen bereit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


# Die drei Kampfsymbole
SYMBOLS = ["BO", "SO", "DO"]
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}


@dataclass
class MonsterCard:
    """Eine Monster-Karte (rot) — wird zum Angreifen verwendet."""

    id: str
    name: str
    kurzbeschreibung: str
    kampfwerte: List[str]  # z.B. ["BO", "SO"]
    beschreibung: str
    zitat: str
    belohnung_expert: str = ""

    @property
    def symbol_vector(self) -> List[int]:
        """Gibt einen 3-dimensionalen Vektor zurück: [BO, SO, DO] counts."""
        vec = [0, 0, 0]
        for s in self.kampfwerte:
            vec[SYMBOL_TO_IDX[s]] += 1
        return vec

    @property
    def difficulty(self) -> int:
        """Anzahl der Symbole = Schwierigkeit."""
        return len(self.kampfwerte)


@dataclass
class WisdomCard:
    """Eine Wissens-Karte (grün) — wird zur Verteidigung verwendet."""

    id: str
    name: str
    kurzbeschreibung: str
    kampfwerte: List[str]  # z.B. ["BO", "DO"]
    beschreibung: str
    zitat: str

    @property
    def symbol_vector(self) -> List[int]:
        """Gibt einen 3-dimensionalen Vektor zurück: [BO, SO, DO] counts."""
        vec = [0, 0, 0]
        for s in self.kampfwerte:
            vec[SYMBOL_TO_IDX[s]] += 1
        return vec

    def matches_symbol(self, symbol: str) -> bool:
        """Prüft ob diese Karte ein bestimmtes Symbol hat."""
        return symbol in self.kampfwerte


@dataclass
class CardPool:
    """Container für alle Karten im Spiel."""

    monsters: List[MonsterCard] = field(default_factory=list)
    wisdoms: List[WisdomCard] = field(default_factory=list)

    @property
    def num_monsters(self) -> int:
        return len(self.monsters)

    @property
    def num_wisdoms(self) -> int:
        return len(self.wisdoms)

    def get_monster_by_id(self, card_id: str) -> Optional[MonsterCard]:
        for m in self.monsters:
            if m.id == card_id:
                return m
        return None

    def get_wisdom_by_id(self, card_id: str) -> Optional[WisdomCard]:
        for w in self.wisdoms:
            if w.id == card_id:
                return w
        return None


class CardLoader:
    """Lädt Kartendaten aus JSON-Dateien.

    Verwendung:
        loader = CardLoader(data_dir="data/")
        pool = loader.load()
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)

    def load(self) -> CardPool:
        """Lädt alle Karten und gibt einen CardPool zurück."""
        monsters = self._load_monsters()
        wisdoms = self._load_wisdoms()
        return CardPool(monsters=monsters, wisdoms=wisdoms)

    def _load_monsters(self) -> List[MonsterCard]:
        path = self.data_dir / "monster_karten.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            MonsterCard(
                id=card["id"],
                name=card["name"],
                kurzbeschreibung=card["kurzbeschreibung"],
                kampfwerte=card["kampfwerte"],
                beschreibung=card["beschreibung"],
                zitat=card["zitat"],
                belohnung_expert=card.get("belohnung_expert", ""),
            )
            for card in data["karten"]
        ]

    def _load_wisdoms(self) -> List[WisdomCard]:
        path = self.data_dir / "wissens_karten.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            WisdomCard(
                id=card["id"],
                name=card["name"],
                kurzbeschreibung=card["kurzbeschreibung"],
                kampfwerte=card["kampfwerte"],
                beschreibung=card["beschreibung"],
                zitat=card["zitat"],
            )
            for card in data["karten"]
        ]
