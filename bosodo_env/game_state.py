"""
Spielzustand-Management für BOSODO.

Verwaltet den gesamten Zustand einer Spielrunde:
Decks, Hände, Trophäen, aktiver Spieler, Ablagestapel.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bosodo_env.card_loader import (
    CardPool,
    MonsterCard,
    WisdomCard,
    SYMBOLS,
    SYMBOL_TO_IDX,
)


@dataclass
class PlayerState:
    """Zustand eines einzelnen Spielers."""

    player_id: int
    monster_hand: List[MonsterCard] = field(default_factory=list)
    wisdom_hand: List[WisdomCard] = field(default_factory=list)
    trophies: List[MonsterCard] = field(default_factory=list)

    @property
    def num_trophies(self) -> int:
        return len(self.trophies)

    def has_minimum_hand(self, min_monsters: int = 2, min_wisdoms: int = 2) -> bool:
        return (
            len(self.monster_hand) >= min_monsters
            and len(self.wisdom_hand) >= min_wisdoms
        )


class GameState:
    """Verwaltet den kompletten Spielzustand einer BOSODO-Runde.

    Implementiert die Casual-Regeln:
    - Jeder Spieler startet mit 2 Monster- und 2 Wissenskarten
    - Angriff → Verteidigung → Nachziehen → optionaler Tausch → Weitergabe
    - Siegbedingung: 3 Trophäen
    """

    def __init__(
        self,
        card_pool: CardPool,
        num_players: int = 4,
        trophies_to_win: int = 3,
        seed: Optional[int] = None,
        llm_cache: Optional[Dict] = None,
        llm_threshold: float = 0.0,
    ):
        self.card_pool = card_pool
        self.num_players = num_players
        self.trophies_to_win = trophies_to_win
        self.rng = random.Random(seed)
        self.llm_cache: Dict = llm_cache or {}
        self.llm_threshold: float = llm_threshold

        # Decks (Nachziehstapel)
        self.monster_deck: List[MonsterCard] = []
        self.wisdom_deck: List[WisdomCard] = []

        # Ablagestapel
        self.monster_discard: List[MonsterCard] = []
        self.wisdom_discard: List[MonsterCard] = []

        # Spieler
        self.players: List[PlayerState] = []
        self.current_attacker_idx: int = 0

        # Rundenzähler
        self.turn_count: int = 0
        self.done: bool = False
        self.winner: Optional[int] = None

        # Letztes Monster, das nicht verteidigt werden konnte
        self.last_unbeatable_monster: Optional[MonsterCard] = None

    def reset(self, seed: Optional[int] = None) -> None:
        """Setzt das Spiel komplett zurück und teilt neue Karten aus."""
        if seed is not None:
            self.rng = random.Random(seed)

        # Decks erstellen und mischen
        self.monster_deck = list(self.card_pool.monsters)
        self.wisdom_deck = list(self.card_pool.wisdoms)
        self.rng.shuffle(self.monster_deck)
        self.rng.shuffle(self.wisdom_deck)

        self.monster_discard = []
        self.wisdom_discard = []

        # Spieler erstellen
        self.players = [PlayerState(player_id=i) for i in range(self.num_players)]

        # Starthände austeilen (je 2 Monster + 2 Wissenskarten)
        for player in self.players:
            for _ in range(2):
                player.monster_hand.append(self._draw_monster())
                player.wisdom_hand.append(self._draw_wisdom())

        self.current_attacker_idx = 0
        self.turn_count = 0
        self.done = False
        self.winner = None
        self.last_unbeatable_monster = None

    def _draw_monster(self) -> MonsterCard:
        """Zieht eine Monster-Karte. Mischt Ablagestapel wenn nötig."""
        if not self.monster_deck:
            if not self.monster_discard:
                # Fallback: Karten recyclen
                self.monster_deck = list(self.card_pool.monsters)
                self.rng.shuffle(self.monster_deck)
            else:
                self.monster_deck = list(self.monster_discard)
                self.monster_discard = []
                self.rng.shuffle(self.monster_deck)
        return self.monster_deck.pop()

    def _draw_wisdom(self) -> WisdomCard:
        """Zieht eine Wissens-Karte. Mischt Ablagestapel wenn nötig."""
        if not self.wisdom_deck:
            if not self.wisdom_discard:
                self.wisdom_deck = list(self.card_pool.wisdoms)
                self.rng.shuffle(self.wisdom_deck)
            else:
                self.wisdom_deck = list(self.wisdom_discard)
                self.wisdom_discard = []
                self.rng.shuffle(self.wisdom_deck)
        return self.wisdom_deck.pop()

    def get_attacker(self) -> PlayerState:
        """Gibt den aktuellen Angreifer zurück."""
        return self.players[self.current_attacker_idx]

    def get_valid_targets(self) -> List[int]:
        """Gibt alle gültigen Angriffsziele zurück (alle außer Angreifer)."""
        return [
            i for i in range(self.num_players) if i != self.current_attacker_idx
        ]

    def can_defend(
        self, defender: PlayerState, monster: MonsterCard
    ) -> Tuple[bool, List[WisdomCard]]:
        """Prüft, ob ein Verteidiger ein Monster besiegen kann (optimales Matching).

        Findet die optimale Kartenzuordnung via Backtracking:
        - Jedes Symbol des Monsters wird einer Wissenskarte zugeteilt
        - Minimiert die Gesamtanzahl genutzter Symbole (spart Multi-Symbol-Karten auf)
        - Wenn llm_threshold > 0: nur Wissenskarten mit LLM-Score >= Threshold erlaubt

        Gibt (kann_verteidigen, verwendete_karten) zurück.
        """
        needed_symbols = list(monster.kampfwerte)
        available = list(defender.wisdom_hand)

        if self.llm_cache and self.llm_threshold > 0.0:
            available = [
                card for card in available
                if self._llm_score(monster.id, card.id) >= self.llm_threshold
            ]

        result = self._find_optimal_defense(needed_symbols, available)
        if result is None:
            return False, []
        return True, result

    def _llm_score(self, monster_id: str, wisdom_id: str) -> float:
        """Gibt den LLM-Score für eine Monster-Wissenskarten-Paarung zurück.

        Gibt 1.0 zurück wenn kein Score vorhanden (kein Symbol-Match → Karte
        wird ohnehin von _find_optimal_defense verworfen).
        """
        entry = self.llm_cache.get(f"{monster_id}_{wisdom_id}")
        if entry is None:
            return 1.0
        return entry["score"]

    def _find_optimal_defense(
        self, needed_symbols: List[str], available: List[WisdomCard]
    ) -> Optional[List[WisdomCard]]:
        """Findet die optimale Verteidigung via Backtracking.

        Optimierungsziel: Minimale Summe der Symbolanzahlen genutzter Karten,
        damit wertvolle Multi-Symbol-Karten für spätere Züge aufgehoben werden.

        Returns None wenn keine Verteidigung möglich.
        """
        best: List = [None]
        best_cost: List = [float("inf")]

        def backtrack(
            sym_idx: int,
            remaining: List[WisdomCard],
            used: List[WisdomCard],
            cost: int,
        ) -> None:
            if sym_idx == len(needed_symbols):
                if cost < best_cost[0]:
                    best_cost[0] = cost
                    best[0] = list(used)
                return
            symbol = needed_symbols[sym_idx]
            for i, card in enumerate(remaining):
                if symbol in card.kampfwerte:
                    backtrack(
                        sym_idx + 1,
                        remaining[:i] + remaining[i + 1 :],
                        used + [card],
                        cost + len(card.kampfwerte),
                    )

        backtrack(0, available, [], 0)
        return best[0]

    def execute_attack(
        self,
        monster_idx: int,
        target_player_idx: int,
    ) -> dict:
        """Führt einen kompletten Angriffsschritt durch.

        Returns:
            dict mit Ergebnissen:
            - "attack_success": bool (Monster besiegt?)
            - "monster": MonsterCard
            - "attacker": int
            - "defender": int
            - "defense_cards": List[WisdomCard]
            - "trophy_awarded": bool
        """
        attacker = self.get_attacker()
        defender = self.players[target_player_idx]

        # Monster-Karte spielen
        monster = attacker.monster_hand[monster_idx]
        attacker.monster_hand.pop(monster_idx)

        # Symbole auf der Hand des Verteidigers (vor Verteidigung)
        defender_symbol_counts = {s: 0 for s in SYMBOLS}
        for card in defender.wisdom_hand:
            for s in card.kampfwerte:
                defender_symbol_counts[s] += 1

        # Symbole für die der Verteidiger gar keine passende Karte hat
        starvation_symbols = [
            s for s in monster.kampfwerte
            if not any(s in card.kampfwerte for card in defender.wisdom_hand)
        ]

        # Verteidigung prüfen
        can_defend, defense_cards = self.can_defend(defender, monster)

        result = {
            "attack_success": not can_defend,
            "monster": monster,
            "attacker": self.current_attacker_idx,
            "defender": target_player_idx,
            "defense_cards": defense_cards,
            "trophy_awarded": False,
            "starvation_symbols": starvation_symbols,
            "defender_symbol_counts": defender_symbol_counts,
        }

        if can_defend:
            # Verteidigung erfolgreich → Trophäe
            for card in defense_cards:
                defender.wisdom_hand.remove(card)
                self.wisdom_discard.append(card)
            defender.trophies.append(monster)
            result["trophy_awarded"] = True

            # Siegbedingung prüfen
            if defender.num_trophies >= self.trophies_to_win:
                self.done = True
                self.winner = target_player_idx
        else:
            # Verteidigung fehlgeschlagen → Monster auf Ablagestapel
            self.monster_discard.append(monster)
            self.last_unbeatable_monster = monster

        return result

    def refill_hands(self) -> None:
        """Alle Spieler ziehen nach bis mindestens 2 Monster + 2 Wissenskarten."""
        for player in self.players:
            while len(player.monster_hand) < 2:
                player.monster_hand.append(self._draw_monster())
            while len(player.wisdom_hand) < 2:
                player.wisdom_hand.append(self._draw_wisdom())

    def optional_swap(self, player_idx: int, card_type: str, card_idx: int) -> None:
        """Optionaler Kartentausch: Eine Karte ablegen und neu ziehen.

        Args:
            player_idx: Index des Spielers
            card_type: "monster" oder "wisdom"
            card_idx: Index der abzulegenden Karte
        """
        player = self.players[player_idx]
        if card_type == "monster" and card_idx < len(player.monster_hand):
            card = player.monster_hand.pop(card_idx)
            self.monster_discard.append(card)
            player.monster_hand.append(self._draw_monster())
        elif card_type == "wisdom" and card_idx < len(player.wisdom_hand):
            card = player.wisdom_hand.pop(card_idx)
            self.wisdom_discard.append(card)
            player.wisdom_hand.append(self._draw_wisdom())

    def advance_turn(self) -> None:
        """Gibt die Angriffskarte an den nächsten Spieler weiter."""
        self.current_attacker_idx = (
            self.current_attacker_idx + 1
        ) % self.num_players
        self.turn_count += 1

    def get_observation_vector(self, perspective_player: int = 0) -> list:
        """Erstellt einen numerischen Beobachtungsvektor für den RL-Agenten.

        Enthält:
        - Eigene Hand (Monster-Symbole, Wissens-Symbole)
        - Anzahl Trophäen pro Spieler
        - Aktueller Angreifer
        - Runde
        """
        player = self.players[perspective_player]
        obs = []

        # Eigene Monster-Hand: Summe der Symbole (max ~6 Karten)
        monster_syms = [0, 0, 0]
        for card in player.monster_hand:
            for i, s in enumerate(SYMBOLS):
                monster_syms[i] += card.symbol_vector[i]
        obs.extend(monster_syms)

        # Eigene Wissens-Hand: Summe der Symbole
        wisdom_syms = [0, 0, 0]
        for card in player.wisdom_hand:
            for i, s in enumerate(SYMBOLS):
                wisdom_syms[i] += card.symbol_vector[i]
        obs.extend(wisdom_syms)

        # Anzahl Handkarten
        obs.append(len(player.monster_hand))
        obs.append(len(player.wisdom_hand))

        # Trophäen aller Spieler
        for p in self.players:
            obs.append(p.num_trophies)

        # Aktueller Angreifer (one-hot)
        for i in range(self.num_players):
            obs.append(1.0 if i == self.current_attacker_idx else 0.0)

        # Normalisierte Runde
        obs.append(min(self.turn_count / 100.0, 1.0))

        return obs
