"""
Modulares Reward-System für BOSODO.

Konfigurierbare Belohnungsfunktionen, die verschiedene Aspekte
des Spielbalancings bewerten:
- Spielbarkeit (kann das Spiel abgeschlossen werden?)
- Fairness (gleichmäßige Verteilung von Siegen)
- Thematische Kohärenz (passen Symbole und Zuordnungen?)
- Spielspaß-Proxies (Spiellänge, Spannungskurve)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from bosodo_env.game_state import GameState
    from bosodo_env.metrics import EpisodeMetrics


@dataclass
class RewardConfig:
    """Konfiguration der Reward-Gewichte.

    Alle Gewichte können angepasst werden, um das Balancing-Ziel
    zu verändern. Standardwerte sind für ausgewogenes Spiel optimiert.
    """

    # Kampf-Rewards
    defense_success: float = 1.0          # Erfolgreich verteidigt
    defense_fail: float = -0.3            # Verteidigung fehlgeschlagen
    trophy_earned: float = 2.0            # Trophäe gewonnen
    game_won: float = 10.0                # Spiel gewonnen
    game_lost: float = -5.0               # Spiel verloren

    # Balancing-Rewards
    game_length_bonus: float = 0.5        # Bonus für optimale Spiellänge
    ideal_game_length: int = 30           # Ideale Anzahl Züge
    game_length_tolerance: int = 15       # Toleranz um den Idealwert

    # Fairness-Rewards
    close_game_bonus: float = 1.0         # Knappes Spielende
    blowout_penalty: float = -1.0         # Einseitiges Ergebnis

    # Symbol-Matching Belohnungen
    symbol_match_bonus: float = 0.3       # Bonus wenn Symbole gut zusammenpassen
    impossible_defense_penalty: float = -0.5  # Wenn Verteidigung unmöglich war

    # Diversitäts-Rewards
    diverse_attacks_bonus: float = 0.2    # Verschiedene Monster werden genutzt
    diverse_targets_bonus: float = 0.1    # Verschiedene Spieler werden angegriffen

    # Zusätzliche Metriken
    stalemate_penalty: float = -2.0       # Spiel dauert viel zu lange


class RewardCalculator:
    """Berechnet Rewards basierend auf Spielaktionen und -zustand."""

    def __init__(self, config: RewardConfig):
        self.config = config

    def calculate(
        self,
        result: Dict[str, Any],
        game_state: "GameState",
        agent_idx: int,
        metrics: "EpisodeMetrics",
    ) -> float:
        """Berechnet den Gesamtreward für eine Aktion.

        Args:
            result: Ergebnis von execute_attack()
            game_state: Aktueller Spielzustand
            agent_idx: Index des RL-Agenten
            metrics: Episoden-Metriken

        Returns:
            float: Gesamtreward
        """
        reward = 0.0

        # --- Kampf-Rewards ---
        if result["defender"] == agent_idx:
            # Agent wurde angegriffen
            if result["trophy_awarded"]:
                reward += self.config.defense_success
                reward += self.config.trophy_earned
            else:
                reward += self.config.defense_fail
        elif result["attacker"] == agent_idx:
            # Agent hat angegriffen
            if result["trophy_awarded"]:
                # Verteidiger hat Trophäe bekommen - schlecht für Angreifer
                reward -= 0.5
            else:
                # Monster nicht besiegt - neutral
                reward += 0.1

        # --- Sieg/Niederlage ---
        if game_state.done:
            if game_state.winner == agent_idx:
                reward += self.config.game_won
            else:
                reward += self.config.game_lost

            # Spiellängen-Bewertung
            reward += self._game_length_reward(game_state.turn_count)

            # Fairness-Bewertung
            reward += self._fairness_reward(game_state)

        # --- Stalemate-Erkennung ---
        if game_state.turn_count > 150:
            reward += self.config.stalemate_penalty

        return reward

    def _game_length_reward(self, turns: int) -> float:
        """Bewertet die Spiellänge.

        Spiele, die zu kurz oder zu lang sind, werden bestraft.
        """
        ideal = self.config.ideal_game_length
        tolerance = self.config.game_length_tolerance
        diff = abs(turns - ideal)

        if diff <= tolerance:
            return self.config.game_length_bonus * (1.0 - diff / tolerance)
        else:
            return -self.config.game_length_bonus * min(diff / ideal, 1.0)

    def _fairness_reward(self, game_state: "GameState") -> float:
        """Bewertet wie knapp das Spiel war.

        Ein knappes Spiel (alle Spieler haben ähnlich viele Trophäen)
        ist besser als ein einseitiger Blowout.
        """
        trophies = [p.num_trophies for p in game_state.players]
        max_trophies = max(trophies)
        min_trophies = min(trophies)
        spread = max_trophies - min_trophies

        if spread <= 1:
            return self.config.close_game_bonus
        elif spread >= 3:
            return self.config.blowout_penalty
        else:
            return 0.0
