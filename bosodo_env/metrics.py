"""
Metriken-Tracking für BOSODO-Episoden.

Sammelt detaillierte Statistiken über Spielverläufe,
die für die Balancing-Analyse verwendet werden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import Counter

from bosodo_env.card_loader import SYMBOLS


@dataclass
class EpisodeMetrics:
    """Sammelt Metriken über eine komplette Spielepisode."""

    total_attacks: int = 0
    successful_defenses: int = 0
    failed_defenses: int = 0
    trophies_awarded: int = 0

    # Tracking welche Karten wie oft genutzt wurden
    monster_usage: Counter = field(default_factory=Counter)
    wisdom_usage: Counter = field(default_factory=Counter)

    # Tracking von Angriffszielen
    attack_targets: Counter = field(default_factory=Counter)

    # Tracking der Verteidigbarkeit
    impossible_defenses: int = 0

    # Abbruch-Tracking
    truncated: bool = False
    abort_reason: Optional[Dict[str, Any]] = None

    # Symbol-Matching-Statistiken
    symbol_matches: List[Dict[str, Any]] = field(default_factory=list)

    # Erweiterte Symbol-Metriken (Abschnitt 4.3)
    symbol_attacks: Counter = field(default_factory=Counter)
    symbol_defense_success: Counter = field(default_factory=Counter)
    symbol_starvation: Counter = field(default_factory=Counter)
    symbol_on_hand_total: Counter = field(default_factory=Counter)
    symbol_on_hand_samples: Counter = field(default_factory=Counter)
    multi_symbol_attacks: int = 0
    multi_symbol_defenses: int = 0

    def record_attack(self, result: Dict[str, Any]) -> None:
        """Zeichnet das Ergebnis eines Angriffs auf."""
        self.total_attacks += 1
        monster = result["monster"]
        defended = result["trophy_awarded"]
        self.monster_usage[monster.id] += 1
        self.attack_targets[result["defender"]] += 1

        if defended:
            self.successful_defenses += 1
            self.trophies_awarded += 1
            for card in result["defense_cards"]:
                self.wisdom_usage[card.id] += 1
        else:
            self.failed_defenses += 1

        # Per-Symbol-Tracking
        for symbol in monster.kampfwerte:
            self.symbol_attacks[symbol] += 1
            if defended:
                self.symbol_defense_success[symbol] += 1

        # Starvation: Symbole ohne passende Karte auf der Hand
        for symbol in result.get("starvation_symbols", []):
            self.symbol_starvation[symbol] += 1

        # Durchschnitt der Symbol-Anzahl auf der Hand des Verteidigers
        for symbol, count in result.get("defender_symbol_counts", {}).items():
            self.symbol_on_hand_total[symbol] += count
            self.symbol_on_hand_samples[symbol] += 1

        # Multi-Symbol-Monster-Tracking
        if len(monster.kampfwerte) >= 2:
            self.multi_symbol_attacks += 1
            if defended:
                self.multi_symbol_defenses += 1

        # Symbol-Matching aufzeichnen
        self.symbol_matches.append(
            {
                "monster_id": monster.id,
                "monster_symbols": monster.kampfwerte,
                "defended": defended,
                "defense_cards": [c.id for c in result["defense_cards"]],
            }
        )

    @property
    def defense_rate(self) -> float:
        """Anteil erfolgreicher Verteidigungen."""
        if self.total_attacks == 0:
            return 0.0
        return self.successful_defenses / self.total_attacks

    @property
    def defense_rate_per_symbol(self) -> Dict[str, float]:
        """Verteidigungsrate pro Symbol (Zielbereich: 40–60%)."""
        return {
            s: round(self.symbol_defense_success[s] / self.symbol_attacks[s], 3)
            if self.symbol_attacks[s] > 0 else 0.0
            for s in SYMBOLS
        }

    @property
    def symbol_starvation_rate(self) -> Dict[str, float]:
        """Anteil der Angriffe, bei denen kein passende Karte auf Hand war (Ziel: <15%)."""
        return {
            s: round(self.symbol_starvation[s] / self.symbol_attacks[s], 3)
            if self.symbol_attacks[s] > 0 else 0.0
            for s in SYMBOLS
        }

    @property
    def avg_symbol_on_hand(self) -> Dict[str, float]:
        """Durchschnittliche Anzahl eines Symbols auf der Verteidigerhand (Ziel: 1.5–3.0)."""
        return {
            s: round(self.symbol_on_hand_total[s] / self.symbol_on_hand_samples[s], 3)
            if self.symbol_on_hand_samples[s] > 0 else 0.0
            for s in SYMBOLS
        }

    @property
    def multi_symbol_defense_rate(self) -> float:
        """Verteidigungsrate bei 2- und 3-Symbol-Monstern (Zielbereich: 30–50%)."""
        if self.multi_symbol_attacks == 0:
            return 0.0
        return round(self.multi_symbol_defenses / self.multi_symbol_attacks, 3)

    @property
    def monster_diversity(self) -> float:
        """Wie gleichmäßig verschiedene Monster genutzt werden (0-1)."""
        if not self.monster_usage:
            return 0.0
        counts = list(self.monster_usage.values())
        n = len(counts)
        if n <= 1:
            return 1.0
        total = sum(counts)
        # Normalisierte Entropie
        import math
        entropy = -sum(
            (c / total) * math.log(c / total) for c in counts if c > 0
        )
        max_entropy = math.log(n)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def summary(
        self,
        truncated: bool = False,
        abort_reason: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.truncated = truncated
        self.abort_reason = abort_reason
        """Gibt eine Zusammenfassung aller Metriken zurück."""
        return {
            "truncated": self.truncated,
            "abort_reason": self.abort_reason,
            "total_attacks": self.total_attacks,
            "successful_defenses": self.successful_defenses,
            "failed_defenses": self.failed_defenses,
            "defense_rate": round(self.defense_rate, 3),
            "trophies_awarded": self.trophies_awarded,
            "monster_diversity": round(self.monster_diversity, 3),
            "unique_monsters_used": len(self.monster_usage),
            "unique_wisdoms_used": len(self.wisdom_usage),
            "most_used_monsters": self.monster_usage.most_common(5),
            "most_used_wisdoms": self.wisdom_usage.most_common(5),
            "all_monster_usage": dict(self.monster_usage),
            "all_wisdom_usage": dict(self.wisdom_usage),
            "attack_target_distribution": dict(self.attack_targets),
            # Erweiterte Symbol-Metriken (Abschnitt 4.3)
            "defense_rate_per_symbol": self.defense_rate_per_symbol,
            "symbol_starvation_rate": self.symbol_starvation_rate,
            "avg_symbol_on_hand": self.avg_symbol_on_hand,
            "multi_symbol_attacks": self.multi_symbol_attacks,
            "multi_symbol_defenses": self.multi_symbol_defenses,
            "multi_symbol_defense_rate": self.multi_symbol_defense_rate,
            # Rohdaten für Aggregation im BalancingAnalyzer
            "symbol_attacks_raw": dict(self.symbol_attacks),
            "symbol_defense_success_raw": dict(self.symbol_defense_success),
            "symbol_starvation_raw": dict(self.symbol_starvation),
            "symbol_on_hand_total_raw": dict(self.symbol_on_hand_total),
            "symbol_on_hand_samples_raw": dict(self.symbol_on_hand_samples),
        }
