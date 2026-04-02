"""
Metriken-Tracking für BOSODO-Episoden.

Sammelt detaillierte Statistiken über Spielverläufe,
die für die Balancing-Analyse verwendet werden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List
from collections import Counter


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

    # Symbol-Matching-Statistiken
    symbol_matches: List[Dict[str, Any]] = field(default_factory=list)

    def record_attack(self, result: Dict[str, Any]) -> None:
        """Zeichnet das Ergebnis eines Angriffs auf."""
        self.total_attacks += 1
        self.monster_usage[result["monster"].id] += 1
        self.attack_targets[result["defender"]] += 1

        if result["trophy_awarded"]:
            self.successful_defenses += 1
            self.trophies_awarded += 1
            for card in result["defense_cards"]:
                self.wisdom_usage[card.id] += 1
        else:
            self.failed_defenses += 1

        # Symbol-Matching aufzeichnen
        monster = result["monster"]
        self.symbol_matches.append(
            {
                "monster_id": monster.id,
                "monster_symbols": monster.kampfwerte,
                "defended": result["trophy_awarded"],
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

    def summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung aller Metriken zurück."""
        return {
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
            "attack_target_distribution": dict(self.attack_targets),
        }
