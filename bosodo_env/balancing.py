"""
Balancing-Analyse für BOSODO.

Analysiert Spielverläufe und identifiziert Balancing-Probleme:
- Übermächtige/zu schwache Monster
- Nutzlose/übermächtige Wissenskarten
- Symbol-Verteilungs-Probleme
- Spiellängen-Anomalien
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

from bosodo_env.card_loader import CardPool, SYMBOLS


@dataclass
class CardBalanceReport:
    """Balancing-Bericht für eine einzelne Karte."""
    card_id: str
    card_name: str
    card_type: str  # "monster" oder "wisdom"
    times_played: int = 0
    times_successful: int = 0  # Verteidigung erfolgreich (für Monster)
    defense_rate: float = 0.0
    usage_percentile: float = 0.0
    balance_score: float = 0.0  # -1 (zu schwach) bis +1 (zu stark)
    issues: List[str] = field(default_factory=list)


@dataclass
class BalancingReport:
    """Gesamtbericht über das Spielbalancing."""
    total_episodes: int = 0
    avg_game_length: float = 0.0
    avg_defense_rate: float = 0.0
    win_distribution: Dict[int, int] = field(default_factory=dict)
    card_reports: Dict[str, CardBalanceReport] = field(default_factory=dict)
    symbol_analysis: Dict[str, Any] = field(default_factory=dict)
    overall_issues: List[str] = field(default_factory=list)
    overall_score: float = 0.0  # 0-100


class BalancingAnalyzer:
    """Analysiert Episoden-Daten und erstellt Balancing-Berichte.

    Verwendung:
        analyzer = BalancingAnalyzer(card_pool)
        for episode_data in episodes:
            analyzer.add_episode(episode_data)
        report = analyzer.analyze()
    """

    def __init__(self, card_pool: CardPool):
        self.card_pool = card_pool
        self.episodes: List[Dict[str, Any]] = []

        # Aggregierte Statistiken
        self.monster_play_count: Counter = Counter()
        self.monster_defense_success: Counter = Counter()
        self.wisdom_play_count: Counter = Counter()
        self.game_lengths: List[int] = []
        self.winners: List[int] = []
        self.defense_rates: List[float] = []

    def add_episode(self, episode_metrics: Dict[str, Any]) -> None:
        """Fügt die Metriken einer abgeschlossenen Episode hinzu."""
        self.episodes.append(episode_metrics)

        # Aggregieren
        if "total_attacks" in episode_metrics:
            self.game_lengths.append(episode_metrics["total_attacks"])
        if "defense_rate" in episode_metrics:
            self.defense_rates.append(episode_metrics["defense_rate"])
        if "most_used_monsters" in episode_metrics:
            for card_id, count in episode_metrics["most_used_monsters"]:
                self.monster_play_count[card_id] += count
        if "most_used_wisdoms" in episode_metrics:
            for card_id, count in episode_metrics["most_used_wisdoms"]:
                self.wisdom_play_count[card_id] += count

    def analyze(self) -> BalancingReport:
        """Erstellt einen umfassenden Balancing-Bericht."""
        report = BalancingReport()
        report.total_episodes = len(self.episodes)

        if not self.episodes:
            report.overall_issues.append("Keine Episoden zum Analysieren.")
            return report

        # Spiellängen-Analyse
        if self.game_lengths:
            report.avg_game_length = sum(self.game_lengths) / len(
                self.game_lengths
            )

        # Verteidigungsrate
        if self.defense_rates:
            report.avg_defense_rate = sum(self.defense_rates) / len(
                self.defense_rates
            )

        # --- Monster-Analyse ---
        for monster in self.card_pool.monsters:
            card_report = CardBalanceReport(
                card_id=monster.id,
                card_name=monster.name,
                card_type="monster",
            )
            card_report.times_played = self.monster_play_count.get(
                monster.id, 0
            )
            report.card_reports[monster.id] = card_report

        # --- Wissenskarten-Analyse ---
        for wisdom in self.card_pool.wisdoms:
            card_report = CardBalanceReport(
                card_id=wisdom.id,
                card_name=wisdom.name,
                card_type="wisdom",
            )
            card_report.times_played = self.wisdom_play_count.get(
                wisdom.id, 0
            )
            report.card_reports[wisdom.id] = card_report

        # --- Symbol-Analyse ---
        report.symbol_analysis = self._analyze_symbols()

        # --- Probleme identifizieren ---
        self._identify_issues(report)

        # --- Gesamtscore berechnen ---
        report.overall_score = self._calculate_overall_score(report)

        return report

    def _analyze_symbols(self) -> Dict[str, Any]:
        """Analysiert die Symbol-Verteilung im Kartendeck."""
        monster_symbols: Counter = Counter()
        wisdom_symbols: Counter = Counter()

        for monster in self.card_pool.monsters:
            for s in monster.kampfwerte:
                monster_symbols[s] += 1

        for wisdom in self.card_pool.wisdoms:
            for s in wisdom.kampfwerte:
                wisdom_symbols[s] += 1

        # Deckungsrate: Wie gut können Wissenskarten Monster abdecken?
        coverage = {}
        for symbol in SYMBOLS:
            m_count = monster_symbols.get(symbol, 0)
            w_count = wisdom_symbols.get(symbol, 0)
            ratio = w_count / m_count if m_count > 0 else float("inf")
            coverage[symbol] = {
                "monster_count": m_count,
                "wisdom_count": w_count,
                "ratio": round(ratio, 2),
                "balanced": 0.8 <= ratio <= 1.5,
            }

        return {
            "monster_symbols": dict(monster_symbols),
            "wisdom_symbols": dict(wisdom_symbols),
            "coverage": coverage,
        }

    def _identify_issues(self, report: BalancingReport) -> None:
        """Identifiziert Balancing-Probleme."""
        # Spiellänge
        if report.avg_game_length < 10:
            report.overall_issues.append(
                f"Spiele sind zu kurz (Ø {report.avg_game_length:.0f} Züge). "
                "Monster sind möglicherweise zu leicht zu besiegen."
            )
        elif report.avg_game_length > 80:
            report.overall_issues.append(
                f"Spiele dauern zu lange (Ø {report.avg_game_length:.0f} Züge). "
                "Verteidigung ist möglicherweise zu schwer."
            )

        # Verteidigungsrate
        if report.avg_defense_rate < 0.2:
            report.overall_issues.append(
                f"Verteidigungsrate sehr niedrig ({report.avg_defense_rate:.1%}). "
                "Wissenskarten passen schlecht zu Monster-Symbolen."
            )
        elif report.avg_defense_rate > 0.8:
            report.overall_issues.append(
                f"Verteidigungsrate sehr hoch ({report.avg_defense_rate:.1%}). "
                "Monster sind zu einfach zu besiegen."
            )

        # Symbol-Balance
        if "coverage" in report.symbol_analysis:
            for symbol, data in report.symbol_analysis["coverage"].items():
                if not data["balanced"]:
                    if data["ratio"] < 0.8:
                        report.overall_issues.append(
                            f"Symbol {symbol}: Zu wenige Wissenskarten "
                            f"(Ratio {data['ratio']}). Verteidigung schwierig."
                        )
                    elif data["ratio"] > 1.5:
                        report.overall_issues.append(
                            f"Symbol {symbol}: Zu viele Wissenskarten "
                            f"(Ratio {data['ratio']}). Verteidigung zu einfach."
                        )

    def _calculate_overall_score(self, report: BalancingReport) -> float:
        """Berechnet einen Gesamtscore (0-100) für das Balancing."""
        score = 100.0

        # Abzüge für Probleme
        score -= len(report.overall_issues) * 10

        # Verteidigungsrate (ideal: 40-60%)
        if report.avg_defense_rate > 0:
            dr_deviation = abs(report.avg_defense_rate - 0.5)
            score -= dr_deviation * 40

        # Spiellänge (ideal: 20-40 Züge)
        if report.avg_game_length > 0:
            if report.avg_game_length < 20:
                score -= (20 - report.avg_game_length) * 2
            elif report.avg_game_length > 40:
                score -= (report.avg_game_length - 40) * 1

        return max(0.0, min(100.0, score))

    def export_report(self, path: str) -> None:
        """Exportiert den Bericht als JSON."""
        report = self.analyze()
        output = {
            "total_episodes": report.total_episodes,
            "avg_game_length": report.avg_game_length,
            "avg_defense_rate": report.avg_defense_rate,
            "overall_score": report.overall_score,
            "overall_issues": report.overall_issues,
            "symbol_analysis": report.symbol_analysis,
            "card_reports": {
                k: {
                    "card_id": v.card_id,
                    "card_name": v.card_name,
                    "card_type": v.card_type,
                    "times_played": v.times_played,
                    "balance_score": v.balance_score,
                    "issues": v.issues,
                }
                for k, v in report.card_reports.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
