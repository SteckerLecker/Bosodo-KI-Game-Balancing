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
class LLMPairingIssue:
    """Eine Paarung, bei der Symbol und Inhalt nicht übereinstimmen."""
    monster_id: str
    monster_name: str
    wisdom_id: str
    wisdom_name: str
    shared_symbols: List[str]
    llm_score: float
    begruendung: str
    category: str  # "fehlzuordnung" | "grauzone" | "perfekt"


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
    # Erweiterte Symbol-Metriken (Abschnitt 4.3)
    defense_rate_per_symbol: Dict[str, float] = field(default_factory=dict)
    symbol_starvation_rate: Dict[str, float] = field(default_factory=dict)
    avg_symbol_on_hand: Dict[str, float] = field(default_factory=dict)
    multi_symbol_defense_rate: float = 0.0
    avg_wisdom_cards_played: float = 0.0  # Ø Wissenskarten gelegt pro Spiel
    # LLM-basierte Inhaltsprüfung (Abschnitt 3.4 Stufe 1)
    llm_analysis: Dict[str, Any] = field(default_factory=dict)


class BalancingAnalyzer:
    """Analysiert Episoden-Daten und erstellt Balancing-Berichte.

    Verwendung:
        analyzer = BalancingAnalyzer(card_pool)
        for episode_data in episodes:
            analyzer.add_episode(episode_data)
        report = analyzer.analyze()
    """

    def __init__(self, card_pool: CardPool, llm_cache: Optional[Dict[str, Any]] = None):
        self.card_pool = card_pool
        self.llm_cache = llm_cache or {}
        self.episodes: List[Dict[str, Any]] = []

        # Aggregierte Statistiken
        self.monster_play_count: Counter = Counter()
        self.monster_defense_success: Counter = Counter()
        self.wisdom_play_count: Counter = Counter()
        self.wisdom_cards_per_episode: List[int] = []
        self.game_lengths: List[int] = []
        self.winners: List[int] = []
        self.defense_rates: List[float] = []

        # Aggregierte Symbol-Rohdaten (über alle Episoden)
        self.agg_symbol_attacks: Counter = Counter()
        self.agg_symbol_defense_success: Counter = Counter()
        self.agg_symbol_starvation: Counter = Counter()
        self.agg_symbol_on_hand_total: Counter = Counter()
        self.agg_symbol_on_hand_samples: Counter = Counter()
        self.agg_multi_symbol_attacks: int = 0
        self.agg_multi_symbol_defenses: int = 0

    def add_episode(self, episode_metrics: Dict[str, Any]) -> None:
        """Fügt die Metriken einer abgeschlossenen Episode hinzu."""
        self.episodes.append(episode_metrics)

        # Aggregieren
        if "total_attacks" in episode_metrics:
            self.game_lengths.append(episode_metrics["total_attacks"])
        if "defense_rate" in episode_metrics:
            self.defense_rates.append(episode_metrics["defense_rate"])
        if "all_monster_usage" in episode_metrics:
            for card_id, count in episode_metrics["all_monster_usage"].items():
                self.monster_play_count[card_id] += count
        elif "most_used_monsters" in episode_metrics:
            # Fallback für ältere Reports (nur Top-5)
            for card_id, count in episode_metrics["most_used_monsters"]:
                self.monster_play_count[card_id] += count
        if "all_wisdom_usage" in episode_metrics:
            ep_wisdom_total = 0
            for card_id, count in episode_metrics["all_wisdom_usage"].items():
                self.wisdom_play_count[card_id] += count
                ep_wisdom_total += count
            self.wisdom_cards_per_episode.append(ep_wisdom_total)
        elif "most_used_wisdoms" in episode_metrics:
            for card_id, count in episode_metrics["most_used_wisdoms"]:
                self.wisdom_play_count[card_id] += count

        # Symbol-Rohdaten aggregieren
        for s, v in episode_metrics.get("symbol_attacks_raw", {}).items():
            self.agg_symbol_attacks[s] += v
        for s, v in episode_metrics.get("symbol_defense_success_raw", {}).items():
            self.agg_symbol_defense_success[s] += v
        for s, v in episode_metrics.get("symbol_starvation_raw", {}).items():
            self.agg_symbol_starvation[s] += v
        for s, v in episode_metrics.get("symbol_on_hand_total_raw", {}).items():
            self.agg_symbol_on_hand_total[s] += v
        for s, v in episode_metrics.get("symbol_on_hand_samples_raw", {}).items():
            self.agg_symbol_on_hand_samples[s] += v
        self.agg_multi_symbol_attacks += episode_metrics.get("multi_symbol_attacks", 0)
        self.agg_multi_symbol_defenses += episode_metrics.get("multi_symbol_defenses", 0)

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

        # Ø Wissenskarten gelegt pro Spiel
        if self.wisdom_cards_per_episode:
            report.avg_wisdom_cards_played = round(
                sum(self.wisdom_cards_per_episode) / len(self.wisdom_cards_per_episode), 2
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

        # --- Erweiterte Symbol-Metriken ---
        report.defense_rate_per_symbol = {
            s: round(self.agg_symbol_defense_success[s] / self.agg_symbol_attacks[s], 3)
            if self.agg_symbol_attacks[s] > 0 else 0.0
            for s in SYMBOLS
        }
        report.symbol_starvation_rate = {
            s: round(self.agg_symbol_starvation[s] / self.agg_symbol_attacks[s], 3)
            if self.agg_symbol_attacks[s] > 0 else 0.0
            for s in SYMBOLS
        }
        report.avg_symbol_on_hand = {
            s: round(self.agg_symbol_on_hand_total[s] / self.agg_symbol_on_hand_samples[s], 3)
            if self.agg_symbol_on_hand_samples[s] > 0 else 0.0
            for s in SYMBOLS
        }
        if self.agg_multi_symbol_attacks > 0:
            report.multi_symbol_defense_rate = round(
                self.agg_multi_symbol_defenses / self.agg_multi_symbol_attacks, 3
            )

        # --- LLM-Inhaltsprüfung ---
        if self.llm_cache:
            report.llm_analysis = self._analyze_llm_scores()

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

    def _analyze_llm_scores(self) -> Dict[str, Any]:
        """Analysiert LLM-Scores für alle symbolisch passenden Paarungen.

        Findet Paarungen, bei denen Symbole matchen aber Inhalt nicht passt
        (Fehlzuordnung) oder nur indirekt passt (Grauzone).

        Kategorien (gemäß Bericht Abschnitt 3.5):
          - perfekt:        0.8 – 1.0
          - grauzone:       0.4 – 0.7
          - fehlzuordnung:  0.0 – 0.3
        """
        fehlzuordnungen: List[Dict] = []
        grauzone: List[Dict] = []
        perfekt: List[Dict] = []

        for monster in self.card_pool.monsters:
            for wisdom in self.card_pool.wisdoms:
                # Nur Paarungen prüfen, bei denen Symbole überlappen
                shared = [s for s in monster.kampfwerte if s in wisdom.kampfwerte]
                if not shared:
                    continue

                key = f"{monster.id}_{wisdom.id}"
                entry = self.llm_cache.get(key)
                if entry is None:
                    continue

                score = entry["score"]
                begruendung = entry.get("begruendung", "")
                pairing = {
                    "monster_id": monster.id,
                    "monster_name": monster.name,
                    "wisdom_id": wisdom.id,
                    "wisdom_name": wisdom.name,
                    "shared_symbols": shared,
                    "llm_score": score,
                    "begruendung": begruendung,
                }

                if score >= 0.8:
                    perfekt.append(pairing)
                elif score >= 0.4:
                    grauzone.append(pairing)
                else:
                    fehlzuordnungen.append(pairing)

        # Sortierung: schlechteste zuerst
        fehlzuordnungen.sort(key=lambda x: x["llm_score"])
        grauzone.sort(key=lambda x: x["llm_score"])

        total = len(fehlzuordnungen) + len(grauzone) + len(perfekt)
        return {
            "total_symbol_matching_pairs": total,
            "perfekt_count": len(perfekt),
            "grauzone_count": len(grauzone),
            "fehlzuordnung_count": len(fehlzuordnungen),
            "fehlzuordnungen": fehlzuordnungen,
            "grauzone": grauzone,
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

        # Symbol-Balance (statisch)
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

        # Erweiterte Symbol-Metriken (Abschnitt 4.3 — nur wenn Daten vorhanden)
        for symbol, rate in report.defense_rate_per_symbol.items():
            if rate > 0 and not (0.4 <= rate <= 0.6):
                if rate < 0.4:
                    report.overall_issues.append(
                        f"Symbol {symbol}: Verteidigungsrate zu niedrig "
                        f"({rate:.1%}, Ziel 40–60%). Zu wenige passende Wissenskarten."
                    )
                else:
                    report.overall_issues.append(
                        f"Symbol {symbol}: Verteidigungsrate zu hoch "
                        f"({rate:.1%}, Ziel 40–60%). Symbol zu leicht abzuwehren."
                    )

        for symbol, rate in report.symbol_starvation_rate.items():
            if rate > 0.15:
                report.overall_issues.append(
                    f"Symbol {symbol}: Hohe Starvation-Rate ({rate:.1%}, Ziel <15%). "
                    "Spieler haben zu selten passende Wissenskarten auf der Hand."
                )

        for symbol, avg in report.avg_symbol_on_hand.items():
            if avg > 0 and not (1.5 <= avg <= 3.0):
                if avg < 1.5:
                    report.overall_issues.append(
                        f"Symbol {symbol}: Zu wenig auf der Hand (Ø {avg:.2f}, Ziel 1.5–3.0)."
                    )
                else:
                    report.overall_issues.append(
                        f"Symbol {symbol}: Zu viel auf der Hand (Ø {avg:.2f}, Ziel 1.5–3.0)."
                    )

        if report.multi_symbol_defense_rate > 0:
            mdr = report.multi_symbol_defense_rate
            if not (0.3 <= mdr <= 0.5):
                if mdr < 0.3:
                    report.overall_issues.append(
                        f"Multi-Symbol-Monster: Verteidigungsrate zu niedrig "
                        f"({mdr:.1%}, Ziel 30–50%). 2-/3-Symbol-Monster sind zu stark."
                    )
                else:
                    report.overall_issues.append(
                        f"Multi-Symbol-Monster: Verteidigungsrate zu hoch "
                        f"({mdr:.1%}, Ziel 30–50%). 2-/3-Symbol-Monster sind zu leicht."
                    )

    def _calculate_overall_score(self, report: BalancingReport) -> float:
        """Berechnet einen Gesamtscore (0-100) für das Balancing.

        Bewertungskomponenten (je 0-100, dann gewichtet):
        - Verteidigungsrate (Ziel 40-60%):     Gewicht 40%
        - Spiellänge (Ziel 20-40 Züge):        Gewicht 30%
        - Symbol-Balance (Starvation, Ø Hand): Gewicht 20%
        - Kritische Issues (Abzüge):            Gewicht 10%
        """
        # --- Teilscores (0-100) ---

        # 1. Verteidigungsrate
        if report.avg_defense_rate > 0:
            dr = report.avg_defense_rate
            if 0.4 <= dr <= 0.6:
                defense_score = 100.0
            else:
                deviation = abs(dr - 0.5) - 0.1  # Abzug ab 10% Abweichung
                defense_score = max(0.0, 100.0 - deviation * 400)
        else:
            defense_score = 0.0

        # 2. Spiellänge
        gl = report.avg_game_length
        if 20 <= gl <= 40:
            length_score = 100.0
        elif gl < 20:
            length_score = max(0.0, 100.0 - (20 - gl) * 4)
        else:
            length_score = max(0.0, 100.0 - (gl - 40) * 2)

        # 3. Symbol-Balance (Starvation + Ø auf Hand)
        symbol_penalties = 0.0
        for rate in report.symbol_starvation_rate.values():
            if rate > 0.15:
                symbol_penalties += (rate - 0.15) * 200  # Max ~17 pro Symbol
        for avg in report.avg_symbol_on_hand.values():
            if avg > 0 and avg < 1.5:
                symbol_penalties += (1.5 - avg) * 30
            elif avg > 3.0:
                symbol_penalties += (avg - 3.0) * 15
        symbol_score = max(0.0, 100.0 - symbol_penalties)

        # 4. Kritische Issues (pauschal, aber gedeckelt)
        critical_keywords = ["zu kurz", "zu lang", "sehr niedrig", "sehr hoch"]
        critical_count = sum(
            1 for issue in report.overall_issues
            if any(kw in issue for kw in critical_keywords)
        )
        issue_score = max(0.0, 100.0 - critical_count * 25)

        # --- Gewichteter Gesamtscore ---
        total = (
            defense_score * 0.40
            + length_score * 0.30
            + symbol_score * 0.20
            + issue_score * 0.10
        )
        return round(max(0.0, min(100.0, total)), 1)

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
            "extended_symbol_metrics": {
                "defense_rate_per_symbol": report.defense_rate_per_symbol,
                "symbol_starvation_rate": report.symbol_starvation_rate,
                "avg_symbol_on_hand": report.avg_symbol_on_hand,
                "multi_symbol_defense_rate": report.multi_symbol_defense_rate,
            },
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
            "llm_analysis": report.llm_analysis,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
