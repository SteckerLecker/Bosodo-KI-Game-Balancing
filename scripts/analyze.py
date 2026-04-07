#!/usr/bin/env python3
"""
BOSODO Balancing-Analyse.

Führt das trainierte Modell aus und analysiert die Spielbalance.
Erstellt einen detaillierten Balancing-Report.

Verwendung:
    python scripts/analyze.py
    python scripts/analyze.py --model output/best_model/best_model.zip
    python scripts/analyze.py --episodes 1000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bosodo_env.env import BosodoEnv
from bosodo_env.card_loader import CardLoader
from bosodo_env.balancing import BalancingAnalyzer
from llm_experts.cache import load_cache


def run_analysis(
    model_path: str,
    data_dir: str,
    n_episodes: int = 500,
    num_players: int = 4,
    output_dir: str = "output/reports/",
    llm_threshold: float = 0.0,
) -> None:
    """Führt die Balancing-Analyse durch."""

    print(f"=== BOSODO Balancing-Analyse ===")
    print(f"Modell: {model_path}")
    print(f"Episoden: {n_episodes}")
    if llm_threshold > 0.0:
        print(f"LLM-Threshold: {llm_threshold}")

    # LLM-Cache laden
    llm_cache = load_cache(data_dir=data_dir)
    if llm_cache:
        print(f"LLM-Cache geladen: {len(llm_cache)} Einträge")
    else:
        print("LLM-Cache nicht gefunden — Inhaltsprüfung wird übersprungen")

    # Karten laden
    loader = CardLoader(data_dir=data_dir)
    card_pool = loader.load()

    # Environment erstellen
    env = BosodoEnv(
        data_dir=data_dir,
        num_players=num_players,
        card_pool=card_pool,
        llm_cache=llm_cache if llm_threshold > 0.0 else None,
        llm_threshold=llm_threshold,
    )

    # Modell laden
    model = PPO.load(model_path)

    # Analyzer erstellen
    analyzer = BalancingAnalyzer(card_pool, llm_cache=llm_cache)

    # Episoden durchspielen
    print(f"\nSpiele {n_episodes} Episoden...")
    truncated_count = 0
    abort_monster_counter: dict = {}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if "episode_metrics" in info:
            metrics = info["episode_metrics"]
            analyzer.add_episode(metrics)

            if metrics.get("truncated"):
                truncated_count += 1
                reason = metrics.get("abort_reason") or {}
                m_id = reason.get("monster_id")
                m_name = reason.get("monster_name") or m_id or "unbekannt"
                m_syms = "/".join(reason.get("monster_symbols", []))
                key = f"{m_name} [{m_syms}]" if m_syms else (m_name or "unbekannt")
                abort_monster_counter[key] = abort_monster_counter.get(key, 0) + 1

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} abgeschlossen")

    # Abbruch-Zusammenfassung
    if truncated_count > 0:
        print(f"\n{'=' * 50}")
        print(f"ABGEBROCHENE EPISODEN")
        print(f"{'=' * 50}")
        print(f"Abgebrochen (Max-Runden erreicht): {truncated_count}/{n_episodes} "
              f"({truncated_count / n_episodes:.1%})")
        if abort_monster_counter:
            print(f"\nHäufigste unbesiegbare Monster bei Abbruch:")
            for monster, count in sorted(
                abort_monster_counter.items(), key=lambda x: -x[1]
            )[:10]:
                print(f"  {count:>4}x  {monster}")

    # Analyse erstellen
    report = analyzer.analyze()

    # Report ausgeben
    print(f"\n{'=' * 50}")
    print(f"BALANCING-REPORT")
    print(f"{'=' * 50}")
    print(f"Episoden analysiert: {report.total_episodes}")
    print(f"Ø Spiellänge: {report.avg_game_length:.1f} Züge")
    print(f"Ø Verteidigungsrate: {report.avg_defense_rate:.1%}")
    print(f"Gesamtscore: {report.overall_score:.1f}/100")

    if report.overall_issues:
        print(f"\nIdentifizierte Probleme:")
        for issue in report.overall_issues:
            print(f"  ⚠ {issue}")

    if report.symbol_analysis.get("coverage"):
        print(f"\nSymbol-Abdeckung (statisch):")
        for symbol, data in report.symbol_analysis["coverage"].items():
            status = "✓" if data["balanced"] else "✗"
            print(
                f"  {status} {symbol}: "
                f"Monster={data['monster_count']}, "
                f"Wissen={data['wisdom_count']}, "
                f"Ratio={data['ratio']}"
            )

    if report.defense_rate_per_symbol:
        print(f"\nErweiterte Symbol-Metriken (Laufzeit):")
        print(f"  {'Symbol':<6} {'Verteidigungs-':<18} {'Starvation-':<16} {'Ø auf Hand':<12}")
        print(f"  {'':6} {'rate (Ziel 40-60%)':<18} {'rate (Ziel <15%)':<16} {'(Ziel 1.5-3.0)':<12}")
        print(f"  {'-' * 54}")
        for s in ["BO", "SO", "DO"]:
            dr = report.defense_rate_per_symbol.get(s, 0.0)
            sr = report.symbol_starvation_rate.get(s, 0.0)
            ah = report.avg_symbol_on_hand.get(s, 0.0)
            print(f"  {s:<6} {dr:.1%}{'':11} {sr:.1%}{'':9} {ah:.2f}")
        print(
            f"\n  Multi-Symbol-Verteidigungsrate: "
            f"{report.multi_symbol_defense_rate:.1%} (Ziel 30–50%)"
        )

    # LLM-Inhaltsprüfung ausgeben
    llm = report.llm_analysis
    if llm:
        print(f"\n{'=' * 50}")
        print(f"LLM-INHALTSPRÜFUNG (Stufe 1)")
        print(f"{'=' * 50}")
        total = llm.get("total_symbol_matching_pairs", 0)
        print(f"Symbolisch passende Paarungen analysiert: {total}")
        print(
            f"  Perfekt     (0.8–1.0): {llm.get('perfekt_count', 0):>3} Paarungen"
        )
        print(
            f"  Grauzone    (0.4–0.7): {llm.get('grauzone_count', 0):>3} Paarungen"
        )
        print(
            f"  Fehlzuordnung (0.0–0.3): {llm.get('fehlzuordnung_count', 0):>3} Paarungen"
        )

        fehlzuordnungen = llm.get("fehlzuordnungen", [])
        if fehlzuordnungen:
            print(f"\nFehlzuordnungen (Symbol matcht, Inhalt passt nicht):")
            for p in fehlzuordnungen[:10]:
                symbols = "/".join(p["shared_symbols"])
                print(
                    f"  [{symbols}] {p['monster_name']} vs. {p['wisdom_name']}"
                    f" — Score {p['llm_score']:.2f}: {p['begruendung']}"
                )
            if len(fehlzuordnungen) > 10:
                print(f"  ... und {len(fehlzuordnungen) - 10} weitere (siehe JSON-Report)")

        grauzone = llm.get("grauzone", [])
        if grauzone:
            print(f"\nGrauzone (indirekter Bezug, Beschreibung evtl. anpassen):")
            for p in grauzone[:5]:
                symbols = "/".join(p["shared_symbols"])
                print(
                    f"  [{symbols}] {p['monster_name']} vs. {p['wisdom_name']}"
                    f" — Score {p['llm_score']:.2f}: {p['begruendung']}"
                )
            if len(grauzone) > 5:
                print(f"  ... und {len(grauzone) - 5} weitere (siehe JSON-Report)")

    # Report speichern
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "balancing_report.json"
    analyzer.export_report(str(report_file))
    print(f"\nReport gespeichert: {report_file}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="BOSODO Balancing-Analyse"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="output/best_model/best_model.zip",
        help="Pfad zum trainierten Modell",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Pfad zu Kartendaten",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Anzahl Episoden für Analyse",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        help="Anzahl Spieler",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/reports/",
        help="Ausgabeverzeichnis für Reports",
    )
    parser.add_argument(
        "--llm-threshold",
        type=float,
        default=0.0,
        help="Min. LLM-Score für Wissenskarte (0.0 = deaktiviert)",
    )
    args = parser.parse_args()

    run_analysis(
        model_path=str(PROJECT_ROOT / args.model),
        data_dir=str(PROJECT_ROOT / args.data_dir),
        n_episodes=args.episodes,
        num_players=args.num_players,
        output_dir=args.output_dir,
        llm_threshold=args.llm_threshold,
    )


if __name__ == "__main__":
    main()
