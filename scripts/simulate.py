#!/usr/bin/env python3
"""
BOSODO Quick-Simulation.

Spielt einige Runden mit zufälligen Aktionen, um das Environment
zu testen und erste Balancing-Metriken zu sammeln.

Verwendung:
    python scripts/simulate.py
    python scripts/simulate.py --episodes 100 --render
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bosodo_env.env import BosodoEnv
from bosodo_env.card_loader import CardLoader
from bosodo_env.balancing import BalancingAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="BOSODO Quick-Simulation"
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num-players", type=int, default=4)
    args = parser.parse_args()

    data_dir = str(PROJECT_ROOT / args.data_dir)
    loader = CardLoader(data_dir=data_dir)
    card_pool = loader.load()

    print(f"Karten geladen: {card_pool.num_monsters} Monster, {card_pool.num_wisdoms} Wissenskarten")

    env = BosodoEnv(
        data_dir=data_dir,
        num_players=args.num_players,
        render_mode="human" if args.render else None,
        card_pool=card_pool,
    )

    analyzer = BalancingAnalyzer(card_pool)

    game_lengths = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            if args.render:
                env.render()

        game_lengths.append(steps)
        if "episode_metrics" in info:
            analyzer.add_episode(info["episode_metrics"])

    env.close()

    # Zusammenfassung
    print(f"\n{'=' * 40}")
    print(f"SIMULATION ZUSAMMENFASSUNG")
    print(f"{'=' * 40}")
    print(f"Episoden: {args.episodes}")
    print(f"Ø Spiellänge: {np.mean(game_lengths):.1f} Schritte")
    print(f"Min/Max: {np.min(game_lengths)}/{np.max(game_lengths)}")

    report = analyzer.analyze()
    print(f"Ø Verteidigungsrate: {report.avg_defense_rate:.1%}")
    print(f"Balancing-Score: {report.overall_score:.1f}/100")
    if report.overall_issues:
        print(f"\nProbleme:")
        for issue in report.overall_issues:
            print(f"  ⚠ {issue}")


if __name__ == "__main__":
    main()
