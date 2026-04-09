#!/usr/bin/env python3
"""
Startet die iterative Balancing-Pipeline (3-Persona-Workflow).

Usage:
    python -m scripts.run_balancing_pipeline
    python -m scripts.run_balancing_pipeline --data-dir data/scrum_edition
    python -m scripts.run_balancing_pipeline --max-iterations 20
"""

import argparse
import sys
from pathlib import Path

# Projekt-Root in den Pfad aufnehmen
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from llm_experts.balancing_pipeline import BalancingPipeline


def main():
    parser = argparse.ArgumentParser(description="Iterative Balancing-Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Pfad zum Datenverzeichnis (default: aus training_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Pfad für Ergebnis-Snapshots (default: <data-dir>/balancing_runs)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximale Anzahl Iterationen (default: 15)",
    )
    args = parser.parse_args()

    # data_dir ermitteln
    data_dir = args.data_dir
    if data_dir is None:
        config_path = _PROJECT_ROOT / "config" / "training_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data_dir = str(_PROJECT_ROOT / cfg.get("game", {}).get("data_dir", "data"))

    # Optionale Overrides
    if args.max_iterations:
        import llm_experts.balancing_pipeline as bp
        bp.MAX_ITERATIONS = args.max_iterations

    pipeline = BalancingPipeline(data_dir=data_dir, output_dir=args.output_dir)
    result = pipeline.run()

    print(f"\nStatus: {result['status']}")
    if result.get("stats"):
        print(f"Ø Matches: {result['stats']['avg_total']}")


if __name__ == "__main__":
    main()
