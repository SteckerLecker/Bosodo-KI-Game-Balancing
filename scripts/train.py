#!/usr/bin/env python3
"""
BOSODO RL-Balancing — Trainings-Skript.

Startet das PPO-Training mit der Konfiguration aus config/training_config.yaml.

Verwendung:
    python scripts/train.py
    python scripts/train.py --config config/my_config.yaml
    python scripts/train.py --timesteps 2000000 --device cuda
"""

import argparse
import sys
from pathlib import Path

import yaml

# Projekt-Root zum Python-Pfad hinzufügen
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import train_agent
from bosodo_env.rewards import RewardConfig
from llm_experts.cache import load_cache


def load_config(config_path: str) -> dict:
    """Lädt die YAML-Konfiguration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_reward_config(cfg: dict) -> RewardConfig:
    """Erstellt eine RewardConfig aus dem YAML-Dictionary."""
    rewards = cfg.get("rewards", {})
    return RewardConfig(**{k: v for k, v in rewards.items() if hasattr(RewardConfig, k)})


def main():
    parser = argparse.ArgumentParser(
        description="BOSODO PPO-Training starten"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Pfad zur Konfigurationsdatei",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Überschreibt total_timesteps aus der Config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Überschreibt device (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Überschreibt Anzahl paralleler Environments",
    )
    args = parser.parse_args()

    # Konfiguration laden
    config_path = PROJECT_ROOT / args.config
    cfg = load_config(str(config_path))

    # Training-Config zusammenbauen
    training_cfg = cfg.get("training", {})
    game_cfg = cfg.get("game", {})
    output_cfg = cfg.get("output", {})

    llm_cache = load_cache()
    if llm_cache:
        print(f"LLM-Cache geladen: {len(llm_cache)} Einträge")
    llm_threshold = game_cfg.get("llm_threshold", 0.0)
    if llm_threshold > 0.0:
        print(f"LLM-Threshold aktiv: {llm_threshold}")
    max_turns = game_cfg.get("max_turns", 200)
    print(f"Max. Runden pro Episode: {max_turns}")
    bot_strategy = game_cfg.get("bot_strategy", "strongest")
    print(f"Bot-Strategie: {bot_strategy}")

    train_config = {
        "data_dir": str(PROJECT_ROOT / game_cfg.get("data_dir", "data/")),
        "num_players": game_cfg.get("num_players", 4),
        "max_turns": max_turns,
        "bot_strategy": bot_strategy,
        "total_timesteps": args.timesteps or training_cfg.get("total_timesteps", 500_000),
        "n_envs": args.n_envs or training_cfg.get("n_envs", 8),
        "learning_rate": training_cfg.get("learning_rate", 3e-4),
        "batch_size": training_cfg.get("batch_size", 256),
        "n_epochs": training_cfg.get("n_epochs", 10),
        "gamma": training_cfg.get("gamma", 0.99),
        "output_dir": str(PROJECT_ROOT / output_cfg.get("dir", "output/")),
        "reward_config": build_reward_config(cfg),
        "device": args.device or training_cfg.get("device", "auto"),
        "llm_cache": llm_cache,
        "llm_threshold": llm_threshold,
    }

    # Training starten
    model = train_agent(train_config)
    print("\nTraining abgeschlossen!")


if __name__ == "__main__":
    main()
