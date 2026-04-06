"""
RL-Agent Training mit Stable Baselines3 (PPO).

Trainiert einen PPO-Agenten in der BOSODO-Spielumgebung und
sammelt dabei Balancing-Metriken.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from bosodo_env.env import BosodoEnv
from bosodo_env.rewards import RewardConfig


class MetricsCallback(BaseCallback):
    """Callback zum Sammeln von Balancing-Metriken während des Trainings.

    Zeichnet nach jeder abgeschlossenen Episode die Spielmetriken auf
    und loggt sie in TensorBoard.
    """

    def __init__(self, log_dir: str = "logs/metrics", verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_metrics: list = []

    def _on_step(self) -> bool:
        # Prüfe ob Episoden beendet wurden
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_metrics" in info:
                metrics = info["episode_metrics"]
                self.episode_metrics.append(metrics)

                # In TensorBoard loggen
                self.logger.record(
                    "game/defense_rate", metrics["defense_rate"]
                )
                self.logger.record(
                    "game/total_attacks", metrics["total_attacks"]
                )
                self.logger.record(
                    "game/trophies_awarded", metrics["trophies_awarded"]
                )
                self.logger.record(
                    "game/monster_diversity", metrics["monster_diversity"]
                )
                self.logger.record(
                    "game/unique_monsters", metrics["unique_monsters_used"]
                )

        return True

    def _on_training_end(self) -> None:
        # Metriken speichern
        output_path = self.log_dir / "training_metrics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.episode_metrics, f, indent=2, ensure_ascii=False)
        if self.verbose:
            print(f"Metriken gespeichert: {output_path}")


def make_env(
    data_dir: str = "data/",
    num_players: int = 4,
    reward_config: Optional[RewardConfig] = None,
    seed: int = 0,
    llm_cache: Optional[dict] = None,
    llm_threshold: float = 0.0,
    max_turns: int = 200,
    bot_strategy: str = "strongest",
) -> BosodoEnv:
    """Factory-Funktion für Environment-Erstellung."""

    def _init():
        env = BosodoEnv(
            data_dir=data_dir,
            num_players=num_players,
            reward_config=reward_config,
            llm_cache=llm_cache,
            llm_threshold=llm_threshold,
            max_turns=max_turns,
            bot_strategy=bot_strategy,
        )
        env.reset(seed=seed)
        return env

    return _init


def train_agent(config: Dict[str, Any]) -> PPO:
    """Trainiert einen PPO-Agenten mit der gegebenen Konfiguration.

    Args:
        config: Dictionary mit Trainingsparametern:
            - data_dir: Pfad zu Kartendaten
            - num_players: Anzahl Spieler
            - total_timesteps: Trainingsschritte
            - n_envs: Parallele Environments
            - learning_rate: Lernrate
            - batch_size: Batch-Größe
            - n_epochs: PPO-Epochen
            - gamma: Diskontierungsfaktor
            - output_dir: Ausgabeverzeichnis
            - reward_config: RewardConfig-Instanz

    Returns:
        Trainiertes PPO-Modell
    """
    data_dir = config.get("data_dir", "data/")
    num_players = config.get("num_players", 4)
    total_timesteps = config.get("total_timesteps", 500_000)
    n_envs = config.get("n_envs", 8)
    learning_rate = config.get("learning_rate", 3e-4)
    batch_size = config.get("batch_size", 256)
    n_epochs = config.get("n_epochs", 10)
    gamma = config.get("gamma", 0.99)
    output_dir = config.get("output_dir", "output/")
    reward_config = config.get("reward_config", RewardConfig())
    device = config.get("device", "auto")
    llm_cache = config.get("llm_cache", {})
    llm_threshold = config.get("llm_threshold", 0.0)
    max_turns = config.get("max_turns", 200)
    bot_strategy = config.get("bot_strategy", "strongest")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parallele Environments erstellen
    env_fns = [
        make_env(
            data_dir=data_dir,
            num_players=num_players,
            reward_config=reward_config,
            seed=i,
            llm_cache=llm_cache,
            llm_threshold=llm_threshold,
            max_turns=max_turns,
            bot_strategy=bot_strategy,
        )
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)

    # Eval-Environment
    eval_env = BosodoEnv(
        data_dir=data_dir,
        num_players=num_players,
        reward_config=reward_config,
        llm_cache=llm_cache,
        llm_threshold=llm_threshold,
        max_turns=max_turns,
        bot_strategy=bot_strategy,
    )

    # PPO-Modell erstellen
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(output_path / "tensorboard"),
        device=device,
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(output_path / "checkpoints"),
        name_prefix="bosodo_ppo",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best_model"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=10_000 // n_envs,
        n_eval_episodes=20,
        deterministic=True,
    )

    metrics_cb = MetricsCallback(
        log_dir=str(output_path / "metrics"),
        verbose=1,
    )

    # Training starten
    print(f"=== BOSODO PPO Training ===")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Environments: {n_envs}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print(f"===========================")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        progress_bar=True,
    )

    # Finales Modell speichern
    final_path = str(output_path / "final_model")
    model.save(final_path)
    print(f"Finales Modell gespeichert: {final_path}")

    vec_env.close()
    eval_env.close()

    return model
