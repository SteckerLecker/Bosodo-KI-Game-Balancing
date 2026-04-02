"""
BOSODO Gymnasium Environment Package.

Registriert die BOSODO-Spielumgebung als Gymnasium-Environment.
"""

from gymnasium.envs.registration import register

register(
    id="Bosodo-v0",
    entry_point="bosodo_env.env:BosodoEnv",
    max_episode_steps=200,
)

from bosodo_env.env import BosodoEnv
from bosodo_env.game_state import GameState
from bosodo_env.card_loader import CardLoader

__all__ = ["BosodoEnv", "GameState", "CardLoader"]
