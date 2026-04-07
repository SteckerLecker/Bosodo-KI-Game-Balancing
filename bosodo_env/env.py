"""
BOSODO Gymnasium Environment.

Wraps die BOSODO-Spielmechanik als Gymnasium-kompatibles Environment
für das Training von RL-Agenten (Stable Baselines3).

Aktionsraum:
    MultiDiscrete([num_monsters_hand, num_players-1])
    → (welche Monster-Karte spielen, wen angreifen)

Beobachtungsraum:
    Box mit Spielzustandsvektor (Handkarten-Symbole, Trophäen, etc.)

Reward-Design:
    Modulares Reward-System über RewardConfig konfigurierbar.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from bosodo_env.card_loader import CardLoader, CardPool, SYMBOLS
from bosodo_env.game_state import GameState
from bosodo_env.rewards import RewardCalculator, RewardConfig
from bosodo_env.metrics import EpisodeMetrics


# Maximale Handgröße (nach Nachziehen + optionalem Tausch)
MAX_HAND_SIZE = 8
# Maximale Spieleranzahl
MAX_PLAYERS = 6


class BosodoEnv(gym.Env):
    """BOSODO Kartenspiel als Gymnasium Environment.

    Der RL-Agent spielt als einer der Spieler und lernt:
    1. Welche Monster-Karte er zum Angreifen wählt
    2. Wen er angreift

    Die anderen Spieler werden durch regelbasierte Bots gesteuert.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        data_dir: str = "data/",
        num_players: int = 4,
        trophies_to_win: int = 3,
        agent_player_idx: int = 0,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
        card_pool: Optional[CardPool] = None,
        llm_cache: Optional[dict] = None,
        llm_threshold: float = 0.0,
        max_turns: int = 200,
        bot_strategy: str = "strongest",
        bot_target_strategy: str = "weakest",
    ):
        """
        Args:
            data_dir: Pfad zum Verzeichnis mit Kartendaten-JSONs.
            num_players: Anzahl Spieler (2-6).
            trophies_to_win: Trophäen zum Sieg (Standard: 3).
            agent_player_idx: Welcher Spieler ist der RL-Agent (0-basiert).
            reward_config: Konfiguration der Reward-Funktion.
            render_mode: "human" oder "ansi" für Textausgabe.
            card_pool: Optionaler vorgefertigter CardPool (überschreibt data_dir).
        """
        super().__init__()

        self.num_players = num_players
        self.trophies_to_win = trophies_to_win
        self.agent_player_idx = agent_player_idx
        self.render_mode = render_mode
        self.max_turns = max_turns
        self.bot_strategy = bot_strategy
        self.bot_target_strategy = bot_target_strategy

        # Karten laden
        if card_pool is not None:
            self.card_pool = card_pool
        else:
            loader = CardLoader(data_dir=data_dir)
            self.card_pool = loader.load()

        # Spielzustand
        self.game_state = GameState(
            card_pool=self.card_pool,
            num_players=num_players,
            trophies_to_win=trophies_to_win,
            llm_cache=llm_cache,
            llm_threshold=llm_threshold,
        )

        # Reward-System
        self.reward_config = reward_config or RewardConfig()
        self.reward_calculator = RewardCalculator(self.reward_config)

        # Metriken
        self.episode_metrics = EpisodeMetrics()

        # --- Aktionsraum ---
        # Aktion = (monster_karte_idx, ziel_spieler_idx)
        # monster_karte_idx: 0 bis MAX_HAND_SIZE-1
        # ziel_spieler_idx: 0 bis num_players-2 (ohne sich selbst)
        self.action_space = spaces.MultiDiscrete(
            [MAX_HAND_SIZE, num_players - 1]
        )

        # --- Beobachtungsraum ---
        # Observation-Vektor-Länge:
        #   3 (Monster-Symbole) + 3 (Wissens-Symbole) +
        #   2 (Handkartenanzahl) + num_players (Trophäen) +
        #   num_players (Angreifer one-hot) + 1 (Runde)
        obs_size = 3 + 3 + 2 + num_players + num_players + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Setzt die Umgebung zurück und startet ein neues Spiel."""
        super().reset(seed=seed)
        self.game_state.reset(seed=seed)
        self.episode_metrics = EpisodeMetrics()

        # Falls der Agent nicht der erste Angreifer ist,
        # Bot-Züge ausführen bis der Agent dran ist
        self._play_bot_turns_until_agent()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Führt einen Spielzug des Agenten aus.

        Args:
            action: [monster_card_idx, target_player_relative_idx]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        monster_idx_raw = int(action[0])
        target_relative = int(action[1])

        # Aktion validieren und clampen
        agent = self.game_state.players[self.agent_player_idx]
        monster_idx = min(monster_idx_raw, len(agent.monster_hand) - 1)
        monster_idx = max(0, monster_idx)

        # Relatives Ziel in absoluten Spieler-Index umrechnen
        valid_targets = self.game_state.get_valid_targets()
        target_idx = valid_targets[min(target_relative, len(valid_targets) - 1)]

        # Angriff ausführen
        result = self.game_state.execute_attack(monster_idx, target_idx)

        # Metriken aktualisieren
        self.episode_metrics.record_attack(result)

        # Nachziehen
        self.game_state.refill_hands()

        # Weitergabe
        self.game_state.advance_turn()

        # Bot-Züge bis Agent wieder dran oder Spiel vorbei
        if not self.game_state.done:
            self._play_bot_turns_until_agent()

        # Reward berechnen
        reward = self.reward_calculator.calculate(
            result=result,
            game_state=self.game_state,
            agent_idx=self.agent_player_idx,
            metrics=self.episode_metrics,
        )

        terminated = self.game_state.done
        truncated = self.game_state.turn_count >= self.max_turns and not terminated
        obs = self._get_obs()
        info = self._get_info()

        if terminated or truncated:
            if truncated:
                m = self.game_state.last_unbeatable_monster
                info["abort_reason"] = {
                    "monster_id": m.id if m else None,
                    "monster_name": getattr(m, "name", None) if m else None,
                    "monster_symbols": list(m.kampfwerte) if m else [],
                }
            info["episode_metrics"] = self.episode_metrics.summary(
                truncated=truncated,
                abort_reason=info.get("abort_reason"),
            )

        return obs, reward, terminated, truncated, info

    def _play_bot_turns_until_agent(self) -> None:
        """Spielt Bot-Züge bis der Agent dran ist oder das Spiel endet/abgebrochen wird."""
        while (
            self.game_state.current_attacker_idx != self.agent_player_idx
            and not self.game_state.done
            and self.game_state.turn_count < self.max_turns
        ):
            self._play_bot_turn()

    def _play_bot_turn(self) -> None:
        """Ein regelbasierter Bot spielt einen Zug.

        Bot-Strategie (konfigurierbar über bot_strategy):
        - "strongest": Stärkstes Monster zuerst (meiste Symbole)
        - "weakest":   Schwächstes Monster zuerst (wenigste Symbole)
        - "random":    Zufälliges Monster aus der Hand

        Bot-Zielwahl (konfigurierbar über bot_target_strategy):
        - "weakest":   Spieler mit wenigsten Trophäen (Standard)
        - "strongest": Spieler mit meisten Trophäen
        - "random":    Zufälliges Ziel
        """
        bot = self.game_state.get_attacker()

        if not bot.monster_hand:
            self.game_state.advance_turn()
            return

        # Monster nach Strategie wählen
        if self.bot_strategy == "random":
            monster_idx = self.game_state.rng.randint(0, len(bot.monster_hand) - 1)
        elif self.bot_strategy == "weakest":
            monster_idx = min(
                range(len(bot.monster_hand)),
                key=lambda i: bot.monster_hand[i].difficulty,
            )
        else:  # "strongest" (Standard)
            monster_idx = max(
                range(len(bot.monster_hand)),
                key=lambda i: bot.monster_hand[i].difficulty,
            )

        # Ziel nach Target-Strategie wählen
        valid_targets = self.game_state.get_valid_targets()
        if self.bot_target_strategy == "random":
            target_idx = self.game_state.rng.choice(valid_targets)
        elif self.bot_target_strategy == "strongest":
            target_idx = max(
                valid_targets,
                key=lambda i: self.game_state.players[i].num_trophies,
            )
        else:  # "weakest" (Standard)
            target_idx = min(
                valid_targets,
                key=lambda i: self.game_state.players[i].num_trophies,
            )

        result = self.game_state.execute_attack(monster_idx, target_idx)
        self.episode_metrics.record_attack(result)

        self.game_state.refill_hands()
        self.game_state.advance_turn()

    def _get_obs(self) -> np.ndarray:
        """Erstellt den Beobachtungsvektor."""
        raw = self.game_state.get_observation_vector(
            perspective_player=self.agent_player_idx
        )
        return np.array(raw, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Erstellt das Info-Dictionary."""
        agent = self.game_state.players[self.agent_player_idx]
        return {
            "turn": self.game_state.turn_count,
            "agent_trophies": agent.num_trophies,
            "agent_monster_hand_size": len(agent.monster_hand),
            "agent_wisdom_hand_size": len(agent.wisdom_hand),
            "current_attacker": self.game_state.current_attacker_idx,
            "winner": self.game_state.winner,
        }

    def render(self) -> Optional[str]:
        """Rendert den aktuellen Spielzustand als Text."""
        if self.render_mode not in ("human", "ansi"):
            return None

        lines = []
        lines.append(f"\n=== BOSODO Runde {self.game_state.turn_count} ===")
        lines.append(
            f"Angreifer: Spieler {self.game_state.current_attacker_idx}"
        )

        for i, player in enumerate(self.game_state.players):
            marker = " (AGENT)" if i == self.agent_player_idx else ""
            lines.append(
                f"  Spieler {i}{marker}: "
                f"{len(player.monster_hand)}M / {len(player.wisdom_hand)}W / "
                f"{player.num_trophies} Trophäen"
            )

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def close(self) -> None:
        """Aufräumen."""
        pass
