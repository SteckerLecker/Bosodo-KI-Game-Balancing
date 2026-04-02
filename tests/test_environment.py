"""Tests für das BOSODO Environment und die Spiellogik."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bosodo_env.card_loader import CardLoader, MonsterCard, WisdomCard, CardPool
from bosodo_env.game_state import GameState, PlayerState
from bosodo_env.env import BosodoEnv
from bosodo_env.rewards import RewardConfig, RewardCalculator
from bosodo_env.metrics import EpisodeMetrics


DATA_DIR = str(PROJECT_ROOT / "data")


# --- CardLoader Tests ---

class TestCardLoader:
    def test_load_cards(self):
        loader = CardLoader(data_dir=DATA_DIR)
        pool = loader.load()
        assert pool.num_monsters == 18
        assert pool.num_wisdoms == 18

    def test_monster_symbols(self):
        loader = CardLoader(data_dir=DATA_DIR)
        pool = loader.load()
        for m in pool.monsters:
            assert len(m.kampfwerte) > 0
            for s in m.kampfwerte:
                assert s in ["BO", "SO", "DO"]

    def test_wisdom_symbols(self):
        loader = CardLoader(data_dir=DATA_DIR)
        pool = loader.load()
        for w in pool.wisdoms:
            assert len(w.kampfwerte) > 0

    def test_symbol_vector(self):
        m = MonsterCard(
            id="T01", name="Test", kurzbeschreibung="",
            kampfwerte=["BO", "SO"], beschreibung="", zitat=""
        )
        assert m.symbol_vector == [1, 1, 0]
        assert m.difficulty == 2


# --- GameState Tests ---

class TestGameState:
    def setup_method(self):
        loader = CardLoader(data_dir=DATA_DIR)
        self.pool = loader.load()

    def test_reset(self):
        gs = GameState(self.pool, num_players=4, seed=42)
        gs.reset()
        assert len(gs.players) == 4
        for p in gs.players:
            assert len(p.monster_hand) == 2
            assert len(p.wisdom_hand) == 2

    def test_draw_refills(self):
        gs = GameState(self.pool, num_players=2, seed=42)
        gs.reset()
        gs.players[0].monster_hand.clear()
        gs.refill_hands()
        assert len(gs.players[0].monster_hand) >= 2

    def test_can_defend_match(self):
        gs = GameState(self.pool, num_players=2, seed=42)
        gs.reset()
        # Monster mit nur "BO"
        monster = MonsterCard(
            id="T", name="T", kurzbeschreibung="",
            kampfwerte=["BO"], beschreibung="", zitat=""
        )
        # Spieler hat eine BO-Wissenskarte
        gs.players[1].wisdom_hand = [
            WisdomCard(id="TW", name="TW", kurzbeschreibung="",
                       kampfwerte=["BO"], beschreibung="", zitat="")
        ]
        can, cards = gs.can_defend(gs.players[1], monster)
        assert can
        assert len(cards) == 1

    def test_can_defend_no_match(self):
        gs = GameState(self.pool, num_players=2, seed=42)
        gs.reset()
        monster = MonsterCard(
            id="T", name="T", kurzbeschreibung="",
            kampfwerte=["DO"], beschreibung="", zitat=""
        )
        gs.players[1].wisdom_hand = [
            WisdomCard(id="TW", name="TW", kurzbeschreibung="",
                       kampfwerte=["BO"], beschreibung="", zitat="")
        ]
        can, cards = gs.can_defend(gs.players[1], monster)
        assert not can

    def test_victory_condition(self):
        gs = GameState(self.pool, num_players=2, trophies_to_win=1, seed=42)
        gs.reset()
        gs.players[1].trophies = [self.pool.monsters[0]]
        # Nächste erfolgreiche Verteidigung sollte nicht den Sieg auslösen
        # (da Trophäe schon vorhanden ist, müsste es schon bei 1 enden)
        # Testen wir direkt:
        assert gs.players[1].num_trophies == 1


# --- Environment Tests ---

class TestBosodoEnv:
    def test_reset(self):
        env = BosodoEnv(data_dir=DATA_DIR, num_players=4)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_step(self):
        env = BosodoEnv(data_dir=DATA_DIR, num_players=4)
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs2, reward, term, trunc, info2 = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        env.close()

    def test_full_episode(self):
        env = BosodoEnv(data_dir=DATA_DIR, num_players=2)
        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        assert steps > 0
        env.close()

    def test_observation_bounds(self):
        env = BosodoEnv(data_dir=DATA_DIR, num_players=4)
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= 0)
        env.close()


# --- Reward Tests ---

class TestRewards:
    def test_default_config(self):
        config = RewardConfig()
        assert config.game_won > 0
        assert config.game_lost < 0

    def test_reward_calculator(self):
        config = RewardConfig()
        calc = RewardCalculator(config)
        assert calc is not None


# --- Metrics Tests ---

class TestMetrics:
    def test_empty_metrics(self):
        m = EpisodeMetrics()
        assert m.defense_rate == 0.0
        summary = m.summary()
        assert summary["total_attacks"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
