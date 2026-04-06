#!/usr/bin/env python3
"""
BOSODO Iteratives Balancing-Skript (Bericht 4.1–4.6).

Automatisierter Zyklus:
    1. Kartendaten laden (aus data/ oder data_vN/)
    2. PPO-Agent trainieren
    3. Balancing analysieren (inkl. LLM-Report)
    4. Konvergenz prüfen
    5. Falls nicht konvergiert: Symbole anpassen (max. 2 Änderungen)
    6. Neue Kartenversion speichern und wiederholen

Verwendung:
    python scripts/iterative_balance.py
    python scripts/iterative_balance.py --max-iterations 15
    python scripts/iterative_balance.py --timesteps 1000000 --episodes 1000
    python scripts/iterative_balance.py --start-version 3   # Ab data_v3/ weitermachen
"""

import argparse
import copy
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# APP_ROOT = Wo der Code liegt (für Imports)
# WORK_DIR = Wo Daten gelesen/geschrieben werden (via --work-dir, default: cwd)
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

# Standard: Projekt-Root (dort liegen data/, config/, output/)
WORK_DIR = APP_ROOT

from agents import train_agent
from bosodo_env.balancing import BalancingAnalyzer, BalancingReport
from bosodo_env.card_loader import SYMBOLS, CardLoader, CardPool
from bosodo_env.env import BosodoEnv
from bosodo_env.rewards import RewardConfig
from llm_experts.cache import load_cache
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

@dataclass
class IterativeConfig:
    """Konfiguration für den iterativen Balancing-Zyklus."""
    max_iterations: int = 10
    timesteps_per_iteration: int = 500_000
    analysis_episodes: int = 500
    num_players: int = 4
    max_turns: int = 200
    bot_strategy: str = "strongest"
    llm_threshold: float = 0.0
    n_envs: int = 8
    device: str = "auto"
    start_version: int = 0
    convergence_window: int = 3  # Aufeinanderfolgende Iterationen

    # Konvergenzkriterien (Bericht 4.6)
    target_defense_rate: Tuple[float, float] = (0.40, 0.60)
    target_starvation_rate: float = 0.15
    target_game_length: Tuple[int, int] = (20, 40)
    target_close_game_ratio: float = 0.50
    target_monster_usage: Tuple[float, float] = (0.02, 0.15)

    # Anpassungsregeln (Bericht 4.4)
    starvation_threshold: float = 0.20
    overdefense_threshold: float = 0.70
    multi_symbol_min: float = 0.25
    max_changes_per_iteration: int = 2


# ---------------------------------------------------------------------------
# Konvergenzprüfung
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """Ergebnis der Konvergenzprüfung einer einzelnen Iteration."""
    defense_rates_ok: bool = False
    starvation_ok: bool = False
    game_length_ok: bool = False
    close_games_ok: bool = False
    monster_usage_ok: bool = False
    details: List[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return all([
            self.defense_rates_ok,
            self.starvation_ok,
            self.game_length_ok,
            self.close_games_ok,
            self.monster_usage_ok,
        ])


def check_convergence(
    report: BalancingReport,
    episode_data: List[Dict[str, Any]],
    card_pool: CardPool,
    cfg: IterativeConfig,
) -> ConvergenceResult:
    """Prüft ob die Balancing-Metriken die Konvergenzkriterien erfüllen."""
    result = ConvergenceResult()
    lo_dr, hi_dr = cfg.target_defense_rate
    lo_gl, hi_gl = cfg.target_game_length
    lo_mu, hi_mu = cfg.target_monster_usage

    # 1. Verteidigungsrate aller Symbole: 40–60 %
    all_dr_ok = True
    for symbol, rate in report.defense_rate_per_symbol.items():
        if not (lo_dr <= rate <= hi_dr):
            all_dr_ok = False
            result.details.append(
                f"Defense-Rate {symbol}: {rate:.1%} (Ziel {lo_dr:.0%}–{hi_dr:.0%})"
            )
    result.defense_rates_ok = all_dr_ok

    # 2. Symbol-Starvation < 15 %
    all_starv_ok = True
    for symbol, rate in report.symbol_starvation_rate.items():
        if rate > cfg.target_starvation_rate:
            all_starv_ok = False
            result.details.append(
                f"Starvation {symbol}: {rate:.1%} (Ziel <{cfg.target_starvation_rate:.0%})"
            )
    result.starvation_ok = all_starv_ok

    # 3. Spiellänge: 20–40 Züge
    gl = report.avg_game_length
    result.game_length_ok = lo_gl <= gl <= hi_gl
    if not result.game_length_ok:
        result.details.append(
            f"Spiellänge: {gl:.1f} (Ziel {lo_gl}–{hi_gl})"
        )

    # 4. ≥ 50 % der Spiele enden mit Trophäen-Unterschied ≤ 1
    close_count = 0
    for ep in episode_data:
        trophies = ep.get("final_trophies", [])
        if len(trophies) >= 2:
            sorted_t = sorted(trophies, reverse=True)
            if sorted_t[0] - sorted_t[1] <= 1:
                close_count += 1
    total = len(episode_data) if episode_data else 1
    close_ratio = close_count / total
    result.close_games_ok = close_ratio >= cfg.target_close_game_ratio
    if not result.close_games_ok:
        result.details.append(
            f"Knappe Spiele: {close_ratio:.1%} (Ziel ≥{cfg.target_close_game_ratio:.0%})"
        )

    # 5. Kein Monster < 2 % oder > 15 % aller Angriffe
    # times_played = absolute Anzahl gespielter Einsätze über alle Episoden.
    # Normierung durch Gesamtangriffe (nicht Episoden), damit Werte 0–1 ergeben.
    total_attacks = sum(
        report.card_reports[m.id].times_played
        for m in card_pool.monsters
        if m.id in report.card_reports
    )
    if total_attacks == 0:
        total_attacks = 1
    all_mu_ok = True
    for monster in card_pool.monsters:
        cr = report.card_reports.get(monster.id)
        if cr is None:
            continue
        usage_ratio = cr.times_played / total_attacks
        if not (lo_mu <= usage_ratio <= hi_mu):
            all_mu_ok = False
            result.details.append(
                f"Monster {monster.id} ({monster.name}): "
                f"Usage {usage_ratio:.1%} (Ziel {lo_mu:.0%}–{hi_mu:.0%})"
            )
    result.monster_usage_ok = all_mu_ok

    return result


# ---------------------------------------------------------------------------
# Symbol-Anpassungsregeln (Bericht 4.4)
# ---------------------------------------------------------------------------

@dataclass
class CardChange:
    """Dokumentiert eine Kartenänderung."""
    card_id: str
    card_name: str
    card_type: str  # "monster" oder "wisdom"
    change_type: str  # "add_symbol" oder "remove_symbol"
    symbol: str
    reason: str
    old_symbols: List[str]
    new_symbols: List[str]


def apply_balancing_rules(
    card_pool: CardPool,
    report: BalancingReport,
    cfg: IterativeConfig,
) -> Tuple[CardPool, List[CardChange]]:
    """Wendet die Anpassungsregeln an und gibt den neuen CardPool zurück.

    Regeln (Bericht 4.4):
        1. Symbol-Hunger: starvation_rate > 20 % → Wissenskarte mit Symbol hinzufügen
        2. Überverteidigung: defense_rate > 70 % → Symbol von Wissenskarte entfernen
        3. Multi-Symbol: multi_symbol_defense_rate < 25 % → mehr Multi-Symbol-Wissenskarten
        4. Max. 2 Kartenänderungen pro Iteration
    """
    changes: List[CardChange] = []

    # Deepcopy der Kartendaten für Mutation
    new_monsters = copy.deepcopy(card_pool.monsters)
    new_wisdoms = copy.deepcopy(card_pool.wisdoms)

    def budget_left() -> bool:
        return len(changes) < cfg.max_changes_per_iteration

    # --- Regel 1: Symbol-Hunger ---
    # Sortiere nach starvation_rate (höchste zuerst)
    starved_symbols = sorted(
        [
            (s, rate)
            for s, rate in report.symbol_starvation_rate.items()
            if rate > cfg.starvation_threshold
        ],
        key=lambda x: -x[1],
    )

    for symbol, rate in starved_symbols:
        if not budget_left():
            break

        # Finde Wissenskarte mit wenigsten Symbolen, die dieses Symbol NICHT hat
        # (bevorzuge 1-Symbol-Karten, um sie zu 2-Symbol-Karten zu machen)
        candidates = [
            w for w in new_wisdoms
            if symbol not in w.kampfwerte and len(w.kampfwerte) < 3
        ]
        if not candidates:
            continue

        # Sortiere: weniger Symbole zuerst, dann nach niedrigster Nutzung
        candidates.sort(key=lambda w: (
            len(w.kampfwerte),
            report.card_reports.get(w.id, CardChange).times_played
            if hasattr(report.card_reports.get(w.id), "times_played")
            else 0,
        ))

        target = candidates[0]
        old_symbols = list(target.kampfwerte)
        target.kampfwerte.append(symbol)

        changes.append(CardChange(
            card_id=target.id,
            card_name=target.name,
            card_type="wisdom",
            change_type="add_symbol",
            symbol=symbol,
            reason=f"Regel 1 — Symbol-Hunger: {symbol} Starvation {rate:.1%} > {cfg.starvation_threshold:.0%}",
            old_symbols=old_symbols,
            new_symbols=list(target.kampfwerte),
        ))

    # --- Regel 2: Überverteidigung ---
    overdefended = sorted(
        [
            (s, rate)
            for s, rate in report.defense_rate_per_symbol.items()
            if rate > cfg.overdefense_threshold
        ],
        key=lambda x: -x[1],
    )

    for symbol, rate in overdefended:
        if not budget_left():
            break

        # Finde Wissenskarte mit diesem Symbol, die mehrere Symbole hat
        # (entferne Symbol von Multi-Symbol-Karte, nie letztes Symbol)
        candidates = [
            w for w in new_wisdoms
            if symbol in w.kampfwerte and len(w.kampfwerte) > 1
        ]
        if not candidates:
            continue

        # Bevorzuge Karten mit den meisten Symbolen (3 vor 2)
        candidates.sort(key=lambda w: -len(w.kampfwerte))
        target = candidates[0]
        old_symbols = list(target.kampfwerte)
        target.kampfwerte.remove(symbol)

        changes.append(CardChange(
            card_id=target.id,
            card_name=target.name,
            card_type="wisdom",
            change_type="remove_symbol",
            symbol=symbol,
            reason=f"Regel 2 — Überverteidigung: {symbol} Defense {rate:.1%} > {cfg.overdefense_threshold:.0%}",
            old_symbols=old_symbols,
            new_symbols=list(target.kampfwerte),
        ))

    # --- Regel 3: Multi-Symbol-Balance ---
    if (
        budget_left()
        and report.multi_symbol_defense_rate < cfg.multi_symbol_min
    ):
        # Finde 1-Symbol-Wissenskarte und mache sie zur 2-Symbol-Karte
        # Wähle das Symbol mit der höchsten Starvation als zweites Symbol
        single_wisdoms = [w for w in new_wisdoms if len(w.kampfwerte) == 1]
        if single_wisdoms:
            # Wähle Symbol mit höchster Starvation, das nicht schon auf der Karte ist
            best_symbol = max(
                report.symbol_starvation_rate.items(),
                key=lambda x: x[1],
            )[0]

            candidates = [
                w for w in single_wisdoms if best_symbol not in w.kampfwerte
            ]
            if candidates:
                target = candidates[0]
                old_symbols = list(target.kampfwerte)
                target.kampfwerte.append(best_symbol)

                changes.append(CardChange(
                    card_id=target.id,
                    card_name=target.name,
                    card_type="wisdom",
                    change_type="add_symbol",
                    symbol=best_symbol,
                    reason=(
                        f"Regel 3 — Multi-Symbol-Balance: "
                        f"Rate {report.multi_symbol_defense_rate:.1%} < {cfg.multi_symbol_min:.0%}"
                    ),
                    old_symbols=old_symbols,
                    new_symbols=list(target.kampfwerte),
                ))

    new_pool = CardPool(monsters=new_monsters, wisdoms=new_wisdoms)
    return new_pool, changes


# ---------------------------------------------------------------------------
# Kartensatz speichern / laden
# ---------------------------------------------------------------------------

def get_data_dir(version: int) -> Path:
    """Gibt den Pfad zum Kartendaten-Verzeichnis für eine Version zurück."""
    if version == 0:
        return WORK_DIR / "data"
    return WORK_DIR / f"data_v{version}"


def save_card_version(card_pool: CardPool, version: int, source_dir: Path) -> Path:
    """Speichert den angepassten Kartensatz als neue Version."""
    target_dir = get_data_dir(version)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Spielregeln kopieren (falls vorhanden)
    spielregeln = source_dir / "spielregeln.json"
    if spielregeln.exists():
        shutil.copy2(spielregeln, target_dir / "spielregeln.json")

    # Monster-Karten speichern
    monster_path = source_dir / "monster_karten.json"
    with open(monster_path, "r", encoding="utf-8") as f:
        monster_data = json.load(f)

    # Kampfwerte aus dem CardPool übernehmen
    for card_data in monster_data["karten"]:
        monster = card_pool.get_monster_by_id(card_data["id"])
        if monster:
            card_data["kampfwerte"] = monster.kampfwerte

    with open(target_dir / "monster_karten.json", "w", encoding="utf-8") as f:
        json.dump(monster_data, f, indent=2, ensure_ascii=False)

    # Wissens-Karten speichern
    wisdom_path = source_dir / "wissens_karten.json"
    with open(wisdom_path, "r", encoding="utf-8") as f:
        wisdom_data = json.load(f)

    for card_data in wisdom_data["karten"]:
        wisdom = card_pool.get_wisdom_by_id(card_data["id"])
        if wisdom:
            card_data["kampfwerte"] = wisdom.kampfwerte

    with open(target_dir / "wissens_karten.json", "w", encoding="utf-8") as f:
        json.dump(wisdom_data, f, indent=2, ensure_ascii=False)

    return target_dir


# ---------------------------------------------------------------------------
# Training & Analyse
# ---------------------------------------------------------------------------

def train_for_iteration(
    data_dir: str,
    iteration: int,
    cfg: IterativeConfig,
    llm_cache: Optional[dict],
    reward_config: RewardConfig,
) -> PPO:
    """Trainiert einen PPO-Agenten für eine Balancing-Iteration."""
    output_dir = str(WORK_DIR / "output" / f"iter_{iteration:02d}")

    train_config = {
        "data_dir": data_dir,
        "num_players": cfg.num_players,
        "max_turns": cfg.max_turns,
        "bot_strategy": cfg.bot_strategy,
        "total_timesteps": cfg.timesteps_per_iteration,
        "n_envs": cfg.n_envs,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "output_dir": output_dir,
        "reward_config": reward_config,
        "device": cfg.device,
        "llm_cache": llm_cache,
        "llm_threshold": cfg.llm_threshold,
    }

    model = train_agent(train_config)
    return model


def run_analysis(
    model: PPO,
    data_dir: str,
    cfg: IterativeConfig,
    llm_cache: Optional[dict],
) -> Tuple[BalancingReport, List[Dict[str, Any]]]:
    """Führt die Balancing-Analyse durch und gibt Report + Episoden-Daten zurück."""
    loader = CardLoader(data_dir=data_dir)
    card_pool = loader.load()

    env = BosodoEnv(
        data_dir=data_dir,
        num_players=cfg.num_players,
        card_pool=card_pool,
        llm_cache=llm_cache if cfg.llm_threshold > 0.0 else None,
        llm_threshold=cfg.llm_threshold,
        max_turns=cfg.max_turns,
        bot_strategy=cfg.bot_strategy,
    )

    analyzer = BalancingAnalyzer(card_pool, llm_cache=llm_cache)
    episode_data: List[Dict[str, Any]] = []

    print(f"  Analyse: {cfg.analysis_episodes} Episoden...")
    for ep in range(cfg.analysis_episodes):
        obs, info = env.reset(seed=ep)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if "episode_metrics" in info:
            metrics = info["episode_metrics"]
            analyzer.add_episode(metrics)
            episode_data.append(metrics)

        if (ep + 1) % 100 == 0:
            print(f"    Episode {ep + 1}/{cfg.analysis_episodes}")

    report = analyzer.analyze()
    env.close()
    return report, episode_data


# ---------------------------------------------------------------------------
# Iterationslog
# ---------------------------------------------------------------------------

def save_iteration_log(
    iteration: int,
    version: int,
    report: BalancingReport,
    convergence: ConvergenceResult,
    changes: List[CardChange],
    output_dir: Path,
) -> None:
    """Speichert das Log einer Iteration als JSON."""
    log = {
        "iteration": iteration,
        "card_version": version,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "overall_score": report.overall_score,
            "avg_game_length": report.avg_game_length,
            "avg_defense_rate": report.avg_defense_rate,
            "defense_rate_per_symbol": report.defense_rate_per_symbol,
            "symbol_starvation_rate": report.symbol_starvation_rate,
            "avg_symbol_on_hand": report.avg_symbol_on_hand,
            "multi_symbol_defense_rate": report.multi_symbol_defense_rate,
        },
        "convergence": {
            "converged": convergence.converged,
            "defense_rates_ok": convergence.defense_rates_ok,
            "starvation_ok": convergence.starvation_ok,
            "game_length_ok": convergence.game_length_ok,
            "close_games_ok": convergence.close_games_ok,
            "monster_usage_ok": convergence.monster_usage_ok,
            "issues": convergence.details,
        },
        "changes": [
            {
                "card_id": c.card_id,
                "card_name": c.card_name,
                "card_type": c.card_type,
                "change_type": c.change_type,
                "symbol": c.symbol,
                "reason": c.reason,
                "old_symbols": c.old_symbols,
                "new_symbols": c.new_symbols,
            }
            for c in changes
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"iteration_{iteration:02d}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # Balancing-Report der Iteration speichern
    report_file = output_dir / f"balancing_report_iter_{iteration:02d}.json"
    # Nutze den Analyzer-Export-Mechanismus über ein dict
    report_data = {
        "iteration": iteration,
        "card_version": version,
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
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Zusammenfassung ausgeben
# ---------------------------------------------------------------------------

def print_iteration_summary(
    iteration: int,
    version: int,
    report: BalancingReport,
    convergence: ConvergenceResult,
    changes: List[CardChange],
) -> None:
    """Gibt eine Zusammenfassung der Iteration auf der Konsole aus."""
    print(f"\n{'=' * 60}")
    print(f"  ITERATION {iteration} — Kartenversion v{version}")
    print(f"{'=' * 60}")
    print(f"  Overall Score:        {report.overall_score:.1f}/100")
    print(f"  Ø Spiellänge:         {report.avg_game_length:.1f} Züge")
    print(f"  Ø Verteidigungsrate:  {report.avg_defense_rate:.1%}")
    print()
    print(f"  {'Symbol':<6} {'Defense':<12} {'Starvation':<14} {'Ø Hand':<10}")
    print(f"  {'-' * 42}")
    for s in SYMBOLS:
        dr = report.defense_rate_per_symbol.get(s, 0.0)
        sr = report.symbol_starvation_rate.get(s, 0.0)
        ah = report.avg_symbol_on_hand.get(s, 0.0)
        print(f"  {s:<6} {dr:<12.1%} {sr:<14.1%} {ah:<10.2f}")
    print(f"  Multi-Symbol Defense: {report.multi_symbol_defense_rate:.1%}")

    print(f"\n  Konvergenz: {'JA' if convergence.converged else 'NEIN'}")
    checks = [
        ("Defense-Rates", convergence.defense_rates_ok),
        ("Starvation", convergence.starvation_ok),
        ("Spiellänge", convergence.game_length_ok),
        ("Knappe Spiele", convergence.close_games_ok),
        ("Monster-Usage", convergence.monster_usage_ok),
    ]
    for name, ok in checks:
        print(f"    {'✓' if ok else '✗'} {name}")

    if convergence.details:
        print(f"\n  Offene Probleme:")
        for detail in convergence.details:
            print(f"    - {detail}")

    if changes:
        print(f"\n  Kartenänderungen ({len(changes)}):")
        for c in changes:
            arrow = "+" if c.change_type == "add_symbol" else "−"
            print(
                f"    {arrow} {c.card_id} ({c.card_name}): "
                f"{c.old_symbols} → {c.new_symbols}"
            )
            print(f"      Grund: {c.reason}")
    else:
        print(f"\n  Keine Kartenänderungen.")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BOSODO Iteratives Balancing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Pfad zur Trainingskonfiguration",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximale Anzahl Iterationen",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Trainings-Timesteps pro Iteration (überschreibt Config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Analyse-Episoden pro Iteration",
    )
    parser.add_argument(
        "--start-version",
        type=int,
        default=0,
        help="Startversion der Kartendaten (0 = data/, N = data_vN/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Arbeitsverzeichnis für Daten und Output (default: aktuelles Verzeichnis)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Analyse ohne Training (nutzt vorhandenes Modell)",
    )
    args = parser.parse_args()

    # Work-Dir setzen (global)
    global WORK_DIR
    if args.work_dir:
        WORK_DIR = Path(args.work_dir).resolve()
    print(f"Arbeitsverzeichnis: {WORK_DIR}")

    # Config laden — zuerst im Work-Dir suchen, dann im App-Dir
    config_path = WORK_DIR / args.config
    if not config_path.exists():
        config_path = APP_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    game_cfg = yaml_cfg.get("game", {})
    training_cfg = yaml_cfg.get("training", {})
    rewards_cfg = yaml_cfg.get("rewards", {})
    reward_config = RewardConfig(**{
        k: v for k, v in rewards_cfg.items() if hasattr(RewardConfig, k)
    })

    # Iterative Config zusammenbauen
    cfg = IterativeConfig(
        max_iterations=args.max_iterations,
        timesteps_per_iteration=args.timesteps or training_cfg.get("total_timesteps", 500_000),
        analysis_episodes=args.episodes,
        num_players=game_cfg.get("num_players", 4),
        max_turns=game_cfg.get("max_turns", 200),
        bot_strategy=game_cfg.get("bot_strategy", "strongest"),
        llm_threshold=game_cfg.get("llm_threshold", 0.0),
        n_envs=training_cfg.get("n_envs", 8),
        device=args.device or training_cfg.get("device", "auto"),
        start_version=args.start_version,
    )

    # LLM-Cache laden
    llm_cache = load_cache()
    if llm_cache:
        print(f"LLM-Cache geladen: {len(llm_cache)} Einträge")
    if cfg.llm_threshold > 0.0:
        print(f"LLM-Threshold aktiv: {cfg.llm_threshold}")

    # Output-Verzeichnis
    output_dir = WORK_DIR / "output" / "iterative_balancing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  BOSODO ITERATIVES BALANCING")
    print(f"{'#' * 60}")
    print(f"  Max. Iterationen:    {cfg.max_iterations}")
    print(f"  Timesteps/Iteration: {cfg.timesteps_per_iteration:,}")
    print(f"  Analyse-Episoden:    {cfg.analysis_episodes}")
    print(f"  Start-Version:       v{cfg.start_version}")
    print(f"  Konvergenz-Fenster:  {cfg.convergence_window} aufeinanderfolgende Iterationen")
    print(f"  Max. Änderungen:     {cfg.max_changes_per_iteration} pro Iteration")
    print(f"  Device:              {cfg.device}")
    print(f"{'#' * 60}\n")

    # Konvergenz-Historie
    convergence_history: List[bool] = []
    iteration_logs: List[Dict] = []

    current_version = cfg.start_version

    for iteration in range(1, cfg.max_iterations + 1):
        data_dir = get_data_dir(current_version)
        print(f"\n>>> Iteration {iteration}/{cfg.max_iterations} "
              f"(Kartendaten: {data_dir.name}) <<<\n")

        if not data_dir.exists():
            print(f"FEHLER: Verzeichnis {data_dir} existiert nicht!")
            sys.exit(1)

        # --- 1. Karten laden ---
        loader = CardLoader(data_dir=str(data_dir))
        card_pool = loader.load()
        print(f"  Karten geladen: {card_pool.num_monsters} Monster, "
              f"{card_pool.num_wisdoms} Wissenskarten")

        # --- 2. Agent trainieren ---
        if args.dry_run:
            model_path = WORK_DIR / "output" / "best_model" / "best_model.zip"
            print(f"  [Dry-Run] Lade vorhandenes Modell: {model_path}")
            model = PPO.load(str(model_path))
        else:
            print(f"  Training: {cfg.timesteps_per_iteration:,} Timesteps...")
            model = train_for_iteration(
                data_dir=str(data_dir),
                iteration=iteration,
                cfg=cfg,
                llm_cache=llm_cache,
                reward_config=reward_config,
            )

        # --- 3. Balancing analysieren ---
        report, episode_data = run_analysis(
            model=model,
            data_dir=str(data_dir),
            cfg=cfg,
            llm_cache=llm_cache,
        )

        # --- 4. Konvergenz prüfen ---
        convergence = check_convergence(report, episode_data, card_pool, cfg)
        convergence_history.append(convergence.converged)

        # Prüfe ob N aufeinanderfolgende Iterationen konvergiert sind
        window = cfg.convergence_window
        if (
            len(convergence_history) >= window
            and all(convergence_history[-window:])
        ):
            print_iteration_summary(iteration, current_version, report, convergence, [])
            save_iteration_log(iteration, current_version, report, convergence, [], output_dir)

            print(f"\n{'*' * 60}")
            print(f"  BALANCING KONVERGIERT nach {iteration} Iterationen!")
            print(f"  {window} aufeinanderfolgende stabile Iterationen erreicht.")
            print(f"  Finale Kartenversion: v{current_version}")
            print(f"  Finale Daten: {get_data_dir(current_version)}")
            print(f"{'*' * 60}\n")
            break

        # --- 5. Symbole anpassen ---
        if not convergence.converged:
            new_pool, changes = apply_balancing_rules(card_pool, report, cfg)
        else:
            new_pool, changes = card_pool, []

        # Zusammenfassung ausgeben
        print_iteration_summary(iteration, current_version, report, convergence, changes)
        save_iteration_log(iteration, current_version, report, convergence, changes, output_dir)

        # --- 6. Neue Version speichern ---
        if changes:
            next_version = current_version + 1
            new_dir = save_card_version(new_pool, next_version, data_dir)
            print(f"  Neue Kartenversion gespeichert: {new_dir}")
            current_version = next_version
        elif not convergence.converged:
            print("  Keine Regeländerungen möglich — Kriterien weiterhin nicht erfüllt.")
            print("  Manuelle Anpassung empfohlen. Beende Zyklus.")
            break

    else:
        # Max. Iterationen erreicht ohne Konvergenz
        print(f"\n{'!' * 60}")
        print(f"  MAX. ITERATIONEN ({cfg.max_iterations}) ERREICHT")
        print(f"  Balancing nicht konvergiert.")
        print(f"  Letzte Kartenversion: v{current_version}")
        print(f"{'!' * 60}\n")

    # Gesamt-Zusammenfassung speichern
    summary = {
        "total_iterations": min(iteration, cfg.max_iterations),
        "converged": len(convergence_history) >= cfg.convergence_window
                     and all(convergence_history[-cfg.convergence_window:]),
        "final_version": current_version,
        "final_data_dir": str(get_data_dir(current_version)),
        "convergence_history": convergence_history,
        "config": {
            "max_iterations": cfg.max_iterations,
            "timesteps_per_iteration": cfg.timesteps_per_iteration,
            "analysis_episodes": cfg.analysis_episodes,
            "convergence_window": cfg.convergence_window,
            "max_changes_per_iteration": cfg.max_changes_per_iteration,
        },
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Zusammenfassung gespeichert: {summary_file}")


if __name__ == "__main__":
    main()
