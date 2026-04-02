# CLAUDE CODE PROMPT — BOSODO RL-Balancing

## Kontext

Du arbeitest an einem Forschungsprojekt der HS Ansbach (SS 2026): "KI-gestütztes Balancing für Lernspiele". Das Ziel ist, das Kartenspiel **BOSODO — Digitale Achtsamkeit (Teen-Edition)** als Gymnasium-Environment nachzubauen und mit einem PPO-Agenten (Stable Baselines3) Spielabläufe zu simulieren, um die Kartenbalance automatisch zu analysieren.

Das Projekt läuft in einem **Docker-Container auf einem Server mit NVIDIA H100 GPU**.

---

## Projekt-Überblick

### Das Spiel (Casual-Regeln)
- 18 Monster-Karten (rot, zum Angreifen) und 18 Wissens-Karten (grün, zur Verteidigung)
- Jede Karte hat 1-3 Kampfsymbole: **BO** (pink), **SO** (cyan), **DO** (gelb)
- 2-6 Spieler, jeder startet mit 2 Monster + 2 Wissenskarten
- **Angriff**: Aktiver Spieler spielt Monster gegen Mitspieler
- **Verteidigung**: Verteidiger muss jedes Monster-Symbol mit passender Wissenskarte matchen
- **Erfolg**: Monster wird Trophäe des Verteidigers
- **Misserfolg**: Monster auf Ablagestapel
- **Sieg**: Erste Person mit 3 Trophäen gewinnt
- Kartenhand wird nach jedem Zug auf mindestens 2+2 aufgefüllt
- Im Uhrzeigersinn, optionaler Kartentausch pro Runde

### Kartendaten
Die JSON-Dateien liegen in `data/`:
- `data/monster_karten.json` — 18 Monster mit id, name, kampfwerte, beschreibung
- `data/wissens_karten.json` — 18 Wissenskarten im gleichen Format
- Symbol-Verteilung: Je 12× BO, SO, DO auf Monster UND Wissenskarten

---

## Bestehende Projektstruktur

```
bosodo-balancing/
├── bosodo_env/                  # Gymnasium Environment
│   ├── __init__.py              # Package + Gym-Registrierung "Bosodo-v0"
│   ├── card_loader.py           # MonsterCard, WisdomCard, CardPool, CardLoader
│   ├── game_state.py            # PlayerState, GameState (Spiellogik)
│   ├── env.py                   # BosodoEnv(gymnasium.Env) — Hauptklasse
│   ├── rewards.py               # RewardConfig, RewardCalculator
│   ├── metrics.py               # EpisodeMetrics
│   └── balancing.py             # BalancingAnalyzer, BalancingReport
├── agents/
│   └── __init__.py              # train_agent(), MetricsCallback, make_env()
├── config/
│   └── training_config.yaml     # Alle Parameter (Game, PPO, Rewards, Output)
├── data/
│   ├── monster_karten.json
│   └── wissens_karten.json
├── scripts/
│   ├── train.py                 # CLI: Training starten
│   ├── analyze.py               # CLI: Balancing-Analyse nach Training
│   └── simulate.py              # CLI: Quick-Simulation ohne Training
├── tests/
│   └── test_environment.py      # pytest-basierte Unit-Tests
├── Dockerfile                   # nvidia/cuda:12.4.1, Python 3.11
├── docker-compose.yml           # Services: training, tensorboard, analyze
├── requirements.txt
├── pyproject.toml
├── README.md
└── Anleitung.md
```

---

## Aufgaben für Claude Code

### Phase 1: Setup & Verifizierung
1. Lies `README.md` und `Anleitung.md` für den Gesamtüberblick
2. Installiere Dependencies: `pip install -r requirements.txt`
3. Führe Tests aus: `pytest tests/ -v`
4. Führe Quick-Simulation aus: `python scripts/simulate.py --episodes 50`
5. Behebe eventuelle Fehler

### Phase 2: Training
1. Starte Training: `python scripts/train.py --timesteps 500000 --device cuda`
2. Beobachte TensorBoard: `tensorboard --logdir output/tensorboard`
3. Prüfe Konvergenz der Rewards und Spielmetriken

### Phase 3: Analyse & Iteration
1. Führe Analyse aus: `python scripts/analyze.py --episodes 500`
2. Lies den Balancing-Report in `output/reports/balancing_report.json`
3. Identifiziere Probleme (zu hohe/niedrige Defense Rate, Spiellänge, Symbol-Imbalance)
4. Passe Reward-Gewichte in `config/training_config.yaml` an
5. Trainiere erneut und vergleiche

### Phase 4: Erweiterungen (optional)
- Bot-Strategien verbessern (aktuell sehr simpel: schwächstes Monster, stärkster Gegner)
- Observation Space erweitern (z.B. Karten der Gegner schätzen, Ablagestapel-Info)
- Expert-Modus implementieren (Charakterkarten, Lebenspunkte)
- Argumentations-Scoring per LLM-API einbauen
- Visualisierung der Spielverläufe erstellen

---

## Technische Details

### Gymnasium Environment API
- `BosodoEnv.reset()` → `(observation, info)`
- `BosodoEnv.step(action)` → `(observation, reward, terminated, truncated, info)`
- Action Space: `MultiDiscrete([8, num_players-1])` — [Monster-Index, Ziel-Index]
- Observation Space: `Box(low=0, high=10, shape=(obs_size,), dtype=float32)`

### PPO-Konfiguration (Stable Baselines3)
- Policy: MlpPolicy (2×64 Hidden Layer)
- Parallelisierung: SubprocVecEnv (8 Environments default)
- Callbacks: Checkpoint, Eval, Custom Metrics
- TensorBoard-Logging aktiv

### Docker (H100)
- Base Image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- Python 3.11
- CUDA 12.4 + cuDNN
- Volumes für data/, output/, config/

### Reward-System
Konfigurierbar über YAML. Wichtigste Metriken:
- `defense_rate` — Anteil erfolgreicher Verteidigungen (ideal: 40-60%)
- `avg_game_length` — Durchschnittliche Spiellänge (ideal: 20-40 Züge)
- `monster_diversity` — Wie gleichmäßig verschiedene Monster genutzt werden
- `overall_score` — Gesamtbalancing-Score (0-100)

---

## Wichtige Design-Entscheidungen

1. **Argumentationskomponente wird übersprungen**: Im echten Spiel muss man erklären, warum das Wissen gegen das Monster hilft. In der Simulation wird nur Symbol-Matching geprüft.

2. **Greedy Symbol-Matching**: Verteidigung nutzt die erste passende Karte. Optimale Multi-Karten-Zuordnung (Constraint Satisfaction) wäre eine Verbesserung.

3. **Einfache Bot-Strategie**: Bots sind bewusst simpel, damit Spiele schnell durchlaufen. Für realistischere Simulation könnten trainierte Bot-Agenten eingesetzt werden.

4. **Agent spielt nur einen Spieler**: Perspektive ist immer Spieler 0. Für vollständigeres Balancing könnten alle Spieler durch Agenten gesteuert werden (Self-Play).

---

## Hinweise zur Codequalität

- Python 3.10+ mit Type Hints
- Docstrings auf allen Klassen und öffentlichen Methoden (Deutsch)
- Konfiguration über YAML, nicht hardcoded
- Tests mit pytest
- Modular: Jede Datei hat eine klare Verantwortung
- Erweiterbar: Neue Kartentypen, Regeln, Algorithmen leicht hinzufügbar
