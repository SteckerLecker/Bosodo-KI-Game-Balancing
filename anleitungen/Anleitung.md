# Anleitung — BOSODO RL-Balancing

Ausführliche Dokumentation des gesamten Projekts. Erklärt jede Komponente, wo sie sich befindet, was sie tut und wie alles zusammenhängt.

---

## Inhaltsverzeichnis

1. [Überblick & Architektur](#1-überblick--architektur)
2. [Das Spiel: BOSODO Casual-Regeln](#2-das-spiel-bosodo-casual-regeln)
3. [Kartendaten (data/)](#3-kartendaten-data)
4. [Das Gymnasium Environment (bosodo_env/)](#4-das-gymnasium-environment-bosodo_env)
5. [Das Reward-System](#5-das-reward-system)
6. [Der RL-Agent (agents/)](#6-der-rl-agent-agents)
7. [Training (scripts/train.py)](#7-training-scriptstrainpy)
8. [Analyse (scripts/analyze.py)](#8-analyse-scriptsanalyzepy)
9. [Docker-Setup](#9-docker-setup)
10. [Tests](#10-tests)
11. [Erweiterbarkeit](#11-erweiterbarkeit)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Überblick & Architektur

Das Projekt bildet das BOSODO-Kartenspiel als digitale Simulation ab, damit ein RL-Agent tausende Spiele durchspielen und dabei lernen kann, wie gut die Karten aufeinander abgestimmt sind.

### Datenfluss

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Kartendaten     │─────▶│  Gymnasium Env   │─────▶│  PPO-Agent       │
│  (JSON-Dateien)  │      │  (Spielsimulation)│◀─────│  (SB3)           │
└─────────────────┘      └──────────────────┘      └──────────────────┘
                                  │                         │
                                  ▼                         ▼
                         ┌──────────────────┐      ┌──────────────────┐
                         │  Episoden-Metriken│─────▶│  Balancing-Report │
                         └──────────────────┘      └──────────────────┘
```

### Ablauf einer Trainings-Episode

1. **Environment wird zurückgesetzt** → Neue Kartendecks werden gemischt, Hände ausgeteilt
2. **Agent wählt Aktion** → Welches Monster spielen? Wen angreifen?
3. **Environment führt Spielzug aus** → Verteidigung wird geprüft, Trophäen vergeben
4. **Reward wird berechnet** → Basierend auf Kampfergebnis, Fairness, Spiellänge
5. **Bot-Spieler ziehen** → Regelbasierte Gegner spielen ihre Züge
6. **Wiederholung** bis jemand 3 Trophäen hat oder das Spiel abgebrochen wird
7. **Metriken werden gesammelt** → Welche Karten wurden gespielt, Verteidigungsraten etc.

---

## 2. Das Spiel: BOSODO Casual-Regeln

Die Simulation implementiert die **Casual-Regeln** des BOSODO-Spiels:

### Aufbau
- 18 Monster-Karten (rot) und 18 Wissens-Karten (grün) werden gemischt
- Jeder Spieler startet mit 2 Monster-Karten und 2 Wissens-Karten

### Kampfsymbole
Jede Karte trägt ein oder mehrere Symbole:
- **BO** (pink/magenta) — Thema: Bildschirmzeit, Konsum, Selbstbild
- **SO** (cyan/türkis) — Thema: Soziale Medien, Nachrichten, Mobbing
- **DO** (gelb) — Thema: Datenschutz, Online-Sicherheit, Grenzen

### Rundenablauf
1. **Angreifen**: Aktiver Spieler spielt eine Monster-Karte gegen einen Mitspieler
2. **Verteidigen**: Verteidiger versucht, alle Symbole des Monsters mit passenden Wissens-Karten zu matchen
3. **Ergebnis**: Erfolgreich → Monster wird Trophäe. Fehlgeschlagen → Monster auf Ablagestapel
4. **Nachziehen**: Alle ziehen auf mindestens 2 Monster + 2 Wissenskarten nach
5. **Weitergabe**: Nächster Spieler im Uhrzeigersinn ist dran

### Siegbedingung
Wer zuerst **3 Monster-Trophäen** gesammelt hat, gewinnt.

### Verteidigungslogik
Um ein Monster zu besiegen, muss der Verteidiger für **jedes Symbol** des Monsters mindestens eine Wissens-Karte mit dem gleichen Symbol ausspielen. Beispiel:

- Monster "Schlafräuber" hat Symbole: `[BO, SO]`
- Verteidiger braucht mindestens eine BO-Karte UND eine SO-Karte
- "Nachrichten limitieren" `[BO, SO]` allein reicht, da sie beide Symbole hat
- Alternativ: "Smart shoppen" `[BO]` + "Mobbing stoppen" `[SO]`

---

## 3. Kartendaten (data/)

### Dateien

| Datei | Inhalt | Anzahl |
|-------|--------|--------|
| `data/monster_karten.json` | Monster-Karten mit Kampfwerten | 18 Karten |
| `data/wissens_karten.json` | Wissens-Karten mit Kampfwerten | 18 Karten |

### JSON-Struktur einer Monster-Karte

```json
{
  "id": "M10",
  "name": "Schlafräuber",
  "kurzbeschreibung": "Schlafprobleme durch Videos",
  "kampfwerte": ["BO", "SO"],
  "beschreibung": "Dieser Räuber stiehlt dir den Schlaf...",
  "zitat": "Nur noch ein Video!",
  "belohnung_expert": "Simons Peitsche"
}
```

### Symbol-Verteilung im aktuellen Deck

**Monster-Karten** (wie oft jedes Symbol vorkommt):
- BO: 12× (in M01-M03, M10-M11, M14-M18)
- SO: 12× (in M04-M06, M10-M13, M16-M18)
- DO: 12× (in M07-M09, M12-M15, M16-M18)

**Wissens-Karten:**
- BO: 12× (in W01-W03, W10-W11, W14-W18)
- SO: 12× (in W04-W06, W10-W13, W16-W18)
- DO: 12× (in W07-W09, W12-W15, W16-W18)

Die Symbole sind aktuell **perfekt symmetrisch verteilt** (je 12× pro Typ). Das bedeutet rein rechnerisch eine ausgeglichene Verteilung — aber ob das Spiel sich auch *fair anfühlt*, hängt von der Kombinatorik ab (welche Karten der Spieler gerade auf der Hand hat).

### Eigene Kartensätze verwenden

Neue JSON-Dateien einfach im gleichen Format in `data/` ablegen. Die Dateinamen müssen `monster_karten.json` und `wissens_karten.json` heißen (oder per `data_dir`-Parameter ein anderes Verzeichnis angeben).

---

## 4. Das Gymnasium Environment (bosodo_env/)

Das Herzstück des Projekts. Hier wird das Kartenspiel als RL-kompatible Simulation umgesetzt.

### 4.1 card_loader.py — Kartendaten laden

**Wo**: `bosodo_env/card_loader.py`

Definiert die Datenklassen für Karten und lädt sie aus JSON:

- `MonsterCard`: Datenklasse mit `id`, `name`, `kampfwerte`, etc. Hat Properties wie `symbol_vector` (gibt `[1, 0, 1]` für `["BO", "DO"]` zurück) und `difficulty` (Anzahl Symbole).
- `WisdomCard`: Analog zu MonsterCard, für Wissens-Karten.
- `CardPool`: Container, der alle Monster und Wisdom-Karten hält.
- `CardLoader`: Lädt die JSON-Dateien und gibt einen `CardPool` zurück.

```python
loader = CardLoader(data_dir="../data/")
pool = loader.load()
print(pool.num_monsters)  # 18
```

### 4.2 game_state.py — Spielzustand

**Wo**: `bosodo_env/game_state.py`

Verwaltet den kompletten Zustand eines Spiels:

- `PlayerState`: Handkarten, Trophäen eines Spielers
- `GameState`: Decks, Ablagestapel, aktiver Spieler, Siegprüfung

Zentrale Methoden:
- `reset()` — Neues Spiel starten, Karten mischen und austeilen
- `execute_attack(monster_idx, target_idx)` — Einen Angriff durchführen
- `can_defend(defender, monster)` — Prüft ob Verteidigung möglich ist
- `refill_hands()` — Alle Spieler ziehen nach
- `advance_turn()` — Nächster Spieler ist dran
- `get_observation_vector()` — Numerischer Zustandsvektor für den RL-Agenten

Die **Verteidigungslogik** implementiert ein Greedy-Matching: Für jedes benötigte Symbol wird die erste verfügbare Wissenskarte mit diesem Symbol verwendet. Das ist eine Vereinfachung — im echten Spiel müsste der Spieler auch argumentieren, *warum* das Wissen gegen das Monster hilft. Diese argumentative Komponente wird in der Simulation ignoriert (bzw. immer als gültig angenommen).

### 4.3 env.py — Das Gymnasium Environment

**Wo**: `bosodo_env/env.py`

Die Hauptklasse `BosodoEnv` implementiert das `gymnasium.Env`-Interface:

**Aktionsraum** (`MultiDiscrete([8, num_players-1])`):
- Dimension 0: Welche Monster-Karte aus der Hand spielen (Index 0-7)
- Dimension 1: Welchen Mitspieler angreifen (relativer Index)

**Beobachtungsraum** (`Box`, float32):
Der Observation-Vektor enthält:
- 3 Werte: Summe der Monster-Symbole auf der Hand [BO, SO, DO]
- 3 Werte: Summe der Wissens-Symbole auf der Hand [BO, SO, DO]
- 2 Werte: Anzahl Monster- und Wissens-Handkarten
- N Werte: Trophäen pro Spieler (N = Spieleranzahl)
- N Werte: Aktueller Angreifer (One-Hot)
- 1 Wert: Normalisierte Rundenzahl

**Bot-Spieler**: Die Mitspieler werden durch einfache Bots gesteuert, die das schwächste Monster wählen und den führenden Spieler angreifen. Dies ist bewusst simpel gehalten — für die Balancing-Analyse ist es wichtiger, viele Spiele durchzuspielen als perfekte Gegner zu haben.

**Wichtig**: Der Agent steuert nur einen der Spieler. Zwischen den Zügen des Agenten spielen die Bots automatisch ihre Züge (`_play_bot_turns_until_agent()`).

### 4.4 rewards.py — Reward-System

**Wo**: `bosodo_env/rewards.py`

Siehe Abschnitt 5 unten für die ausführliche Erklärung.

### 4.5 metrics.py — Episoden-Metriken

**Wo**: `bosodo_env/metrics.py`

`EpisodeMetrics` sammelt detaillierte Statistiken während eines Spiels:
- Wie oft jede Monster-/Wissenskarte gespielt wurde
- Verteidigungsrate (erfolgreich vs. fehlgeschlagen)
- Monster-Diversität (werden alle Monster gleichmäßig genutzt?)
- Wer wie oft angegriffen wurde

Am Spielende wird ein `summary()`-Dictionary erstellt, das vom Balancing-Analyzer weiterverarbeitet wird.

### 4.6 balancing.py — Balancing-Analyse

**Wo**: `bosodo_env/balancing.py`

`BalancingAnalyzer` aggregiert die Metriken vieler Episoden und erstellt einen `BalancingReport`:

- **Symbol-Abdeckungsanalyse**: Passen Monster- und Wissens-Symbole zusammen?
- **Karten-Reports**: Wie oft wird jede Karte gespielt? Gibt es Ausreißer?
- **Problem-Identifikation**: Automatische Erkennung von Balancing-Problemen
- **Gesamtscore**: 0-100 Punkte für die Gesamtbalance

Der Report wird als JSON exportiert und kann für weitere Analyse verwendet werden.

---

## 5. Das Reward-System

Das Reward-System ist der **zentrale Hebel** für das Balancing. Es definiert, was der Agent als "gutes Spiel" lernt.

### Reward-Komponenten

| Komponente | Standard-Gewicht | Zweck |
|------------|-----------------|-------|
| `defense_success` | +1.0 | Belohnt erfolgreiche Verteidigung |
| `defense_fail` | -0.3 | Leichte Strafe für fehlgeschlagene Verteidigung |
| `trophy_earned` | +2.0 | Bonus für gewonnene Trophäe |
| `game_won` | +10.0 | Großer Bonus für Spielsieg |
| `game_lost` | -5.0 | Strafe für Niederlage |
| `game_length_bonus` | +0.5 | Bonus wenn Spiellänge im Idealbereich |
| `close_game_bonus` | +1.0 | Bonus für knappes Ergebnis |
| `blowout_penalty` | -1.0 | Strafe für einseitiges Ergebnis |
| `stalemate_penalty` | -2.0 | Strafe wenn Spiel >150 Züge dauert |

### Anpassung der Rewards

Rewards werden über `config/training_config.yaml` konfiguriert. Um beispielsweise längere Spiele zu bevorzugen:

```yaml
rewards:
  ideal_game_length: 50        # Von 30 auf 50 erhöhen
  game_length_tolerance: 20    # Mehr Toleranz
  stalemate_penalty: -1.0      # Weniger Strafe für lange Spiele
```

### Balancing-Philosophie

Die Rewards sind so designed, dass der Agent lernt, Spiele zu spielen, die:
- **nicht zu schnell enden** (≥15 Züge)
- **nicht endlos dauern** (≤45 Züge)
- **knapp ausgehen** (alle Spieler haben ähnlich viele Trophäen)
- **verschiedene Karten nutzen** (kein "eine Karte dominiert alles")

Wenn der trainierte Agent nach vielen Episoden bestimmte Karten nie nutzt oder Spiele immer nach 5 Zügen enden, deutet das auf ein Balancing-Problem hin.

---

## 6. Der RL-Agent (agents/)

**Wo**: `agents/__init__.py`

### PPO (Proximal Policy Optimization)

PPO ist ein Policy-Gradient-Algorithmus, der gut mit Multi-Discrete Action Spaces umgehen kann. Er ist stabil im Training und erfordert weniger Hyperparameter-Tuning als viele andere RL-Algorithmen.

### Technische Details

- **Policy**: `MlpPolicy` (Multilayer Perceptron mit 2 Hidden Layers à 64 Neuronen)
- **Parallelisierung**: `SubprocVecEnv` für echte Multiprocessing-Parallelisierung
- **Callbacks**:
  - `CheckpointCallback` — Speichert Modell alle N Schritte
  - `EvalCallback` — Evaluiert auf separatem Environment, speichert bestes Modell
  - `MetricsCallback` — Sammelt Balancing-Metriken und loggt in TensorBoard

### Factory-Funktion

```python
from agents import train_agent

model = train_agent({
    "data_dir": "data/",
    "total_timesteps": 1_000_000,
    "n_envs": 8,
    "device": "cuda",
})
```

---

## 7. Training (scripts/train.py)

**Wo**: `scripts/train.py`

### Ablauf

1. Liest `config/training_config.yaml`
2. Erstellt `RewardConfig` aus den YAML-Werten
3. Ruft `train_agent()` auf
4. Speichert Modell und Metriken in `output/`

### CLI-Parameter

```bash
python scripts/train.py \
    --config config/training_config.yaml \
    --timesteps 2000000 \
    --device cuda \
    --n-envs 16
```

### Ausgabestruktur

Nach dem Training enthält `output/`:

```
output/
├── best_model/
│   └── best_model.zip          # Bestes Modell (nach Eval-Reward)
├── checkpoints/
│   ├── bosodo_ppo_50000_steps.zip
│   ├── bosodo_ppo_100000_steps.zip
│   └── ...
├── eval_logs/
│   └── evaluations.npz         # Evaluations-Ergebnisse
├── metrics/
│   └── training_metrics.json   # Alle Episoden-Metriken
└── tensorboard/
    └── PPO_1/                  # TensorBoard-Logs
```

### TensorBoard

```bash
tensorboard --logdir output/tensorboard
```

Zeigt:
- Episoden-Reward über Zeit
- Verteidigungsrate pro Episode
- Durchschnittliche Spiellänge
- Monster-Diversität
- Policy/Value Loss

---

## 8. Analyse (scripts/analyze.py)

**Wo**: `scripts/analyze.py`

### Was es tut

1. Lädt das trainierte Modell
2. Spielt N Episoden mit deterministischen Aktionen
3. Sammelt Metriken mit dem `BalancingAnalyzer`
4. Gibt Report auf der Konsole aus
5. Speichert JSON-Report in `output/reports/`

### Ausgabe-Beispiel

```
=== BOSODO Balancing-Analyse ===
Episoden analysiert: 500
Ø Spiellänge: 28.3 Züge
Ø Verteidigungsrate: 47.2%
Gesamtscore: 82.5/100

Symbol-Abdeckung:
  ✓ BO: Monster=12, Wissen=12, Ratio=1.0
  ✓ SO: Monster=12, Wissen=12, Ratio=1.0
  ✓ DO: Monster=12, Wissen=12, Ratio=1.0
```

### Quick-Simulation (scripts/simulate.py)

Für schnelle Tests ohne Training:

```bash
python scripts/simulate.py --episodes 100 --render
```

Spielt Runden mit zufälligen Aktionen und gibt erste Balancing-Einschätzungen.

---

## 9. Docker-Setup

### Dockerfile

Basiert auf `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` mit Python 3.11. Optimiert für H100 GPUs.

### Docker Compose Services

| Service | Zweck | Port |
|---------|-------|------|
| `training` | PPO-Training mit GPU | — |
| `tensorboard` | Training-Monitoring | 6006 |
| `analyze` | Balancing-Analyse (on-demand) | — |

### Befehle

```bash
# Training starten
docker compose up training

# Training im Hintergrund
docker compose up -d training

# TensorBoard starten
docker compose up tensorboard

# Analyse ausführen (nach Training)
docker compose run --rm analyze

# Logs ansehen
docker compose logs -f training

# Alles stoppen
docker compose down
```

### GPU-Verifizierung

```bash
docker compose run --rm training python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Volumes

- `./data` → `/app/data` (read-only) — Kartendaten
- `./output` → `/app/output` — Trainings-Ergebnisse
- `./config` → `/app/config` (read-only) — Konfiguration

---

## 10. Tests

**Wo**: `tests/test_environment.py`

### Test-Kategorien

- **CardLoader**: Karten korrekt geladen? Symbole valide?
- **GameState**: Spielstart, Nachziehen, Verteidigung, Siegbedingung
- **BosodoEnv**: Reset, Step, komplette Episode, Observation-Bounds
- **Rewards**: Konfiguration, Berechnung
- **Metrics**: Leere Metriken, Zusammenfassung

### Ausführen

```bash
# Alle Tests
pytest tests/ -v

# Einzelne Testklasse
pytest tests/test_environment.py::TestGameState -v

# Mit Output
pytest tests/ -v -s
```

---

## 11. Erweiterbarkeit

Das Projekt ist bewusst modular aufgebaut, um Erweiterungen zu erleichtern.

### Neue Kartensätze testen

1. Neue JSON-Dateien erstellen (gleiches Format wie in `data/`)
2. In ein neues Verzeichnis legen, z.B. `data_v2/`
3. Training starten mit `--data-dir data_v2/` (CLI) oder in YAML anpassen

### Expert-Modus implementieren

Der Expert-Modus (mit Charakter- und Lebenspunktekarten) kann durch Erweitern von `GameState` implementiert werden:

1. `CharacterCard` und `LifePointCard` Datenklassen in `card_loader.py` hinzufügen
2. `PlayerState` um `character` und `life_points` erweitern
3. `GameState` um Expert-Logik ergänzen (Spezialfähigkeiten, Lebenspunkte)

### Andere RL-Algorithmen

In `agents/__init__.py` kann PPO durch andere SB3-Algorithmen ersetzt werden:

```python
from stable_baselines3 import A2C, DQN

# A2C statt PPO
model = A2C("MlpPolicy", env, ...)

# DQN (nur für Discrete action spaces — erfordert Änderung am Action Space)
```

### Argumentations-Komponente

Im echten Spiel muss der Verteidiger *argumentieren*, warum das Wissen gegen das Monster hilft. Dies könnte als LLM-basierte Bewertung implementiert werden:

1. Neues Modul `llm_experts/` anlegen
2. LLM-Prompt erstellen, der Monster-Beschreibung + Wissenskarten-Beschreibung erhält
3. LLM bewertet, ob die Argumentation schlüssig ist (0-1 Score)
4. Score als zusätzlichen Faktor in `can_defend()` einbauen

### Balancing-Vorschläge generieren

Der `BalancingAnalyzer` könnte erweitert werden, um automatisch Vorschläge zu machen:

```python
# In balancing.py ergänzen:
def suggest_changes(self, report: BalancingReport) -> List[str]:
    suggestions = []
    if report.avg_defense_rate < 0.3:
        suggestions.append("Füge mehr Wissenskarten mit SO-Symbol hinzu")
    if report.avg_game_length < 15:
        suggestions.append("Erhöhe die Trophäen-Anforderung auf 4")
    return suggestions
```

---

## 12. Troubleshooting

### CUDA nicht erkannt

```
RuntimeError: No CUDA GPUs are available
```

→ Prüfe `nvidia-smi` auf dem Host. Stelle sicher, dass das NVIDIA Container Toolkit installiert ist:
```bash
nvidia-smi
docker run --rm --runtime=nvidia nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Out of Memory (GPU)

→ Reduziere `n_envs` in der Config oder per CLI:
```bash
python scripts/train.py --n-envs 4
```

PPO ist hauptsächlich CPU-intensiv (Environment-Simulation). Die GPU wird nur für das neuronale Netz genutzt, das hier klein ist.

### Training konvergiert nicht

→ Mögliche Ursachen:
- Reward-Gewichte zu extrem: `game_won: 100` dominiert alle anderen Signale
- Zu wenige Timesteps: Für ein Kartenspiel mit Zufallselement braucht man ≥500k Steps
- Lernrate zu hoch: Probiere `learning_rate: 0.0001`

### Spiele enden sofort

→ Prüfe die Kartendaten: Wenn ein Symbol bei Monstern vorkommt, aber keine Wissenskarte es hat, kann nie verteidigt werden. Nutze `scripts/simulate.py` zum schnellen Testen.

### JSON-Ladefehler

→ Stelle sicher, dass `data/monster_karten.json` und `data/wissens_karten.json` existieren und valides JSON enthalten.

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **BO/SO/DO** | Die drei Kampfsymbole des Spiels |
| **Trophäe** | Erfolgreich besiegtes Monster (= gewonnene Verteidigung) |
| **Episode** | Ein komplettes Spiel von Start bis Sieg |
| **Timestep** | Ein einzelner Schritt des RL-Agenten |
| **Reward** | Belohnungssignal, das dem Agenten sagt, wie gut seine Aktion war |
| **Observation** | Numerischer Vektor, der den Spielzustand beschreibt |
| **Defense Rate** | Anteil der Angriffe, bei denen der Verteidiger erfolgreich war |
| **PPO** | Proximal Policy Optimization — der verwendete RL-Algorithmus |
| **SB3** | Stable Baselines3 — die RL-Bibliothek |
| **Gymnasium** | Standard-API für RL-Environments (Nachfolger von OpenAI Gym) |
