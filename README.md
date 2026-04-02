# BOSODO RL-Balancing

**KI-gestütztes Balancing für das BOSODO-Lernkartenspiel mittels Reinforcement Learning**

Hochschule Ansbach — Sommersemester 2026 | Strang A: Reinforcement Learning für Balancing

---

## Was ist das?

Dieses Projekt simuliert das Kartenspiel **BOSODO — Digitale Achtsamkeit (Teen-Edition)** als [Gymnasium](https://gymnasium.farama.org/)-Environment und trainiert einen **PPO-Agenten** (Proximal Policy Optimization) mit [Stable Baselines3](https://stable-baselines3.readthedocs.io/), um Spielverläufe zu simulieren und die Kartenbalance zu analysieren.

Das Ziel: Durch tausende simulierte Spiele automatisch erkennen, ob Monster zu stark oder zu schwach sind, ob Wissenskarten gut zu den Monstern passen und ob die Symbol-Verteilung (BO/SO/DO) fair ist.

---

## Quickstart

### Voraussetzungen

- Docker + Docker Compose
- NVIDIA GPU mit aktuellem Treiber (getestet auf H100)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### In 3 Schritten zum Training

```bash
# 1. Repository klonen
git clone <repo-url> && cd bosodo-balancing

# 2. Training starten (baut Docker-Image automatisch)
docker compose up training

# 3. TensorBoard öffnen (in zweitem Terminal)
docker compose up tensorboard
# → http://localhost:6006
```

### Nach dem Training: Balancing-Analyse

```bash
docker compose run --rm analyze
# → Bericht wird in output/reports/balancing_report.json gespeichert
```

---

## Ohne Docker (lokale Installation)

```bash
# Python 3.10+ und CUDA erforderlich
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Training starten
python scripts/train.py

# Quick-Simulation (ohne Training, zufällige Aktionen)
python scripts/simulate.py --episodes 100

# Analyse nach dem Training
python scripts/analyze.py --model output/best_model/best_model.zip
```

---

## Projektstruktur

```
bosodo-balancing/
├── bosodo_env/                 # Gymnasium Environment (Kernmodul)
│   ├── __init__.py             # Package + Gym-Registrierung
│   ├── card_loader.py          # Karten-Datenmodell & JSON-Loader
│   ├── game_state.py           # Spielzustand-Verwaltung
│   ├── env.py                  # Gymnasium Environment (Hauptklasse)
│   ├── rewards.py              # Modulares Reward-System
│   ├── metrics.py              # Episoden-Metriken
│   └── balancing.py            # Balancing-Analyse & Reports
│
├── agents/
│   └── __init__.py             # PPO-Training mit Stable Baselines3
│
├── config/
│   └── training_config.yaml    # Alle Trainingsparameter (YAML)
│
├── data/                       # Kartendaten (JSON)
│   ├── monster_karten.json     # 18 Monster-Karten
│   └── wissens_karten.json     # 18 Wissens-Karten
│
├── scripts/
│   ├── train.py                # Training starten
│   ├── analyze.py              # Balancing-Analyse
│   └── simulate.py             # Quick-Simulation
│
├── tests/
│   └── test_environment.py     # Unit-Tests
│
├── Dockerfile                  # Docker-Image (CUDA 12.4)
├── docker-compose.yml          # Training + TensorBoard + Analyse
├── requirements.txt            # Python-Dependencies
├── pyproject.toml              # Build-Konfiguration
├── Anleitung.md                # Ausführliche Projektdokumentation
└── README.md                   # ← Du bist hier
```

---

## Konfiguration

Alle Parameter werden über `config/training_config.yaml` gesteuert. Die wichtigsten:

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `training.total_timesteps` | 1.000.000 | Trainingsschritte gesamt |
| `training.n_envs` | 8 | Parallele Environments |
| `training.device` | `auto` | `auto` / `cuda` / `cpu` |
| `game.num_players` | 4 | Spieler pro Runde |
| `game.trophies_to_win` | 3 | Trophäen zum Sieg |
| `rewards.*` | Siehe YAML | Reward-Gewichte für Balancing |

CLI-Parameter überschreiben die YAML-Config:

```bash
python scripts/train.py --timesteps 2000000 --device cuda --n-envs 16
```

---

## Reward-Design

Das Reward-System ist modular und über die YAML-Config anpassbar. Es bewertet:

- **Kampf-Ergebnisse**: Verteidigung erfolgreich/fehlgeschlagen, Trophäen
- **Spielspaß-Proxies**: Optimale Spiellänge (nicht zu kurz, nicht zu lang)
- **Fairness**: Knappe Spiele werden belohnt, Blowouts bestraft
- **Stalemate-Vermeidung**: Extrem lange Spiele werden bestraft

→ Details in `Anleitung.md`, Abschnitt "Reward-System".

---

## Tests

```bash
# Alle Tests
pytest tests/ -v

# Mit Coverage
pytest tests/ -v --cov=bosodo_env
```

---

## Erweiterbarkeit

Das Projekt ist bewusst modular aufgebaut:

- **Neue Kartensätze**: Einfach neue JSON-Dateien in `data/` ablegen
- **Andere Spielregeln**: `GameState` anpassen (z.B. Expert-Modus)
- **Andere RL-Algorithmen**: In `agents/__init__.py` z.B. DQN oder A2C verwenden
- **Andere Reward-Funktionen**: `RewardConfig` erweitern oder `RewardCalculator` subclassen
- **LLM-Integration** (Strang B): Eigenes Modul unter `llm_experts/` anlegen

→ Ausführliche Erweiterungshinweise in `Anleitung.md`.

---

## Kontext: Forschungsprojekt

Dieses Repository ist **Strang A** des Forschungsprojekts "KI-gestütztes Balancing für Lernspiele" der HS Ansbach (SS 2026). Es wird ergänzt durch:

- **Strang B**: LLM-Expertensimulation (separates Repository)
- **Strang C**: Menschliche Spieltests & Vergleichsstudie

Primäre Forschungsfrage: *Kann ein KI-gestützter Balancing-Prozess die Qualität und Effizienz des Kartenspiel-Balancings gegenüber rein manuellem Balancing verbessern?*
