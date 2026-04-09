# Balancing-Pipeline ausführen

Die Balancing-Pipeline überarbeitet iterativ Kartentexte, damit jede Karte mit genau 4–6 Partnern stark matcht (Score ≥ 0.75). Sie erkennt und behebt zwei Probleme:

- **Zu starke Karten** (> 5 Matches) — zu generisch, matchen mit fast allem → spezifischer formulieren
- **Zu schwache Karten** (< 2 Matches) — matchen mit fast niemandem → besser formulieren, damit sie mit passenden Partnern matchen

Dazu nutzt sie drei LLM-Personas in einer Schleife:

1. **Analyst** — analysiert die Score-Matrix, identifiziert zu starke und zu schwache Karten (Priorität: zu schwache Karten)
2. **Game Designer** — formuliert max. 3 Kartentexte pro Iteration um (spezifischer oder breiter, je nach Problem)
3. **Matcher** — berechnet die Scores für geänderte Paare neu

---

## Voraussetzungen

### 1. LLM-Provider konfigurieren

Die Pipeline nutzt den gleichen Provider wie `batch_evaluate`. In der `.env` im Projektstamm muss konfiguriert sein:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Alternativ mit Ollama (lokal):

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:4b
```

### 2. LLM-Cache muss existieren

Der `llm_cache.json` muss bereits erzeugt worden sein (siehe [llm_cache_erzeugen.md](llm_cache_erzeugen.md)). Die Pipeline baut darauf auf und modifiziert ihn iterativ.

### 3. Kartendaten vorhanden

Im Data-Verzeichnis müssen `monster_karten.json` und `wissens_karten.json` liegen:

```
data/scrum_edition/
├── monster_karten.json
├── wissens_karten.json
└── llm_cache.json
```

---

## Pipeline starten

### Standard (liest `data_dir` aus `config/training_config.yaml`):

```bash
python -m scripts.run_balancing_pipeline
```

### Mit explizitem Data-Verzeichnis:

```bash
python -m scripts.run_balancing_pipeline --data-dir data/scrum_edition
```

### Mit angepasster Iterationsanzahl:

```bash
python -m scripts.run_balancing_pipeline --max-iterations 20
```

### Alle Optionen kombiniert:

```bash
python -m scripts.run_balancing_pipeline --data-dir data/scrum_edition --output-dir results/balancing_run_01 --max-iterations 10
```

---

## Was passiert während der Ausführung?

Pro Iteration durchläuft die Pipeline folgende Schritte:

```
1. ANALYST    → Analysiert Score-Matrix, findet Top-3 Problemkarten
                (zu stark: > 5 Matches, zu schwach: < 2 Matches)
2. DESIGNER   → Überarbeitet max. 3 Kartentexte
                (zu stark → spezifischer, zu schwach → besser formulieren)
3. MATCHER    → Bewertet alle Paare der geänderten Karten neu via LLM
4. CHECK      → Prüft ob Ø Matches/Karte ≤ 5 UND keine Karte < 2 Matches
                → Ja: Fertig / Nein: Weiter
```

### Konsolenausgabe (Beispiel):

```
============================================================
  BOSODO Balancing-Pipeline
  Daten: data/scrum_edition
  LLM: openai/gpt-4o-mini
  Ziel: Ø ≤ 5 Matches/Karte, keine Karte < 2 (Threshold 0.75)
============================================================

Initiale Statistik:
  Ø Monster-Matches: 6.89
  Ø Wissen-Matches:  6.89
  Ø Gesamt:          6.89
  ⚠ Zu schwache Karten (< 2 Matches): 1

────────────────────────────────────────────────────────────
  Iteration 1
────────────────────────────────────────────────────────────

[1/3] Analyst: Analysiere Score-Matrix...
  → M12 (0 Matches, ↓ zu schwach): Beschreibung klarer formulieren...
  → K01 (13 Matches, ↑ zu stark): Fokus auf einen konkreten Aspekt...
  → K05 (12 Matches, ↑ zu stark): Einschränken auf spezifische Technik...

[2/3] Game Designer: Kartentexte überarbeiten...
  → M12: Klarer auf passendes Szenario fokussiert
  → K01: Spezifischer auf testbare Software fokussiert
  → K05: Auf Sprint-Burndown eingeschränkt

[3/3] Matcher: 3 Karten neu bewerten...
    [1/54] M01_K01: 0.45 – Kein direkter Bezug...
    ...

  Ergebnis: Ø 5.8 (vorher: 6.89)
  ✓ Verbesserung: schwache Karten: 1 → 0, Ø: 6.89 → 5.8
```

---

## Ergebnisse & Ausgabedateien

Die Pipeline speichert alle Zwischenstände unter `<data-dir>/balancing_runs/` (oder dem mit `--output-dir` angegebenen Pfad):

```
balancing_runs/
├── iteration_01/
│   ├── monster_karten_vor.json      # Kartentexte vor der Iteration
│   ├── wissens_karten_vor.json
│   ├── llm_cache_vor.json           # Cache vor der Iteration
│   ├── monster_karten_nach.json     # Kartentexte nach der Iteration
│   ├── wissens_karten_nach.json
│   ├── llm_cache_nach.json          # Cache nach der Iteration
│   └── balance_report.md            # Markdown-Report mit Tabellen
├── iteration_02/
│   └── ...
├── monster_karten_best.json         # Bestes Ergebnis (Karten)
├── wissens_karten_best.json
├── llm_cache_best.json              # Bester Cache
└── pipeline_summary.json            # Gesamtzusammenfassung
```

> Die Original-Dateien unter `data/scrum_edition/` werden **nicht** verändert.

### balance_report.md

Jeder Iterationsreport enthält:
- Übersichtstabelle (Ø Matches Monster/Wissen/Gesamt vs. Ziel)
- Monster-Match-Tabelle (sortiert nach Anzahl Matches)
- Wissen-Match-Tabelle (sortiert nach Anzahl Matches)
- Durchgeführte Änderungen (alt/neu/Begründung)

### pipeline_summary.json

Am Ende der Pipeline wird eine Zusammenfassung geschrieben:

```json
{
  "status": "erfolg",
  "iterationen": 4,
  "initial_avg": 6.89,
  "final_avg": 4.8,
  "history": [
    {
      "iteration": 1,
      "avg_before": 6.89,
      "avg_after": 5.8,
      "changed_cards": ["K01", "K05", "M08"]
    }
  ]
}
```

---

## Abbruchbedingungen

| Bedingung | Verhalten |
|-----------|-----------|
| Ø Matches/Karte ≤ 5 **und** keine Karte < 2 Matches | Pipeline stoppt mit Status `erfolg` |
| 10 Iterationen ohne Verbesserung | Pipeline stoppt mit Status `abbruch` |
| Iteration verschlechtert Metrik | Automatischer Rollback auf vorherige Version |
| Max. 15 Iterationen erreicht | Pipeline stoppt mit Status `abbruch` |

**Verbesserung** wird in folgender Priorität bewertet:
1. Weniger zu schwache Karten (< 2 Matches) — höchste Priorität
2. Niedrigerer Ø Matches/Karte — bei gleich vielen schwachen Karten

Bei einem Rollback werden Kartentexte und Cache auf den letzten besseren Stand zurückgesetzt. Die Pipeline versucht dann in der nächsten Iteration andere Karten zu optimieren.

---

## Wichtige Hinweise

- **Kosten**: Pro Iteration werden ca. 3 LLM-Calls (Analyst + Designer) + bis zu 54 Scorer-Calls (3 Karten × 18 Partner) gemacht. Bei OpenRouter mit gpt-4o-mini sind das ca. $0.01–0.02 pro Iteration.
- **Dauer**: Eine Iteration dauert je nach Provider 1–5 Minuten.
- **Originaldaten bleiben unverändert**: Die Pipeline schreibt nie in das Original-Datenverzeichnis (`data/scrum_edition/`). Das beste Ergebnis wird als `*_best.json` im Output-Verzeichnis gespeichert.
- **Fortsetzen**: Die Pipeline ist aktuell nicht fortsetzbar — bei einem Abbruch (Ctrl+C) startet sie von vorne. Die Snapshots bleiben aber erhalten.

---

## Typische Probleme

| Problem | Ursache | Lösung |
|---------|---------|--------|
| `OPENROUTER_API_KEY ist nicht gesetzt` | `.env` fehlt oder Provider falsch | `.env` im Projektstamm prüfen |
| `llm_cache.json not found` | Cache wurde noch nicht erzeugt | Erst `python -m llm_experts.batch_evaluate` ausführen |
| Pipeline ändert nichts | LLM liefert keine verwertbaren JSON-Antworten | Anderes/größeres Modell in `.env` konfigurieren |
| Rollback in jeder Iteration | Änderungen verschlechtern konsistent | Threshold oder Ziel-Matches in `balancing_pipeline.py` anpassen |
| Karte hat 0 Matches | Karte ist zu spezifisch oder schlecht formuliert | Pipeline erkennt dies automatisch und priorisiert die Überarbeitung |
| Sehr langsam | Viele Re-Scoring-Calls bei Ollama | Auf OpenRouter wechseln oder kleineres Modell nutzen |
