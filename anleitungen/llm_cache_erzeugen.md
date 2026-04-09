# LLM-Cache erzeugen

Der LLM-Cache (`llm_cache.json`) speichert für jede Monster-Wissenskarten-Kombination einen Relevanz-Score (0.0–1.0), der vom LLM berechnet wird. Da das Scoring API-Calls kostet und Zeit braucht, wird der Cache einmal gebaut und dann wiederverwendet.

---

## Voraussetzungen

### 1. Kartenstruktur im Data-Verzeichnis

Das Zielverzeichnis braucht zwei Dateien:

```
data/meine_edition/
├── monster_karten.json
└── wissens_karten.json
```

Jede Datei muss eine `karten`-Liste enthalten:

```json
{
  "karten": [
    {
      "id": "M01",
      "name": "Kartenname",
      "beschreibung": "Ausführliche Beschreibung des Problems...",
      "kampfwerte": ["BO", "SO"]
    }
  ]
}
```

> Das Feld `beschreibung` ist entscheidend — das LLM bewertet anhand dieser Texte.

### 2. LLM-Provider konfigurieren

Erstelle eine `.env`-Datei im Projektstamm (falls noch nicht vorhanden):

**Option A — Ollama (lokal, kostenlos):**

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:4b
```

Ollama muss laufen und das Modell muss heruntergeladen sein:
```bash
ollama pull qwen2.5:4b
```

**Option B — OpenRouter (cloud, kostenpflichtig):**

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

---

## Data-Dir in der Config setzen (optional)

Wenn du keine Parameter übergibst, liest das Skript `data_dir` aus `config/training_config.yaml`:

```yaml
game:
  data_dir: "data/meine_edition/"
```

---

## Cache erzeugen

### Standard (data_dir aus Config)

```bash
python -m llm_experts.batch_evaluate
```

### Mit expliziten Pfaden (überschreibt Config)

```bash
# Eingabe- und Ausgabeordner identisch
python -m llm_experts.batch_evaluate --data-dir data/meine_edition

# Kurze Schreibweise
python -m llm_experts.batch_evaluate -d data/meine_edition

# Eingabe- und Ausgabeordner separat
python -m llm_experts.batch_evaluate --data-dir data/meine_edition --output-dir data/cache_output

# Kurze Schreibweise
python -m llm_experts.batch_evaluate -d data/meine_edition -o data/cache_output
```

**Parameter:**

| Parameter | Kurz | Beschreibung |
|-----------|------|--------------|
| `--data-dir` | `-d` | Ordner mit `monster_karten.json` und `wissens_karten.json` |
| `--output-dir` | `-o` | Ordner, in dem `llm_cache.json` gespeichert wird (Standard: identisch mit `--data-dir`) |

Das Skript:
1. Liest den Eingabeordner aus `--data-dir` oder `config/training_config.yaml`
2. Lädt alle Monster- und Wissenskarten
3. Berechnet den Score für jede Kombination (M × W Paare)
4. Speichert den Fortschritt nach jeder Paarung in `<output-dir>/llm_cache.json`

Ausgabe-Beispiel:
```
Provider   : qwen2.5:4b
Eingabe    : /app/data/meine_edition
Ausgabe    : /app/data/meine_edition
Gesamt     : 324 | Cache: 0 | Neu: 324

100%|████| 324/324 [12:34<00:00, Schlafräuber_Nachrichten limitieren  0.92  Adressiert direkt...]

Fertig. data/meine_edition/llm_cache.json (324 Einträge)
```

---

## Abbruch & Fortsetzung

Das Skript ist **idempotent**: bereits bewertete Paare werden übersprungen. Bei einem Abbruch (Ctrl+C, Netzwerkfehler, etc.) einfach neu starten — der Fortschritt bleibt erhalten.

---

## Cache-Inhalt verstehen

```json
{
  "M01_W03": {
    "score": 0.87,
    "begruendung": "Wissenskarte adressiert direkt das Problem der Monster-Karte"
  },
  "M01_W04": {
    "score": 0.12,
    "begruendung": "Kein inhaltlicher Bezug zwischen den Karten"
  }
}
```

Der `score` wird im Spiel gegen `llm_threshold` aus der Config verglichen. Liegt der Score darunter, gilt die Wissenskarte als nicht spielbar gegen dieses Monster.

---

## Cache analysieren

Nach dem Erzeugen kann der Cache mit dem Analyse-Skript ausgewertet werden:

```bash
python llm_experts/analyse_llmCache.py
```

Das zeigt z.B. welche Monster schwer zu verteidigen sind oder welche Wissenskarten kaum Relevanz haben.

---

## Typische Probleme

| Problem | Ursache | Lösung |
|---------|---------|--------|
| `OPENROUTER_API_KEY ist nicht gesetzt` | `.env` fehlt oder falsch benannt | `.env` im Projektstamm anlegen |
| `Leere Antwort vom Modell` | Qwen3-Modell mit aktivem Thinking-Modus | `OLLAMA_MODEL=qwen2.5:4b` statt `qwen3` verwenden |
| Timeout-Fehler | Ollama-Modell zu langsam / nicht gestartet | `ollama serve` starten, kleineres Modell wählen |
| `monster_karten.json not found` | Falscher `data_dir`-Pfad in der Config | Pfad in `config/training_config.yaml` prüfen |
| Score immer 0.5 | Beschreibung der Karten zu kurz/generisch | `beschreibung`-Felder in den JSON-Dateien ausbauen |
