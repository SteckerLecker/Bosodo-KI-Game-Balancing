# Lernkartenspiel — Iterative Balancing-Pipeline

## Projektkontext

Ich habe ein **Lernkartenspiel** mit zwei Kartentypen:

- **Monsterkarten (M01–M18):** Beschreiben agile Anti-Patterns / Problemszenarien
- **Wissenskarten (K01–K18):** Beschreiben agile Methoden / Lösungsstrategien

Jedes Monster-Wissenskarten-Paar hat einen **Match-Score (0–1)**, der angibt, wie gut die Wissenskarte das Monster bekämpft. Diese Scores liegen in `data/scrum_edition/llm_cache.json` als Key-Value-Paare (z.B. `"M01_K01": {"score": 0.9, "begruendung": "..."}`).

### Problem

Das Spiel ist aktuell **schlecht balanciert**: 80% der Paare matchen (Score ≥ 0.65). Manche Monster matchen mit allen 18 Karten, manche Wissenskarten ebenso. Die Karten sind zu generisch formuliert.

### Ziel

Durch iteratives Überarbeiten der Kartentexte soll jede Karte nur noch mit **4–6 Partnern** stark matchen (Score ≥ 0.75), sodass Spieler gezielt die richtige Karte auswählen müssen.

---

## Architektur: 3-Persona-Pipeline

Implementiere den folgenden Workflow als **automatisierten Loop** mit Zwischenspeicherung jeder Iteration.

### Persona 1 — Balancing-Analyst

**Rolle:** Analysiert die aktuelle Score-Matrix und identifiziert Problemstellen.

**Input:** `llm_cache.json` (Score-Matrix), Kartentexte (aus separater Datei oder Struktur)

**Aufgabe:**
- Erstelle eine Übersicht: Wie viele Matches (≥ 0.75) hat jedes Monster / jede Wissenskarte?
- Identifiziere die **Top-3 problematischsten Karten** (die mit den meisten Matches)
- Für jede problematische Karte: Analysiere *warum* sie so breit matcht (zu allgemein? zu viele Überschneidungen?)
- Gib dem Game Designer **konkrete Anweisungen**, z.B.:
  - "M01 matcht 18/18 — zu generisch. Fokus einengen auf ein spezifisches Planungsproblem."
  - "K10 matcht 18/18 — zu breit. Auf eine konkrete Technik einschränken."

**Output:** JSON mit Analyse + Überarbeitungsanweisungen

```json
{
  "iteration": 3,
  "zusammenfassung": { "avg_matches_monster": 14.2, "avg_matches_karte": 14.5 },
  "problemkarten": [
    {
      "id": "M01",
      "typ": "monster",
      "aktuelle_matches": 18,
      "analyse": "Zu allgemein formuliert...",
      "anweisung": "Fokus auf spezifisches Szenario..."
    }
  ]
}
```

### Persona 2 — Game Designer

**Rolle:** Überarbeitet Kartentexte basierend auf den Anweisungen des Analysten.

**Input:** Aktuelle Kartentexte + Anweisungen vom Analysten

**Regeln:**
- Ändere **maximal 3 Karten pro Iteration** (Änderungen isoliert nachvollziehbar halten)
- Behalte den **Lerninhalt** bei — nur die Formulierung wird spezifischer
- Monster: Konkretes Szenario beschreiben (Wer? Was genau? Welcher Kontext?)
- Wissenskarten: Konkrete Technik/Methode statt allgemeines Prinzip
- Gib die alten und neuen Texte als Diff zurück

**Output:** JSON mit geänderten Karten

```json
{
  "aenderungen": [
    {
      "id": "M01",
      "alt": "Bisheriger Text...",
      "neu": "Neuer spezifischerer Text...",
      "begruendung": "Warum diese Änderung..."
    }
  ]
}
```

### Persona 3 — Matcher

**Rolle:** Berechnet die Match-Scores für die geänderten Karten neu.

**Input:** Geänderter Kartentext + alle potenziellen Partner

**Aufgabe:**
- Berechne nur die Scores für die **geänderten Paare** (nicht die gesamte Matrix)
- Verwende den gleichen Bewertungsmaßstab wie in der bestehenden `llm_cache.json`
- Aktualisiere die `llm_cache.json` mit den neuen Scores

**Output:** Aktualisierte Score-Einträge

---

## Iterationsablauf

```
┌─────────────────────────────────────────────┐
│  Start: llm_cache.json + Kartentexte laden  │
└──────────────────┬──────────────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  1. ANALYST: Matrix analysieren      │
│     → Problemkarten identifizieren   │
│     → Anweisungen formulieren        │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  2. GAME DESIGNER: Karten anpassen   │
│     → Max. 3 Karten pro Iteration    │
│     → Spezifischer formulieren       │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  3. MATCHER: Neue Scores berechnen   │
│     → Nur geänderte Paare            │
│     → Cache aktualisieren            │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  4. Zielmetrik prüfen               │
│     → Ø Matches/Karte ≤ 5?          │
│     → JA → Fertig                   │
│     → NEIN → Zurück zu Schritt 1    │
└──────────────────────────────────────┘
```

## Abbruchbedingungen

- **Erfolg:** Durchschnittlich ≤ 5 Matches pro Karte (bei Threshold 0.75)
- **Abbruch:** Nach 10 Iterationen ohne signifikante Verbesserung
- **Rollback:** Wenn eine Iteration die Metrik verschlechtert, letzte Version wiederherstellen

## Technische Hinweise

- Speichere **jeden Zwischenstand** versioniert (Snapshots der Cache + Kartentexte)
- Logge pro Iteration: welche Karten geändert, alter/neuer Score, Delta
- Erstelle nach jeder Iteration einen kurzen **Balance-Report** (Markdown-Tabelle)
- LLM-Calls für Matching können parallelisiert werden (nur geänderte Paare)

## Erster Schritt

Lade `llm_cache.json` und die Kartentexte. Falls keine separate `cards.json` existiert, extrahiere die Karteninhalte aus den Begründungen in der Cache-Datei und frage mich nach den vollständigen Kartentexten.
