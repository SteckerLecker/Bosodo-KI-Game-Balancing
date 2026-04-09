# Iteratives LLM-Balancing — Verbesserungen

Stand: 09. April 2026

Basierend auf Analyse von `results/balancing_run_01` (7 Iterationen, Ziel erreicht) und `results/balancing_run_02` (10 Iterationen, Ziel verfehlt).

---

## Erkenntnisse aus den Runs

| Metrik | Run 01 | Run 02 |
|--------|--------|--------|
| Start avg. Matches | 6.61 | 6.89 |
| End avg. Matches | 4.61 (Ziel erreicht) | 6.56 (Ziel verfehlt) |
| Verbesserung | -30% | -4.8% |
| Hauptproblem | — | Oszillation & Stagnation |

**Kernprobleme:**
- Oszillation: Karten werden hin- und hergeändert (M16: 9 Änderungen, Matches schwanken 0-3)
- Rewording statt Rebalancing: Ab Iteration ~5 nur noch kosmetische Umformulierungen
- Unterverbundene Karten werden ignoriert (M05, K02)
- Über-Spezifizierung (K05 → "Webanwendung")
- Kein Gedächtnis über Iterationen hinweg

---

## Umgesetzte Verbesserungen

Alle Änderungen in **`llm_experts/balancing_pipeline.py`** (Pipeline v2).

### ✅ 1. Change-History im Prompt (Prio: HOCH)
**Betrifft:** `BalancingAnalyst.analyze()`, `GameDesigner.redesign()`

Beide LLM-Personas erhalten jetzt die letzten 3 Änderungen pro Karte inkl. Ergebnis (✓ verbessert / ✗ gescheitert). Gescheiterte Ansätze werden als Negativbeispiele explizit markiert.

**Umsetzung:**
- `change_history: dict[str, list[dict]]` — Wird in `Pipeline._record_history()` nach jeder Iteration gepflegt
- Analyst-Prompt enthält Sektion "BISHERIGE ÄNDERUNGEN" mit den letzten 3 Versuchen pro Karte
- Designer-Prompt enthält analoge Historie mit Erfolgs-/Misserfolgsmarkierung

---

### ✅ 2. Card-Lock nach Stabilisierung (Prio: MITTEL)
**Betrifft:** `Pipeline._update_card_locks()`, `BalancingAnalyst.analyze()`

Karten, die 2 Iterationen in Folge im Zielbereich (2–6 Matches) liegen, werden gesperrt.

**Umsetzung:**
- `locked_cards: set[str]` — Gesperrte Karten-IDs
- `_stable_counts: dict[str, int]` — Zähler pro Karte, wie viele Iterationen in Folge stabil
- Konstante `LOCK_AFTER_STABLE_ITERS = 2`, `STABLE_RANGE = (2, 6)`
- Analyst überspringt gesperrte Karten bei der Problemkarten-Suche
- Analyst-Prompt enthält Sektion "GESPERRTE KARTEN" mit Liste
- Report zeigt gesperrte Karten an

---

### ✅ 3. Standardabweichung als Metrik (Prio: HOCH)
**Betrifft:** `_compute_match_stats()`, `Pipeline.run()`

**Umsetzung:**
- `_compute_match_stats()` berechnet `std_dev` über alle Monster- und Wissens-Match-Counts
- Wird im Analyst-Prompt angezeigt ("Ziel: möglichst niedrig, ideal < 2.0")
- Wird im Balance-Report pro Iteration ausgegeben
- Dient als Tiebreaker bei Rollback-Check: Bei gleichem Avg + gleicher Weak-Zahl gewinnt niedrigere Std
- Pipeline trackt `best_std` analog zu `best_avg`

---

### ✅ 4. Match-Partner im Designer-Prompt (Prio: HOCH)
**Betrifft:** `GameDesigner.redesign()`

**Umsetzung:**
- Pipeline übergibt `match_details` (aus `monster_matches` + `wissen_matches`) an den Designer
- Designer-Prompt zeigt für jede zu starke Karte die konkreten Partner mit ID und Name: `"aktuelle_partner": ["K03 (Sprint-Fokus)", "K12 (Klare Kriterien)", ...]`
- Prompt-Anweisung: "Überlege gezielt, mit welchen Partnern die Karte NICHT mehr matchen soll"

---

### ✅ 5. Temperatur-Eskalation bei Stagnation (Prio: MITTEL)
**Betrifft:** `Pipeline._get_temperature()`, `Pipeline.run()`

**Umsetzung:**
- Basis-Temperatur: `TEMP_BASE = 0.4`
- Nach `STALE_ESCALATION_THRESHOLD = 3` stagnierenden Iterationen: +0.15 pro weitere Stagnation
- Maximum: `TEMP_MAX = 0.8`
- Temperatur wird an `_llm_json_call()`, Analyst und Designer weitergereicht
- Aktuelle Temperatur wird im Iterations-Header und in der History angezeigt

---

### ✅ 6. Semantische Diff-Prüfung (Prio: MITTEL)
**Betrifft:** `_word_diff_ratio()`, `Pipeline._filter_semantic_diffs()`

**Umsetzung:**
- `_word_diff_ratio(text_a, text_b)` — Jaccard-Distanz auf Wort-Ebene (symmetrische Differenz / Vereinigung)
- Schwellenwert: `MIN_SEMANTIC_DIFF_RATIO = 0.15` (15%)
- Änderungen unterhalb des Schwellenwerts werden als kosmetisch abgelehnt
- Werden alle Änderungen einer Iteration abgelehnt, zählt das als Stagnation
- Report zeigt semantische Differenz pro Änderung an

**Kalibrierung (getestet):**
- "zeigt" → "beweist" (1 Wort in langem Satz): ~17% → grenzwertig akzeptiert
- Kompletter Rewrite: ~93% → klar akzeptiert
- Identischer Text: 0% → abgelehnt

---

### ✅ 7. Rollback-Blacklist (Prio: MITTEL)
**Betrifft:** `Pipeline._record_history()`, `BalancingAnalyst.analyze()`

**Umsetzung:**
- `failed_changes: list[dict]` — Sammelt alle gescheiterten Änderungen `{id, iteration, alt, neu}`
- `_record_history()` wird nach jedem Rollback-Check aufgerufen und markiert Änderungen als `improved=True/False`
- Analyst-Prompt enthält Sektion "GESCHEITERTE ANSÄTZE" mit den letzten 5 relevanten Fehlversuchen
- Pipeline-Summary enthält `total_changes` und `failed_changes` Zähler

---

### ✅ 8. Anti-Über-Spezifizierung (Bonus)
**Betrifft:** Analyst-Prompt, Designer-Prompt

Beide Prompts enthalten jetzt die explizite Warnung:
> "Vermeide Über-Spezifizierung auf Technologien (z.B. 'Webanwendung', 'React', 'API'). Verengte den Kontext auf agile Situationen und Scrum-Rollen stattdessen."

---

## Neue Konstanten (Übersicht)

| Konstante | Wert | Beschreibung |
|-----------|------|--------------|
| `STABLE_RANGE` | (2, 6) | Zielbereich für Card-Lock |
| `LOCK_AFTER_STABLE_ITERS` | 2 | Iterationen im Zielbereich bis Lock |
| `TEMP_BASE` | 0.4 | Basis-LLM-Temperatur |
| `TEMP_ESCALATION` | 0.15 | Temperaturerhöhung pro Stagnation |
| `TEMP_MAX` | 0.8 | Maximale Temperatur |
| `STALE_ESCALATION_THRESHOLD` | 3 | Stagnations-Iterationen bis Eskalation |
| `MIN_SEMANTIC_DIFF_RATIO` | 0.15 | Mindest-Wortdifferenz für Änderungsannahme |

---

## Zurückgestellt

### 🔜 Dynamischer Match-Threshold
Weichere Gewichtung statt harter 0.75-Grenze. Erst nach Validierung der obigen Maßnahmen.

### 🔜 Mehr Karten pro Iteration bei Stagnation
Scope von 3 auf 5-6 erweitern bei Stagnation. Risiko: schwieriger zu isolieren was hilft.
