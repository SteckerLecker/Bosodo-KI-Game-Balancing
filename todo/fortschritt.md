# BOSODO Balancing — Fortschritt & Nächste Schritte

Stand: April 2026

---

## Erledigte Aufgaben

### ✅ 1. Erweiterte Metriken pro Symbol (Bericht 4.3)
**Datei:** `bosodo_env/metrics.py`

Alle vier Metriken aus dem Bericht sind implementiert und werden in jeder Episode getrackt:
- `defense_rate_per_symbol` (Ziel: 40–60 %)
- `symbol_starvation_rate` (Ziel: < 15 %)
- `avg_symbol_on_hand` (Ziel: 1.5–3.0)
- `multi_symbol_defense_rate` (Ziel: 30–50 %)

Der `BalancingAnalyzer` aggregiert die Rohdaten über alle Episoden und gibt sie im Report aus.

---

### ✅ 2. Optimiertes Verteidigungsmatching (Bericht 2.1)
**Datei:** `bosodo_env/game_state.py` — Methode `_find_optimal_defense()`

Greedy-Matching wurde durch Backtracking ersetzt. Optimierungsziel: minimale Summe der Symbolanzahlen genutzter Karten, damit wertvolle Multi-Symbol-Karten für spätere Züge aufgespart werden.

---

### ✅ 3. LLM-Batch-Bewertung (Bericht 3.1–3.3)
**Dateien:** `llm_experts/scorer.py`, `llm_experts/cache.py`, `llm_experts/batch_evaluate.py`

- Alle 18×18 = 324 Monster-Wissenskarten-Paarungen wurden bewertet
- Cache liegt als `llm_cache.json` vor
- Unterstützt Ollama (lokal) und OpenRouter (Cloud) via `.env`

---

### ✅ 4. LLM-Cache in Balancing-Report einbinden (Bericht 3.4 Stufe 1)
**Dateien:** `bosodo_env/balancing.py`, `scripts/analyze.py`

Der Cache wird beim Aufruf von `scripts/analyze.py` automatisch geladen.
`BalancingAnalyzer` analysiert alle symbolisch passenden Paarungen und kategorisiert:

| Kategorie     | Score   | Anzahl (aktueller Kartensatz) |
|---------------|---------|-------------------------------|
| Perfekt        | 0.8–1.0 | 21                            |
| Grauzone       | 0.4–0.7 | 85                            |
| Fehlzuordnung  | 0.0–0.3 | 128                           |

Die Fehlzuordnungen und Grauzone-Fälle werden in der Konsolenausgabe und im `balancing_report.json` unter dem Schlüssel `llm_analysis` ausgegeben.

---

## Offene Aufgaben (priorisiert)

### 🔜 Nächster Schritt: Iteratives Balancing-Skript (Bericht 4.1–4.6)
**Zu erstellen:** `scripts/iterative_balance.py`

Das Herzstück des automatisierten Balancings. Ablauf:
1. Kartendaten laden (aus `data_vN/`)
2. Agent trainieren (z.B. 500k Timesteps)
3. Balancing analysieren (inkl. LLM-Report)
4. Konvergenz prüfen (Kriterien aus Bericht 4.6)
5. Falls nicht konvergiert: Symbole anpassen nach Regeln aus Bericht 4.4
6. Neue Kartenversion speichern (`data_v{N+1}/`) und wiederholen

**Konvergenzkriterien** (alle drei aufeinanderfolgenden Iterationen erfüllt):
- Verteidigungsrate aller Symbole: 40–60 %
- Symbol-Starvation-Rate: < 15 %
- Spiellänge: 20–40 Züge
- ≥ 50 % der Spiele enden mit Trophäen-Unterschied ≤ 1
- Kein Monster < 2 % oder > 15 % aller Spiele

**Anpassungsregeln** (Bericht 4.4):
- Regel 1 — Symbol-Hunger: `starvation_rate > 20 %` → Wissenskarte mit Symbol hinzufügen
- Regel 2 — Überverteidigung: `defense_rate > 70 %` → Wissenskarten-Symbol entfernen
- Regel 3 — Multi-Symbol: `multi_symbol_defense_rate < 25 %` → mehr Multi-Symbol-Wissenskarten
- Regel 4 — Max. 2 Kartenänderungen pro Iteration

---

### 🔜 Danach: Erweiterte Bot-Strategien (Bericht 2.3)
**Zu ändern:** `bosodo_env/env.py` (Bot-Logik)

Aktuell: nur eine Strategie (schwächstes Monster, stärkster Gegner).
Ziel: drei Profile — Aggressiv / Defensiv / Strategisch — zufällig oder konfigurierbar per Bot.

---

### 🔄 Mittelfristig: LLM in Gameplay integrieren (Bericht 3.4 Stufe 2)
**Zu ändern:** `bosodo_env/game_state.py` — `can_defend()`

LLM-Score als Multiplikator: Verteidigung gilt als ungültig wenn Score < Schwellenwert (z.B. 0.3), auch wenn Symbole matchen. Schwellenwert über Config steuerbar.

**Voraussetzung:** Erst das iterative Skript fertigstellen, damit der Effekt messbar ist.

---

### 🔄 Langfristig: Self-Play Multi-Agent (Bericht 2.2)

Alle Spieler durch trainierte Agenten steuern (PettingZoo oder sequentielles Self-Play gegen frühere Versionen). Hoher Aufwand — erst nach dem iterativen Zyklus angehen.

---

### 🔄 Langfristig: Reward-System Erweiterung (Bericht 2.5)
**Zu ändern:** `bosodo_env/rewards.py`

Geplante Ergänzungen:
- Karten-Diversitäts-Bonus
- Differenzierte Verteidigungsqualität (knappe vs. überwältigende Verteidigung)
- Spannungskurven-Bonus (Wechselführung)
