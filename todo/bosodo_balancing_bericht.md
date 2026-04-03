# BOSODO RL-Balancing — Analyse & Verbesserungsvorschläge

**Faires Balancing durch KI-gestützte Simulation, LLM-basierte Inhaltsprüfung und iterative Farb-Optimierung**

Hochschule Ansbach — Sommersemester 2026 | Erstellt: April 2026

---

## 1. Projektübersicht & Ist-Analyse

Das Projekt **BOSODO RL-Balancing** simuliert das Lernkartenspiel „BOSODO — Digitale Achtsamkeit (Teen-Edition)" als Gymnasium-Environment und trainiert einen PPO-Agenten (Proximal Policy Optimization) mittels Stable Baselines3. Das Ziel ist, durch tausende simulierte Spiele automatisch zu erkennen, ob Monster zu stark oder zu schwach sind, ob Wissenskarten inhaltlich zu den Monstern passen und ob die Symbol-Verteilung (BO/SO/DO) fair ist.

### 1.1 Spielmechanik (Casual-Regeln)

Das Spiel verwendet 18 Monster-Karten (rot, Angriff) und 18 Wissens-Karten (grün, Verteidigung). Jede Karte trägt 1–3 Kampfsymbole: **BO** (pink, Thema Bildschirmzeit/Konsum/Selbstbild), **SO** (cyan, Soziale Medien/Nachrichten/Mobbing) und **DO** (gelb, Datenschutz/Online-Sicherheit/Grenzen). Spieler starten mit je 2 Monster + 2 Wissenskarten, greifen Mitspieler mit Monstern an, und Verteidiger müssen alle Symbole des Monsters mit passenden Wissenskarten matchen. Drei Trophäen führen zum Sieg.

### 1.2 Aktuelle Symbol-Verteilung

Die Symbole sind aktuell perfekt symmetrisch verteilt — je 12× pro Typ auf beiden Kartentypen. Das bedeutet rein rechnerisch eine ausgeglichene Verteilung. Die zentrale Frage ist jedoch, ob sich das Spiel auch fair *anfühlt*, da die Kombinatorik (welche Karten der Spieler auf der Hand hat) entscheidend ist.

| Symbol | Monster-Karten | Wissens-Karten | Ratio |
|--------|---------------|----------------|-------|
| **BO** (pink) | 12× | 12× | 1.0 ✔ |
| **SO** (cyan) | 12× | 12× | 1.0 ✔ |
| **DO** (gelb) | 12× | 12× | 1.0 ✔ |

### 1.3 Identifizierte Schwachstellen

Aus der Analyse des Codes und der Architektur ergeben sich folgende zentrale Problemfelder:

- **Fehlende Argumentationskomponente:** Im echten Spiel muss der Verteidiger argumentieren, warum das Wissen gegen das Monster hilft. Die Simulation prüft nur Symbol-Matching — die inhaltliche Passung wird komplett ignoriert.
- **Greedy Symbol-Matching:** Die Verteidigung nutzt die erste passende Karte statt optimaler Zuordnung. Bei Monstern mit mehreren Symbolen kann das zu suboptimalen Verteidigungen führen.
- **Einfache Bot-Strategie:** Bots wählen immer das schwächste Monster und greifen den führenden Spieler an. Das simuliert kein realistisches Spielverhalten.
- **Einzelspieler-Perspektive:** Nur Spieler 0 wird durch den Agenten gesteuert, kein Self-Play. Dies begrenzt die Aussagekraft der Balancing-Analyse.
- **Kombinatorische Blindheit:** Die symmetrische 12:12:12-Verteilung garantiert nicht, dass Multi-Symbol-Monster (z.B. [BO,SO,DO]) gleich schwer zu verteidigen sind wie Ein-Symbol-Monster.

---

## 2. Verbesserungsvorschläge für das Balancing

### 2.1 Optimiertes Verteidigungsmatching

Das aktuelle Greedy-Matching sollte durch ein optimales Constraint-Satisfaction-Verfahren ersetzt werden. Statt die erste passende Karte zu verwenden, sollte ein Algorithmus alle möglichen Kombinationen prüfen und die optimale Verteidigung wählen — also diejenige, die die wertvollsten Multi-Symbol-Karten für spätere Züge aufhebt.

**Konkrete Umsetzung in `game_state.py`:**

1. Die Methode `can_defend()` von Greedy auf Bipartite-Matching umstellen (z.B. mit `scipy.optimize.linear_sum_assignment` oder einem eigenen Backtracking-Algorithmus).
2. Kosten-Matrix aufbauen: Jede Wissenskarte bekommt einen "Wert" basierend auf der Anzahl ihrer Symbole — Multi-Symbol-Karten sind wertvoller und sollten nach Möglichkeit aufgespart werden.
3. Die Verteidigung sollte die Kombination wählen, die den geringsten Gesamtwert verbraucht.

### 2.2 Self-Play statt Ein-Spieler-Agent

Aktuell steuert der PPO-Agent nur Spieler 0, während Bots die anderen Spieler kontrollieren. Für aussagekräftigeres Balancing sollte ein Self-Play-Ansatz implementiert werden, bei dem alle Spieler durch trainierte Agenten gesteuert werden. Dies erfordert eine Anpassung des Environments, sodass der Agent in einem Multi-Agent-Setup (z.B. PettingZoo) operiert, oder alternativ ein sequentielles Self-Play, bei dem der aktuelle Agent gegen frühere Versionen seiner selbst spielt.

### 2.3 Erweiterte Bot-Strategien

Als Zwischenschritt vor Self-Play können die Bot-Strategien diversifiziert werden. Statt nur „schwächstes Monster, stärkster Gegner" sollten verschiedene Strategieprofile implementiert werden: **Aggressiv** (stärkstes Monster, zufälliger Gegner), **Defensiv** (schwächstes Monster, schwächster Gegner), **Strategisch** (Monster passend zur vermuteten Handkartenverteilung des Gegners). Dies ermöglicht realistischere Spielsimulationen.

### 2.4 Schwierigkeitsgewichtung für Multi-Symbol-Monster

Monster mit 3 Symbolen (z.B. [BO, SO, DO]) sind deutlich schwerer zu verteidigen als Ein-Symbol-Monster. Der Balancing-Analyzer sollte die Verteidigungsrate pro Schwierigkeitsgrad (1-Symbol, 2-Symbol, 3-Symbol) separat tracken. Falls 3-Symbol-Monster eine Verteidigungsrate unter 20% zeigen, ist das ein klares Balancing-Problem. Mögliche Gegenmaßnahmen: Mehr Multi-Symbol-Wissenskarten hinzufügen oder die Handkartengröße erhöhen.

### 2.5 Reward-System verfeinern

Das bestehende Reward-System sollte um folgende Komponenten ergänzt werden:

- **Karten-Diversität-Bonus:** Belohnung dafür, viele verschiedene Karten zu nutzen statt immer dieselben.
- **Verteidigungsqualität:** Differenziertere Bewertung — knappe Verteidigung (gerade so geschafft) vs. überwältigende Verteidigung (drei Karten für ein Symbol).
- **Spannungskurve:** Belohnung für Spiele mit Wechselführung — wenn verschiedene Spieler im Laufe des Spiels in Führung liegen.

---

## 3. LLM-basierte Prüfung der inhaltlichen Passung

Die wichtigste fehlende Komponente im aktuellen System ist die Argumentationsprüfung. Im echten Spiel genügt es nicht, ein Symbol zu matchen — der Verteidiger muss argumentieren, warum das Wissen gegen das Monster hilft. Ein LLM kann diese inhaltliche Plausibilitätsprüfung automatisieren.

### 3.1 Konzept: LLM-Argumentations-Scorer

Für jede Monster-Wissenskarten-Kombination bewertet ein LLM auf einer Skala von 0.0 bis 1.0, wie gut die Wissenskarte inhaltlich als Verteidigung gegen das Monster funktioniert. Diese Bewertung ist unabhängig vom Symbol-Matching und bewertet die thematische Passung der Beschreibungen.

### 3.2 Architektur des LLM-Moduls

Erstelle ein neues Modul `llm_experts/` mit folgender Struktur:

1. **`llm_experts/scorer.py`** — Hauptklasse `ArgumentationScorer` mit Methode `score(monster, wissenskarte) → float`. Nutzt die Anthropic API (Claude Sonnet) oder OpenAI API.
2. **`llm_experts/cache.py`** — Persistenter JSON-Cache für alle 18×18 = 324 möglichen Paarungen. Da die Kartenmenge fix ist, muss jede Paarung nur einmal bewertet werden.
3. **`llm_experts/batch_evaluate.py`** — Skript das alle 324 Paarungen auf einmal bewertet und den Cache befüllt. Geschätzte Kosten: ca. $2–5 mit Claude Sonnet.

### 3.3 Prompt-Design für die Bewertung

Der LLM-Prompt sollte folgende Struktur haben:

```
Du bist ein Experte für digitale Medienkompetenz bei Jugendlichen.
Bewerte, wie gut das folgende Wissen als Verteidigung gegen das
beschriebene Monster-Problem funktioniert.

Monster: "{monster.name}" — {monster.beschreibung}
Wissenskarte: "{wissen.name}" — {wissen.beschreibung}

Bewerte auf einer Skala von 0.0 bis 1.0:
0.0 = Keinerlei Bezug, das Wissen hilft überhaupt nicht
0.5 = Indirekter Bezug, könnte teilweise helfen
1.0 = Perfekte Verteidigung, das Wissen adressiert exakt das Problem

Antworte NUR mit einer Zahl zwischen 0.0 und 1.0.
```

### 3.4 Integration in die Spielsimulation

Die LLM-Scores werden in zwei Stufen integriert:

**Stufe 1 — Analyse-Only (empfohlen als erster Schritt):** Die LLM-Scores werden ausschließlich im Balancing-Report verwendet, um aufzuzeigen, welche Monster-Wissenskarten-Paarungen inhaltlich nicht passen, obwohl sie symbolisch matchen. Dies erfordert keine Änderung am Environment.

**Stufe 2 — Gameplay-Integration:** Der LLM-Score wird als Multiplikator in `can_defend()` eingebaut. Bei einem Score unter einem Schwellenwert (z.B. 0.3) gilt die Verteidigung als ungültig, selbst wenn die Symbole matchen. Der Schwellenwert wird über die YAML-Config steuerbar gemacht.

### 3.5 Erwartete Erkenntnisse

Die LLM-Analyse wird voraussichtlich drei Kategorien von Paarungen aufdecken:

| Kategorie | LLM-Score | Bedeutung |
|-----------|-----------|-----------|
| **Perfekt** | 0.8 – 1.0 | Symbol und Inhalt passen — keine Anpassung nötig |
| **Grauzone** | 0.4 – 0.7 | Indirekt passend — Beschreibung evtl. anpassen |
| **Fehlzuordnung** | 0.0 – 0.3 | Symbol matcht, aber Inhalt passt nicht — Symbol oder Beschreibung ändern |

---

## 4. Iteratives Farb-Balancing (BO/SO/DO)

Obwohl die aktuelle 12:12:12-Verteilung rechnerisch symmetrisch ist, kann die Spielerfahrung trotzdem unfair sein. Die Frage ist: Welche Verteilung führt zu den fairsten, spannendsten Spielen? Das folgende iterative Verfahren beantwortet diese Frage datenbasiert.

### 4.1 Überblick: Der Iterative Balancing-Zyklus

Der Prozess folgt einem Zyklus aus Simulation, Analyse und Anpassung, der so oft wiederholt wird, bis die Balancing-Metriken stabil im Zielbereich liegen.

> **Kartendaten → RL-Simulation → Balancing-Analyse → Symbol-Anpassung → (Wiederholung)**

### 4.2 Schritt 1: Baseline-Messung

Vor jeder Änderung die Baseline mit dem aktuellen Kartensatz ermitteln:

1. Training mit 1M Timesteps und der aktuellen Konfiguration durchführen.
2. 500 Analyse-Episoden spielen und den Balancing-Report generieren.
3. Folgende Metriken als Baseline festhalten: Verteidigungsrate gesamt und pro Symbol, durchschnittliche Spiellänge, Karten-Nutzungsgleichmäßigkeit (Gini-Koeffizient), Anteil knapper Spiele (Trophäen-Differenz ≤ 1).

### 4.3 Schritt 2: Symbol-spezifische Analyse

Der `BalancingAnalyzer` muss erweitert werden, um folgende Metriken pro Symbol zu tracken:

| Metrik | Zielbereich | Bedeutung |
|--------|-------------|-----------|
| `defense_rate_per_symbol` | 40–60% | Anteil der Angriffe mit Symbol X, die erfolgreich verteidigt wurden |
| `symbol_starvation_rate` | < 15% | Wie oft ein Spieler ein Symbol nicht verteidigen konnte, weil keine passende Karte auf der Hand war |
| `avg_symbol_on_hand` | 1.5–3.0 | Durchschnittliche Anzahl eines Symbols auf der Hand |
| `multi_symbol_defense_rate` | 30–50% | Verteidigungsrate bei 2- und 3-Symbol-Monstern |

### 4.4 Schritt 3: Automatische Symbol-Anpassung

Basierend auf den Metriken werden die Symbole automatisch angepasst. Folgende Regeln bilden den Kern des iterativen Algorithmus:

- **Regel 1 — Symbol-Hunger:** Wenn `symbol_starvation_rate` für ein Symbol > 20%, füge eine zusätzliche Wissenskarte mit diesem Symbol hinzu (entweder neue Karte oder bestehendes Symbol zu einer Karte hinzufügen).
- **Regel 2 — Überverteidigung:** Wenn `defense_rate` für ein Symbol > 70%, entferne ein Symbol von einer Wissenskarte oder füge ein zusätzliches Monster-Symbol hinzu.
- **Regel 3 — Multi-Symbol-Balance:** Wenn `multi_symbol_defense_rate` < 25%, erhöhe die Anzahl der Multi-Symbol-Wissenskarten oder reduziere die Anzahl der 3-Symbol-Monster.
- **Regel 4 — Minimaländerungen:** Pro Iteration höchstens 2 Karten ändern, um die Auswirkungen klar nachvollziehen zu können.

### 4.5 Schritt 4: Erneute Simulation & Vergleich

Nach jeder Symbol-Anpassung wird der gesamte Trainings- und Analyse-Zyklus wiederholt. Die Ergebnisse werden gegen die Baseline und die vorherige Iteration verglichen. Wichtig: Es sollte ein Versionierungs-System für die Kartensätze eingeführt werden (z.B. `data_v1/`, `data_v2/`, `data_v3/`), damit jede Iteration nachvollziehbar bleibt.

### 4.6 Konvergenzkriterien

Das iterative Balancing gilt als konvergiert, wenn über drei aufeinanderfolgende Iterationen folgende Bedingungen erfüllt sind:

- Verteidigungsrate aller Symbole liegt im Bereich 40–60%.
- Symbol-Starvation-Rate liegt unter 15% für alle Symbole.
- Durchschnittliche Spiellänge liegt zwischen 20 und 40 Zügen.
- Mindestens 50% der Spiele enden mit einem Trophäen-Unterschied von höchstens 1.
- Kein einzelnes Monster wird in weniger als 2% oder mehr als 15% aller Spiele gespielt.

---

## 5. Automatisiertes Balancing-Skript

Das folgende Skript fasst den gesamten iterativen Prozess zusammen. Es sollte als `scripts/iterative_balance.py` implementiert werden.

### 5.1 Pseudocode

```python
MAX_ITERATIONS = 10
TIMESTEPS_PER_ITERATION = 500_000
ANALYSIS_EPISODES = 500

for iteration in range(MAX_ITERATIONS):
    # 1. Kartendaten laden
    cards = load_cards(f"data_v{iteration}/")

    # 2. Agent trainieren
    model = train_agent(cards, TIMESTEPS_PER_ITERATION)

    # 3. Balancing analysieren
    report = analyze(model, ANALYSIS_EPISODES)

    # 4. Konvergenz prüfen
    if report.is_balanced():
        print(f"Balancing erreicht nach {iteration+1} Iterationen")
        break

    # 5. Symbole anpassen und neue Version speichern
    new_cards = apply_balancing_rules(cards, report)
    save_cards(new_cards, f"data_v{iteration+1}/")
    log_iteration(iteration, report)
```

### 5.2 Ergänzung: LLM-Bewertung im Zyklus

Die LLM-Bewertung kann als zusätzlicher Schritt nach der Symbol-Analyse integriert werden. Wenn Symbole angepasst werden, wird der LLM-Cache für die geänderten Karten invalidiert und die neuen Paarungen automatisch neu bewertet. So ist sichergestellt, dass jede Symbol-Änderung auch inhaltlich geprüft wird.

---

## 6. Zusammenfassung & Priorisierung

Die folgende Tabelle fasst alle Verbesserungsvorschläge zusammen, priorisiert nach Aufwand und erwartetem Nutzen für die Fairness des Spiels.

| # | Maßnahme | Aufwand | Nutzen | Priorität |
|---|----------|---------|--------|-----------|
| 1 | Erweiterte Metriken pro Symbol (Abschnitt 4.3) | Gering | Hoch | **Sofort umsetzen** |
| 2 | Optimiertes Verteidigungsmatching (2.1) | Gering | Hoch | **Sofort umsetzen** |
| 3 | LLM-Batch-Bewertung (3.1–3.3) | Mittel | Hoch | **Nächster Sprint** |
| 4 | Iteratives Balancing-Skript (4.1–4.6) | Mittel | Sehr hoch | **Nächster Sprint** |
| 5 | Erweiterte Bot-Strategien (2.3) | Mittel | Mittel | Mittelfristig |
| 6 | LLM-Integration in Gameplay (3.4 Stufe 2) | Hoch | Mittel | Mittelfristig |
| 7 | Self-Play Multi-Agent (2.2) | Hoch | Hoch | Langfristig |
| 8 | Reward-System Erweiterung (2.5) | Mittel | Mittel | Langfristig |

**Empfohlene Reihenfolge:** Beginne mit den erweiterten Metriken und dem optimierten Matching (1–2), da diese mit geringem Aufwand sofort bessere Datengrundlagen liefern. Danach die LLM-Bewertung und das iterative Skript (3–4) implementieren — das bildet den Kern des verbesserten Balancings. Die restlichen Maßnahmen können dann basierend auf den gewonnenen Erkenntnissen priorisiert werden.
