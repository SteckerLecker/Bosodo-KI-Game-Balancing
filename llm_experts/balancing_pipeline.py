"""
Iterative Balancing-Pipeline mit 3 LLM-Personas:
  1. Balancing-Analyst  – Score-Matrix analysieren, Problemkarten finden
  2. Game Designer      – Kartentexte spezifischer formulieren (max. 3/Iteration)
  3. Matcher            – Geänderte Paare neu scoren

Ablauf:
  Start → Analyst → Designer → Matcher → Zielmetrik prüfen → ggf. Loop
"""

import copy
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from llm_experts.cache import load_cache
from llm_experts.scorer import ArgumentationScorer

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

MATCH_THRESHOLD = 0.6
TARGET_AVG_MATCHES = 5
MIN_MATCHES = 2  # Karten mit weniger Matches gelten als zu schwach
MAX_ITERATIONS = 15
MAX_STALE_ITERATIONS = 10
STABLE_RANGE = (MIN_MATCHES, TARGET_AVG_MATCHES + 1)  # 2-6 Matches = Zielbereich
LOCK_AFTER_STABLE_ITERS = 2  # Karte sperren nach N Iterationen im Zielbereich
TEMP_BASE = 0.4  # Basis-Temperatur für LLM-Calls
TEMP_ESCALATION = 0.15  # Temperatur-Erhöhung pro Stagnation
TEMP_MAX = 0.8  # Maximale Temperatur
STALE_ESCALATION_THRESHOLD = 3  # Ab wann Temperatur eskaliert wird
MIN_SEMANTIC_DIFF_RATIO = 0.15  # Mindest-Unterschied (Wort-Ebene) für Änderungsannahme


def _build_llm_client() -> tuple[OpenAI, str]:
    """Baut einen OpenAI-kompatiblen Client (gleiche Provider-Logik wie scorer.py)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            timeout=180.0,
        )
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:4b")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY ist nicht gesetzt")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    else:
        raise ValueError(f"Unbekannter LLM_PROVIDER: '{provider}'")
    return client, model


def _llm_json_call(client: OpenAI, model: str, prompt: str, retries: int = 3,
                    temperature: float = TEMP_BASE) -> dict:
    """Sendet einen Prompt und parsed die JSON-Antwort."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            if not content:
                extras = getattr(resp.choices[0].message, "model_extra", {}) or {}
                content = extras.get("reasoning_content", "")
            return json.loads(content)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return {}


def _word_diff_ratio(text_a: str, text_b: str) -> float:
    """Berechnet den Anteil unterschiedlicher Wörter zwischen zwei Texten (0-1)."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 0.0
    union = words_a | words_b
    if not union:
        return 0.0
    diff = words_a.symmetric_difference(words_b)
    return len(diff) / len(union)


def _compute_match_stats(
    cache: dict, monster_ids: list[str], wissen_ids: list[str]
) -> dict:
    """Berechnet Match-Statistiken für die aktuelle Cache-Matrix."""
    monster_matches: dict[str, list[str]] = {m: [] for m in monster_ids}
    wissen_matches: dict[str, list[str]] = {k: [] for k in wissen_ids}

    for m_id in monster_ids:
        for k_id in wissen_ids:
            key = f"{m_id}_{k_id}"
            entry = cache.get(key)
            if entry and entry["score"] >= MATCH_THRESHOLD:
                monster_matches[m_id].append(k_id)
                wissen_matches[k_id].append(m_id)

    monster_counts = {m: len(v) for m, v in monster_matches.items()}
    wissen_counts = {k: len(v) for k, v in wissen_matches.items()}

    avg_monster = sum(monster_counts.values()) / max(len(monster_counts), 1)
    avg_wissen = sum(wissen_counts.values()) / max(len(wissen_counts), 1)

    # Karten mit zu wenig Matches zählen
    all_counts = list(monster_counts.values()) + list(wissen_counts.values())
    too_weak_count = sum(1 for c in all_counts if c < MIN_MATCHES)

    # Standardabweichung der Match-Verteilung
    avg_total = (avg_monster + avg_wissen) / 2
    if all_counts:
        variance = sum((c - avg_total) ** 2 for c in all_counts) / len(all_counts)
        std_dev = round(math.sqrt(variance), 2)
    else:
        std_dev = 0.0

    return {
        "monster_matches": monster_matches,
        "wissen_matches": wissen_matches,
        "monster_counts": monster_counts,
        "wissen_counts": wissen_counts,
        "avg_matches_monster": round(avg_monster, 2),
        "avg_matches_wissen": round(avg_wissen, 2),
        "avg_total": round(avg_total, 2),
        "std_dev": std_dev,
        "too_weak_count": too_weak_count,
    }


def _cards_by_id(karten_list: list[dict]) -> dict[str, dict]:
    return {k["id"]: k for k in karten_list}


# ---------------------------------------------------------------------------
# Persona 1 – Balancing-Analyst
# ---------------------------------------------------------------------------

class BalancingAnalyst:
    """Analysiert die Score-Matrix und gibt Überarbeitungsanweisungen."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def analyze(
        self,
        cache: dict,
        monster_karten: list[dict],
        wissens_karten: list[dict],
        iteration: int,
        change_history: Optional[dict[str, list[dict]]] = None,
        locked_cards: Optional[set[str]] = None,
        failed_changes: Optional[list[dict]] = None,
        temperature: float = TEMP_BASE,
    ) -> dict:
        monster_ids = [m["id"] for m in monster_karten]
        wissen_ids = [k["id"] for k in wissens_karten]
        stats = _compute_match_stats(cache, monster_ids, wissen_ids)

        m_by_id = _cards_by_id(monster_karten)
        k_by_id = _cards_by_id(wissens_karten)
        locked = locked_cards or set()

        # Top-Problemkarten sammeln: zu starke (> TARGET) UND zu schwache (< MIN)
        too_strong = []
        too_weak = []

        for m_id, count in stats["monster_counts"].items():
            if m_id in locked:
                continue
            card = m_by_id[m_id]
            entry = {
                "id": m_id,
                "typ": "monster",
                "name": card["name"],
                "beschreibung": card["beschreibung"],
                "aktuelle_matches": count,
                "partner": stats["monster_matches"][m_id],
            }
            if count > TARGET_AVG_MATCHES:
                entry["problem"] = "zu_stark"
                too_strong.append(entry)
            elif count < MIN_MATCHES:
                entry["problem"] = "zu_schwach"
                too_weak.append(entry)

        for k_id, count in stats["wissen_counts"].items():
            if k_id in locked:
                continue
            card = k_by_id[k_id]
            entry = {
                "id": k_id,
                "typ": "wissen",
                "name": card["name"],
                "beschreibung": card["beschreibung"],
                "aktuelle_matches": count,
                "partner": stats["wissen_matches"][k_id],
            }
            if count > TARGET_AVG_MATCHES:
                entry["problem"] = "zu_stark"
                too_strong.append(entry)
            elif count < MIN_MATCHES:
                entry["problem"] = "zu_schwach"
                too_weak.append(entry)

        too_strong.sort(key=lambda x: x["aktuelle_matches"], reverse=True)
        too_weak.sort(key=lambda x: x["aktuelle_matches"])

        # Zu schwache Karten haben Priorität (0 Matches = unspielbar), dann zu starke
        # Max 6 Kandidaten für Kontext, davon mindestens die schwachen
        top_problems = too_weak[:3] + too_strong[:max(0, 6 - len(too_weak[:3]))]

        if not top_problems:
            return {
                "iteration": iteration,
                "zusammenfassung": {
                    "avg_matches_monster": stats["avg_matches_monster"],
                    "avg_matches_wissen": stats["avg_matches_wissen"],
                    "std_dev": stats["std_dev"],
                    "too_weak_count": stats["too_weak_count"],
                },
                "problemkarten": [],
                "ziel_erreicht": True,
            }

        # LLM-Call: Analyse & Anweisungen
        problem_json = json.dumps(top_problems, ensure_ascii=False, indent=2)

        weak_count = len([p for p in top_problems if p.get("problem") == "zu_schwach"])
        strong_count = len([p for p in top_problems if p.get("problem") == "zu_stark"])
        problem_beschreibung = ""
        if weak_count > 0 and strong_count > 0:
            problem_beschreibung = f"Es gibt {weak_count} zu schwache Karten (< {MIN_MATCHES} Matches) und {strong_count} zu starke Karten (> {TARGET_AVG_MATCHES} Matches)."
        elif weak_count > 0:
            problem_beschreibung = f"Es gibt {weak_count} zu schwache Karten (< {MIN_MATCHES} Matches), die mit fast keinem Partner matchen."
        else:
            problem_beschreibung = f"Es gibt {strong_count} zu starke Karten (> {TARGET_AVG_MATCHES} Matches), die zu breit matchen."

        # Change-History für relevante Karten aufbereiten
        history_section = ""
        if change_history:
            relevant_ids = {p["id"] for p in top_problems}
            history_entries = []
            for card_id in relevant_ids:
                if card_id in change_history and change_history[card_id]:
                    changes = change_history[card_id]
                    entry_lines = [f"  {card_id} ({len(changes)} bisherige Änderungen):"]
                    for ch in changes[-3:]:  # Maximal letzte 3 Änderungen zeigen
                        result_label = "✓ verbessert" if ch.get("improved") else "✗ verschlechtert/neutral"
                        entry_lines.append(
                            f"    - Iter {ch['iteration']}: \"{ch['alt'][:60]}...\" → \"{ch['neu'][:60]}...\" ({result_label})"
                        )
                    history_entries.append("\n".join(entry_lines))
            if history_entries:
                history_section = "\n\nBISHERIGE ÄNDERUNGEN an diesen Karten (vermeide Wiederholung gescheiterter Ansätze!):\n" + "\n".join(history_entries)

        # Gescheiterte Änderungen als Negativbeispiele
        blacklist_section = ""
        if failed_changes:
            relevant_ids = {p["id"] for p in top_problems}
            relevant_fails = [f for f in failed_changes if f["id"] in relevant_ids]
            if relevant_fails:
                fail_lines = ["GESCHEITERTE ANSÄTZE (diese Änderungen haben NICHT funktioniert, versuche etwas Anderes!):"]
                for f in relevant_fails[-5:]:
                    fail_lines.append(f"  - {f['id']} Iter {f['iteration']}: \"{f['neu'][:80]}...\"")
                blacklist_section = "\n\n" + "\n".join(fail_lines)

        # Gesperrte Karten erwähnen
        lock_section = ""
        if locked:
            lock_section = f"\n\nGESPERRTE KARTEN (diese Karten sind im Zielbereich und dürfen NICHT verändert werden): {', '.join(sorted(locked))}"

        prompt = f"""Du bist ein Balancing-Analyst für ein Lernkartenspiel über agile Methoden (Scrum).

Das Spiel hat Monster-Karten (agile Anti-Patterns) und Wissens-Karten (agile Methoden).
Jedes Paar hat einen Match-Score (0-1). Ein Match zählt ab Score ≥ {MATCH_THRESHOLD}.

ZIEL: Jede Karte soll mit 4-6 Partnern stark matchen (Score ≥ {MATCH_THRESHOLD}).
- Karten mit > {TARGET_AVG_MATCHES} Matches sind ZU STARK (zu generisch, matchen mit zu vielen Partnern)
- Karten mit < {MIN_MATCHES} Matches sind ZU SCHWACH (zu spezifisch oder schlecht formuliert, matchen mit fast niemandem)
Aktuell ist der Durchschnitt: Monster {stats['avg_matches_monster']}, Wissen {stats['avg_matches_wissen']}.
Standardabweichung der Matches: {stats['std_dev']} (Ziel: möglichst niedrig, ideal < 2.0).

{problem_beschreibung}{history_section}{blacklist_section}{lock_section}

Hier sind die problematischen Karten:

{problem_json}

Wähle die TOP 3 Karten aus, die am dringendsten überarbeitet werden müssen.
Priorisiere dabei zu schwache Karten (0-1 Matches), da diese im Spiel unbrauchbar sind.

Für jede Karte:
1. Analysiere das Problem:
   - ZU STARK: Warum matcht sie so breit? (zu allgemein? Überlappungen?)
   - ZU SCHWACH: Warum matcht sie mit fast niemandem? (zu spezifisch? schlecht formuliert? thematisch isoliert?)
2. Gib eine KONKRETE Anweisung:
   - ZU STARK: Text spezifischer formulieren, Fokus einengen
   - ZU SCHWACH: Text so umformulieren, dass er klar zu 4-6 passenden Partnern matcht, ohne zu allgemein zu werden

WICHTIG: Vermeide Über-Spezifizierung auf bestimmte Technologien (z.B. "Webanwendung", "React", "API").
Verengte den Kontext stattdessen auf spezifische agile Situationen oder Scrum-Rollen.

Antworte als JSON:
{{
  "problemkarten": [
    {{
      "id": "M01",
      "typ": "monster|wissen",
      "problem": "zu_stark|zu_schwach",
      "aktuelle_matches": 0,
      "analyse": "Warum die Karte problematisch ist...",
      "anweisung": "Konkrete Überarbeitungsanweisung..."
    }}
  ]
}}"""

        result = _llm_json_call(self.client, self.model, prompt, temperature=temperature)

        return {
            "iteration": iteration,
            "zusammenfassung": {
                "avg_matches_monster": stats["avg_matches_monster"],
                "avg_matches_wissen": stats["avg_matches_wissen"],
                "std_dev": stats["std_dev"],
            },
            "problemkarten": result.get("problemkarten", []),
            "ziel_erreicht": False,
        }


# ---------------------------------------------------------------------------
# Persona 2 – Game Designer
# ---------------------------------------------------------------------------

class GameDesigner:
    """Überarbeitet Kartentexte basierend auf Analyst-Anweisungen."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def redesign(
        self,
        anweisungen: list[dict],
        monster_karten: list[dict],
        wissens_karten: list[dict],
        match_details: Optional[dict] = None,
        change_history: Optional[dict[str, list[dict]]] = None,
        temperature: float = TEMP_BASE,
    ) -> list[dict]:
        if not anweisungen:
            return []

        # Max 3 Karten pro Iteration
        anweisungen = anweisungen[:3]

        m_by_id = _cards_by_id(monster_karten)
        k_by_id = _cards_by_id(wissens_karten)
        all_by_id = {**m_by_id, **k_by_id}

        cards_context = []
        for aw in anweisungen:
            card_id = aw["id"]
            if card_id.startswith("M"):
                card = m_by_id.get(card_id)
            else:
                card = k_by_id.get(card_id)
            if card:
                entry = {
                    "id": card_id,
                    "name": card["name"],
                    "thema": card.get("thema", ""),
                    "beschreibung": card["beschreibung"],
                    "problem": aw.get("problem", "zu_stark"),
                    "anweisung": aw.get("anweisung", ""),
                    "analyse": aw.get("analyse", ""),
                }
                # Match-Partner mit Namen anzeigen
                if match_details and card_id in match_details:
                    partner_infos = []
                    for p_id in match_details[card_id]:
                        p_card = all_by_id.get(p_id)
                        if p_card:
                            partner_infos.append(f"{p_id} ({p_card['name']})")
                    entry["aktuelle_partner"] = partner_infos
                cards_context.append(entry)

        context_json = json.dumps(cards_context, ensure_ascii=False, indent=2)

        # Change-History aufbereiten
        history_section = ""
        if change_history:
            history_lines = []
            for aw in anweisungen:
                card_id = aw["id"]
                if card_id in change_history and change_history[card_id]:
                    changes = change_history[card_id]
                    history_lines.append(f"\n{card_id} — bisherige Änderungen:")
                    for ch in changes[-3:]:
                        result_label = "✓ verbessert" if ch.get("improved") else "✗ gescheitert"
                        history_lines.append(f"  Iter {ch['iteration']} ({result_label}): \"{ch['neu'][:80]}\"")
            if history_lines:
                history_section = "\n\nBISHERIGE ÄNDERUNGEN (vermeide Wiederholung gescheiterter Ansätze!):" + "\n".join(history_lines)

        prompt = f"""Du bist ein Game Designer für ein Scrum-Lernkartenspiel.

Deine Aufgabe: Überarbeite die folgenden Kartentexte, damit jede Karte mit genau 4-6 Partnern matcht.

Es gibt zwei Arten von Problemen:
- **ZU STARK** (zu viele Matches): Die Karte ist zu generisch und matcht mit fast allem → SPEZIFISCHER formulieren
- **ZU SCHWACH** (zu wenige Matches, 0-1): Die Karte matcht mit fast niemandem → so umformulieren, dass sie klar zu passenden Partnern matcht

REGELN:
- Behalte den LERNINHALT bei – nur die Formulierung wird angepasst
- Monster-Karten: Beschreibe ein KONKRETES Szenario (Wer? Was genau? Welcher Kontext?)
- Wissens-Karten: Beschreibe eine KONKRETE Technik/Methode statt allgemeines Prinzip
- Bei ZU SCHWACHEN Karten: Stelle sicher, dass die Beschreibung klar erkennen lässt, welche Probleme/Lösungen dazu passen. Die Karte darf breiter werden, aber nicht so breit, dass sie mit allem matcht.
- Der Name und das Thema der Karte bleiben gleich
- Die Beschreibung soll weiterhin unterhaltsam und im gleichen Stil sein
- Beschreibungen sollen ähnlich lang bleiben (1-3 Sätze)
- WICHTIG: Vermeide Über-Spezifizierung auf Technologien (z.B. "Webanwendung", "React", "API"). Verengte den Kontext auf agile Situationen und Scrum-Rollen stattdessen.
- Die neue Beschreibung muss sich INHALTLICH deutlich vom alten Text unterscheiden, nicht nur umformuliert sein!

Bei ZU STARKEN Karten siehst du die aktuellen Match-Partner. Überlege gezielt, mit welchen Partnern die Karte NICHT mehr matchen soll, und formuliere entsprechend spezifischer.

Hier sind die Karten mit Überarbeitungsanweisungen:

{context_json}{history_section}

Antworte als JSON:
{{
  "aenderungen": [
    {{
      "id": "M01",
      "alt": "Bisheriger Beschreibungstext...",
      "neu": "Neuer angepasster Beschreibungstext...",
      "begruendung": "Warum diese Änderung die Matches verbessert..."
    }}
  ]
}}"""

        result = _llm_json_call(self.client, self.model, prompt, temperature=temperature)
        return result.get("aenderungen", [])


# ---------------------------------------------------------------------------
# Persona 3 – Matcher
# ---------------------------------------------------------------------------

class Matcher:
    """Re-scored geänderte Paare mit dem bestehenden ArgumentationScorer."""

    def __init__(self):
        self.scorer = ArgumentationScorer()

    def rescore_changed(
        self,
        changed_ids: list[str],
        monster_karten: list[dict],
        wissens_karten: list[dict],
        cache: dict,
    ) -> dict[str, dict]:
        """Berechnet Scores nur für Paare, an denen geänderte Karten beteiligt sind."""
        updated_entries = {}

        changed_monster_ids = [cid for cid in changed_ids if cid.startswith("M")]
        changed_wissen_ids = [cid for cid in changed_ids if cid.startswith("K")]

        m_by_id = _cards_by_id(monster_karten)
        k_by_id = _cards_by_id(wissens_karten)

        pairs_to_score = set()

        # Geändertes Monster → gegen alle Wissenskarten scoren
        for m_id in changed_monster_ids:
            for k in wissens_karten:
                pairs_to_score.add((m_id, k["id"]))

        # Geänderte Wissenskarte → gegen alle Monster scoren
        for k_id in changed_wissen_ids:
            for m in monster_karten:
                pairs_to_score.add((m["id"], k_id))

        total = len(pairs_to_score)
        print(f"  Matcher: {total} Paare neu bewerten...")

        for i, (m_id, k_id) in enumerate(sorted(pairs_to_score), 1):
            monster = m_by_id[m_id]
            wissen = k_by_id[k_id]
            key = f"{m_id}_{k_id}"

            score, begruendung = self.scorer.score(monster, wissen)
            updated_entries[key] = {"score": score, "begruendung": begruendung}
            cache[key] = updated_entries[key]

            print(f"    [{i}/{total}] {key}: {score:.2f} – {begruendung[:60]}")
            time.sleep(0.05)

        return updated_entries


# ---------------------------------------------------------------------------
# Pipeline-Orchestrierung
# ---------------------------------------------------------------------------

class BalancingPipeline:
    """Orchestriert den iterativen Balancing-Loop."""

    def __init__(self, data_dir: str, output_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "balancing_runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client, self.model = _build_llm_client()
        self.analyst = BalancingAnalyst(self.client, self.model)
        self.designer = GameDesigner(self.client, self.model)
        self.matcher = Matcher()

        # Kartendaten laden
        self.monster_karten = json.loads(
            (self.data_dir / "monster_karten.json").read_text(encoding="utf-8")
        )
        self.wissens_karten = json.loads(
            (self.data_dir / "wissens_karten.json").read_text(encoding="utf-8")
        )
        self.cache = load_cache(data_dir=str(self.data_dir))

        # Neue Tracking-Strukturen
        self.change_history: dict[str, list[dict]] = {}  # card_id → [{iteration, alt, neu, improved}]
        self.failed_changes: list[dict] = []  # [{id, iteration, alt, neu}]
        self.locked_cards: set[str] = set()  # Karten im Zielbereich, die nicht mehr geändert werden
        self._stable_counts: dict[str, int] = {}  # card_id → Anzahl aufeinanderfolgender Iterationen im Zielbereich

    def _snapshot(self, iteration: int, label: str) -> Path:
        """Speichert einen Snapshot der aktuellen Daten."""
        snap_dir = self.output_dir / f"iteration_{iteration:02d}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        (snap_dir / f"monster_karten_{label}.json").write_text(
            json.dumps(self.monster_karten, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (snap_dir / f"wissens_karten_{label}.json").write_text(
            json.dumps(self.wissens_karten, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (snap_dir / f"llm_cache_{label}.json").write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return snap_dir

    def _apply_changes(self, aenderungen: list[dict]):
        """Wendet Textänderungen auf die Kartenlisten an."""
        m_list = self.monster_karten["karten"]
        k_list = self.wissens_karten["karten"]

        for change in aenderungen:
            card_id = change["id"]
            new_text = change.get("neu", "")
            if not new_text:
                continue

            if card_id.startswith("M"):
                for card in m_list:
                    if card["id"] == card_id:
                        card["beschreibung"] = new_text
                        break
            else:
                for card in k_list:
                    if card["id"] == card_id:
                        card["beschreibung"] = new_text
                        break

    def _save_best(self):
        """Speichert das beste Ergebnis als *_best.json im Output-Verzeichnis (Original bleibt unverändert)."""
        (self.output_dir / "monster_karten_best.json").write_text(
            json.dumps(self.monster_karten, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "wissens_karten_best.json").write_text(
            json.dumps(self.wissens_karten, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "llm_cache_best.json").write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_report(self, iteration: int, stats: dict, analyse: dict, aenderungen: list[dict], snap_dir: Path):
        """Schreibt einen Markdown-Balance-Report."""
        monster_ids = [m["id"] for m in self.monster_karten["karten"]]
        wissen_ids = [k["id"] for k in self.wissens_karten["karten"]]

        lines = [
            f"# Balance-Report – Iteration {iteration}",
            f"_Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
            "## Übersicht",
            f"| Metrik | Wert |",
            f"|--------|------|",
            f"| Ø Matches Monster | {stats['avg_matches_monster']} |",
            f"| Ø Matches Wissen | {stats['avg_matches_wissen']} |",
            f"| Ø Gesamt | {stats['avg_total']} |",
            f"| Std-Abweichung | {stats.get('std_dev', '–')} |",
            f"| Ziel | ≤ {TARGET_AVG_MATCHES} |",
            "",
            "## Monster-Matches",
            "| ID | Name | Matches |",
            "|----|------|---------|",
        ]
        m_by_id = _cards_by_id(self.monster_karten["karten"])
        for m_id in sorted(stats["monster_counts"], key=lambda x: stats["monster_counts"][x], reverse=True):
            name = m_by_id[m_id]["name"]
            count = stats["monster_counts"][m_id]
            lines.append(f"| {m_id} | {name} | {count} |")

        lines += [
            "",
            "## Wissens-Matches",
            "| ID | Name | Matches |",
            "|----|------|---------|",
        ]
        k_by_id = _cards_by_id(self.wissens_karten["karten"])
        for k_id in sorted(stats["wissen_counts"], key=lambda x: stats["wissen_counts"][x], reverse=True):
            name = k_by_id[k_id]["name"]
            count = stats["wissen_counts"][k_id]
            lines.append(f"| {k_id} | {name} | {count} |")

        if self.locked_cards:
            lines += ["", f"## Gesperrte Karten ({len(self.locked_cards)})",
                       ", ".join(sorted(self.locked_cards))]

        if aenderungen:
            lines += ["", "## Änderungen in dieser Iteration"]
            for ch in aenderungen:
                lines.append(f"\n### {ch['id']}")
                lines.append(f"**Alt:** {ch.get('alt', '–')}")
                lines.append(f"**Neu:** {ch.get('neu', '–')}")
                lines.append(f"**Begründung:** {ch.get('begruendung', '–')}")
                # Semantische Diff-Info
                if ch.get("alt") and ch.get("neu"):
                    diff_ratio = _word_diff_ratio(ch["alt"], ch["neu"])
                    lines.append(f"**Semantische Differenz:** {diff_ratio:.0%}")

        report_path = snap_dir / "balance_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _update_card_locks(self, stats: dict):
        """Sperrt Karten, die N Iterationen in Folge im Zielbereich liegen."""
        all_counts = {**stats["monster_counts"], **stats["wissen_counts"]}
        for card_id, count in all_counts.items():
            if card_id in self.locked_cards:
                continue
            if STABLE_RANGE[0] <= count <= STABLE_RANGE[1]:
                self._stable_counts[card_id] = self._stable_counts.get(card_id, 0) + 1
                if self._stable_counts[card_id] >= LOCK_AFTER_STABLE_ITERS:
                    self.locked_cards.add(card_id)
                    print(f"  🔒 {card_id} gesperrt (stabil im Zielbereich seit {LOCK_AFTER_STABLE_ITERS} Iterationen)")
            else:
                self._stable_counts[card_id] = 0

    def _get_temperature(self, stale_count: int) -> float:
        """Berechnet die aktuelle LLM-Temperatur basierend auf Stagnation."""
        if stale_count < STALE_ESCALATION_THRESHOLD:
            return TEMP_BASE
        escalation_steps = stale_count - STALE_ESCALATION_THRESHOLD + 1
        return min(TEMP_BASE + escalation_steps * TEMP_ESCALATION, TEMP_MAX)

    def _filter_semantic_diffs(self, aenderungen: list[dict]) -> list[dict]:
        """Filtert Änderungen heraus, die nur kosmetische Umformulierungen sind."""
        accepted = []
        for ch in aenderungen:
            alt = ch.get("alt", "")
            neu = ch.get("neu", "")
            if not alt or not neu:
                accepted.append(ch)
                continue
            diff_ratio = _word_diff_ratio(alt, neu)
            if diff_ratio >= MIN_SEMANTIC_DIFF_RATIO:
                accepted.append(ch)
                print(f"  ✓ {ch['id']}: Semantische Differenz {diff_ratio:.0%} – akzeptiert")
            else:
                print(f"  ✗ {ch['id']}: Semantische Differenz nur {diff_ratio:.0%} – abgelehnt (Minimum: {MIN_SEMANTIC_DIFF_RATIO:.0%})")
        return accepted

    def _record_history(self, iteration: int, aenderungen: list[dict], improved: bool):
        """Zeichnet Änderungen in der Change-History auf."""
        for ch in aenderungen:
            card_id = ch["id"]
            if card_id not in self.change_history:
                self.change_history[card_id] = []
            self.change_history[card_id].append({
                "iteration": iteration,
                "alt": ch.get("alt", ""),
                "neu": ch.get("neu", ""),
                "improved": improved,
            })
            if not improved:
                self.failed_changes.append({
                    "id": card_id,
                    "iteration": iteration,
                    "alt": ch.get("alt", ""),
                    "neu": ch.get("neu", ""),
                })

    def run(self) -> dict:
        """Führt die iterative Balancing-Pipeline aus."""
        print("=" * 60)
        print("  BOSODO Balancing-Pipeline (v2 – mit History/Lock/Diff)")
        print(f"  Daten: {self.data_dir}")
        print(f"  LLM: {self.model}")
        print(f"  Ziel: Ø ≤ {TARGET_AVG_MATCHES} Matches/Karte (Threshold {MATCH_THRESHOLD})")
        print("=" * 60)

        monster_ids = [m["id"] for m in self.monster_karten["karten"]]
        wissen_ids = [k["id"] for k in self.wissens_karten["karten"]]

        # Initiale Statistik
        initial_stats = _compute_match_stats(self.cache, monster_ids, wissen_ids)
        print(f"\nInitiale Statistik:")
        print(f"  Ø Monster-Matches: {initial_stats['avg_matches_monster']}")
        print(f"  Ø Wissen-Matches:  {initial_stats['avg_matches_wissen']}")
        print(f"  Ø Gesamt:          {initial_stats['avg_total']}")
        print(f"  Std-Abweichung:    {initial_stats['std_dev']}")
        if initial_stats["too_weak_count"] > 0:
            print(f"  ⚠ Zu schwache Karten (< {MIN_MATCHES} Matches): {initial_stats['too_weak_count']}")

        best_avg = initial_stats["avg_total"]
        best_std = initial_stats["std_dev"]
        best_weak = initial_stats["too_weak_count"]
        best_cache = copy.deepcopy(self.cache)
        best_monster = copy.deepcopy(self.monster_karten)
        best_wissen = copy.deepcopy(self.wissens_karten)
        stale_count = 0

        history = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            current_temp = self._get_temperature(stale_count)
            temp_info = f" (Temperatur: {current_temp:.2f})" if current_temp > TEMP_BASE else ""

            print(f"\n{'─' * 60}")
            print(f"  Iteration {iteration}{temp_info}")
            if self.locked_cards:
                print(f"  Gesperrte Karten: {', '.join(sorted(self.locked_cards))}")
            print(f"{'─' * 60}")

            # Snapshot VOR der Iteration
            self._snapshot(iteration, "vor")

            # Card-Lock aktualisieren
            current_stats = _compute_match_stats(self.cache, monster_ids, wissen_ids)
            self._update_card_locks(current_stats)

            # Match-Details für Designer vorbereiten
            match_details = {**current_stats["monster_matches"], **current_stats["wissen_matches"]}

            # --- Persona 1: Analyst ---
            print("\n[1/3] Analyst: Analysiere Score-Matrix...")
            analyse = self.analyst.analyze(
                self.cache,
                self.monster_karten["karten"],
                self.wissens_karten["karten"],
                iteration,
                change_history=self.change_history,
                locked_cards=self.locked_cards,
                failed_changes=self.failed_changes,
                temperature=current_temp,
            )

            if analyse.get("ziel_erreicht"):
                print("  ✓ Ziel erreicht! Keine Problemkarten mehr.")
                snap_dir = self._snapshot(iteration, "final")
                stats = _compute_match_stats(self.cache, monster_ids, wissen_ids)
                self._write_report(iteration, stats, analyse, [], snap_dir)
                self._save_best()
                return {"status": "erfolg", "iterationen": iteration, "stats": stats}

            for pk in analyse.get("problemkarten", []):
                problem_label = "↑ zu stark" if pk.get("problem") != "zu_schwach" else "↓ zu schwach"
                print(f"  → {pk['id']} ({pk.get('aktuelle_matches', '?')} Matches, {problem_label}): {pk.get('anweisung', '')[:80]}")

            # --- Persona 2: Game Designer ---
            print("\n[2/3] Game Designer: Kartentexte überarbeiten...")
            aenderungen = self.designer.redesign(
                analyse.get("problemkarten", []),
                self.monster_karten["karten"],
                self.wissens_karten["karten"],
                match_details=match_details,
                change_history=self.change_history,
                temperature=current_temp,
            )

            if not aenderungen:
                print("  Keine Änderungen vorgeschlagen. Abbruch.")
                break

            # Semantische Diff-Prüfung
            print("\n  Semantische Diff-Prüfung:")
            aenderungen = self._filter_semantic_diffs(aenderungen)

            if not aenderungen:
                print("  Alle Änderungen als kosmetisch abgelehnt. Zählt als Stagnation.")
                stale_count += 1
                snap_dir = self._snapshot(iteration, "nach")
                self._write_report(iteration, current_stats, analyse, [], snap_dir)
                history.append({
                    "iteration": iteration,
                    "avg_before": best_avg,
                    "avg_after": best_avg,
                    "changed_cards": [],
                    "note": "alle Änderungen als kosmetisch abgelehnt",
                })
                if stale_count >= MAX_STALE_ITERATIONS:
                    print(f"\n  ✗ Abbruch: {MAX_STALE_ITERATIONS} Iterationen ohne Verbesserung.")
                    break
                continue

            changed_ids = []
            for ch in aenderungen:
                print(f"  → {ch['id']}: {ch.get('begruendung', '')[:80]}")
                changed_ids.append(ch["id"])

            self._apply_changes(aenderungen)

            # --- Persona 3: Matcher ---
            print(f"\n[3/3] Matcher: {len(changed_ids)} Karten neu bewerten...")
            self.matcher.rescore_changed(
                changed_ids,
                self.monster_karten["karten"],
                self.wissens_karten["karten"],
                self.cache,
            )

            # Neue Statistik berechnen
            new_stats = _compute_match_stats(self.cache, monster_ids, wissen_ids)
            new_avg = new_stats["avg_total"]
            new_std = new_stats["std_dev"]

            print(f"\n  Ergebnis: Ø {new_avg} (vorher: {best_avg}), Std {new_std} (vorher: {best_std})")
            if new_stats["too_weak_count"] > 0:
                print(f"  ⚠ Zu schwache Karten (< {MIN_MATCHES} Matches): {new_stats['too_weak_count']}")

            # Snapshot NACH der Iteration
            snap_dir = self._snapshot(iteration, "nach")
            self._write_report(iteration, new_stats, analyse, aenderungen, snap_dir)

            history.append({
                "iteration": iteration,
                "avg_before": best_avg,
                "avg_after": new_avg,
                "std_before": best_std,
                "std_after": new_std,
                "changed_cards": changed_ids,
                "temperature": current_temp,
            })

            # Rollback-Check: Verbesserung = weniger schwache Karten ODER (gleich viele + niedrigerer Avg)
            # Neu: Bei gleichem Avg UND gleicher Weak-Zahl → niedrigere Std-Abweichung ist besser
            new_weak = new_stats["too_weak_count"]
            is_better = (
                new_weak < best_weak
                or (new_weak == best_weak and new_avg < best_avg)
                or (new_weak == best_weak and new_avg == best_avg and new_std < best_std)
            )
            is_worse = (
                new_weak > best_weak
                or (new_weak == best_weak and new_avg > best_avg)
            )

            if is_worse:
                print(f"  ✗ Verschlechterung! Rollback auf vorherige Version.")
                self._record_history(iteration, aenderungen, improved=False)
                self.cache = copy.deepcopy(best_cache)
                self.monster_karten = copy.deepcopy(best_monster)
                self.wissens_karten = copy.deepcopy(best_wissen)
                stale_count += 1
            elif is_better:
                details = []
                if new_weak < best_weak:
                    details.append(f"schwache Karten: {best_weak} → {new_weak}")
                if new_avg != best_avg:
                    details.append(f"Ø: {best_avg} → {new_avg}")
                if new_std != best_std:
                    details.append(f"Std: {best_std} → {new_std}")
                print(f"  ✓ Verbesserung: {', '.join(details)}")
                self._record_history(iteration, aenderungen, improved=True)
                best_avg = new_avg
                best_std = new_std
                best_weak = new_weak
                best_cache = copy.deepcopy(self.cache)
                best_monster = copy.deepcopy(self.monster_karten)
                best_wissen = copy.deepcopy(self.wissens_karten)
                stale_count = 0
            else:
                print(f"  – Keine Veränderung.")
                self._record_history(iteration, aenderungen, improved=False)
                stale_count += 1

            # Ziel erreicht? (Durchschnitt OK UND keine zu schwachen Karten)
            if new_avg <= TARGET_AVG_MATCHES and new_stats["too_weak_count"] == 0:
                print(f"\n  ★ ZIEL ERREICHT! Ø {new_avg} ≤ {TARGET_AVG_MATCHES}, Std {new_std}, keine zu schwachen Karten")
                self._save_best()
                return {"status": "erfolg", "iterationen": iteration, "stats": new_stats, "history": history}

            # Abbruch bei Stagnation
            if stale_count >= MAX_STALE_ITERATIONS:
                print(f"\n  ✗ Abbruch: {MAX_STALE_ITERATIONS} Iterationen ohne Verbesserung.")
                break

        # Bestes Ergebnis wiederherstellen und im Output-Verzeichnis speichern
        self.cache = best_cache
        self.monster_karten = best_monster
        self.wissens_karten = best_wissen
        self._save_best()

        final_stats = _compute_match_stats(self.cache, monster_ids, wissen_ids)

        # Gesamtreport schreiben
        summary_path = self.output_dir / "pipeline_summary.json"
        summary_path.write_text(json.dumps({
            "status": "abbruch",
            "iterationen": len(history),
            "initial_avg": initial_stats["avg_total"],
            "final_avg": final_stats["avg_total"],
            "initial_std": initial_stats["std_dev"],
            "final_std": final_stats["std_dev"],
            "locked_cards": sorted(self.locked_cards),
            "total_changes": sum(len(v) for v in self.change_history.values()),
            "failed_changes": len(self.failed_changes),
            "history": history,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{'=' * 60}")
        print(f"  Pipeline beendet nach {len(history)} Iterationen")
        print(f"  Initial: Ø {initial_stats['avg_total']} (Std {initial_stats['std_dev']}) → Final: Ø {final_stats['avg_total']} (Std {final_stats['std_dev']})")
        print(f"  Gesperrte Karten: {len(self.locked_cards)}, Gescheiterte Änderungen: {len(self.failed_changes)}")
        print(f"  Ergebnisse: {self.output_dir}")
        print(f"{'=' * 60}")

        return {"status": "abbruch", "iterationen": len(history), "stats": final_stats, "history": history}
