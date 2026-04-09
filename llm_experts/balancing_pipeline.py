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

MATCH_THRESHOLD = 0.75
TARGET_AVG_MATCHES = 5
MAX_ITERATIONS = 15
MAX_STALE_ITERATIONS = 10


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


def _llm_json_call(client: OpenAI, model: str, prompt: str, retries: int = 3) -> dict:
    """Sendet einen Prompt und parsed die JSON-Antwort."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.4,
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

    return {
        "monster_matches": monster_matches,
        "wissen_matches": wissen_matches,
        "monster_counts": monster_counts,
        "wissen_counts": wissen_counts,
        "avg_matches_monster": round(avg_monster, 2),
        "avg_matches_wissen": round(avg_wissen, 2),
        "avg_total": round((avg_monster + avg_wissen) / 2, 2),
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
    ) -> dict:
        monster_ids = [m["id"] for m in monster_karten]
        wissen_ids = [k["id"] for k in wissens_karten]
        stats = _compute_match_stats(cache, monster_ids, wissen_ids)

        m_by_id = _cards_by_id(monster_karten)
        k_by_id = _cards_by_id(wissens_karten)

        # Top-Problemkarten sammeln (Monster + Wissen gemischt, nach Anzahl Matches)
        problem_candidates = []
        for m_id, count in stats["monster_counts"].items():
            if count > TARGET_AVG_MATCHES:
                partners = stats["monster_matches"][m_id]
                card = m_by_id[m_id]
                problem_candidates.append({
                    "id": m_id,
                    "typ": "monster",
                    "name": card["name"],
                    "beschreibung": card["beschreibung"],
                    "aktuelle_matches": count,
                    "partner": partners,
                })
        for k_id, count in stats["wissen_counts"].items():
            if count > TARGET_AVG_MATCHES:
                partners = stats["wissen_matches"][k_id]
                card = k_by_id[k_id]
                problem_candidates.append({
                    "id": k_id,
                    "typ": "wissen",
                    "name": card["name"],
                    "beschreibung": card["beschreibung"],
                    "aktuelle_matches": count,
                    "partner": partners,
                })

        problem_candidates.sort(key=lambda x: x["aktuelle_matches"], reverse=True)
        top_problems = problem_candidates[:6]  # Mehr Kontext für das LLM

        if not top_problems:
            return {
                "iteration": iteration,
                "zusammenfassung": {
                    "avg_matches_monster": stats["avg_matches_monster"],
                    "avg_matches_wissen": stats["avg_matches_wissen"],
                },
                "problemkarten": [],
                "ziel_erreicht": True,
            }

        # LLM-Call: Analyse & Anweisungen
        problem_json = json.dumps(top_problems, ensure_ascii=False, indent=2)
        prompt = f"""Du bist ein Balancing-Analyst für ein Lernkartenspiel über agile Methoden (Scrum).

Das Spiel hat Monster-Karten (agile Anti-Patterns) und Wissens-Karten (agile Methoden).
Jedes Paar hat einen Match-Score (0-1). Ein Match zählt ab Score ≥ {MATCH_THRESHOLD}.

ZIEL: Jede Karte soll nur mit 4-6 Partnern stark matchen (Score ≥ {MATCH_THRESHOLD}).
Aktuell ist der Durchschnitt: Monster {stats['avg_matches_monster']}, Wissen {stats['avg_matches_wissen']}.

Hier sind die problematischsten Karten (zu viele Matches):

{problem_json}

Wähle die TOP 3 Karten aus, die am dringendsten überarbeitet werden müssen.
Für jede Karte:
1. Analysiere WARUM sie so breit matcht (zu allgemein? Überlappungen?)
2. Gib eine KONKRETE Anweisung, wie der Text spezifischer formuliert werden soll

Antworte als JSON:
{{
  "problemkarten": [
    {{
      "id": "M01",
      "typ": "monster|wissen",
      "aktuelle_matches": 18,
      "analyse": "Warum die Karte zu breit matcht...",
      "anweisung": "Konkrete Überarbeitungsanweisung..."
    }}
  ]
}}"""

        result = _llm_json_call(self.client, self.model, prompt)

        return {
            "iteration": iteration,
            "zusammenfassung": {
                "avg_matches_monster": stats["avg_matches_monster"],
                "avg_matches_wissen": stats["avg_matches_wissen"],
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
    ) -> list[dict]:
        if not anweisungen:
            return []

        # Max 3 Karten pro Iteration
        anweisungen = anweisungen[:3]

        m_by_id = _cards_by_id(monster_karten)
        k_by_id = _cards_by_id(wissens_karten)

        cards_context = []
        for aw in anweisungen:
            card_id = aw["id"]
            if card_id.startswith("M"):
                card = m_by_id.get(card_id)
            else:
                card = k_by_id.get(card_id)
            if card:
                cards_context.append({
                    "id": card_id,
                    "name": card["name"],
                    "thema": card.get("thema", ""),
                    "beschreibung": card["beschreibung"],
                    "anweisung": aw.get("anweisung", ""),
                    "analyse": aw.get("analyse", ""),
                })

        context_json = json.dumps(cards_context, ensure_ascii=False, indent=2)

        prompt = f"""Du bist ein Game Designer für ein Scrum-Lernkartenspiel.

Deine Aufgabe: Überarbeite die folgenden Kartentexte, um sie SPEZIFISCHER zu machen.
Das Ziel ist, dass jede Karte nur noch mit wenigen (4-6) Partnerkarten stark matcht,
statt mit fast allen.

REGELN:
- Behalte den LERNINHALT bei – nur die Formulierung wird spezifischer
- Monster-Karten: Beschreibe ein KONKRETES Szenario (Wer? Was genau? Welcher Kontext?)
- Wissens-Karten: Beschreibe eine KONKRETE Technik/Methode statt allgemeines Prinzip
- Der Name und das Thema der Karte bleiben gleich
- Die Beschreibung soll weiterhin unterhaltsam und im gleichen Stil sein
- Beschreibungen sollen ähnlich lang bleiben (1-3 Sätze)

Hier sind die Karten mit Überarbeitungsanweisungen:

{context_json}

Antworte als JSON:
{{
  "aenderungen": [
    {{
      "id": "M01",
      "alt": "Bisheriger Beschreibungstext...",
      "neu": "Neuer spezifischerer Beschreibungstext...",
      "begruendung": "Warum diese Änderung die Matches reduziert..."
    }}
  ]
}}"""

        result = _llm_json_call(self.client, self.model, prompt)
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

        if aenderungen:
            lines += ["", "## Änderungen in dieser Iteration"]
            for ch in aenderungen:
                lines.append(f"\n### {ch['id']}")
                lines.append(f"**Alt:** {ch.get('alt', '–')}")
                lines.append(f"**Neu:** {ch.get('neu', '–')}")
                lines.append(f"**Begründung:** {ch.get('begruendung', '–')}")

        report_path = snap_dir / "balance_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def run(self) -> dict:
        """Führt die iterative Balancing-Pipeline aus."""
        print("=" * 60)
        print("  BOSODO Balancing-Pipeline")
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

        best_avg = initial_stats["avg_total"]
        best_cache = copy.deepcopy(self.cache)
        best_monster = copy.deepcopy(self.monster_karten)
        best_wissen = copy.deepcopy(self.wissens_karten)
        stale_count = 0

        history = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n{'─' * 60}")
            print(f"  Iteration {iteration}")
            print(f"{'─' * 60}")

            # Snapshot VOR der Iteration
            self._snapshot(iteration, "vor")

            # --- Persona 1: Analyst ---
            print("\n[1/3] Analyst: Analysiere Score-Matrix...")
            analyse = self.analyst.analyze(
                self.cache,
                self.monster_karten["karten"],
                self.wissens_karten["karten"],
                iteration,
            )

            if analyse.get("ziel_erreicht"):
                print("  ✓ Ziel erreicht! Keine Problemkarten mehr.")
                snap_dir = self._snapshot(iteration, "final")
                stats = _compute_match_stats(self.cache, monster_ids, wissen_ids)
                self._write_report(iteration, stats, analyse, [], snap_dir)
                self._save_best()
                return {"status": "erfolg", "iterationen": iteration, "stats": stats}

            for pk in analyse.get("problemkarten", []):
                print(f"  → {pk['id']} ({pk.get('aktuelle_matches', '?')} Matches): {pk.get('anweisung', '')[:80]}")

            # --- Persona 2: Game Designer ---
            print("\n[2/3] Game Designer: Kartentexte überarbeiten...")
            aenderungen = self.designer.redesign(
                analyse.get("problemkarten", []),
                self.monster_karten["karten"],
                self.wissens_karten["karten"],
            )

            if not aenderungen:
                print("  Keine Änderungen vorgeschlagen. Abbruch.")
                break

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

            print(f"\n  Ergebnis: Ø {new_avg} (vorher: {best_avg})")

            # Snapshot NACH der Iteration
            snap_dir = self._snapshot(iteration, "nach")
            self._write_report(iteration, new_stats, analyse, aenderungen, snap_dir)

            history.append({
                "iteration": iteration,
                "avg_before": best_avg,
                "avg_after": new_avg,
                "changed_cards": changed_ids,
            })

            # Rollback-Check
            if new_avg > best_avg:
                print(f"  ✗ Verschlechterung! Rollback auf vorherige Version.")
                self.cache = copy.deepcopy(best_cache)
                self.monster_karten = copy.deepcopy(best_monster)
                self.wissens_karten = copy.deepcopy(best_wissen)
                stale_count += 1
            elif new_avg < best_avg:
                print(f"  ✓ Verbesserung: {best_avg} → {new_avg}")
                best_avg = new_avg
                best_cache = copy.deepcopy(self.cache)
                best_monster = copy.deepcopy(self.monster_karten)
                best_wissen = copy.deepcopy(self.wissens_karten)
                stale_count = 0
            else:
                print(f"  – Keine Veränderung.")
                stale_count += 1

            # Ziel erreicht?
            if new_avg <= TARGET_AVG_MATCHES:
                print(f"\n  ★ ZIEL ERREICHT! Ø {new_avg} ≤ {TARGET_AVG_MATCHES}")
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
            "history": history,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{'=' * 60}")
        print(f"  Pipeline beendet nach {len(history)} Iterationen")
        print(f"  Initial: Ø {initial_stats['avg_total']} → Final: Ø {final_stats['avg_total']}")
        print(f"  Ergebnisse: {self.output_dir}")
        print(f"{'=' * 60}")

        return {"status": "abbruch", "iterationen": len(history), "stats": final_stats, "history": history}
