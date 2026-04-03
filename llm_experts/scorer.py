import json
import os
import time

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI

load_dotenv()


def _build_client() -> tuple[OpenAI, str, bool]:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            timeout=180.0,
        )
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:4b")
        # Qwen3-Modelle haben einen Thinking-Modus der content leer lässt
        disable_thinking = "qwen3" in model.lower()

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY ist nicht gesetzt")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        disable_thinking = False

    else:
        raise ValueError(f"Unbekannter LLM_PROVIDER: '{provider}'. Erlaubt: 'ollama', 'openrouter'")

    return client, model, disable_thinking


class ArgumentationScorer:
    def __init__(self):
        self.client, self.model, self._disable_thinking = _build_client()

    def score(self, monster: dict, wissen: dict) -> tuple[float, str]:
        prompt = f"""Du bist ein Experte für digitale Medienkompetenz bei Jugendlichen.
Bewerte, wie gut diese Wissenskarte als Verteidigung gegen das Monster-Problem funktioniert.

Monster: "{monster['name']}" — {monster['beschreibung']}
Wissenskarte: "{wissen['name']}" — {wissen['beschreibung']}

Antworte ausschließlich mit einem JSON-Objekt mit genau diesen zwei Feldern:
- "score": Zahl zwischen 0.0 und 1.0
  (0.0 = keinerlei Bezug | 0.5 = indirekter Bezug | 1.0 = perfekte Verteidigung)
- "begruendung": Ein Satz (max. 80 Zeichen), warum dieser Score

Beispiel: {{"score": 0.85, "begruendung": "Wissen adressiert direkt das Problem der Karte"}}"""

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        if self._disable_thinking:
            kwargs["extra_body"] = {"think": False}

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except APITimeoutError:
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))

        message = response.choices[0].message
        content = message.content
        if not content:
            # Fallback: Qwen3 thinking mode leaks output into reasoning_content
            extras = getattr(message, "model_extra", {}) or {}
            content = extras.get("reasoning_content", "")
        if not content:
            raise ValueError(f"Leere Antwort vom Modell. Vollständige Nachricht: {message}")
        data = json.loads(content)
        score = max(0.0, min(1.0, float(data["score"])))
        begruendung = data.get("begruendung", "")
        return score, begruendung
