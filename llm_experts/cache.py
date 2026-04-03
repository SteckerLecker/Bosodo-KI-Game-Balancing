import json
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent / "llm_cache.json"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_score(cache: dict, monster_id: str, wissen_id: str) -> float | None:
    entry = cache.get(f"{monster_id}_{wissen_id}")
    if entry is not None:
        return entry["score"]
    return None
