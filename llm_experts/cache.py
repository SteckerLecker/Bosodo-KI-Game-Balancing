import json
from pathlib import Path
from typing import Optional

# Fallback wenn kein data_dir übergeben wird
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_cache.json"


def _cache_path(data_dir: Optional[str] = None) -> Path:
    if data_dir is not None:
        return Path(data_dir) / "llm_cache.json"
    return _DEFAULT_CACHE_PATH


def load_cache(data_dir: Optional[str] = None) -> dict:
    path = _cache_path(data_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, data_dir: Optional[str] = None) -> None:
    path = _cache_path(data_dir)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_score(cache: dict, monster_id: str, wissen_id: str) -> float | None:
    entry = cache.get(f"{monster_id}_{wissen_id}")
    if entry is not None:
        return entry["score"]
    return None
