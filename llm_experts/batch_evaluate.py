import json
import time
from pathlib import Path

from tqdm import tqdm

from llm_experts.cache import load_cache, save_cache
from llm_experts.scorer import ArgumentationScorer


def main():
    monster_karten = json.loads(Path("data/monster_karten.json").read_text(encoding="utf-8"))["karten"]
    wissens_karten = json.loads(Path("data/wissens_karten.json").read_text(encoding="utf-8"))["karten"]

    cache = load_cache()
    scorer = ArgumentationScorer()

    total = len(monster_karten) * len(wissens_karten)
    skipped = sum(
        1 for m in monster_karten for w in wissens_karten
        if f"{m['id']}_{w['id']}" in cache
    )
    to_evaluate = total - skipped

    print(f"Provider : {scorer.model}")
    print(f"Gesamt   : {total} | Cache: {skipped} | Neu: {to_evaluate}\n")

    pbar = tqdm(total=to_evaluate, unit="Paarung", dynamic_ncols=True)

    for monster in monster_karten:
        for wissen in wissens_karten:
            key = f"{monster['id']}_{wissen['id']}"

            if key in cache:
                continue

            score, begruendung = scorer.score(monster, wissen)
            cache[key] = {"score": score, "begruendung": begruendung}
            save_cache(cache)

            pbar.set_postfix_str(f"{key}  {score:.2f}  {begruendung[:50]}")
            pbar.update(1)
            time.sleep(0.05)

    pbar.close()
    print(f"\nFertig. llm_cache.json ({total} Einträge)")


if __name__ == "__main__":
    main()
