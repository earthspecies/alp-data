"""Per-species / negative-handling analysis of DCLDE2013 multilabel predictions.

Reads a raw-predictions JSONL emitted by NatureLM `cli.py chat-tasks` for the
`dclde2013-multilabel-species` split and reports, per species, precision /
recall / F1 (binary presence), plus the false-positive rate on true-negative
clips and a predicted-vs-target confusion breakdown.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SPECIES = [
    "Balaenoptera physalus",  # fin
    "Megaptera novaeangliae",  # humpback
    "Eubalaena glacialis",  # North Atlantic right
    "Balaenoptera musculus",  # blue
]
COMMON = {
    "Balaenoptera physalus": "fin",
    "Megaptera novaeangliae": "humpback",
    "Eubalaena glacialis": "NARW",
    "Balaenoptera musculus": "blue",
}


def _to_set(text: str) -> set[str]:
    text = (text or "").strip()
    if not text or text.lower() == "none":
        return set()
    return {p.strip() for p in text.split(",") if p.strip() and p.strip().lower() != "none"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]

    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()

    n = len(rows)
    n_neg = 0
    neg_fp = 0  # negative clips where model predicted >=1 species
    exact = 0
    confusion = defaultdict(Counter)  # target species -> Counter(predicted species)

    for r in rows:
        pred = _to_set(r.get("processed_prediction") or r.get("prediction") or "")
        tgt = _to_set(r.get("target") or "")

        if pred == tgt:
            exact += 1

        if not tgt:
            n_neg += 1
            if pred:
                neg_fp += 1

        for sp in SPECIES:
            in_t = sp in tgt
            in_p = sp in pred
            if in_t:
                support[sp] += 1
            if in_t and in_p:
                tp[sp] += 1
            elif in_p and not in_t:
                fp[sp] += 1
            elif in_t and not in_p:
                fn[sp] += 1

        # confusion: for each true species, what did the model predict
        for t in (tgt or {"None"}):
            for p in (pred or {"None"}):
                confusion[t][p] += 1

    def prf(sp: str) -> tuple[float, float, float]:
        p = tp[sp] / (tp[sp] + fp[sp]) if (tp[sp] + fp[sp]) else 0.0
        rcl = tp[sp] / (tp[sp] + fn[sp]) if (tp[sp] + fn[sp]) else 0.0
        f1 = 2 * p * rcl / (p + rcl) if (p + rcl) else 0.0
        return p, rcl, f1

    print(f"file: {args.jsonl}")
    print(f"clips: {n}  | true-negative clips: {n_neg}")
    print(f"exact-match accuracy: {exact / n:.4f}")
    print(
        f"false-positive rate on negatives: {neg_fp}/{n_neg} = "
        f"{(neg_fp / n_neg if n_neg else 0):.4f}\n"
    )

    print(f"{'species':28s} {'common':9s} {'supp':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
          f"{'prec':>6s} {'rec':>6s} {'F1':>6s}")
    macro = []
    for sp in SPECIES:
        p, rcl, f1 = prf(sp)
        macro.append(f1)
        print(f"{sp:28s} {COMMON[sp]:9s} {support[sp]:5d} {tp[sp]:5d} {fp[sp]:5d} "
              f"{fn[sp]:5d} {p:6.3f} {rcl:6.3f} {f1:6.3f}")
    print(f"\nmacro-F1 over 4 species: {sum(macro) / len(macro):.4f}")

    print("\nconfusion (rows=true label incl. None, cols=predicted):")
    cols = SPECIES + ["None"]
    print(f"{'true \\ pred':22s} " + " ".join(f"{COMMON.get(c, c)[:8]:>8s}" for c in cols))
    for t in cols:
        row = confusion.get(t, Counter())
        print(f"{COMMON.get(t, t):22s} " + " ".join(f"{row.get(c, 0):8d}" for c in cols))


if __name__ == "__main__":
    main()
