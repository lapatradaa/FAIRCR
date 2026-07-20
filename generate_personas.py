"""
Persona generator (k=1..k_max), depends only on persona_lib's ATTRS + VALUE_POOL --
not on the dataset. Fixed seeds make it reproducible.

  k=1: every attribute swept through all 5 of its VALUE_POOL values (15*5 = 75 rows).
  k>=2: every C(15, k) combo of attributes -- EXHAUSTIVE as long as C(15, k) fits within
        combo_budget (true for k=2..4 at the default budget=1365: 105, 455, 1365 combos).
        Once C(15, k) exceeds combo_budget (k>=5 by default), combos are instead a
        seeded RANDOM SAMPLE of combo_budget distinct k-sized subsets -- capping the
        combinatorial blowup (sum_{k=1..15} C(15,k) = 32,767) instead of enumerating
        every combo. Each combo gets one seeded joint value draw (one value per
        attribute in the combo, drawn from VALUE_POOL).

`build()` is the pipeline entrypoint (run_experiment.py imports it directly and builds
personas in memory every run -- never reads a cached CSV). This file's CLI only dumps
data/personas.csv for manual inspection.
"""
import argparse
import csv
import itertools
import math
import os
import random

from persona_lib import ATTRS, VALUE_POOL, combo_prompt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_COMBO_BUDGET = 1365  # = C(15, 4), the largest tier in the original exhaustive design


def _combos_for_k(k, combo_budget, rng):
    """Every C(15, k) combo if it fits in combo_budget, else combo_budget distinct
    k-sized subsets chosen by rejection sampling (cheap here: even the worst case,
    k=7/8 with C(15,k)=6435, sampling 1365 distinct combos has a low collision rate)."""
    total = math.comb(len(ATTRS), k)
    if total <= combo_budget:
        return list(itertools.combinations(ATTRS, k))
    seen = set()
    combos = []
    while len(combos) < combo_budget:
        c = tuple(sorted(rng.sample(ATTRS, k)))
        if c not in seen:
            seen.add(c)
            combos.append(c)
    return combos


def build(k_min=1, k_max=4, value_seed=12, combo_seed=11, combo_budget=DEFAULT_COMBO_BUDGET, n_draws=1):
    """Return the persona rows for k_min..k_max as a list of dicts, each with keys:
    RowID, k, combo_id, combo_attrs, combo_values, <15 ATTRS cols>, PromptText.

    n_draws: for k>=2, how many independent random value-draws to take per combo
    (k=1 always sweeps all 5 values per attribute regardless of n_draws -- that
    behavior needs no draw count, it's already exhaustive)."""
    rng_value = random.Random(value_seed)
    rng_combo = random.Random(combo_seed)
    rows = []
    rid = 1
    combo_id = 0
    for k in range(k_min, k_max + 1):
        if k == 1:
            for attr in ATTRS:
                for val in VALUE_POOL[attr]:
                    row = {a: "" for a in ATTRS}
                    row[attr] = val
                    row["RowID"] = rid
                    row["k"] = k
                    row["combo_id"] = ""
                    row["combo_attrs"] = attr
                    row["combo_values"] = val
                    row["PromptText"] = combo_prompt([(attr, val)])
                    rows.append(row)
                    rid += 1
            continue
        for combo in _combos_for_k(k, combo_budget, rng_combo):
            for _ in range(n_draws):
                combo_id += 1
                draw = {a: rng_value.choice(VALUE_POOL[a]) for a in combo}
                row = {a: "" for a in ATTRS}
                row.update(draw)
                row["RowID"] = rid
                row["k"] = k
                row["combo_id"] = combo_id
                row["combo_attrs"] = "|".join(combo)
                row["combo_values"] = "|".join(draw[a] for a in combo)
                row["PromptText"] = combo_prompt([(a, draw[a]) for a in combo])
                rows.append(row)
                rid += 1
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Dump the exhaustive personas to a CSV for inspection only "
                    "(run_experiment.py builds personas in memory and never reads this file).")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=4, help="up to 15 (all attributes at once)")
    ap.add_argument("--value-seed", type=int, default=12)
    ap.add_argument("--combo-seed", type=int, default=11, help="seed for which combos are sampled once C(15,k) > --combo-budget")
    ap.add_argument("--combo-budget", type=int, default=DEFAULT_COMBO_BUDGET,
                     help="max combos per k; below this, k is exhaustive (every C(15,k)); above, a random sample")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "data", "personas.csv"))
    args = ap.parse_args()

    rows = build(args.k_min, args.k_max, args.value_seed, args.combo_seed, args.combo_budget)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["RowID", "k", "combo_id", "combo_attrs", "combo_values"] + ATTRS + ["PromptText"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    per_k = {k: sum(1 for r in rows if r["k"] == k) for k in range(args.k_min, args.k_max + 1)}
    print(f"Wrote {len(rows)} personas to {args.out}")
    print("  k breakdown: " + ", ".join(f"k={k}:{n}" for k, n in per_k.items()))


if __name__ == "__main__":
    main()
