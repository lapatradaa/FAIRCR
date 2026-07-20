# Claude Code task — FAIRCR2

## Project
FAIRCR2 measures how a code-review persona shifts an LLM's issue-detection vs. a
no-persona baseline, for attribute combos k = 0..4. Clean rebuild of an older, messier
project (FAIRCR). The 15 attributes and prompt phrasing live in `persona_lib.py`.

Design (exhaustive, no sampling/nesting):
| k | attrs | 15Ck | values/combo | rows |
|---|-------|------|--------------|------|
| 0 | baseline (no persona) | — | — | 50 snippet rows |
| 1 | 1 | 15 | 5 | 75 |
| 2 | 2 | 105 | 1 | 105 |
| 3 | 3 | 455 | 1 | 455 |
| 4 | 4 | 1365 | 1 | 1365 |
Total 2,000 personas.

## Files
- `persona_lib.py` — ATTRS, VALUE_POOL (5 values/attr), `humanize`, `combo_prompt`,
  `TASK_BLOCK`. Single source of truth for phrasing. No ontology/rdflib dependency.
- `generate_personas.py` — `build(k_min,k_max)` returns the exhaustive persona rows;
  CLI dumps `data/personas.csv`.
- `run_experiment.py` — one run = k=0 baseline + all personas, SAME model + SAME snippets
  + SAME session. Sync (resumable, `--workers`) or `--batch` (Batch API, ~50% cost).
  Output cap via `--max-output-tokens`. Dry-run by default; `--live` / `--batch`.
- `collect_batch.py` — turn a finished Batch API job into `results/*.csv`.
- `analyze.py` — accuracy/precision/recall/f1 (vs ground truth) + pass_rate (vs the paired
  baseline). Refuses pass_rate unless baseline & results share identical snippets + gt.

## The rule that must never break
`pass_rate` = fraction of (persona, snippet) rows whose predicted_label equals the
no-persona baseline's label on the SAME snippet. Valid only when baseline and personas are
from the same model + snippets + session. `run_experiment.py` generates the k=0 baseline in
the same run; `analyze.py` has a guardrail. Do not reintroduce a cross-session baseline.

## Change 1 — dataset moved INTO this project
The 583MB dataset now lives at `issue_location/dataset-issue-location.csv` inside FAIRCR2
(no longer the sibling ../FAIRCR). Update `DEFAULT_DATASET` in `run_experiment.py` to
`os.path.join(SCRIPT_DIR, "issue_location", "dataset-issue-location.csv")` and verify the
dry run still finds it.

## Change 2 — make persona generation systematic (option 2)
Right now `run_experiment.py` reads `data/personas.csv`, a derived cache that goes STALE if
`ATTRS`/`VALUE_POOL` change and you forget to regenerate it. Fix:
- `run_experiment.py` should build personas IN MEMORY every run via
  `from generate_personas import build; personas = build(1, 4)` — never read a cached CSV.
- Keep `generate_personas.py` runnable to OPTIONALLY dump `data/personas.csv` for
  inspection only (not a pipeline input).
- Personas depend only on ATTRS + VALUE_POOL (fixed seeds -> reproducible), NOT on the
  dataset. Confirm resume still works (it keys off results.csv, not personas.csv).

## Token efficiency (already designed for; keep it)
Cost is ~all input tokens (100k calls). Savers: resume (skip done pairs), output cap,
Batch API (~50%), concurrency. Storing vs. building the prompt does NOT change API tokens —
it's only a local-file concern.

## Your tasks
1. Apply Change 1 (local dataset path) and Change 2 (in-memory personas).
2. `pip install openai python-dotenv pandas` if missing; confirm `.env` has OPENAI_API_KEY.
3. Dry run: `python3 run_experiment.py` — confirm it finds the local dataset and builds
   2,000 personas in memory.
4. Smoke run: `python3 run_experiment.py --live --n-snippets 5 --workers 8` then
   `python3 analyze.py` — sanity-check parse rate and that k=0 baseline + pass_rate appear.
5. For the full run, prefer `--batch --submit` then `collect_batch.py <id>` (50% cheaper).

## Constraints
- Model default `gpt-5.4-mini`, 50 snippets, seed 0.
- Keep prompt phrasing identical to `combo_prompt` / `TASK_BLOCK`.
- Don't reintroduce a stale persona cache or a cross-session baseline.
