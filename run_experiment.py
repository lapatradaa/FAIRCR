"""
One run = the k=0 no-persona baseline + all 2,000 personas (k=1..4), over the SAME
model + SAME sampled snippets + SAME session -- so pass_rate (analyze.py) is always
comparing like-for-like. Personas are built IN MEMORY every run from persona_lib's
ATTRS/VALUE_POOL via generate_personas.build() -- never read from a cached CSV, so they
can never go stale.

Two ways to run:
  Sync (resumable, concurrent):
    python3 run_experiment.py --live --workers 8
  Batch API (~50% cheaper, for the full 2,000 x 50 = 100,000-call run):
    python3 run_experiment.py --batch --submit
    python3 collect_batch.py <batch_id>

Dry-run by default (no API calls, no cost) -- pass --live or --batch to execute.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from generate_personas import DEFAULT_COMBO_BUDGET, build
from persona_lib import ATTRS, TASK_BLOCK

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "issue_location", "dataset-issue-location.csv")
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

RESULT_COLS = (["snippet_idx", "persona_row_id", "k", "combo_id", "combo_attrs", "combo_values"] +
               ATTRS + ["ground_truth", "predicted_label", "raw_response"])

BASELINE_ROW = {"RowID": 0, "k": 0, "combo_id": "", "combo_attrs": "baseline",
                 "combo_values": "", "PromptText": TASK_BLOCK,
                 **{a: "" for a in ATTRS}}


def load_api_key(api_key_env="OPENAI_API_KEY"):
    """Project .env wins over an inherited shell env var -- otherwise a stale/
    placeholder value exported in the shell profile silently shadows the real key."""
    if os.path.exists(ENV_PATH):
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH, override=True)
    return os.environ.get(api_key_env) or None


def sample_snippets(dataset_path, n, seed):
    """Reservoir-sample n snippets balanced 50/50 by ground-truth class (issue 0/1),
    via one reservoir per class in a single pass over the dataset. n odd gives the
    extra snippet to class 0."""
    rng = random.Random(seed)
    n0 = n // 2 + (n % 2)
    n1 = n // 2
    caps = {"0": n0, "1": n1}
    reservoirs = {"0": [], "1": []}
    seen = {"0": 0, "1": 0}
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row["issue"]
            cap = caps.get(cls, 0)
            if cap == 0:
                continue
            i = seen[cls]
            if len(reservoirs[cls]) < cap:
                reservoirs[cls].append(row)
            else:
                j = rng.randint(0, i)
                if j < cap:
                    reservoirs[cls][j] = row
            seen[cls] += 1
    combined = reservoirs["0"] + reservoirs["1"]
    rng.shuffle(combined)
    return combined


LABEL_RE = re.compile(r"LABEL:\s*([01])")


def call_openai(client, model, system_prompt, code, max_output_tokens=None):
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": code},
        ],
        temperature=0,
    )
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    m = LABEL_RE.search(text)
    return (int(m.group(1)) if m else None), text


def load_done_pairs(results_path):
    """(persona_row_id, snippet_idx) pairs already present in results.csv, for resume."""
    if not os.path.exists(results_path):
        return set()
    done = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((int(row["persona_row_id"]), int(row["snippet_idx"])))
    return done


def iter_pairs(personas, snippets, done_pairs):
    for p in personas:
        for si, s in enumerate(snippets):
            if (p["RowID"], si) in done_pairs:
                continue
            yield p, si, s


def write_run_info(out_dir, model, extra, filename="run_info.json"):
    info = {"model": model, "timestamp": datetime.now(timezone.utc).isoformat(), **extra}
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path


def run_sync(client, model, personas, snippets, results_path, workers, max_output_tokens):
    done_pairs = load_done_pairs(results_path)
    pairs = list(iter_pairs(personas, snippets, done_pairs))
    total_planned = len(personas) * len(snippets)
    if done_pairs:
        print(f"Resuming: {len(done_pairs):,}/{total_planned:,} pairs already done, "
              f"{len(pairs):,} remaining.")
    if not pairs:
        print("Nothing to do -- all pairs already present in results.csv.")
        return

    file_exists = os.path.exists(results_path)
    lock = threading.Lock()
    f = open(results_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(RESULT_COLS)
        f.flush()

    done = 0

    def work(item):
        p, si, s = item
        pred, raw = call_openai(client, model, p["PromptText"], s["code"], max_output_tokens)
        return p, si, s, pred, raw

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(work, item) for item in pairs]
            for fut in as_completed(futures):
                p, si, s, pred, raw = fut.result()
                row = ([si, p["RowID"], p["k"], p["combo_id"], p["combo_attrs"], p["combo_values"]] +
                       [p[a] for a in ATTRS] +
                       [s["issue"], pred, raw.replace("\n", " ")])
                with lock:
                    writer.writerow(row)
                    done += 1
                    if done % 100 == 0:
                        f.flush()
                        print(f"  {done:,}/{len(pairs):,} calls done...")
    finally:
        f.close()
    print(f"\nWrote {done:,} new rows to {results_path}")

    if done:
        sort_results_file(results_path)


def sort_results_file(results_path):
    """Concurrent workers write rows in whatever order responses arrive, not submission
    order -- re-sort the finished file by (persona_row_id, snippet_idx) so each
    persona's snippets read out in ascending order."""
    with open(results_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda r: (int(r["persona_row_id"]), int(r["snippet_idx"])))
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        writer.writerows(rows)


def build_batch_requests(personas, snippets, model, max_output_tokens):
    """Yield (custom_id, request_dict, manifest_row) for every (persona, snippet) pair,
    including the baseline (RowID 0). Not filtered by resume -- batch jobs are meant to
    be submitted once as a whole; use sync mode if you need incremental resume."""
    for p in personas:
        for si, s in enumerate(snippets):
            custom_id = f"{p['RowID']}-{si}"
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": p["PromptText"]},
                    {"role": "user", "content": s["code"]},
                ],
                "temperature": 0,
            }
            if max_output_tokens:
                body["max_tokens"] = max_output_tokens
            request = {"custom_id": custom_id, "method": "POST",
                       "url": "/v1/chat/completions", "body": body}
            manifest_row = ([p["RowID"], p["k"], p["combo_id"], p["combo_attrs"], p["combo_values"]] +
                            [p[a] for a in ATTRS] + [si, s["issue"], custom_id])
            yield custom_id, request, manifest_row


def prepare_batch(personas, snippets, model, max_output_tokens, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    input_path = os.path.join(out_dir, "batch_input.jsonl")
    manifest_path = os.path.join(out_dir, "batch_manifest.csv")
    with open(input_path, "w", encoding="utf-8") as jf, \
         open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["persona_row_id", "k", "combo_id", "combo_attrs", "combo_values"] +
                        ATTRS + ["snippet_idx", "ground_truth", "custom_id"])
        n = 0
        for custom_id, request, manifest_row in build_batch_requests(personas, snippets, model, max_output_tokens):
            jf.write(json.dumps(request) + "\n")
            writer.writerow(manifest_row)
            n += 1
    return input_path, manifest_path, n


def submit_batch(client, input_path, out_dir, model, extra, run_info_name="run_info.json"):
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Submitted batch job: {batch.id} (status={batch.status})")
    write_run_info(out_dir, model, {**extra, "batch_id": batch.id, "input_file_id": uploaded.id},
                    filename=run_info_name)
    print(f"Save this batch id -- collect it later with: python3 collect_batch.py {batch.id}")
    return batch


def main():
    ap = argparse.ArgumentParser(description="Run k=0 baseline + all personas (k=1..4) against the model.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--k-min", type=int, default=1, help="lowest k to build (k=0 baseline is always included separately)")
    ap.add_argument("--k-max", type=int, default=4, help="up to 15 (all attributes at once)")
    ap.add_argument("--n-snippets", type=int, default=50)
    ap.add_argument("--snippet-seed", type=int, default=0)
    ap.add_argument("--value-seed", type=int, default=12, help="seed for persona value draws (k>=2)")
    ap.add_argument("--combo-seed", type=int, default=11, help="seed for which combos are sampled once C(15,k) > --combo-budget")
    ap.add_argument("--combo-budget", type=int, default=None,
                     help="max attribute combos per k (default: generate_personas.DEFAULT_COMBO_BUDGET, 1365); "
                          "below this, k is exhaustive (every C(15,k)); above, a random sample")
    ap.add_argument("--n-draws", type=int, default=1,
                     help="for k>=2, how many independent random value-draws to take per combo "
                          "(k=1 already sweeps all 5 values per attribute regardless of this)")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--base-url", default=None,
                     help="OpenAI-compatible endpoint override, e.g. Gemini's "
                          "https://generativelanguage.googleapis.com/v1beta/openai/ "
                          "(default: none, uses OpenAI's own API)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                     help="name of the .env / environment variable holding the API key for --base-url "
                          "(e.g. GEMINI_API_KEY when pointing --base-url at Gemini)")
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests (sync mode only)")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--results-name", default="results.csv",
                     help="filename for the results CSV within --out-dir (use this to split k ranges into separate files)")
    ap.add_argument("--run-info-name", default="run_info.json",
                     help="filename for the run-info JSON within --out-dir (set this alongside --results-name so split runs don't clobber each other's metadata)")
    ap.add_argument("--no-baseline", action="store_true",
                     help="skip the k=0 baseline row (useful when splitting k ranges across files that already have a baseline elsewhere)")
    ap.add_argument("--live", action="store_true", help="sync mode: actually call the API (costs money)")
    ap.add_argument("--batch", action="store_true", help="prepare a Batch API job (~50%% cheaper)")
    ap.add_argument("--submit", action="store_true", help="with --batch: actually upload + submit the job")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        sys.exit(f"ERROR: dataset not found at {args.dataset}")

    combo_budget = args.combo_budget if args.combo_budget is not None else DEFAULT_COMBO_BUDGET
    personas = build(args.k_min, args.k_max, value_seed=args.value_seed, combo_seed=args.combo_seed,
                     combo_budget=combo_budget, n_draws=args.n_draws)
    n_personas = len(personas)
    if not args.no_baseline:
        personas = [BASELINE_ROW] + personas
    baseline_note = "" if args.no_baseline else " + 1 baseline (k=0)"
    print(f"Built {n_personas:,} personas in memory (k={args.k_min}..{args.k_max}, combo_budget={combo_budget})"
          f"{baseline_note}.")

    print(f"Sampling {args.n_snippets} snippets from {os.path.relpath(args.dataset, SCRIPT_DIR)} "
          f"(seed={args.snippet_seed})...")
    snippets = sample_snippets(args.dataset, args.n_snippets, args.snippet_seed)
    print(f"Sampled {len(snippets)} snippets.")

    os.makedirs(args.out_dir, exist_ok=True)
    snippets_path = os.path.join(args.out_dir, "snippets.csv")
    with open(snippets_path, "w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(["snippet_idx", "issue", "code"])
        for si, s in enumerate(snippets):
            writer.writerow([si, s["issue"], s["code"]])
    print(f"Wrote sampled snippets: {snippets_path}")

    total_calls = len(personas) * len(snippets)
    print(f"Model: {args.model}")
    print(f"Total planned LLM calls: {len(personas):,} x {len(snippets)} = {total_calls:,}")
    est_in, est_out = total_calls * 400, total_calls * 20
    print(f"Rough token estimate: ~{est_in:,} input, ~{est_out:,} output.")

    results_path = os.path.join(args.out_dir, args.results_name)

    if args.batch:
        if args.base_url:
            print(f"WARNING: --batch with --base-url={args.base_url} is untested -- the Batch API "
                  "(files.create/batches.create) is OpenAI-specific and may not work against "
                  "other OpenAI-compatible endpoints. Prefer --live for non-OpenAI providers.")
        input_path, manifest_path, n = prepare_batch(personas, snippets, args.model,
                                                       args.max_output_tokens, args.out_dir)
        print(f"\nWrote batch input: {input_path}  ({n:,} requests)")
        print(f"Wrote batch manifest: {manifest_path}")
        if not args.submit:
            print("\nDRY RUN — batch job NOT submitted. Re-run with --submit to upload and start it.")
            return
        api_key = load_api_key(args.api_key_env)
        if not api_key:
            sys.exit(f"ERROR: {args.api_key_env} not found in environment or .env file.")
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)
        submit_batch(client, input_path, args.out_dir, args.model,
                     {"dataset": os.path.relpath(args.dataset, SCRIPT_DIR),
                      "n_snippets": args.n_snippets, "snippet_seed": args.snippet_seed,
                      "k_min": args.k_min, "k_max": args.k_max, "value_seed": args.value_seed,
                      "combo_seed": args.combo_seed, "combo_budget": combo_budget},
                     run_info_name=args.run_info_name)
        return

    if not args.live:
        print("\nDRY RUN — no API calls made. Re-run with --live (sync) or --batch (Batch API).")
        p, s = personas[1], snippets[0]
        print(f"\nSample persona x snippet pairing:")
        print(f"  k={p['k']} combo_id={p['combo_id']}  {p['combo_attrs']} = {p['combo_values']}")
        print(f"  System prompt:\n{p['PromptText']}\n")
        print(f"  User (code):\n{s['code'][:300]}")
        print(f"  Ground truth issue label: {s['issue']}")
        print(f"\nBaseline system prompt (k=0):\n{personas[0]['PromptText']}")
        print(f"\nWould write results to: {results_path}")
        return

    api_key = load_api_key(args.api_key_env)
    if not api_key:
        sys.exit(f"ERROR: {args.api_key_env} not found in environment or .env file.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)

    run_sync(client, args.model, personas, snippets, results_path, args.workers, args.max_output_tokens)
    write_run_info(args.out_dir, args.model,
                   {"dataset": os.path.relpath(args.dataset, SCRIPT_DIR),
                    "n_snippets": args.n_snippets, "snippet_seed": args.snippet_seed,
                    "k_min": args.k_min, "k_max": args.k_max, "value_seed": args.value_seed,
                    "combo_seed": args.combo_seed, "combo_budget": combo_budget},
                   filename=args.run_info_name)
    print(f"\nNext: python3 analyze.py {results_path}")


if __name__ == "__main__":
    main()
