"""
Turn a finished Batch API job (submitted via `run_experiment.py --batch --submit`) into
a results.csv with the exact same schema as the sync runner, so analyze.py can read
either one identically.

Usage:
  python3 collect_batch.py <batch_id>
  python3 collect_batch.py <batch_id> --manifest results/batch_manifest.csv --out results/results.csv
"""
import argparse
import csv
import json
import os
import re
import sys

from persona_lib import ATTRS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")

RESULT_COLS = (["snippet_idx", "persona_row_id", "k", "combo_id", "combo_attrs", "combo_values"] +
               ATTRS + ["ground_truth", "predicted_label", "raw_response"])

LABEL_RE = re.compile(r"LABEL:\s*([01])")


def load_api_key():
    """Project .env wins over an inherited shell OPENAI_API_KEY -- otherwise a stale/
    placeholder value exported in the shell profile silently shadows the real key."""
    if os.path.exists(ENV_PATH):
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH, override=True)
    return os.environ.get("OPENAI_API_KEY") or None


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["custom_id"]: row for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description="Collect a finished Batch API job into results/results.csv.")
    ap.add_argument("batch_id")
    ap.add_argument("--manifest", default=os.path.join(RESULTS_DIR, "batch_manifest.csv"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "results.csv"))
    args = ap.parse_args()

    api_key = load_api_key()
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not found in environment or .env file.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    batch = client.batches.retrieve(args.batch_id)
    print(f"Batch {batch.id}: status={batch.status}  "
          f"completed={batch.request_counts.completed}/{batch.request_counts.total}  "
          f"failed={batch.request_counts.failed}")
    if batch.status != "completed":
        print("Not finished yet -- re-run this once status is 'completed'.")
        return

    if not os.path.exists(args.manifest):
        sys.exit(f"ERROR: manifest not found at {args.manifest} (from the --batch prepare step).")
    manifest = load_manifest(args.manifest)

    output_content = client.files.content(batch.output_file_id).text
    parsed = {}
    for line in output_content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        body = obj.get("response", {}).get("body", {})
        choices = body.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""
        text = text or ""
        m = LABEL_RE.search(text)
        parsed[custom_id] = (int(m.group(1)) if m else None, text.replace("\n", " "))

    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id).text
        n_errors = sum(1 for line in error_content.splitlines() if line.strip())
        print(f"WARNING: {n_errors} requests failed (see error file {batch.error_file_id}); "
              f"they'll be missing from {args.out}.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_written, n_missing = 0, 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_COLS)
        for custom_id, row in manifest.items():
            if custom_id not in parsed:
                n_missing += 1
                continue
            pred, raw = parsed[custom_id]
            writer.writerow(
                [row["snippet_idx"], row["persona_row_id"], row["k"], row["combo_id"], row["combo_attrs"], row["combo_values"]] +
                [row[a] for a in ATTRS] +
                [row["ground_truth"], pred, raw]
            )
            n_written += 1

    print(f"Wrote {n_written:,} rows to {args.out}"
          + (f" ({n_missing:,} manifest rows had no matching output)" if n_missing else ""))
    print(f"Next: python3 analyze.py {args.out}")


if __name__ == "__main__":
    main()
