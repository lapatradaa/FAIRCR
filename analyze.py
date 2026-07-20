"""
Analyze a results.csv (from run_experiment.py's sync mode or collect_batch.py):
  - accuracy / precision / recall / f1 vs ground_truth, per persona (k, combo_id,
    combo_attrs, combo_values) and per k, plus an overall row.
  - pass_rate vs the paired k=0 baseline: fraction of (persona, snippet) rows whose
    predicted_label equals the baseline's predicted_label on the SAME snippet_idx.

GUARDRAIL: pass_rate is only valid when the baseline and the persona rows being scored
came from the same model + same snippets + same session. Since run_experiment.py always
writes the k=0 baseline into the SAME results.csv as the personas it's paired with, that
guarantee holds automatically here -- this script just double-checks it (identical
snippet_idx set, identical ground_truth per snippet_idx) and refuses to compute
pass_rate if it doesn't hold, rather than silently comparing against a stale baseline
from a different run.

Usage:
  python3 analyze.py [results.csv]
"""
import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(SCRIPT_DIR, "results", "results.csv")

GROUP_COLS = ["k", "combo_id", "combo_attrs", "combo_values"]


def confusion_counts(df):
    """Return TP, FP, TN, FN, unparsed (predicted_label is null), and derived metrics."""
    unparsed = df["predicted_label"].isna().sum()
    parsed = df.dropna(subset=["predicted_label"])
    tp = ((parsed["ground_truth"] == 1) & (parsed["predicted_label"] == 1)).sum()
    fp = ((parsed["ground_truth"] == 0) & (parsed["predicted_label"] == 1)).sum()
    tn = ((parsed["ground_truth"] == 0) & (parsed["predicted_label"] == 0)).sum()
    fn = ((parsed["ground_truth"] == 1) & (parsed["predicted_label"] == 0)).sum()
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and precision == precision and recall == recall else float("nan"))
    return {"n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn, "unparsed": unparsed,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def split_baseline(df):
    """Split results.csv into (baseline_df, persona_df) by k==0."""
    baseline_df = df[df["k"] == 0]
    persona_df = df[df["k"] != 0]
    return baseline_df, persona_df


def validate_paired_baseline(persona_df, baseline_df):
    """Refuse pass_rate unless baseline and persona rows cover the identical snippet_idx
    set with identical ground_truth -- the check that guards against ever pairing
    personas against a baseline from a different session/model/snippet sample."""
    if baseline_df.empty:
        return "no k=0 baseline rows found in this results file -- run_experiment.py always " \
               "generates one in the same session; re-run it rather than editing results.csv by hand."
    b_gt = baseline_df[["snippet_idx", "ground_truth"]].drop_duplicates()
    p_gt = persona_df[["snippet_idx", "ground_truth"]].drop_duplicates()
    b_idx, p_idx = set(b_gt["snippet_idx"]), set(p_gt["snippet_idx"])
    if b_idx != p_idx:
        return ("baseline and persona rows cover different snippet_idx sets "
                f"(persona-only: {sorted(p_idx - b_idx)}, baseline-only: {sorted(b_idx - p_idx)}). "
                "They must come from the SAME sampled snippets/session.")
    merged = p_gt.merge(b_gt, on="snippet_idx", suffixes=("_persona", "_baseline"))
    mismatches = merged[merged["ground_truth_persona"] != merged["ground_truth_baseline"]]
    if len(mismatches):
        return ("ground_truth differs between baseline and persona rows for snippet_idx "
                f"{mismatches['snippet_idx'].tolist()} -- different dataset samples/snapshots; "
                "pass_rate would be meaningless.")
    return None


def compute_pass_rate(persona_df, baseline_df, group_cols=None):
    baseline_pred = baseline_df[["snippet_idx", "predicted_label"]].rename(
        columns={"predicted_label": "baseline_predicted_label"})
    merged = persona_df.merge(baseline_pred, on="snippet_idx", how="inner").dropna(
        subset=["predicted_label", "baseline_predicted_label"])
    merged = merged.copy()
    merged["passed"] = merged["predicted_label"] == merged["baseline_predicted_label"]
    if group_cols is None:
        return merged["passed"].mean() if len(merged) else float("nan")
    return merged.groupby(group_cols)["passed"].mean().rename("pass_rate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", default=None, help="default: alongside the results file")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit(f"ERROR: {args.results} not found. Run run_experiment.py or collect_batch.py first.")

    df = pd.read_csv(args.results)
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    df["combo_id"] = df["combo_id"].fillna("").astype(str)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))

    n_unparsed = df["predicted_label"].isna().sum()
    print(f"Loaded {len(df):,} rows from {args.results} ({n_unparsed:,} unparsed predictions).")

    baseline_df, persona_df = split_baseline(df)
    print(f"Baseline (k=0) rows: {len(baseline_df):,}   Persona (k>=1) rows: {len(persona_df):,}")

    error = validate_paired_baseline(persona_df, baseline_df) if not persona_df.empty else None
    if error:
        print(f"\nWARNING: refusing to compute pass_rate -- {error}")

    # per-persona (finest grain: one row per k/combo_id/combo_attrs/combo_values)
    if not persona_df.empty:
        by_persona = persona_df.groupby(GROUP_COLS, dropna=False).apply(
            lambda g: pd.Series(confusion_counts(g)), include_groups=False
        ).reset_index()
        if not error:
            pr = compute_pass_rate(persona_df, baseline_df, group_cols=GROUP_COLS).reset_index()
            by_persona = by_persona.merge(pr, on=GROUP_COLS, how="left")
        else:
            by_persona["pass_rate"] = float("nan")
        metric_cols = ["accuracy", "precision", "recall", "f1", "pass_rate"]
        for c in metric_cols:
            by_persona[c] = by_persona[c].round(4)
        by_persona = by_persona[GROUP_COLS + ["n", "unparsed"] + metric_cols]
        by_persona_path = os.path.join(out_dir, "summary_by_persona.csv")
        by_persona.to_csv(by_persona_path, index=False)
        print(f"Wrote {by_persona_path}  ({len(by_persona)} personas)")

        # per-k overview
        rows = []
        for k, g in persona_df.groupby("k"):
            cc = confusion_counts(g)
            pass_rate = round(compute_pass_rate(g, baseline_df), 4) if not error else float("nan")
            rows.append({"k": k, "n": cc["n"], "accuracy": round(cc["accuracy"], 4),
                         "precision": round(cc["precision"], 4), "recall": round(cc["recall"], 4),
                         "f1": round(cc["f1"], 4), "pass_rate": pass_rate})
        overview = pd.DataFrame(rows).sort_values("k")
        overview_path = os.path.join(out_dir, "summary_overview.csv")
        overview.to_csv(overview_path, index=False)
        print(f"Wrote {overview_path}")
        print("\nPer-k overview (personas only):")
        print(overview.to_string(index=False))

    if not baseline_df.empty:
        bcc = confusion_counts(baseline_df)
        print(f"\nBaseline (k=0, no persona): n={bcc['n']} accuracy={bcc['accuracy']:.4f} "
              f"precision={bcc['precision']:.4f} recall={bcc['recall']:.4f} f1={bcc['f1']:.4f} "
              f"unparsed={bcc['unparsed']}")


if __name__ == "__main__":
    main()
