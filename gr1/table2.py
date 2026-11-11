"""Table II of the paper (humanoid results) from worker output or from the bundled records (standard library only).

Counts of 120 scenes per condition: NS = non-start, Att. = attempt, NC = non-contact motion, Compl. = any placement,
Desig. = designated-task success, Commit = episodes in which the gate let the policy act. Each cell is printed as
computed / paper.

Usage:
  python3 gr1/table2.py                      the bundled records in data/gr1/ (gate rows: first run; gate/template_only
                                             ran only in the repeat)
  python3 gr1/table2.py --episodes DIR       worker output, DIR/**/episodes_*.jsonl written by gr1/gr1_closed_loop.py

The full analysis of the bundled records (statistical tests, decision rules, comparison of the two gate runs) is
gr1/analyze_gr1.py.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gr1 import PAPER, RES, TABLE_ROWS  # noqa: E402

N = 120


def bundled():
    rows = {}
    with open(os.path.join(RES, "gr1_tabletop_episodes.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["cond"], []).append(r)
    with open(os.path.join(RES, "gr1_gate_episodes.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["run"] == ("repeat" if r["cond"] == "gate/template_only" else "first"):
                rows.setdefault(r["cond"], []).append(r)
    return rows


def from_worker(root):
    rows = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "episodes_*.jsonl"), recursive=True)):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["cond"] not in TABLE_ROWS:
                    continue
                rows.setdefault(e["cond"], []).append({
                    "category": e["category"], "outcome": e["outcome"], "designated": e["designated"],
                    "gate_commit": "" if e.get("gate_commit") is None else str(e["gate_commit"])})
    return rows


def cell(rr):
    count = lambda c: sum(1 for r in rr if r["category"] == c)
    commit = sum(r["gate_commit"] == "True" for r in rr) if rr[0].get("gate_commit") not in (None, "") else None
    return (count("nonstart"), count("attempt"), count("ncmotion"), sum(r["outcome"] != "none" for r in rr),
            sum(int(r["designated"]) for r in rr), commit)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=None, help="directory with worker output (episodes_*.jsonl)")
    a = ap.parse_args(argv)
    rows = from_worker(a.episodes) if a.episodes else bundled()
    show = lambda v, p: "---" if p is None else f"{v}/{p}"
    print(f"{'Condition':24s} {'n':>4s} {'NS':>9s} {'Att.':>9s} {'NC':>9s} {'Compl.':>9s} {'Desig.':>9s} {'Commit':>9s}   (computed / paper)")
    equal = total = 0
    for c in TABLE_ROWS:
        if c not in rows:
            print(f"{c:24s}    0   not run")
            continue
        got, want = cell(rows[c]), PAPER[c]
        print(f"{c:24s} {len(rows[c]):4d} " + " ".join(f"{show(g, w):>9s}" for g, w in zip(got, want))
              + ("" if len(rows[c]) == N else f"   expected {N} episodes"))
        for g, w in zip(got, want):
            if w is not None:
                total += 1
                equal += g == w
    print(f"cells equal to the paper: {equal} of {total}")
    return 1 if (not a.episodes and equal != total) else 0


if __name__ == "__main__":
    sys.exit(main())
