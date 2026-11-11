"""Table I of the paper from the episode tables in data/libero_goal/ (numpy only; no GPU, simulator, or model).

Each file data/libero_goal/<checkpoint>.csv holds the conditions without a mechanism on the 200 shared initial states
(ten task files, initial states j = 0..19 of each). The two GR00T N1.7 files hold all six conditions (1,200 episodes).
The files of pi0.5, SmolVLA, Fast-WAM, Evo-1, and OpenVLA-OFT hold four conditions (800 episodes); their Other-task and
Imp. columns are not in the bundled records and come from running the experiment (docs/RUN.md). For those cells the
script prints the paper's value followed by "*".

Printed per checkpoint (percent of 200):
  Orig.       designated-task success under none/original
  Empty       any-task completion under none/empty, with the 95% cluster-bootstrap interval and, in parentheses,
              designated-task success on the same episodes
  Neutral     any-task completion under none/neutral ("do something")
  Nonsense    any-task completion under none/nonsense ("xxx")
  Other-task  the task named by the instruction was completed under none/other_task
  Imp.        any-task completion under none/impossible_action
  Non-start   episodes without completion, without contact, and with less than 5 cm of end-effector displacement,
              summed over the conditions present in the records

Cluster bootstrap: 10,000 resamples that draw the ten task files with replacement and, within each drawn file, its 20
initial states with replacement (numpy default_rng(20261019), percentile interval).

Usage: python3 libero_goal/table1.py [--data data/libero_goal] [--B 10000] [--episodes DIR --name NAME]
  --episodes DIR  also score worker output (DIR/**/episodes_*.jsonl written by `python -m wnta.worker`) and print it as
                  an extra row named NAME.
Exit code 1 if a value computed from the bundled tables differs from the paper's value. Values marked "*" are not
computed and are not compared.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261019
GS = [k * 50 + j for k in range(10) for j in range(20)]
CONDS = ["none/original", "none/empty", "none/neutral", "none/nonsense", "none/other_task", "none/impossible_action"]
NONSTART_DISP_M = 0.05
# (file stem, name in the paper)
CHECKPOINTS = [("gr00t_n17_ft", "GR00T N1.7 (ft)"), ("gr00t_n17_official", "GR00T N1.7 (official)"), ("pi05", "pi0.5"),
               ("smolvla", "SmolVLA"), ("fastwam", "Fast-WAM"), ("evo1", "Evo-1"), ("openvla_oft", "OpenVLA-OFT")]
# Table I of the paper: Orig., Empty, [interval], (designated on Empty), Neutral, Nonsense, Other-task, Imp.
PAPER = {
    "gr00t_n17_ft": (92.5, 63.0, (54.0, 71.5), 6.5, 60.5, 63.0, 92.0, 61.0),
    "gr00t_n17_official": (96.0, 43.5, (35.0, 52.0), 3.0, 51.5, 49.5, 96.0, 73.0),
    "pi05": (96.0, 94.0, (90.0, 97.5), 9.0, 90.0, 86.5, 96.5, 91.0),
    "smolvla": (75.5, 2.5, (0.5, 5.5), 0.0, 0.0, 1.0, 75.0, 3.0),
    "fastwam": (95.5, 30.5, (21.5, 40.5), 2.5, 37.5, 34.0, 96.0, 50.5),
    "evo1": (92.0, 56.5, (47.0, 65.5), 5.0, 68.0, 64.5, 93.0, 71.0),
    "openvla_oft": (97.5, 78.0, (71.0, 85.0), 8.0, 78.0, 78.5, 97.0, 83.5),
}


def cluster_idx(B, rng):
    groups = [[i for i, g in enumerate(GS) if g // 50 == k] for k in range(10)]
    out = np.empty((B, len(GS)), dtype=np.int64)
    ks = rng.integers(0, 10, size=(B, 10))
    for b in range(B):
        out[b] = np.concatenate([np.asarray(groups[k])[rng.integers(0, 20, size=20)] for k in ks[b]])
    return out


def load_csv(path):
    """{cond: {g: row}} with the columns used below."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["cond"], {})[int(r["g"])] = {
                "task": r["task"], "named_task": r["named_task"], "outcome": r["outcome"],
                "contact": r["first_contact"] != "", "max_disp": float(r["max_disp"])}
    return out


def load_jsonl(root):
    """The same structure from worker output (one JSON line per episode)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "episodes_*.jsonl"), recursive=True)):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                named = r.get("named_task")
                out.setdefault(r["cond"], {})[int(r["g"])] = {
                    "task": str(r["task"]), "named_task": "" if named is None else str(named), "outcome": str(r["outcome"]),
                    "contact": bool(r.get("contact_first_step")) or r.get("first_contact") is not None,
                    "max_disp": float(r["max_disp"])}
    return out


def flags(d, cond, kind):
    rows = [d[cond][g] for g in GS]
    if kind == "designated":
        return np.array([float(r["outcome"] == r["task"]) for r in rows])
    if kind == "named":
        return np.array([float(r["outcome"] == r["named_task"]) for r in rows])
    if kind == "completion":
        return np.array([float(r["outcome"] != "none") for r in rows])
    if kind == "non_start":
        return np.array([float(r["outcome"] == "none" and not r["contact"] and r["max_disp"] < NONSTART_DISP_M) for r in rows])
    raise ValueError(kind)


REQUIRED = CONDS[:4]


def row(d, idx):
    """Table values from the conditions present. None for Other-task or Imp. if that condition is not in the records."""
    complete = lambda c: c in d and sorted(d[c]) == GS
    missing = [c for c in REQUIRED if not complete(c)]
    if missing:
        return None, missing
    partial = [c for c in CONDS[4:] if c in d and not complete(c)]
    if partial:
        return None, partial
    pct = lambda x: 100.0 * float(x.mean())
    empty = flags(d, "none/empty", "completion")
    boot = 100.0 * empty[idx].mean(1)
    other = pct(flags(d, "none/other_task", "named")) if "none/other_task" in d else None
    imp = pct(flags(d, "none/impossible_action", "completion")) if "none/impossible_action" in d else None
    vals = (pct(flags(d, "none/original", "designated")), pct(empty),
            (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            pct(flags(d, "none/empty", "designated")), pct(flags(d, "none/neutral", "completion")),
            pct(flags(d, "none/nonsense", "completion")), other, imp)
    present = [c for c in CONDS if c in d]
    nonstart = int(sum(flags(d, c, "non_start").sum() for c in present))
    return (vals, nonstart, 200 * len(present)), []


def same(vals, ref):
    flat = lambda v: [v[0], v[1], v[2][0], v[2][1], *v[3:]]
    return all(a is None or abs(a - b) < 0.05 for a, b in zip(flat(vals), flat(ref)))


def fmt(name, vals, nonstart, n, ref=None):
    def cell(v, r):
        if v is not None:
            return f"{v:5.1f} "
        return f"{r:5.1f}*" if r is not None else "    - "
    r6, r7 = (ref[6], ref[7]) if ref else (None, None)
    return (f"{name:24s} {vals[0]:5.1f}  {vals[1]:5.1f} [{vals[2][0]:4.1f}-{vals[2][1]:4.1f}] ({vals[3]:3.1f})  {vals[4]:5.1f}  "
            f"{vals[5]:5.1f}  {cell(vals[6], r6)} {cell(vals[7], r7)} {nonstart:4d} of {n:,}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "libero_goal"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--episodes", default=None, help="directory with worker output (episodes_*.jsonl)")
    ap.add_argument("--name", default="your run")
    a = ap.parse_args(argv)
    idx = cluster_idx(a.B, np.random.default_rng(SEED))
    print(f"{'Checkpoint':24s} {'Orig.':>5s}  {'Empty [95% interval] (desig.)':29s} {'Neut.':>5s}  {'Nons.':>5s}  {'Other':>5s}  "
          f"{'Imp.':>5s}  {'Non-start'}")
    bad = 0
    cells = starred = 0
    for stem, name in CHECKPOINTS:
        res, missing = row(load_csv(os.path.join(a.data, stem + ".csv")), idx)
        if res is None:
            print(f"{name}: conditions missing or incomplete (200 initial states each): {', '.join(missing)}")
            bad += 1
            continue
        vals, nonstart, n = res
        ok = same(vals, PAPER[stem]) and nonstart == 0
        bad += not ok
        starred += (vals[6] is None) + (vals[7] is None)
        cells += 8 - (vals[6] is None) - (vals[7] is None)
        print(fmt(name, vals, nonstart, n, PAPER[stem]), "" if ok else "  DIFFERS FROM THE PAPER")
    print(f"rows whose computed values equal Table I of the paper: {len(CHECKPOINTS) - bad} of {len(CHECKPOINTS)} "
          f"({cells} values computed from the bundled records)")
    if starred:
        print(f"* {starred} values are the paper's values, not in the bundled records; run the experiment to obtain them "
              f"(docs/RUN.md).")
    if a.episodes:
        res, missing = row(load_jsonl(a.episodes), idx)
        if res is None:
            print(f"{a.name}: conditions missing or incomplete (200 initial states each): {', '.join(missing)}")
            return 2
        print(fmt(a.name, *res))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
