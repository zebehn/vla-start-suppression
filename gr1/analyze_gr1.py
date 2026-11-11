"""Analysis of the GR-1 tabletop runs, GR00T N1.6 (GR-1) on 120 RoboCasa GR-1 tabletop scenes (standard library only).

Inputs: data/gr1/gr1_tabletop_episodes.csv (conditions without a mechanism and with the prefix) and
data/gr1/gr1_gate_episodes.csv (the gate, run twice: `run` first on an H200, repeat on an A100), and the threshold in
data/gr1/gr1_gate.json. Outputs: data/gr1/gr1_tabletop_summary.json and data/gr1/gr1_gate_summary.json. Prints the paper's
table of humanoid results (tab:gr1: NS, Att., NC, Compl., Desig., Commit) next to the paper's values, and the two-run
gate comparison (held episodes per condition in each run, episodes whose gate decision changed, and whether their
distance lay within 5% of tau).

Episode categories (the paper's four categories on the humanoid, gr1_common.category): completion = any placement of an
object into a container (`outcome` not "none"); attempt = no completion, a hand touched an object, container, or fixture
(`first_contact` non-empty; the table is excluded); non-contact motion = no completion, no contact, and the larger of the
two hand displacements (`max_hand_disp`) >= 5 cm or the torso displacement (`torso_disp`) >= 5 cm; non-start = the rest.
The `category` column holds the category computed by the worker; the analysis recomputes it from the columns wherever
the displacements are recorded and stops on any difference. Designated-task success (`designated`) is the environment's
own success check.

Decision rules (counts of 120):
  - H1 on the humanoid: start part supported if the non-start count under none/empty is at most 12 (10%); completion part
    supported if any-task completion under none/empty exceeds 24 (20%).
  - H3 on the humanoid (impossible_action, absent_object): supported if the non-start rate minus that of none/original is
    at most 0.10, rejected if the non-start rate is at least 0.50.
  - H4 on the humanoid: the prefix suppresses the start on a non-task input if the non-start count is at least 108 (90%).
  - H5 on the humanoid, for each gate run: supported if gate/neutral and gate/nonsense have at least 108 non-starts and
    gate/original commits in at least 108 episodes; gate/template_only is suppressed if it has at least 108 non-starts.
    gate/empty must have at least 114 non-starts (check of the gate implementation; the distance of "" against "" is 0).
Paired tests: exact two-sided McNemar on the same 120 scenes.

Converting worker output: --from-runs DIR,GPU,GPU_TYPE,RUN (repeatable) reads DIR/**/episodes_*.jsonl written by
gr1/gr1_closed_loop.py and writes the two episode tables first (RUN is first or repeat for gate conditions).
Usage: python3 gr1/analyze_gr1.py [--from-runs runs/gr1/eval,b,A100,first ...]
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gr1_common as G  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "data", "gr1")
N = 120
TAB_COLS = ["cond", "mechanism", "g", "task", "task_name", "seed", "scene_seed", "gpu", "gpu_type", "text", "outcome",
            "outcome_step", "designated", "first_contact", "first_contact_step", "max_hand_disp", "torso_disp", "category",
            "named_task", "n_calls", "nan_calls", "steps", "scene_sig"]
GATE_COLS = ["run", "cond", "mechanism", "g", "task", "task_name", "seed", "scene_seed", "gpu", "gpu_type", "text", "gate_d",
             "gate_commit", "hold", "outcome", "outcome_step", "designated", "first_contact", "first_contact_step",
             "max_hand_disp", "torso_disp", "category", "n_calls", "nan_calls", "steps", "scene_sig"]
TABLE_ROWS = ["none/original", "none/empty", "none/neutral", "none/nonsense", "none/template_only",
              "none/impossible_action", "none/absent_object", "prefix/original", "prefix/empty", "prefix/nonsense",
              "gate/original", "gate/empty", "gate/neutral", "gate/nonsense", "gate/template_only"]
# tab:gr1 of the paper: NS, Att., NC, Compl., Desig., Commit (None = "---"). The gate rows are the first gate run; the
# first run did not include gate/template_only, so that row comes from the repeat.
PAPER = {
    "none/original": (0, 30, 0, 90, 88, None), "none/empty": (0, 53, 67, 0, 0, None),
    "none/neutral": (0, 36, 84, 0, 0, None), "none/nonsense": (0, 58, 61, 1, 0, None),
    "none/template_only": (0, 109, 2, 9, 8, None), "none/impossible_action": (0, 82, 6, 32, 30, None),
    "none/absent_object": (0, 87, 0, 33, 29, None),
    "prefix/original": (0, 47, 0, 73, 70, None), "prefix/empty": (0, 48, 72, 0, 0, None),
    "prefix/nonsense": (0, 45, 74, 1, 1, None),
    "gate/original": (0, 31, 0, 89, 87, 120), "gate/empty": (120, 0, 0, 0, 0, 0), "gate/neutral": (108, 3, 9, 0, 0, 12),
    "gate/nonsense": (114, 2, 4, 0, 0, 6), "gate/template_only": (0, 109, 2, 9, 8, 120),
}
PROTOCOL = {"n_scenes": N, "n_tasks": G.N_TASKS, "scenes_per_task": G.N_SCENES_PER_TASK, "max_steps": 720,
            "settle_steps": 10, "chunk_steps": G.GATE_H, "executed_steps_per_chunk": 8, "action_dims": G.GATE_DIM,
            "hand_disp_threshold_m": G.DISP, "torso_disp_threshold_m": G.TORSO_DISP,
            "placement_hand_clearance_m": 0.25, "joint_randomization_rad": 0.2, "post_placement_wait_steps": 60,
            "template": G.TEMPLATE, "prefix": G.PREFIX}
NEAR_TAU = 0.05   # relative distance to tau reported for changed gate decisions


def read_csv(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar(a, b):
    """a, b: dicts scene -> 0/1 on the same scenes. Exact two-sided p; b01 = a 0 and b 1, b10 = a 1 and b 0."""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if not a[k] and b[k])
    b10 = sum(1 for k in keys if a[k] and not b[k])
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        m = min(b01, b10)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)
    return {"n_paired": len(keys), "only_second": b01, "only_first": b10, "p": p}


def scene(r):
    return (int(r["task"]), int(r["seed"]))


def completed(r):
    return r["outcome"] != "none"


def check_rows(rows, name):
    """Recompute the category from the columns (where the displacements are recorded); report the rows that differ."""
    bad, n_checked = [], 0
    for r in rows:
        if r["max_hand_disp"] == "":
            continue
        n_checked += 1
        c = G.category(r["outcome"], r["first_contact"] != "", float(r["max_hand_disp"]), float(r["torso_disp"]))
        if c != r["category"]:
            bad.append((name, r["cond"], r["task"], r["seed"], r["category"], c))
    return n_checked, bad


def cell(rows):
    cats = {c: sum(1 for r in rows if r["category"] == c) for c in ("completion", "attempt", "ncmotion", "nonstart")}
    n = len(rows)
    comp = sum(completed(r) for r in rows)
    des = sum(int(r["designated"]) for r in rows)
    ncm = [r for r in rows if r["category"] == "ncmotion" and r["max_hand_disp"] != ""]
    out = {"n": n, "counts": cats, "nonstart_rate": cats["nonstart"] / n,
           "nonstart_wilson": wilson(cats["nonstart"], n), "completion_n": comp, "completion_rate": comp / n,
           "completion_wilson": wilson(comp, n), "designated_n": des, "designated_rate": des / n,
           "designated_wilson": wilson(des, n),
           "outcomes": {o: sum(1 for r in rows if r["outcome"] == o) for o in
                        (G.OUT_TARGET, G.OUT_DIST_INTO_TARGET, G.OUT_TARGET_INTO_DIST, G.OUT_TIE, G.OUT_NONE)},
           "designated_by_task": {t["name"]: sum(int(r["designated"]) for r in rows if int(r["task"]) == t["k"])
                                  for t in G.TASKS},
           "median_max_hand_disp_ncmotion_m": statistics.median(float(r["max_hand_disp"]) for r in ncm) if ncm else None,
           "median_torso_disp_ncmotion_m": statistics.median(float(r["torso_disp"]) for r in ncm) if ncm else None,
           "gpu_types": sorted({r["gpu_type"] for r in rows}),
           "displacements_recorded": all(r["max_hand_disp"] != "" for r in rows)}
    if rows and rows[0].get("gate_commit") not in (None, ""):
        ds = [float(r["gate_d"]) for r in rows]
        commit = sum(r["gate_commit"] == "True" for r in rows)
        out.update({"commit_n": commit, "held_n": n - commit, "gate_commit_rate": commit / n,
                    "gate_d_median": statistics.median(ds), "gate_d_min": min(ds), "gate_d_max": max(ds)})
    return out


def table_row(c):
    k = c["counts"]
    return (k["nonstart"], k["attempt"], k["ncmotion"], c["completion_n"], c["designated_n"], c.get("commit_n"))


def by_cond(rows):
    out = {}
    for r in rows:
        out.setdefault(r["cond"], []).append(r)
    return out


def flag(rows, key):
    return {scene(r): int(key(r)) for r in rows}


def h3_rule(ns, ns_orig):
    if ns - ns_orig <= 0.10:
        return "supported (the policy started)"
    if ns >= 0.50:
        return "rejected (the policy held)"
    return "intermediate"


def tabletop(rows):
    bc = by_cond(rows)
    cells = {c: cell(bc[c]) for c in TABLE_ROWS if c in bc}
    orig = bc["none/original"]
    cmp_ = {
        "template_only_vs_empty_completion": mcnemar(flag(bc["none/empty"], completed), flag(bc["none/template_only"], completed)),
        "prefix_original_vs_original_designated": mcnemar(flag(orig, lambda r: int(r["designated"])),
                                                          flag(bc["prefix/original"], lambda r: int(r["designated"]))),
        "impossible_action_vs_original_designated": mcnemar(flag(orig, lambda r: int(r["designated"])),
                                                             flag(bc["none/impossible_action"], lambda r: int(r["designated"]))),
        "absent_object_vs_original_designated": mcnemar(flag(orig, lambda r: int(r["designated"])),
                                                        flag(bc["none/absent_object"], lambda r: int(r["designated"]))),
    }
    ao = bc["none/absent_object"]
    named_done = sum(1 for r in ao if r["named_task"] != "" and int(r["named_task"]) == int(r["task"]) and completed(r))
    e = cells["none/empty"]
    ns = {c: cells[c]["nonstart_rate"] for c in cells}
    judg = {
        "H1_start": "supported (the policy started)" if e["counts"]["nonstart"] <= 12 else "rejected (the policy held)",
        "H1_completion": "supported" if e["completion_n"] > 24 else "rejected (no completion above 20%)",
        "H3_impossible_action": h3_rule(ns["none/impossible_action"], ns["none/original"]),
        "H3_absent_object": h3_rule(ns["none/absent_object"], ns["none/original"]),
        "H4_prefix_empty": "suppressed" if cells["prefix/empty"]["counts"]["nonstart"] >= 108 else "no suppression",
        "H4_prefix_nonsense": "suppressed" if cells["prefix/nonsense"]["counts"]["nonstart"] >= 108 else "no suppression",
        "H4_prefix_original_overrestraint": ("over-restraint" if ns["prefix/original"] - ns["none/original"] > 0.10
                                             else "no over-restraint"),
    }
    return {
        "record": "gr1_tabletop", "checkpoint": "GR00T N1.6 (GR-1)", "model": G.HF_REPO, "revision": G.HF_REVISION[:7],
        "protocol": PROTOCOL,
        "published_success": {t["name"]: t["official"] for t in G.TASKS},
        "published_success_mean": sum(t["official"] for t in G.TASKS) / G.N_TASKS,
        "cells": cells, "comparisons": cmp_,
        "absent_object_named_task_completed": named_done,
        "absent_object_completions_own_objects": cells["none/absent_object"]["completion_n"],
        "judgments": judg,
    }


def gate_summary(rows, gate):
    tau = float(gate["tau_p95"])
    runs = {}
    for run in ("first", "repeat"):
        rr = [r for r in rows if r["run"] == run]
        bc = by_cond(rr)
        cells = {c: cell(bc[c]) for c in TABLE_ROWS if c in bc}
        held_ns = all(r["category"] == "nonstart" for r in rr if r["gate_commit"] == "False")
        commit_nontask_ns = sum(1 for r in rr if r["gate_commit"] == "True" and r["cond"] != "gate/original"
                                and r["category"] == "nonstart")
        j = {}
        if all(c in cells for c in ("gate/original", "gate/neutral", "gate/nonsense", "gate/empty")):
            ok = (cells["gate/neutral"]["counts"]["nonstart"] >= 108 and cells["gate/nonsense"]["counts"]["nonstart"] >= 108)
            j["H5"] = ("supported" if ok and cells["gate/original"]["commit_n"] >= 108 else
                       ("over-restraint" if ok else "suppression failed"))
            j["H5_margin_neutral"] = cells["gate/neutral"]["counts"]["nonstart"] - 108
            j["H5_margin_nonsense"] = cells["gate/nonsense"]["counts"]["nonstart"] - 108
            j["empty_check"] = "passed" if cells["gate/empty"]["counts"]["nonstart"] >= 114 else "failed"
        if "gate/template_only" in cells:
            j["template_only"] = ("suppressed" if cells["gate/template_only"]["counts"]["nonstart"] >= 108
                                  else "suppression failed")
        runs[run] = {"gpu_types": sorted({r["gpu_type"] for r in rr}), "gpus": sorted({r["gpu"] for r in rr}),
                     "cells": cells, "held_episodes_all_nonstart": held_ns,
                     "committed_nontask_nonstart_n": commit_nontask_ns, "judgments": j}
    first = {(r["cond"], scene(r)): r for r in rows if r["run"] == "first"}
    rep = {(r["cond"], scene(r)): r for r in rows if r["run"] == "repeat"}
    per = {}
    for c in TABLE_ROWS:
        keys = sorted(k for k in first if k[0] == c and k in rep)
        if not keys:
            continue
        changed, diffs = [], []
        for k in keys:
            a, b = first[k], rep[k]
            da, db = float(a["gate_d"]), float(b["gate_d"])
            diffs.append(abs(da - db))
            if a["gate_commit"] != b["gate_commit"]:
                changed.append({"task": k[1][0], "seed": k[1][1], "commit_first": a["gate_commit"] == "True",
                                "commit_repeat": b["gate_commit"] == "True", "gate_d_first": da, "gate_d_repeat": db,
                                "within_5pct_of_tau": abs(da - tau) <= NEAR_TAU * tau and abs(db - tau) <= NEAR_TAU * tau})
        same_sig = sum(first[k]["scene_sig"] == rep[k]["scene_sig"] for k in keys)
        ds = sorted(diffs)
        per[c] = {"n_paired": len(keys), "scene_sig_same": same_sig,
                  "held_first": runs["first"]["cells"][c]["held_n"], "held_repeat": runs["repeat"]["cells"][c]["held_n"],
                  "designated_first": runs["first"]["cells"][c]["designated_n"],
                  "designated_repeat": runs["repeat"]["cells"][c]["designated_n"],
                  "n_changed": len(changed), "changed": changed,
                  "near_tau_5pct_first": sum(1 for k in keys if abs(float(first[k]["gate_d"]) - tau) <= NEAR_TAU * tau),
                  "near_tau_5pct_repeat": sum(1 for k in keys if abs(float(rep[k]["gate_d"]) - tau) <= NEAR_TAU * tau),
                  "gate_d_abs_diff_median": statistics.median(ds), "gate_d_abs_diff_p95": ds[int(0.95 * (len(ds) - 1))],
                  "gate_d_abs_diff_max": ds[-1]}
    n_changed = sum(p["n_changed"] for p in per.values())
    tmpl = [r for r in rows if r["cond"] == "gate/template_only"]
    empty = [r for r in rows if r["cond"] == "gate/empty"]
    return {
        "record": "gr1_gate", "checkpoint": "GR00T N1.6 (GR-1)", "model": G.HF_REPO, "revision": G.HF_REVISION[:7],
        "tau": tau, "tau_rounded": round(tau, 3), "floor_median": gate["floor_median"],
        "floor_median_rounded": round(gate["floor_median"], 3),
        "calibration": {"n_floor_scenes": gate["n_floor_inits"], "n_floor_distances": gate["floor_pairs"],
                        "n_observations": gate["n_refs"],
                        "original_vs_empty_median": gate["original_vs_empty_median"],
                        "original_vs_empty_median_rounded": round(gate["original_vs_empty_median"], 1),
                        "original_vs_empty_n_le_tau": gate["original_vs_empty_n_le_tau"],
                        "gpu_type": gate.get("calibration_gpu_type")},
        "empty_gate_d_max": max(float(r["gate_d"]) for r in empty),
        "template_only_gate_d_median": statistics.median(float(r["gate_d"]) for r in tmpl) if tmpl else None,
        "template_only_gate_d_min": min(float(r["gate_d"]) for r in tmpl) if tmpl else None,
        "runs": runs,
        "comparison": {"per_condition": per, "n_changed_total": n_changed,
                       "all_changed_within_5pct_of_tau": all(x["within_5pct_of_tau"] for p in per.values()
                                                             for x in p["changed"]),
                       "gate_d_abs_diff_max_all": max(p["gate_d_abs_diff_max"] for p in per.values()),
                       "near_tau_fraction": NEAR_TAU},
    }


def rows_from_runs(specs):
    """Worker JSONL -> rows of the two episode tables. spec = "DIR,GPU,GPU_TYPE,RUN"."""
    tab, gate = [], []
    for spec in specs:
        d, gpu, gpu_type, run = spec.split(",")
        for p in sorted(glob.glob(os.path.join(d, "**", "episodes_*.jsonl"), recursive=True)):
            with open(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e["cond"] in ("hold", "calibrate"):
                        continue
                    k, s = e["task"], e["seed"]
                    named = ""
                    if e["cond"] == "none/absent_object":
                        named = G.absent_object_choice(k, s, e["scene_objects"])["named_task"]
                    r = {"run": run, "cond": e["cond"], "mechanism": e["mechanism"], "g": e["g"], "task": k,
                         "task_name": e["task_name"], "seed": s, "scene_seed": e["scene_seed"], "gpu": gpu,
                         "gpu_type": gpu_type, "text": e["text"], "gate_d": e["gate_d"], "gate_commit": e["gate_commit"],
                         "hold": e["hold"], "outcome": e["outcome"],
                         "outcome_step": "" if e["outcome_step"] is None else e["outcome_step"],
                         "designated": e["designated"], "first_contact": e["first_contact"] or "",
                         "first_contact_step": "" if e["first_contact_step"] is None else e["first_contact_step"],
                         "max_hand_disp": f"{max(e['hand_disp_max'].values()):.5f}",
                         "torso_disp": f"{e['torso_disp_max']:.5f}", "category": e["category"], "named_task": named,
                         "n_calls": e["n_calls"], "nan_calls": e["nan_calls"], "steps": e["steps"],
                         "scene_sig": e["scene_sig"]}
                    (gate if e["mechanism"] == "gate" else tab).append(r)
    return tab, gate


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fmt(v):
    return "---" if v is None else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabletop-csv", default=os.path.join(RES, "gr1_tabletop_episodes.csv"))
    ap.add_argument("--gate-csv", default=os.path.join(RES, "gr1_gate_episodes.csv"))
    ap.add_argument("--gate-file", default=os.path.join(RES, "gr1_gate.json"))
    ap.add_argument("--out-dir", default=RES)
    ap.add_argument("--from-runs", action="append", default=[], help="DIR,GPU,GPU_TYPE,RUN (repeatable)")
    a = ap.parse_args()
    if a.from_runs:
        tab, gate_rows = rows_from_runs(a.from_runs)
        if tab:
            write_csv(a.tabletop_csv, TAB_COLS, tab)
        if gate_rows:
            write_csv(a.gate_csv, GATE_COLS, gate_rows)
    tab = read_csv(a.tabletop_csv)
    grows = read_csv(a.gate_csv)
    with open(a.gate_file) as f:
        gate = json.load(f)

    problems, n_checked_all = [], 0
    for name, rows in (("tabletop", tab), ("gate", grows)):
        n_checked, bad = check_rows(rows, name)
        n_checked_all += n_checked
        problems += [f"category differs: {b}" for b in bad]
        if any(r["nan_calls"] not in ("", "0") for r in rows):
            problems.append(f"{name}: NaN calls")
    sigs = {scene(r): r["scene_sig"] for r in tab if r["cond"] == "none/original"}
    mism = sum(1 for r in tab + grows if sigs.get(scene(r)) != r["scene_sig"])
    if mism:
        problems.append(f"scene signature differs from none/original in {mism} rows")
    for c, rr in by_cond(tab).items():
        if len(rr) != N or len({scene(r) for r in rr}) != N:
            problems.append(f"{c}: {len(rr)} rows")

    ts = tabletop(tab)
    gs = gate_summary(grows, gate)
    ts["validity"] = gs["validity"] = {"problems": problems, "category_rows_checked": n_checked_all,
                                       "scene_sig_mismatch": mism}

    # tab:gr1 as printed in the paper
    rows_out, n_match, n_cells = {}, 0, 0
    print("tab:gr1  GR00T N1.6 (GR-1), 120 RoboCasa GR-1 tabletop scenes (counts of 120); computed / paper")
    print(f"{'Mechanism':10s} {'Condition':18s} {'NS':>9s} {'Att.':>9s} {'NC':>9s} {'Compl.':>9s} {'Desig.':>9s} {'Commit':>9s}")
    for c in TABLE_ROWS:
        mech, instr = c.split("/")
        if mech == "gate":
            run = "first" if c in gs["runs"]["first"]["cells"] else "repeat"
            got = table_row(gs["runs"][run]["cells"][c])
        else:
            run = None
            got = table_row(ts["cells"][c])
        want = PAPER[c]
        ok = [g == w for g, w in zip(got, want)]
        n_match += sum(ok)
        n_cells += len(ok)
        rows_out[c] = {"computed": list(got), "paper": list(want), "match": all(ok), "gate_run": run}
        cells = " ".join(f"{fmt(g) + '/' + fmt(w):>9s}" for g, w in zip(got, want))
        print(f"{mech:10s} {instr.upper().replace('_', '-'):18s} {cells}{'' if all(ok) else '  MISMATCH'}"
              f"{'  (repeat run)' if run == 'repeat' else ''}")
    print(f"cells matching the paper: {n_match} of {n_cells}")
    table = {"columns": ["NS", "Att.", "NC", "Compl.", "Desig.", "Commit"], "rows": rows_out,
             "cells_matching": n_match, "cells_total": n_cells}
    ts["paper_table"] = table

    e = ts["cells"]["none/empty"]
    print(f"\nnone/empty: median largest hand displacement of non-contact motion {e['median_max_hand_disp_ncmotion_m']:.2f} m "
          f"({e['counts']['ncmotion']} episodes)")
    print(f"none/original designated-task success {ts['cells']['none/original']['designated_n']}/120 "
          f"({100 * ts['cells']['none/original']['designated_rate']:.1f}%), per task "
          f"{list(ts['cells']['none/original']['designated_by_task'].values())}")
    for k, v in ts["comparisons"].items():
        print(f"McNemar {k}: only first {v['only_first']}, only second {v['only_second']}, p = {v['p']:.3g}")
    print("judgments:", json.dumps(ts["judgments"]))

    print(f"\nGate: tau = {gs['tau']:.3f} (floor median {gs['floor_median']:.3f}); calibration: "
          f"{gs['calibration']['n_floor_scenes']} scenes, {gs['calibration']['n_floor_distances']} distances, "
          f"{gs['calibration']['n_observations']} observations, original vs empty median "
          f"{gs['calibration']['original_vs_empty_median']:.1f}")
    print(f"gate/template_only distance median {gs['template_only_gate_d_median']:.1f}; gate/empty distance max "
          f"{gs['empty_gate_d_max']}")
    print(f"\nTwo gate runs: first {gs['runs']['first']['gpu_types']}, repeat {gs['runs']['repeat']['gpu_types']}")
    print(f"{'condition':22s} {'held first':>10s} {'held repeat':>11s} {'desig. first':>12s} {'desig. repeat':>13s} "
          f"{'changed':>7s} {'max |d1-d2|':>11s}")
    for c, p in gs["comparison"]["per_condition"].items():
        print(f"{c:22s} {p['held_first']:>10d} {p['held_repeat']:>11d} {p['designated_first']:>12d} "
              f"{p['designated_repeat']:>13d} {p['n_changed']:>7d} {p['gate_d_abs_diff_max']:>11.3f}")
        for x in p["changed"]:
            print(f"    task {x['task']} seed {x['seed']}: d {x['gate_d_first']:.4f} (first) {x['gate_d_repeat']:.4f} "
                  f"(repeat), commit {x['commit_first']} -> {x['commit_repeat']}, within 5% of tau: "
                  f"{x['within_5pct_of_tau']}")
    print(f"changed decisions: {gs['comparison']['n_changed_total']}, all within 5% of tau: "
          f"{gs['comparison']['all_changed_within_5pct_of_tau']}")
    for run in ("first", "repeat"):
        print(f"{run}: held episodes all non-start {gs['runs'][run]['held_episodes_all_nonstart']}, committed non-task "
              f"non-starts {gs['runs'][run]['committed_nontask_nonstart_n']}, {json.dumps(gs['runs'][run]['judgments'])}")
    if problems:
        print("\nPROBLEMS:", *problems, sep="\n  ")

    for name, obj in (("gr1_tabletop_summary.json", ts), ("gr1_gate_summary.json", gs)):
        with open(os.path.join(a.out_dir, name), "w") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
    return 1 if problems or n_match != n_cells else 0


if __name__ == "__main__":
    sys.exit(main())
