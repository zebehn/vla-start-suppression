"""Gate threshold for GR00T N1.6 (GR-1) from calibration observations (CPU, numpy). Writes data/gr1/gr1_gate.json.

Collection: gr1/gr1_closed_loop.py --calibrate (gr1/run_gr1.sh calibrate) on the 504 observations of
gr1_common.calibration_list() (6 tasks x seeds 200-283, none of them an evaluation scene). One JSON line per observation:
floor (10, 16, 29) = the original instruction with call seeds call_seed(g, q), q = 0..9; empty (16, 29) = "" with
call_seed(g, 0).
Computation (the same operations as the LIBERO-Goal gate calibration of the paper, with the sample adapted to the humanoid suite):
  - Reference: floor[0] of every observation (original instruction, call_seed(g, 0)). Per-dimension std over the
    504 x 16 reference steps; dimensions with std below 0.001 are left out. A scene of this suite offers one task, so
    the original instruction of the scene is the only reference (the LIBERO calibration used the ten tasks' instructions).
  - Floor: the 54 observations of seeds 200-208 x 45 pairs = 2,430 distances; tau = 95th percentile.
    The floor over all 504 observations (22,680 distances) is reported only.
  - Reported only: distance between the original-instruction chunk and the "" chunk with the same call seed
    (original_vs_empty_*).
Usage: python gr1/gr1_gate_calib.py --in <dir with episodes_*.jsonl> [--in ...] --out data/gr1/gr1_gate.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gr1_common as G  # noqa: E402

METHOD = ("gr1/gr1_gate_calib.py: 504 first observations (6 tasks x seeds 200-283, not evaluation scenes); std over "
          "the 504 original-instruction chunks (16 steps x 29 dimensions); tau = 95th percentile of the 45 pairwise "
          "distances of 10 call seeds on the 54 observations of seeds 200-208 (2,430 distances)")


def load(dirs):
    rows = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "episodes_*.jsonl"))):
            with open(p) as f:
                rows.extend(json.loads(x) for x in f if x.strip())
    rows.sort(key=lambda r: (r["task"], r["seed"]))
    return rows


def check(rows):
    want = set(G.calibration_list())
    got = [(r["task"], r["seed"]) for r in rows]
    return {"n": len(rows), "expected": len(want), "duplicates": len(got) - len(set(got)),
            "missing": sorted(want - set(got))[:20], "unexpected": sorted(set(got) - want)[:20],
            "nan_calls": int(sum(r["nan_calls"] for r in rows)),
            "lang_bad": [[r["task"], r["seed"]] for r in rows if not r["lang_ok"]],
            "initial_contact": int(sum(1 for r in rows if r.get("initial_contact")))}


def compute(rows):
    floor = np.array([r["floor"] for r in rows], dtype=np.float64)          # (n, 10, 16, 29)
    refs = floor[:, 0]
    empty = np.array([r["empty"] for r in rows], dtype=np.float64)
    mask = np.array([r["seed"] in G.CAL_FLOOR_SEEDS for r in rows])
    out = G.gate_compute(refs, floor, mask, empty)
    out["gate_keys"] = [k for k, _ in G.GATE_KEYS]
    out["n_dims_kept"] = int(np.sum(out["dims_kept"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ins", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-type", default="", help="GPU type of the calibration run, written to the json")
    a = ap.parse_args()
    rows = load(a.ins)
    chk = check(rows)
    if chk["n"] != chk["expected"] or chk["duplicates"] or chk["missing"] or chk["unexpected"] or chk["nan_calls"] \
            or chk["lang_bad"]:
        print("calibration check failed:", json.dumps(chk))
        sys.exit(2)
    res = compute(rows)
    res.update({"model": G.HF_REPO, "revision": G.HF_REVISION[:7], "profile": "gr1",
                "calibration_gpu_type": a.gpu_type, "method": METHOD, "check": chk})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("std", "dims_kept")}))


if __name__ == "__main__":
    main()
