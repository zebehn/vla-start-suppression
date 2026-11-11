"""Command line: python -m <package> <command> ...

  score <episodes dir> [--gate-profile f] [--out f]   metrics per condition from episode records (any layout, episodes_*.jsonl)
  conformance --port N [...]                           adapter conformance test (needs pyzmq, msgpack_numpy)
  stub-server --port N [--deterministic ...]           stub adapter for trying the worker and the conformance test
"""
import argparse
import glob
import json
import os
import sys

from . import score as SC
from . import spec as S
from ._name import NAME, SPEC_ID


def cmd_score(argv):
    from .gate import GateProfile
    ap = argparse.ArgumentParser(prog=f"{NAME} score")
    ap.add_argument("path")
    ap.add_argument("--gate-profile", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args(argv)
    rows = {}
    files = sorted(glob.glob(os.path.join(a.path, "**", "episodes_*.jsonl"), recursive=True))
    if not os.path.isdir(a.path):
        print(f"score: {a.path} is not a directory", file=sys.stderr)
        return 2
    if not files:
        print(f"score: no episode records under {a.path}. The worker writes one episodes_<worker>.jsonl per process "
              f"(any directory layout); a sample is in libero_goal/examples/sample_episodes.", file=sys.stderr)
        return 2
    for f in files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows.setdefault(r["cond"], []).append(r)
    if not rows:
        print(f"score: the {len(files)} episode files under {a.path} are empty", file=sys.stderr)
        return 2
    tau = GateProfile.from_json(a.gate_profile).tau if a.gate_profile else None
    out = {"spec": SPEC_ID, "path": a.path, "gate_profile": a.gate_profile, "conditions": {}, "problems": []}
    for c, rs in sorted(rows.items()):
        expect = S.core_inits() if c in S.CORE_CONDITIONS or c in S.APPENDIX_CONDITIONS else sorted({r["g"] for r in rs})
        probs = SC.check_integrity(rs, c, expect, max_steps=S.MAX_POLICY_STEPS)
        out["problems"] += probs
        out["conditions"][c] = SC.condition_metrics(rs, tau=tau if c.startswith("gate/") else None)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps(out, indent=1))
    for c, m in out["conditions"].items():
        print(f"{c:22s} n={m['n']:3d} completion={m['completion_n']:3d} success={m['success_n']:3d} attempt={m['attempt_n']:3d} "
              f"non_contact_motion={m['non_contact_motion_n']:3d} non_start={m['non_start_n']:3d} no_contact={m['no_contact_n']:3d}"
              + (f" commit={m['commit_n']:3d}" if "commit_n" in m else ""))
    if out["problems"]:
        print("PROBLEMS", *out["problems"][:20], sep="\n  ")
        return 0 if a.allow_incomplete else 2
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.replace("<package>", NAME))
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "score":
        return cmd_score(rest)
    if cmd == "conformance":
        from . import conformance
        return conformance.main(rest)
    if cmd == "stub-server":
        from .adapters import base
        return base.main(rest)
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
