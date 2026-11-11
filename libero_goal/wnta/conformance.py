"""Adapter conformance test. Runs against a live adapter server; needs no simulator and no GPU on the client side.

    python -m <package>.conformance --port 5799 [--obs init_000.npz] [--exec-steps 8] [--horizon 16] [--server-log server.log]

Checks (status pass | fail | warn | info):
  K01 ping
  K02 meta endpoint and required fields (servers without it: warn, fields parsed from --server-log)
  K03 infer reply: (T, 7) float32, T >= max(exec_steps, horizon), nan flag consistent, gripper g' in [0, 1]
      (warn for small overshoots, which the worker's binarization absorbs; fail beyond 0.25)
  K04 same seed twice gives a bit-identical chunk (paired design requires it)
  K05 different seeds: stochastic or not, compared with meta.stochastic; deterministic -> gate floor must be settle_obs
  K06 empty instruction: the text the server received is exactly "", empty_handling is true_empty, the prompt for ""
      carries no substituted sentence (compares with the prompt for a non-empty command)
  K07 the instruction changes the first chunk (original command vs "") - info only, a model property
  K08 normalization: the image (VISUAL) normalization in use equals the installed framework default, or the difference is
      a declared override (norm_map_source=override with norm_map_note); state/action differences are listed as a warning
  K09 mask=True is rejected (or declared supported)
  K10 unknown endpoint returns an error reply
Exit code 0 if no check failed, 2 otherwise. The report is printed as JSON (and written with --out).
"""
import argparse
import json
import re
import sys

import numpy as np

from . import protocol as P

COMMAND = "put the bowl on the plate"
SEED = 20261016


def zero_obs():
    o = {}
    for k, (shape, dt) in P.OBS_KEYS.items():
        o[k] = np.zeros(shape, dt)
    return o


def meta_from_log(path):
    """Fields printed on the load line by servers without a meta endpoint (the policy servers of the paper's runs)."""
    t = open(path, errors="replace").read()
    m = {}
    for key, pat in (("empty", r"empty=(\w+)"), ("norm_map_source", r"norm_map_source=(\w+)"),
                     ("image_resolution", r"image_resolution=(\(\d+, \d+\))"), ("chunk_len", r"chunk=(\d+)"),
                     ("chunk_out", r"out=(\d+)"), ("visual_identity", r"visual_identity=(\w+)")):
        v = sorted(set(re.findall(pat, t)))
        if v:
            m[key] = v[0] if len(v) == 1 else v
    if "empty" in m:
        m["empty_handling"] = {"true": "true_empty", "default": "framework_default"}.get(m["empty"], m["empty"])
    return m


def run(port, obs=None, exec_steps=8, horizon=16, server_log=None, timeout_ms=600000):
    cl = P.PolicyClient(port, timeout_ms=timeout_ms)
    rep = []

    def add(cid, status, detail):
        rep.append({"id": cid, "status": status, "detail": detail})
    obs = zero_obs() if obs is None else obs
    op = P.validate_obs(obs)
    if op:
        add("K00", "fail", f"test observation invalid: {op}")
        return rep
    if not cl.ping():
        add("K01", "fail", "no reply to ping")
        return rep
    add("K01", "pass", "ping ok")
    meta = cl.call("meta")
    if "error" in meta:
        meta = meta_from_log(server_log) if server_log else {}
        add("K02", "warn", f"no meta endpoint (server without meta); fields from log: {meta}")
    else:
        miss = [k for k in P.META_REQUIRED if k not in meta]
        add("K02", "fail" if miss else "pass", f"missing meta fields {miss}" if miss else "meta complete")
    need = max(exec_steps, horizon or 0)
    r1 = cl.call("infer", {"obs": obs, "text": COMMAND, "seed": SEED, "mask": False})
    p = P.validate_chunk(r1, need)
    ov = None if p else P.gripper_overshoot(r1)
    add("K03", "fail" if p else ("warn" if ov else "pass"),
        p or (f"chunk {np.asarray(r1['chunk']).shape}; gripper g' slightly outside [0, 1] {ov} (harmless: the worker binarizes)"
              if ov else f"chunk {np.asarray(r1['chunk']).shape}, needs >= {need} steps"))
    if p and "error" in r1:
        return rep
    a = np.asarray(r1["chunk"])
    b = np.asarray(cl.call("infer", {"obs": obs, "text": COMMAND, "seed": SEED, "mask": False})["chunk"])
    same = a.shape == b.shape and np.array_equal(a, b)
    add("K04", "pass" if same else "fail",
        "same seed bit-identical" if same else f"same seed differs, max|diff| {float(np.abs(a - b).max()):.3g}")
    c = np.asarray(cl.call("infer", {"obs": obs, "text": COMMAND, "seed": SEED + 1, "mask": False})["chunk"])
    stochastic = not (a.shape == c.shape and np.array_equal(a, c))
    declared = meta.get("stochastic")
    if declared is None:
        add("K05", "warn", f"stochastic={stochastic} (not declared); gate floor source must be "
                           f"{'seed' if stochastic else 'settle_obs'}")
    else:
        add("K05", "pass" if bool(declared) == stochastic else "fail",
            f"measured stochastic={stochastic}, declared {declared}; gate floor source {'seed' if stochastic else 'settle_obs'}")
    e = np.asarray(cl.call("infer", {"obs": obs, "text": "", "seed": SEED, "mask": False})["chunk"])
    tok_e = cl.call("tokens")
    cl.call("infer", {"obs": obs, "text": COMMAND, "seed": SEED, "mask": False})
    tok_c = cl.call("tokens")
    issues = []
    if tok_e.get("text") != "":
        issues.append(f"server recorded text {tok_e.get('text')!r} for the empty instruction")
    eh = meta.get("empty_handling")
    if eh not in P.EMPTY_HANDLING:
        issues.append(f"empty_handling {eh!r} not declared as one of {P.EMPTY_HANDLING}")
    elif eh != "true_empty":
        issues.append("empty_handling is framework_default: core condition none/empty requires the empty string to reach the model")
    pe, pc = tok_e.get("prompt"), tok_c.get("prompt")
    if (pe is not None and not isinstance(pe, str)) or (pc is not None and not isinstance(pc, str)):
        issues.append(f"tokens prompt must be a string, got {type(pe).__name__}")
        pe = pc = None
    if pe is not None and pc is not None and COMMAND in pc:
        residue = pc.replace(COMMAND, "")
        if pe != residue:
            issues.append(f"prompt for '' is {pe!r}, expected the command-free template {residue!r} (substitution?)")
    add("K06", "fail" if issues else ("pass" if pe is not None else "warn"),
        issues or ("empty prompt matches the command-free template" if pe is not None else "prompt not exposed by tokens"))
    dd = float(np.abs(a - e).max()) if a.shape == e.shape else None
    add("K07", "info", f"max|chunk(command) - chunk('')| at the same seed = {dd}")
    nm, nd, src = meta.get("norm_map"), meta.get("norm_map_installed_default"), meta.get("norm_map_source")
    if nm is None or nd is None:
        add("K08", "warn", f"normalization not declared (norm_map={nm}, installed default={nd}, source={src})")
    else:
        diff = {k: (nm.get(k), nd.get(k)) for k in sorted(set(nm) | set(nd)) if nm.get(k) != nd.get(k)}
        vis = {k: v for k, v in diff.items() if "VISUAL" in k.upper()}
        other = {k: v for k, v in diff.items() if k not in vis}
        declared = src == "override" and bool(meta.get("norm_map_note"))
        if vis and not declared:
            add("K08", "fail", f"image normalization differs from the installed default {vis} without a declared override "
                               "(risk: the installed model applies its own image transform a second time)")
        elif other and not declared:
            add("K08", "warn", f"state/action normalization differs from the config-class default {other}; these are applied and "
                               "inverted by the stored statistics, so the confirmation run decides")
        elif diff:
            add("K08", "pass", f"differences {diff} declared: {meta.get('norm_map_note')}")
        else:
            add("K08", "pass", f"normalization {nm} equals the installed default (source {src})")
    rm = cl.call("infer", {"obs": obs, "text": COMMAND, "seed": SEED, "mask": True})
    add("K09", "pass" if "error" in rm else "info", "mask rejected" if "error" in rm else "mask accepted (adapter declares support)")
    ru = cl.call("no_such_endpoint")
    add("K10", "pass" if "error" in ru else "fail", "unknown endpoint -> error" if "error" in ru else f"reply {ru}")
    cl.close()
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--obs", default=None, help="observation npz with the wire keys (e.g. a calibration observation)")
    ap.add_argument("--exec-steps", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    obs = {k: v for k, v in np.load(a.obs).items()} if a.obs else None
    rep = run(a.port, obs, a.exec_steps, a.horizon, a.server_log)
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    return 2 if any(r["status"] == "fail" for r in rep) else 0


if __name__ == "__main__":
    sys.exit(main())
