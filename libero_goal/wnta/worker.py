"""Episode worker (LIBERO worker environment). One process runs a list of initial states for one condition.

    MUJOCO_EGL_DEVICE_ID=<card> CUDA_VISIBLE_DEVICES=<card> <libero python> -m <package>.worker \
        --cond none/empty --gs 0,2,4 --port 5741 --out <dir> [--gate-profile <gate profile json>] [--seed-offset 0] [--worker w0]

Writes one JSON line per episode to <out>/episodes_<worker>.jsonl with the v0 record fields plus spec,
gate_profile_sha256 and adapter_meta.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from . import protocol as P
from . import spec as S
from ._name import SPEC_ID
from .env_libero import GoalScene, process_obs
from .gate import GateProfile


def infer(client, obs, text, seed):
    return client.infer(process_obs(obs), text, seed)


def run_episode(env, client, cond, k, j, state, gate=None, seed_offset=0, exec_steps=S.EXEC_STEPS, max_steps=S.MAX_POLICY_STEPS):
    g = S.gidx(k, j)
    text, mech, named = S.condition_text(cond, k, j, env.languages)
    if mech == "gate" and gate is None:
        raise ValueError("gate conditions need a gate profile")
    np.random.seed(S.scene_seed(g))
    obs = env.reset_to(state)
    start_true = env.goals_true()
    eef0 = np.asarray(obs["robot0_eef_pos"]).copy()
    rec = {"spec": SPEC_ID, "cond": cond, "mechanism": mech, "g": g, "task": k, "j": j, "text": text, "card": env.card, "outcome": "none",
           "outcome_step": None, "tie_tasks": None, "start_true": start_true, "first_contact": None, "first_contact_step": None,
           "contact_first_step": {}, "eef_path": 0.0, "max_disp": 0.0, "scene_seed": S.scene_seed(g), "named_task": named,
           "seed_offset": seed_offset, "exec_steps": exec_steps, "gripper_binarize": True, "n_calls": 0, "nan_calls": 0,
           "gate_d": None, "gate_commit": None, "hold": False, "first_chunk": None,
           "gate_floor_source": gate.floor_source if gate else None, "gate_profile_sha256": gate.sha256() if gate else None}
    chunk, hold, prev, t = None, False, eef0.copy(), 0
    while t < max_steps:
        if not hold and t % exec_steps == 0:
            seed = S.call_seed(g, t // exec_steps, seed_offset)
            chunk, nan = infer(client, obs, text, seed)
            rec["n_calls"] += 1
            rec["nan_calls"] += int(nan)
            if t == 0:
                rec["first_chunk"] = np.round(chunk, 6).tolist()
                if mech == "gate":
                    ref, nan2 = infer(client, obs, "", seed)
                    rec["n_calls"] += 1
                    rec["nan_calls"] += int(nan2)
                    d, commit = gate.commit(chunk, ref)
                    rec["gate_d"], rec["gate_commit"] = d, commit
                    if not commit:
                        hold = rec["hold"] = True
        obs = env.step_env(S.HOLD_ACTION) if hold else env.step_wire(chunk[t % exec_steps])
        t += 1
        eef = np.asarray(obs["robot0_eef_pos"])
        rec["eef_path"] += float(np.linalg.norm(eef - prev))
        rec["max_disp"] = max(rec["max_disp"], float(np.linalg.norm(eef - eef0)))
        prev = eef.copy()
        for o in sorted(env.finger_contacts()):
            if o not in rec["contact_first_step"]:
                rec["contact_first_step"][o] = t
                if rec["first_contact"] is None:
                    rec["first_contact"], rec["first_contact_step"] = o, t
        newly = [x for x in env.goals_true() if x not in start_true]
        if newly:
            rec["outcome"] = newly[0] if len(newly) == 1 else "tie"
            rec["tie_tasks"] = newly if len(newly) > 1 else None
            rec["outcome_step"] = t
            break
    rec["steps"] = t
    rec["noaction"] = bool(rec["max_disp"] < S.NONSTART_DISP_M and not rec["contact_first_step"])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=S.CORE_CONDITIONS + S.APPENDIX_CONDITIONS)
    ap.add_argument("--gs", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--gate-profile", default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=S.MAX_POLICY_STEPS)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    gate = GateProfile.from_json(a.gate_profile) if a.gate_profile else None
    client = P.PolicyClient(a.port)
    if not client.ping():
        raise SystemExit("server not reachable")
    meta = client.call("meta")
    meta = None if "error" in meta else meta
    env = GoalScene()
    states = [env.init_states(k) for k in range(S.N_TASKS)]
    path = os.path.join(a.out, f"episodes_{a.worker}.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"exists: {path}")
    for g in [int(x) for x in a.gs.split(",") if x]:
        k, j = S.split_g(g)
        t0 = time.time()
        rec = run_episode(env, client, a.cond, k, j, states[k][j], gate, a.seed_offset, max_steps=a.max_steps)
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["max_steps"] = a.max_steps
        rec["adapter_meta_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest() if meta else None
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{a.cond} {a.worker} card{env.card}] g{g} -> {rec['outcome']} disp {rec['max_disp']:.3f} gate {rec['gate_d']} {rec['wall_s']}s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
