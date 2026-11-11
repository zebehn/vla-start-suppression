"""Closed-loop worker for the GR-1 tabletop runs (GR00T N1.6 (GR-1)); run with the python of the robocasa_uv environment
that Isaac-GR00T n1.6.1 sets up (gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv). The policy runs in
gr1/gr1_server.py.

Episode protocol (one episode per (task, seed); the same (task, seed) gives the same scene and call seeds in every condition):
  - The environment is created for each episode with seed = scene_seed(task, seed) and reset with the same seed.
    The scene layout (placements, object types, textures, joint randomization of +-0.2 rad) uses robosuite's
    np.random.default_rng(seed) (robosuite v1.5.1 environments/base.py:138-139); the public evaluation creates the
    environment without a seed, so its scenes are random. PYTHONHASHSEED must be fixed (gr1/run_gr1.sh sets 0),
    because the scene generation depends on Python's hash order.
  - Ten settling steps with the hold action, then up to 720 policy steps. The policy is called every 8 steps and the first
    8 steps of each 16-step chunk are executed (rollout_policy.py:225-232, multistep_wrapper.py:249-310).
  - The language field "annotation.human.coarse_action" of the observation is replaced by the condition string
    (gr1_common.instruction_string) right before the observation is sent to the server. The environment writes
    "unlocked_waist: " + instruction into this field (gymnasium_groot.py:112-115), so none/empty is the empty string and
    none/template_only keeps the template alone. The server returns the string the model actually received, and the
    record's lang_ok checks it.
  - Gate (gate/*): on the first call, the policy is called with the condition string and with "" on the same observation
    and the same call seed; d = gr1_common.chunk_dist of the two 16 x 29 chunks with the std and kept dimensions of the
    gate file. If d > tau, the episode runs as without the gate. If d <= tau, the hold action is sent for the whole
    episode and the policy is not called again.
  - Hold action (settling, the hold baseline, and a held gate episode): hand targets = the controller's current targets
    (robocasa/models/robots/__init__.py:74-97 reconstruct_latest_actions, mapped by the key converter), arm and waist
    targets = the current joint positions (state.*).
Measurements after every step:
  - Placement: object o is in container r (check_obj_in_receptacle with the container's highest spawn region, as in
    tabletop_24dc.py:626-665), both hands are at least 0.25 m from the object (any_gripper_obj_far), and the object does
    not touch the table. The first placement pair that becomes true sets `outcome` (gr1_common.FirstTrue); after it the
    episode waits up to 60 steps for the environment's own success check (`env_success_step`, designated-task success).
  - Contact: the 12 collision geometries of each Fourier hand (palm and fingers) against scene objects, containers, and
    fixtures; contacts with the table (the Counter fixture) are recorded separately and are not contacts. A hand contact
    present right after settling is not counted until it has been released once (none occurred in the gate runs and
    the hold baseline on the 120 evaluation scenes).
  - Displacement: the largest distance of each hand end (robot.eef_site_id) and of the head link robot0_head_pitch
    (torso reference, moved by the waist) from their positions after settling.
The episode category follows gr1_common.category.

Usage:
  python gr1/gr1_closed_loop.py --cond none/empty --scenes gr1/configs/gr1_scenes.json --port 5761 --out <dir> --worker w0
  python gr1/gr1_closed_loop.py --cond gate/neutral --gate-file data/gr1/gr1_gate.json --eps 0:0,0:1 --port 5761 --out <dir>
  python gr1/gr1_closed_loop.py --cond none/absent_object --strings gr1/configs/gr1_instruction_strings.json ...
  python gr1/gr1_closed_loop.py --cond hold --scenes gr1/configs/gr1_scenes.json --out <dir>      (no policy)
  python gr1/gr1_closed_loop.py --calibrate --eps <k:s list> --port 5761 --out <dir>          (gate calibration)
  python gr1/gr1_closed_loop.py --check-scene --eps 0:100,1:100 --out <dir>                    (scene reproducibility)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gr1_common as G  # noqa: E402

VIDEO_KEY = "video.ego_view_pad_res256_freq20"                # saved video (gymnasium_groot.py:99-101)
MODEL_VIDEO_KEY = "video.ego_view_bg_crop_pad_res256_freq20"  # model input (processor_config.json gr1.video)
STATE_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
SIDES = ("left", "right")
SETTLE_STEPS = 10
EXEC_STEPS = 8
MAX_STEPS = 720
POST_PLACE_WAIT = 60       # steps to wait for the environment's success check after the first placement
STRING_TABLE_CONDS = ("none/impossible_action", "none/absent_object", "prefix/original")


def load_gate(path):
    """Gate file (data/gr1/gr1_gate.json or the output of gr1/gr1_gate_calib.py)."""
    with open(path) as f:
        j = json.load(f)
    return {"tau": float(j["tau_p95"]), "std": np.asarray(j["std"], dtype=np.float64),
            "keep": np.asarray(j["dims_kept"], dtype=bool), "horizon": int(j.get("horizon", G.GATE_H))}


def build_policy_obs(obs, text):
    """The flat observation that the public rollout sends to the server (B=1, T=1); the language field is the string."""
    out = {MODEL_VIDEO_KEY: np.ascontiguousarray(np.asarray(obs[MODEL_VIDEO_KEY], dtype=np.uint8)[None, None])}
    for k in STATE_KEYS:
        out[f"state.{k}"] = np.asarray(obs[f"state.{k}"], dtype=np.float32)[None, None]
    out[G.LANG_KEY] = [text]
    return out


def save_video(path, frames):
    try:
        import av
        cont = av.open(path, mode="w")
        st = cont.add_stream("h264", rate=20)
        st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                cont.mux(pkt)
        for pkt in st.encode():
            cont.mux(pkt)
        cont.close()
    except Exception as e:  # the video is optional; keep the frames as npz
        np.savez_compressed(path.replace(".mp4", ".npz"), frames=np.stack(frames))
        print("video fallback npz:", e, flush=True)


class GR1Env:
    """One RoboCasa GR-1 environment (created for each episode)."""

    def __init__(self, k, seed):
        import gymnasium as gym
        import robocasa  # noqa: F401  (makes the gym ids available, as in rollout_policy.py:85-87)
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401
        import robocasa.utils.object_utils as OU
        from robocasa.models.robots import reconstruct_latest_actions
        self.OU, self._reconstruct = OU, reconstruct_latest_actions
        self.k, self.seed = k, seed
        self.genv = gym.make(G.env_name(k), enable_render=True, seed=seed)
        self.ge = self.genv.unwrapped
        self.raw = self.ge.env

    def _index(self):
        """Lookup tables, rebuilt after every reset (the model is reloaded on reset, tabletop.py:162)."""
        raw, sim = self.raw, self.raw.sim
        robot = raw.robots[0]
        self.robot = robot
        self.names = sorted(raw.objects.keys())
        self.pairs, self.roles = G.candidate_pairs(self.names)
        self.hand_geoms = {}
        for side in SIDES:
            ids = set()
            for n in robot.gripper[side].contact_geoms:
                try:
                    ids.add(int(sim.model.geom_name2id(n)))
                except Exception:
                    pass
            self.hand_geoms[side] = ids
        self.hand_all = set().union(*self.hand_geoms.values())
        roots = {}
        for name, bid in raw.obj_body_id.items():
            roots[int(bid)] = "obj:" + name
        self.unmapped_fixtures = []
        for fname, fx in raw.fixtures.items():
            try:
                bid = int(sim.model.body_name2id(fx.root_body))
            except Exception:
                self.unmapped_fixtures.append(fname)
                continue
            is_counter = any(c.__name__ == "Counter" for c in type(fx).__mro__)
            roots[bid] = ("counter:" if is_counter else "fx:") + fname
        try:
            roots[int(sim.model.body_name2id(robot.robot_model.root_body))] = "robot"
        except Exception:
            pass
        self.roots = roots
        self.parent = np.asarray(sim.model.body_parentid)
        self.geom_body = np.asarray(sim.model.geom_bodyid)
        pf = robot.robot_model.naming_prefix
        self.torso_body = None
        for cand in ("head_pitch", "head_yaw", "torso_waist_roll"):
            try:
                self.torso_bid = int(sim.model.body_name2id(pf + cand))
                self.torso_body = pf + cand
                break
            except Exception:
                continue
        if self.torso_body is None:
            raise RuntimeError("torso reference body not found (expected robot0_head_pitch)")
        cats = {}
        for cfg in getattr(raw, "object_cfgs", []):
            if "info" in cfg and cfg.get("name") in raw.objects:
                cats[cfg["name"]] = cfg["info"].get("cat")
        self.cats = cats

    def reset(self):
        obs, _ = self.genv.reset(seed=self.seed)
        self._index()
        return obs

    def step(self, action):
        obs, _r, _term, _trunc, _info = self.genv.step({k: np.asarray(v, dtype=np.float32) for k, v in action.items()})
        return obs

    def hold_action(self, obs):
        """Hands: the controller's current targets; arms and waist: the current joint positions."""
        act = self.ge.key_converter.map_action(self._reconstruct(self.raw))
        hold = {"action." + k[5:]: np.asarray(v, dtype=np.float32).copy() for k, v in act.items()}
        for k in ("left_arm", "right_arm", "waist"):
            hold[f"action.{k}"] = np.asarray(obs[f"state.{k}"], dtype=np.float32).copy()
        return hold

    def _spawns(self, r):
        recep = self.raw.objects[r]
        return [self.OU.get_highest_spawn_region(self.raw, recep)] if len(recep.spawns) > 0 else [None]

    def placements(self):
        raw, OU = self.raw, self.OU
        out = set()
        for o, r in self.pairs:
            if not OU.check_obj_in_receptacle(raw, o, r, spawn_regions=self._spawns(r)):
                continue
            if not OU.any_gripper_obj_far(raw, obj_name=o):
                continue
            if OU.check_obj_fixture_contact(raw, o, raw.counter):
                continue
            out.add((o, r))
        return out

    def env_success(self):
        return bool(self.raw._check_success())

    def hand_contacts(self):
        """{side: set(owner)}; owner is "obj:<name>", "fx:<name>", "counter:<name>", "robot", or "other:<body name>"."""
        sim = self.raw.sim
        out = {s: set() for s in SIDES}
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for side in SIDES:
                hg = self.hand_geoms[side]
                if g1 in hg and g2 not in self.hand_all:
                    other = g2
                elif g2 in hg and g1 not in self.hand_all:
                    other = g1
                else:
                    continue
                b = int(self.geom_body[other])
                own = G.owner_of_body(b, self.parent, self.roots)
                if own is None:
                    own = "other:" + str(sim.model.body_id2name(b))
                out[side].add(own)
        return out

    def hand_pos(self):
        sim = self.raw.sim
        return {s: np.array(sim.data.site_xpos[self.robot.eef_site_id[s]]) for s in SIDES}

    def torso_pos(self):
        return np.array(self.raw.sim.data.body_xpos[self.torso_bid])

    def scene_items(self):
        sim = self.raw.sim
        items = [(n, self.cats.get(n), *sim.data.body_xpos[self.raw.obj_body_id[n]]) for n in self.names]
        for fname, fx in self.raw.fixtures.items():
            try:
                bid = int(sim.model.body_name2id(fx.root_body))
                items.append((fname, type(fx).__name__, *sim.data.body_xpos[bid]))
            except Exception:
                pass
        return items

    def close(self):
        try:
            self.genv.close()
        except Exception:
            pass


def settle(env):
    obs = env.reset()
    hold = env.hold_action(obs)
    for _ in range(SETTLE_STEPS):
        obs = env.step(hold)
    return obs, hold


def condition_text(cond, env_text, k, s, env, strings):
    """Condition string; for the conditions in STRING_TABLE_CONDS it must equal the row of the strings table."""
    text = G.instruction_string(cond, env_text, k, s, env.cats)
    if cond in STRING_TABLE_CONDS:
        if strings is None:
            raise SystemExit(f"{cond} needs --strings gr1/configs/gr1_instruction_strings.json")
        row = strings["scenes"].get(f"{k}:{s}")
        if row is None:
            raise SystemExit(f"strings table has no scene {k}:{s}")
        want = row[cond.split("/")[1] if cond.startswith("none/") else cond]["text"]
        if want != text:
            raise SystemExit(f"{cond} {k}:{s}: computed string {text!r} differs from the table {want!r}")
    return text


def run_episode(env, client, cond, k, s, gate=None, strings=None, out_dir=None, want_video=False, max_steps=MAX_STEPS):
    """One episode of condition cond on (task k, seed s); returns the record (one JSON line)."""
    g = G.gidx(k, s)
    mech, _instr = G.split_cond(cond)
    obs, hold = settle(env)
    init_active = {(sd, o) for sd, owners in env.hand_contacts().items() for o in owners
                   if o.startswith("obj:") or o.startswith("fx:") or o.startswith("counter:")}
    start_true = sorted(G.pair_key(o, r) for o, r in env.placements())
    env_text = obs[G.LANG_KEY]
    is_hold, is_gate = mech == "hold", mech == "gate"
    text = None if is_hold else condition_text(cond, env_text, k, s, env, strings)
    if is_gate and gate is None:
        raise SystemExit("gate conditions need --gate-file")
    held = False
    hp0, tp0 = env.hand_pos(), env.torso_pos()
    st0 = {kk: np.asarray(obs[f"state.{kk}"], dtype=np.float64).copy() for kk in STATE_KEYS}
    rec = {"cond": cond, "mechanism": mech, "g": g, "task": k, "task_name": G.TASKS[k]["name"], "seed": s,
           "scene_seed": G.scene_seed(k, s), "text": text, "env_text": env_text,
           "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "outcome": G.OUT_NONE, "outcome_step": None,
           "outcome_pairs": None, "tie_pairs": None, "start_true": start_true, "env_success_step": None,
           "other_first_true": {}, "first_contact": None, "first_contact_step": None, "contact_first_step": {},
           "hand_first_contact": {sd: None for sd in SIDES}, "counter_contact_first_step": None, "contact_other": {},
           "hand_disp_max": {sd: 0.0 for sd in SIDES}, "torso_disp_max": 0.0, "torso_body": env.torso_body,
           "waist_delta_max": 0.0, "finger_delta_max": {"left_hand": 0.0, "right_hand": 0.0},
           "initial_contact": sorted(f"{sd}:{o}" for sd, o in init_active), "post_place_steps": None,
           "gate_d": None, "gate_commit": None, "gate_tau": None, "n_calls": 0, "nan_calls": 0, "hold": is_hold,
           "first_chunk": None, "lang_in_set": [], "lang_model_set": [], "prompt0": None, "lang_ok": None,
           "expected_model_text": None if text is None else G.formalize(text),
           "roles": {n: env.roles[n] for n in env.names}, "scene_objects": env.cats,
           "scene_sig": G.scene_signature(env.scene_items()), "unmapped_fixtures": env.unmapped_fixtures,
           "upstream": {"isaac_gr00t": G.GR00T_COMMIT, "robocasa_gr1_tabletop_tasks": G.RGT_COMMIT}}
    ft = G.FirstTrue(env.roles, start_true)
    frames, chunk = [], None
    lang_in, lang_model = set(), set()
    t, place_t = 0, None
    while t < max_steps:
        if not is_hold and not held and t % EXEC_STEPS == 0:
            seed = G.call_seed(g, t // EXEC_STEPS)
            resp = client.call_endpoint("infer", {"observation": build_policy_obs(obs, text), "seed": seed,
                                                  "want_prompt": t == 0})
            chunk = {kk: np.asarray(v) for kk, v in resp["action"].items()}
            rec["n_calls"] += 1
            rec["nan_calls"] += int(bool(resp["nan"]))
            lang_in.update(resp.get("text_in") or [])
            lang_model.update(resp.get("lang_model") or [])
            if t == 0:
                rec["first_chunk"] = {kk: np.round(v[0], 5).tolist() for kk, v in chunk.items()}
                rec["prompt0"] = (resp.get("prompt") or [None])[0]
            if is_gate and t == 0:
                ref = client.call_endpoint("infer", {"observation": build_policy_obs(obs, ""), "seed": seed,
                                                     "want_prompt": False})
                rec["n_calls"] += 1
                rec["nan_calls"] += int(bool(ref["nan"]))
                d = G.chunk_dist(G.chunk_vec(resp["action"]), G.chunk_vec(ref["action"]), gate["std"], gate["keep"],
                                 gate["horizon"])
                rec["gate_d"], rec["gate_tau"], rec["gate_commit"] = d, gate["tau"], bool(d > gate["tau"])
                if not rec["gate_commit"]:
                    held = rec["hold"] = True
        if want_video:
            frames.append(np.ascontiguousarray(obs[VIDEO_KEY]))
        act = hold if (is_hold or held) else {kk: v[0, t % EXEC_STEPS] for kk, v in chunk.items()}
        obs = env.step(act)
        t += 1
        hp, tp = env.hand_pos(), env.torso_pos()
        for sd in SIDES:
            rec["hand_disp_max"][sd] = max(rec["hand_disp_max"][sd], float(np.linalg.norm(hp[sd] - hp0[sd])))
        rec["torso_disp_max"] = max(rec["torso_disp_max"], float(np.linalg.norm(tp - tp0)))
        rec["waist_delta_max"] = max(rec["waist_delta_max"],
                                     float(np.max(np.abs(np.asarray(obs["state.waist"], np.float64) - st0["waist"]))))
        for hk in ("left_hand", "right_hand"):
            rec["finger_delta_max"][hk] = max(rec["finger_delta_max"][hk],
                                              float(np.max(np.abs(np.asarray(obs[f"state.{hk}"], np.float64) - st0[hk]))))
        cur = env.hand_contacts()
        init_active &= {(sd, o) for sd, owners in cur.items() for o in owners}   # released initial contacts count again
        for sd, owners in cur.items():
            for o in sorted(owners):
                if (sd, o) in init_active:
                    continue
                if o.startswith("counter:"):
                    if rec["counter_contact_first_step"] is None:
                        rec["counter_contact_first_step"] = t
                elif o.startswith("obj:") or o.startswith("fx:"):
                    name = o.split(":", 1)[1]
                    if name not in rec["contact_first_step"]:
                        rec["contact_first_step"][name] = t
                        if rec["first_contact"] is None:
                            rec["first_contact"], rec["first_contact_step"] = name, t
                    if rec["hand_first_contact"][sd] is None:
                        rec["hand_first_contact"][sd] = [name, t]
                elif o != "robot":
                    rec["contact_other"].setdefault(o, t)
        if rec["env_success_step"] is None and env.env_success():
            rec["env_success_step"] = t
        if place_t is None and ft.update(env.placements(), t):
            place_t = t
        if place_t is not None and (rec["env_success_step"] is not None or t - place_t >= POST_PLACE_WAIT):
            rec["post_place_steps"] = t - place_t
            break
    if place_t is not None and rec["post_place_steps"] is None:
        rec["post_place_steps"] = t - place_t
    rec["outcome"], rec["outcome_step"] = ft.outcome, ft.step
    rec["outcome_pairs"], rec["tie_pairs"], rec["other_first_true"] = ft.pairs, ft.tie_pairs, ft.other_first_true
    rec["designated"] = int(rec["env_success_step"] is not None)
    rec["steps"] = t
    rec["lang_in_set"], rec["lang_model_set"] = sorted(lang_in), sorted(lang_model)
    if not is_hold:
        rec["lang_ok"] = rec["lang_model_set"] == [rec["expected_model_text"]] and rec["lang_in_set"] == [text]
    rec["category"] = G.categorize(rec)
    rec["hand_actuation"] = G.hand_actuation(rec)
    if want_video and frames and out_dir:
        vd = os.path.join(out_dir, "videos")
        os.makedirs(vd, exist_ok=True)
        save_video(os.path.join(vd, f"{cond.replace('/', '_')}_k{k}_s{s:03d}.mp4"), frames[::2])
    return rec


def calib_episode(env, client, k, s):
    """Gate calibration observation (no closed loop): after the same reset and settling as an episode, the original
    instruction with call_seed(g, q), q = 0..9, and "" with call_seed(g, 0)."""
    g = G.gidx(k, s)
    obs, _hold = settle(env)
    init = sorted(f"{sd}:{o}" for sd, owners in env.hand_contacts().items() for o in owners
                  if o.startswith("obj:") or o.startswith("fx:") or o.startswith("counter:"))
    text = obs[G.LANG_KEY]
    floor, nan, lang_in, lang_model = [], 0, set(), set()
    for q in range(G.N_FLOOR_CALLS):
        resp = client.call_endpoint("infer", {"observation": build_policy_obs(obs, text), "seed": G.call_seed(g, q),
                                              "want_prompt": False})
        floor.append(np.round(G.chunk_vec(resp["action"]), 6).tolist())
        nan += int(bool(resp["nan"]))
        lang_in.update(resp.get("text_in") or [])
        lang_model.update(resp.get("lang_model") or [])
    ref = client.call_endpoint("infer", {"observation": build_policy_obs(obs, ""), "seed": G.call_seed(g, 0),
                                         "want_prompt": False})
    nan += int(bool(ref["nan"]))
    rec = {"cond": "calibrate", "task": k, "task_name": G.TASKS[k]["name"], "seed": s, "g": g,
           "scene_seed": G.scene_seed(k, s), "env_text": text, "initial_contact": init, "floor": floor,
           "empty": np.round(G.chunk_vec(ref["action"]), 6).tolist(), "empty_lang_model": sorted(ref.get("lang_model") or []),
           "n_calls": G.N_FLOOR_CALLS + 1, "nan_calls": nan, "lang_in_set": sorted(lang_in),
           "lang_model_set": sorted(lang_model), "scene_sig": G.scene_signature(env.scene_items()),
           "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID")}
    rec["lang_ok"] = (rec["lang_in_set"] == [text] and rec["lang_model_set"] == [G.formalize(text)]
                      and rec["empty_lang_model"] == [""])
    return rec


def check_scene(eps, out):
    """Scene reproducibility (no policy): two environments with the same seed give the same scene signature."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "scene_check.jsonl")
    for k, s in eps:
        sigs, langs = [], []
        for _ in range(2):
            env = GR1Env(k, G.scene_seed(k, s))
            try:
                obs = env.reset()
                sigs.append(G.scene_signature(env.scene_items()))
                langs.append(obs[G.LANG_KEY])
                roles = {n: env.roles[n] for n in env.names}
                cats, torso = env.cats, env.torso_body
                nhand = {sd: len(env.hand_geoms[sd]) for sd in SIDES}
            finally:
                env.close()
        r = {"task": k, "seed": s, "sig": sigs, "same": sigs[0] == sigs[1] and langs[0] == langs[1], "env_text": langs[0],
             "roles": roles, "scene_objects": cats, "torso_body": torso, "hand_geoms": nhand}
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[scene] k{k} s{s} same={r['same']} {langs[0]!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=G.CONDITIONS)
    ap.add_argument("--calibrate", action="store_true", help="collect gate calibration observations instead of episodes")
    ap.add_argument("--check-scene", action="store_true")
    ap.add_argument("--eps", default="", help='"task:seed" comma list')
    ap.add_argument("--scenes", default="", help="scene list file (gr1/configs/gr1_scenes.json); used when --eps is empty")
    ap.add_argument("--port", type=int, default=0, help="policy server port (not used by hold and --check-scene)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--gate-file", default="", help="data/gr1/gr1_gate.json; required for gate/*")
    ap.add_argument("--strings", default="", help="gr1/configs/gr1_instruction_strings.json; required for " +
                    ", ".join(STRING_TABLE_CONDS))
    ap.add_argument("--video-eps", default="")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    a = ap.parse_args()
    if a.eps:
        eps = G.parse_eps(a.eps)
    elif a.scenes:
        eps = G.scene_list(a.scenes)
    else:
        raise SystemExit("give --eps or --scenes")
    if a.check_scene:
        check_scene(eps, a.out)
        return
    if not a.calibrate and not a.cond:
        raise SystemExit("--cond, --calibrate, or --check-scene required")
    gate = load_gate(a.gate_file) if a.gate_file else None
    strings = None
    if a.strings:
        with open(a.strings) as f:
            strings = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    vids = set(G.parse_eps(a.video_eps))
    client = None
    if a.calibrate or a.cond != "hold":
        from gr00t.policy.server_client import PolicyClient  # the client of rollout_policy.py:424-426
        client = PolicyClient(host="127.0.0.1", port=a.port, timeout_ms=300000)
        assert client.ping(), f"policy server on port {a.port} not reachable"
    path = os.path.join(a.out, f"episodes_{a.worker}.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"output exists: {path}")
    for k, s in eps:
        t0 = time.time()
        env = GR1Env(k, G.scene_seed(k, s))
        try:
            if a.calibrate:
                rec = calib_episode(env, client, k, s)
            else:
                rec = run_episode(env, client, a.cond, k, s, gate, strings, a.out, (k, s) in vids, a.max_steps)
        finally:
            env.close()
        rec["wall_s"] = round(time.time() - t0, 1)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if a.calibrate:
            print(f"[calibrate {a.worker}] k{k} s{s} nan {rec['nan_calls']} lang_ok {rec['lang_ok']} {rec['wall_s']}s",
                  flush=True)
            continue
        print(f"[{a.cond} {a.worker}] k{k} s{s} -> {rec['outcome']} {rec['category']} "
              f"hand {max(rec['hand_disp_max'].values()):.3f} torso {rec['torso_disp_max']:.3f} lang_ok {rec['lang_ok']} "
              f"gate_d {rec.get('gate_d')} steps {rec['steps']} {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
