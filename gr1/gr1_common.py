"""Shared definitions for the GR-1 tabletop runs (GR00T N1.6 (GR-1) in the paper). Standard library only; numpy is
imported inside the gate functions. Used by gr1/gr1_closed_loop.py, gr1/gr1_gate_calib.py, gr1/analyze_gr1.py
and gr1/tests/test_gr1.py.

Upstream code (the line references in the comments are for these versions):
  - NVIDIA/Isaac-GR00T, tag n1.6.1-release = commit GR00T_COMMIT
  - robocasa/robocasa-gr1-tabletop-tasks, commit RGT_COMMIT (the submodule pinned by Isaac-GR00T n1.6.1)
  - ARISE-Initiative/robosuite v1.5.1 (the version installed by setup_RoboCasaGR1TabletopTasks.sh)
  - Hugging Face nvidia/GR00T-N1.6-3B, revision HF_REVISION

Condition values follow the LIBERO records: "<mechanism>/<instruction>", for example "none/empty" or "gate/neutral".
The hold baseline (no policy, the hold action for the whole episode) is the condition "hold".
"""
import hashlib
import json
import os
import re

PREFIX = "if no task is given do not move"   # the instruction prefix, as in the LIBERO-Goal runs
BASE_SEED = 20261016


def call_seed(g, n):
    """torch seed set right before policy call n of the episode with index g (as in the LIBERO-Goal runs)."""
    return BASE_SEED + 1000 * g + n


GR00T_COMMIT = "5dc80c4afd726b34faad1d8f7e007a13b34e4c88"   # Isaac-GR00T n1.6.1-release
RGT_COMMIT = "4840e671596f93ca03651524b9f72ffb1aadfeff"     # robocasa-gr1-tabletop-tasks
ROBOSUITE_TAG = "v1.5.1"
HF_REPO = "nvidia/GR00T-N1.6-3B"
HF_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"
# sha256 of the checkpoint files at HF_REVISION (weights: Hugging Face LFS oid; json files: measured on a download).
CKPT_SHA256 = {
    "model-00001-of-00002.safetensors": "9710d31d331b79cfa229fd23605ba9ad47e207cb0f4c7722fe0bacdc666c8326",
    "model-00002-of-00002.safetensors": "eb79bbd7893068897cda18c86ed91064e92e20e356dea97bc398bf9c8bf2fa35",
    "model.safetensors.index.json": "4baab54d48f5492daeea0a18d29bf8956661367820d6a19feb0dab10848ef5ab",
    "config.json": "f2702d11bbde7ebf250b81f28bbef606bf1aa2acdb5ce4f5304a2a29054c3c7b",
    "processor_config.json": "3c38bc24a9bbb333a567f9093abb959c1205a04acdb0a0229f86262ada4e0274",
    "statistics.json": "cefa9f4c612a350a4d406a501c6640d0a2371004d63d7b7e3c9703aac41a455b",
    "embodiment_id.json": "d30b5175fdf87b6b40f6b62f8f0e131fcd4b59759a666956190e8110ac61629f",
}

ENV_SUFFIX = "SplitA_GR1ArmsAndWaistFourierHands_Env"
# The six tasks with the highest published success of this checkpoint (examples/robocasa-gr1-tabletop-tasks/README.md:24-36).
# regions = the distractor regions of each task (tabletop_24dc.py:891-905, distractor_config_for_posttrain). Every task also
# gets FIXED_DISTRACTOR_CONFIG["back_edge"] (tabletop_24dc.py:112-139, zero or one decorative object at the back edge).
TASKS = [
    {"k": 0, "name": "PlateToPlate", "source": "plate", "target": "plate", "official": 0.787, "regions": []},
    {"k": 1, "name": "TrayToPlate", "source": "tray", "target": "plate", "official": 0.710,
     "regions": ["distractor_target_container"]},
    {"k": 2, "name": "CuttingboardToPan", "source": "cutting_board", "target": "pan", "official": 0.685,
     "regions": ["distractor_source_container"]},
    {"k": 3, "name": "CuttingboardToPot", "source": "cutting_board", "target": "pot", "official": 0.650,
     "regions": ["distractor_target_container"]},
    {"k": 4, "name": "TrayToPot", "source": "tray", "target": "pot", "official": 0.645, "regions": []},
    {"k": 5, "name": "PlacematToPlate", "source": "placemat", "target": "plate", "official": 0.630,
     "regions": ["distractor_obj"]},
]
N_TASKS = len(TASKS)
N_SCENES_PER_TASK = 20


def env_name(k):
    """Gym id of task k (gymnasium_groot.py:140-168 also lists the gr1_arms_waist_fourier_hands robot under gr1_unified/)."""
    return f"gr1_unified/PosttrainPnPNovelFrom{TASKS[k]['name']}{ENV_SUFFIX}"


# ---------------- scenes and seeds (paired design: the same (task, seed) gives the same scene and call seeds in every condition)
SCENE_SEED0 = 20261105
G0 = 300000            # global index g = G0 + 1000 * task + seed (does not overlap the LIBERO indices 0..499)
CAL_SEEDS = list(range(200, 284))          # calibration observations: 6 tasks x 84 seeds = 504 (not evaluation scenes)
CAL_FLOOR_SEEDS = list(range(200, 209))    # observations that set tau: 6 tasks x 9 seeds = 54
N_FLOOR_CALLS = 10                         # floor: the original instruction with call_seed(g, q), q = 0..9 -> 45 pairs
CHECK_SEEDS = list(range(100, 102))        # scene reproducibility check (no policy)


def scene_seed(k, s):
    """Seed given to the environment constructor and to reset() for task k, seed s."""
    return SCENE_SEED0 + 1000 * k + s


def gidx(k, s):
    """Global index g of (task k, seed s); policy call n of the episode uses call_seed(g, n)."""
    return G0 + 1000 * k + s


def parse_eps(spec):
    """"k:s,k:s,..." -> [(k, s), ...]"""
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x:
            k, s = x.split(":")
            out.append((int(k), int(s)))
    return out


def scene_list(path):
    """gr1/configs/gr1_scenes.json -> [(task, seed), ...] in task order and file order (6 tasks x 20 distinct seeds)."""
    with open(path) as f:
        seeds = json.load(f)["seeds"]
    assert sorted(seeds) == [str(t["k"]) for t in TASKS], sorted(seeds)
    out = []
    for t in TASKS:
        ss = [int(x) for x in seeds[str(t["k"])]]
        assert len(ss) == N_SCENES_PER_TASK and len(set(ss)) == len(ss), (t["k"], ss)
        out += [(t["k"], s) for s in ss]
    return out


def calibration_list():
    return [(t["k"], s) for t in TASKS for s in CAL_SEEDS]


# ---------------- instruction strings ----------------
# The environment gives raw_obs["language"] = ep_meta["lang"] (gymnasium_basic.py:209; tabletop_pnp.py:110-112:
# "pick the {obj} from the {source} and place it in the {target}"), and gymnasium_groot.py:112-115 wraps it as
# obs["annotation.human.coarse_action"] = f"unlocked_waist: {lang}" for the GR1ArmsAndWaist robot.
TEMPLATE = "unlocked_waist: "
LANG_KEY = "annotation.human.coarse_action"
# None = the environment's string (with the template); the other strings replace the whole language field (no template).
INSTRUCTIONS = {"original": None, "empty": "", "neutral": "do something", "nonsense": "xxx", "template_only": TEMPLATE}
RULE_INSTRUCTIONS = ("impossible_action", "absent_object")   # strings from the scene (instruction_string below)
CONDITIONS = (["none/" + i for i in ("original", "empty", "neutral", "nonsense", "template_only", "impossible_action",
                                     "absent_object")]
              + ["prefix/" + i for i in ("original", "empty", "nonsense")]
              + ["gate/" + i for i in ("original", "empty", "neutral", "nonsense", "template_only")]
              + ["hold"])
NONTASK = ("empty", "neutral", "nonsense", "template_only")

ENV_RE = re.compile(r"^(unlocked_waist: )?pick the (.+) from the (.+) and place it in the (.+)$")
# ABSENT-OBJECT object names: the ten designated objects of the suite's training data (alphabetical).
OBJ_POOL = ["bell pepper", "can", "croissant", "cupcake", "eggplant", "lemon", "pear", "squash", "sweet potato", "tomato"]
# IMPOSSIBLE-ACTION: scene g receives template g mod 6; {obj} designated object, {source} source container,
# {target} target container, read from the environment's instruction.
IMPOSSIBLE_ACTION_TEMPLATES = ["eat the {obj}", "wash the {target} in the sink", "pour water into the {target}",
                               "cut the {obj} in half", "paint the {source} red", "throw the {obj} at the wall"]


def split_cond(cond):
    """"gate/neutral" -> ("gate", "neutral"); "hold" -> ("hold", None)."""
    if cond == "hold":
        return "hold", None
    mech, instr = cond.split("/", 1)
    return mech, instr


def words(name):
    """Category names (underscores) and instruction words (spaces) in the same form."""
    return str(name).replace("_", " ").strip().lower()


def parse_env_text(env_text):
    """Environment instruction (with or without the template) -> (object, source container, target container)."""
    m = ENV_RE.match(env_text or "")
    if not m:
        raise ValueError(f"unexpected environment instruction: {env_text!r}")
    return m.group(2), m.group(3), m.group(4)


def absent_object_choice(k, s, scene_objects):
    """Task and object specified by the ABSENT-OBJECT instruction of (task k, seed s).
    Task: among the other tasks whose source and target containers both are not in the scene, number g mod (count) in task
    order (if no task qualifies, the tasks with the fewest containers in the scene). Object: among the OBJ_POOL names
    not in the scene, number (g + that task) mod (count). scene_objects maps scene object names to their categories."""
    g = gidx(k, s)
    present = {words(c) for c in (scene_objects or {}).values() if c}
    overlap = {}
    for t in TASKS:
        if t["k"] != k:
            overlap[t["k"]] = sorted({w for w in (words(t["source"]), words(t["target"])) if w in present})
    best = min(len(v) for v in overlap.values())
    cands = [k2 for k2 in sorted(overlap) if len(overlap[k2]) == best]
    k2 = cands[g % len(cands)]
    objs = [o for o in OBJ_POOL if o not in present]
    obj = objs[(g + k2) % len(objs)]
    return {"named_task": k2, "named_task_name": TASKS[k2]["name"], "named_object": obj,
            "named_source": words(TASKS[k2]["source"]), "named_target": words(TASKS[k2]["target"]),
            "overlap": overlap[k2]}


def instruction_string(cond, env_text, k, s, scene_objects=None):
    """String put into the language field for condition cond; None for the hold baseline (no policy call).
    - none/original and gate/original: the environment's string; empty, neutral, nonsense: the string alone (no template);
      template_only: "unlocked_waist: ".
    - impossible_action: template + IMPOSSIBLE_ACTION_TEMPLATES[g mod 6] filled with the scene's names.
    - absent_object: template + "pick the {obj'} from the {source'} and place it in the {target'}" (absent_object_choice).
    - prefix/x: PREFIX + " " + the string of x (PREFIX alone when x is empty).
    - gate/x: the same string as none/x."""
    mech, instr = split_cond(cond)
    if mech == "hold":
        return None
    if instr == "impossible_action":
        obj, source, target = parse_env_text(env_text)
        tpl = IMPOSSIBLE_ACTION_TEMPLATES[gidx(k, s) % len(IMPOSSIBLE_ACTION_TEMPLATES)]
        text = TEMPLATE + tpl.format(obj=obj, source=source, target=target)
    elif instr == "absent_object":
        c = absent_object_choice(k, s, scene_objects)
        text = TEMPLATE + f"pick the {c['named_object']} from the {c['named_source']} and place it in the {c['named_target']}"
    else:
        t = INSTRUCTIONS[instr]
        text = env_text if t is None else t
    if mech == "prefix":
        text = PREFIX if text == "" else f"{PREFIX} {text}"
    return text


def formalize(text):
    """What the model sees (processing_gr00t_n1d6.py:369-373, formalize_language=True): lower() and [^\\w\\s] removed."""
    return re.sub(r"[^\w\s]", "", text.lower())


# ---------------- object roles (naming rules of tabletop_24dc.py) ----------------
# _get_obj_cfgs (tabletop_24dc.py:471-508): target container "container" (:481), designated object "obj" (:493), source
# container "obj" + "_container" (:399). Distractor objects f"distractor_obj_{region}_type_{i}_num_{j}" (:597); in the
# distractor_source_container region the object's container is that name + "_container" (:399). The "object" of the
# distractor_target_container region is a target-container type (:96-107), so it counts as a distractor container.
ROLE_TARGET_OBJ, ROLE_DIST_OBJ = "target_obj", "distractor_obj"
ROLE_TARGET_RECEP, ROLE_SOURCE_RECEP, ROLE_DIST_RECEP = "target_recep", "source_recep", "distractor_recep"


def role_of(name):
    if name == "obj":
        return ROLE_TARGET_OBJ
    if name == "container":
        return ROLE_TARGET_RECEP
    if name == "obj_container":
        return ROLE_SOURCE_RECEP
    if name.startswith("distractor_obj_distractor_target_container"):
        return ROLE_DIST_RECEP
    if name.startswith("distractor_obj_distractor_source_container"):
        return ROLE_DIST_RECEP if name.endswith("_container") else ROLE_DIST_OBJ
    if name.startswith("distractor_obj_distractor_obj") or name.startswith("distractor_obj_back_edge"):
        return ROLE_DIST_OBJ
    return "other"


# Placement outcomes: designated object into the target container, distractor object into the target container,
# designated object into a distractor container; "tie" = two of these on the same step (counted as completion).
OUT_TARGET, OUT_DIST_INTO_TARGET, OUT_TARGET_INTO_DIST, OUT_NONE, OUT_TIE = (
    "target", "distractor_into_target", "target_into_distractor", "none", "tie")
PLACEMENT_OUTCOMES = (OUT_TARGET, OUT_DIST_INTO_TARGET, OUT_TARGET_INTO_DIST, OUT_TIE)


def pair_outcome(obj_role, recep_role):
    """(object role, container role) -> outcome; None for pairs that are recorded only (distractor into distractor
    container, anything into the source container)."""
    if obj_role == ROLE_TARGET_OBJ and recep_role == ROLE_TARGET_RECEP:
        return OUT_TARGET
    if obj_role == ROLE_DIST_OBJ and recep_role == ROLE_TARGET_RECEP:
        return OUT_DIST_INTO_TARGET
    if obj_role == ROLE_TARGET_OBJ and recep_role == ROLE_DIST_RECEP:
        return OUT_TARGET_INTO_DIST
    return None


def candidate_pairs(names):
    """Scene object names -> (object, container) pairs checked after every step, and the role of each name."""
    roles = {n: role_of(n) for n in names}
    objs = [n for n, r in roles.items() if r in (ROLE_TARGET_OBJ, ROLE_DIST_OBJ)]
    receps = [n for n, r in roles.items() if r in (ROLE_TARGET_RECEP, ROLE_SOURCE_RECEP, ROLE_DIST_RECEP)]
    return [(o, r) for o in sorted(objs) for r in sorted(receps)], roles


def pair_key(o, r):
    return f"{o}->{r}"


class FirstTrue:
    """First-true rule: pairs already true after settling (start_true) are not counted. The first placement pair that
    becomes true sets the outcome; two different outcomes on the same step give "tie"."""

    def __init__(self, roles, start_true):
        self.roles = roles
        self.start_true = set(start_true)
        self.outcome, self.step, self.tie_pairs, self.pairs = OUT_NONE, None, None, None
        self.other_first_true = {}

    def update(self, true_pairs, t):
        newly = [p for p in sorted(true_pairs) if pair_key(*p) not in self.start_true]
        hits = []
        for o, r in newly:
            out = pair_outcome(self.roles.get(o), self.roles.get(r))
            if out is None:
                self.other_first_true.setdefault(pair_key(o, r), t)
            else:
                hits.append((out, pair_key(o, r)))
        if hits and self.outcome == OUT_NONE:
            outs = sorted({h[0] for h in hits})
            self.outcome = outs[0] if len(outs) == 1 else OUT_TIE
            self.tie_pairs = [h[1] for h in hits] if len(outs) > 1 else None
            self.pairs = [h[1] for h in hits]
            self.step = t
            return True
        return False


# ---------------- episode categories (the paper's four categories on the humanoid) ----------------
DISP = 0.05            # m, largest displacement of either hand end from its post-settling position
TORSO_DISP = 0.05      # m, displacement of the torso reference body (head link)
HAND_ACT_RAD = 0.2     # rad, finger joint change recorded as hand actuation (not part of the categories)


def category(outcome, contact, max_hand_disp, torso_disp):
    """completion: a placement outcome; attempt: no completion, a hand touched an object, container, or fixture (table
    excluded); ncmotion (non-contact motion): no completion, no contact, the larger hand displacement >= 5 cm or the torso
    displacement >= 5 cm; nonstart: the rest."""
    if outcome in PLACEMENT_OUTCOMES:
        return "completion"
    if contact:
        return "attempt"
    if max_hand_disp >= DISP or torso_disp >= TORSO_DISP:
        return "ncmotion"
    return "nonstart"


def categorize(rec):
    return category(rec["outcome"], bool(rec["contact_first_step"]), max(rec["hand_disp_max"].values()),
                    rec["torso_disp_max"])


def hand_actuation(rec):
    """Fingers closed without arm motion (recorded separately; not part of the non-start rule)."""
    return max(rec["finger_delta_max"].values()) > HAND_ACT_RAD


# ---------------- contact owners (MuJoCo body tree, pure functions) ----------------
def owner_of_body(body, parent, roots):
    """Walk up from a body id through `parent` to the first id in `roots` (id -> owner name). None at the world body."""
    seen = 0
    while body is not None and body >= 0 and seen < 10000:
        if body in roots:
            return roots[body]
        if body == 0:
            return None
        body = int(parent[body])
        seen += 1
    return None


def scene_signature(items):
    """sha1 of [(name, category, x, y, z), ...] rounded to mm; checks that a (task, seed) gives the same scene in every
    condition."""
    rows = sorted((str(n), str(c), round(float(x) * 1000), round(float(y) * 1000), round(float(z) * 1000))
                  for n, c, x, y, z in items)
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


# ---------------- gate ----------------
# Action chunk: the server's keys concatenated in this order (29 dimensions x 16 steps, the order of processor_config.json
# gr1.action).
GATE_KEYS = [("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3)]
GATE_DIM = sum(d for _, d in GATE_KEYS)   # 29
GATE_H = 16
STD_MIN = 1e-3                             # dimensions with a smaller std are left out (as in the LIBERO-Goal gate calibration)


def chunk_vec(action):
    """Server action dict ({"action.<key>": (1,16,d) or (16,d)}) -> (16, 29) array."""
    import numpy as np
    parts = []
    for k, d in GATE_KEYS:
        v = np.asarray(action[f"action.{k}"], dtype=np.float64)
        if v.ndim == 3:
            v = v[0]
        assert v.shape[-1] == d, (k, v.shape)
        parts.append(v[:GATE_H])
    return np.concatenate(parts, axis=-1)


def chunk_dist(a, b, std, keep, horizon=GATE_H):
    """Gate distance: divide by the per-dimension std, L2 over the kept dimensions, mean over the chunk steps."""
    import numpy as np
    sd = np.where(np.asarray(keep), np.asarray(std, dtype=np.float64), 1.0)
    a, b = np.asarray(a, dtype=np.float64)[:horizon], np.asarray(b, dtype=np.float64)[:horizon]
    z = ((a - b) / sd)[:, np.asarray(keep, dtype=bool)]
    return float(np.mean(np.linalg.norm(z, axis=-1)))


def gate_compute(refs, floor, floor_main_mask=None, empty=None):
    """refs (n,H,29): original-instruction chunks with call_seed(g, 0); floor (n,R,H,29): the original instruction with R
    call seeds; floor_main_mask (n,) bool: observations that set tau (CAL_FLOOR_SEEDS); empty (n,H,29): the empty-string
    chunk with call_seed(g, 0) (reported only). tau = 95th percentile of the floor distances."""
    import numpy as np
    refs, floor = np.asarray(refs, dtype=np.float64), np.asarray(floor, dtype=np.float64)
    std = refs.reshape(-1, refs.shape[-1]).std(0)
    keep = std >= STD_MIN
    r = floor.shape[1]

    def dists(idx):
        return [chunk_dist(floor[i, p], floor[i, q], std, keep) for i in idx for p in range(r) for q in range(p + 1, r)]
    n = len(floor)
    main_idx = [i for i in range(n) if floor_main_mask is None or floor_main_mask[i]]
    fd = dists(main_idx)
    tau = float(np.percentile(fd, 95))
    fd_all = dists(range(n))
    out = {"tau_p95": tau, "std": std.tolist(), "dims_kept": keep.tolist(), "horizon": GATE_H, "n_refs": int(len(refs)),
           "n_floor_inits": len(main_idx), "floor_pairs": len(fd), "floor_median": float(np.median(fd)),
           "tau_p95_all_inits": float(np.percentile(fd_all, 95)), "floor_pairs_all_inits": len(fd_all)}
    if empty is not None:
        de = np.array([chunk_dist(refs[i], np.asarray(empty[i], dtype=np.float64), std, keep) for i in range(len(refs))])
        out["original_vs_empty_median"] = float(np.median(de))
        out["original_vs_empty_n_le_tau"] = int(np.sum(de <= tau))
        out["original_vs_empty_n"] = int(len(de))
    return out
