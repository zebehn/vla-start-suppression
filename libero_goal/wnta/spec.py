"""Frozen protocol values (spec/v0.json) and the rules that turn them into episodes.

Pure Python plus json; numpy is needed only for load_init_states().
"""
import hashlib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(_HERE, "spec")


def load(version="v0"):
    with open(os.path.join(SPEC_DIR, f"{version}.json")) as f:
        return json.load(f)


SPEC = load()
SCENE = SPEC["scene"]
EXEC = SPEC["execution"]
COND = SPEC["conditions"]
N_TASKS = SCENE["n_tasks"]
N_INIT_FILE = SCENE["init_per_task_file"]
TASK_LANGUAGE = list(SCENE["task_language"])
CORE_CONDITIONS = list(COND["core"])
APPENDIX_CONDITIONS = list(COND["appendix"])
MECHANISMS = tuple(COND["mechanisms"])  # ("none", "prefix", "gate")
PREFIX = COND["prefix_sentence"]
INSTRUCTIONS = dict(COND["instructions"])  # None = the task file's own instruction
SCENE_SEED_BASE = SCENE["scene_seed_base"]
CALL_SEED_BASE = EXEC["call_seed_base"]
SEED_OFFSET_REPLICATE = EXEC["seed_offset_replicate"]
EXEC_STEPS = EXEC["exec_steps"]
MAX_POLICY_STEPS = SCENE["max_policy_steps"]
SETTLE_STEPS = SCENE["settle_steps"]
HOLD_ACTION = list(EXEC["hold_action"])
NONSTART_DISP_M = SPEC["metrics"]["non_start_disp_m"]


def gidx(k, j):
    """Global initial-state index g = 50 k + j."""
    if not (0 <= k < N_TASKS and 0 <= j < N_INIT_FILE):
        raise ValueError(f"task {k} / j {j} out of range")
    return N_INIT_FILE * k + j


def split_g(g):
    return divmod(int(g), N_INIT_FILE)


def core_inits():
    """The 200 initial states of the core track: every task, j = 0..19, ascending g."""
    return [gidx(k, j) for k in range(N_TASKS) for j in SCENE["core_init_j"]]


def scene_seed(g):
    return SCENE_SEED_BASE + int(g)


def call_seed(g, call_index, seed_offset=0):
    return CALL_SEED_BASE + 1000 * int(g) + int(call_index) + int(seed_offset)


def other_task(k, j):
    """Appendix none/other_task: the other task whose instruction is given at task file k, index j."""
    return (k + 1 + (j % 9)) % 10


def condition_text(cond, k, j, langs=None):
    """(text sent to the policy, mechanism, named task or None) for a condition "<mechanism>/<instruction>" at task file k, index j.

    langs defaults to the frozen LIBERO task language; pass LIBERO's task.language list to check both sources agree.
    """
    langs = TASK_LANGUAGE if langs is None else langs
    g = gidx(k, j)
    if cond in COND["appendix"]:
        a = COND["appendix"][cond]
        if cond == "none/paraphrase":
            return a["paraphrases"][str(k)], "none", None
        if cond in ("none/neutral_new", "none/absent_object", "none/impossible_action"):
            return a["sentences"][g % 6], "none", None
        if cond == "none/other_task":
            k2 = other_task(k, j)
            return langs[k2], "none", k2
    mech, _, instr = cond.partition("/")
    if mech not in MECHANISMS or instr not in INSTRUCTIONS:
        raise ValueError(f"unknown condition {cond}")
    t = langs[k] if INSTRUCTIONS[instr] is None else INSTRUCTIONS[instr]
    if mech == "prefix":
        t = PREFIX if t == "" else f"{PREFIX} {t}"
    return t, mech, None


def shard(gs, i, n):
    """Deterministic split of initial states for parallel runs, independent of the condition.

    Invariant needed for paired comparisons: the same g goes to the same shard in every condition.
    """
    return [g for g in gs if (int(g) % N_INIT_FILE) % n == i]


def load_init_states():
    """(10, 50, 79) float64 array of LIBERO-Goal init states; verifies the committed sha256."""
    import numpy as np
    with open(os.path.join(SPEC_DIR, "init_states_v0.json")) as f:
        meta = json.load(f)
    arr = np.load(os.path.join(SPEC_DIR, "init_states_v0.npz"))["states"]
    h = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    if h != meta["array_sha256"]:
        raise RuntimeError(f"init_states_v0.npz sha256 {h} != {meta['array_sha256']}")
    return arr


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
