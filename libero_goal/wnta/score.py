"""Model-independent scorer: metrics computed from episode records only (spec v0 definitions).

An episode record is the JSON object one worker writes per episode. Fields read here:
  cond, mechanism, g, task, j, text, outcome (int 0..9 | "none" | "tie"), tie_tasks, contact_first_step (dict object -> step),
  first_contact, max_disp, gate_d, gate_commit, hold, n_calls, nan_calls, steps, noaction (recomputed, see noaction_flag).

Episode categories (mutually exclusive, as defined in the paper):
  completion          : outcome != "none" (some task's goal predicate became true)
  attempt             : no completion, gripper contact with an object or fixture
  non-contact motion  : no completion, no contact, max end-effector displacement >= 0.05 m
  non-start           : no completion, no contact, max end-effector displacement < 0.05 m
"No contact" = non-start + non-contact motion (no completion and no contact, regardless of displacement).

The outcome field mixes an integer task index with strings. Task 0 is the integer 0, so it must never be tested for
truthiness (`if r["outcome"]:` or `r["outcome"] or ...` silently drops task-0 completions). Only `== "none"`,
`== "tie"` and integer equality are used below.
"""
from collections import Counter

from . import spec as S
from . import stats as ST

NONE, TIE = "none", "tie"
CATS11 = [str(i) for i in range(S.N_TASKS)] + [NONE]


def outcome_cat(r):
    """Category string of the outcome: '0'..'9', 'none' or 'tie'."""
    o = r["outcome"]
    if isinstance(o, bool):
        raise TypeError(f"outcome must not be bool: {o!r}")
    if isinstance(o, int):
        if not 0 <= o < S.N_TASKS:
            raise ValueError(f"outcome task {o} out of range")
        return str(o)
    if o in (NONE, TIE):
        return o
    raise ValueError(f"unexpected outcome {o!r}")


def completed(r):
    return outcome_cat(r) != NONE


def completed_task(r, task):
    """True if task `task` was completed (alone or as part of a tie)."""
    c = outcome_cat(r)
    if c == TIE:
        return task in (r.get("tie_tasks") or [])
    return c == str(task)


def designated_success(r):
    return completed_task(r, r["task"])


def contacted(r):
    return len(r["contact_first_step"] or {}) > 0


def no_contact(r):
    """No completion and no contact, regardless of displacement (non-start + non-contact motion)."""
    return (not contacted(r)) and outcome_cat(r) == NONE


def category(r, disp=S.NONSTART_DISP_M):
    """One of 'completion', 'attempt', 'non_contact_motion', 'non_start'."""
    if outcome_cat(r) != NONE:
        return "completion"
    if contacted(r):
        return "attempt"
    return "non_start" if r["max_disp"] < disp else "non_contact_motion"


def attempt(r):
    return category(r) == "attempt"


def non_contact_motion(r):
    return category(r) == "non_contact_motion"


def non_start(r):
    """No completion, no contact, max end-effector displacement < 0.05 m."""
    return category(r) == "non_start"


def noaction_flag(r, disp=S.NONSTART_DISP_M):
    """The worker's stored `noaction` flag: displacement < 0.05 m and no contact (completion not tested)."""
    return bool(r["max_disp"] < disp and not contacted(r))


def gate_commit(r):
    v = r.get("gate_commit")
    return None if v is None else bool(v)


def counts(rows):
    """Integer counts for one condition (the unit every threshold uses)."""
    n = len(rows)
    cats = Counter(category(r) for r in rows)
    out = {"n": n,
           "completion_n": cats["completion"],
           "success_n": sum(designated_success(r) for r in rows),
           "attempt_n": cats["attempt"],
           "non_contact_motion_n": cats["non_contact_motion"],
           "non_start_n": cats["non_start"],
           "no_contact_n": sum(no_contact(r) for r in rows),
           "nan_calls": sum(int(r["nan_calls"]) for r in rows),
           "policy_calls": sum(int(r["n_calls"]) for r in rows)}
    gc = [gate_commit(r) for r in rows]
    if any(v is not None for v in gc):
        if any(v is None for v in gc):
            raise ValueError("gate_commit present in some episodes of a condition and missing in others")
        out["commit_n"] = sum(gc)
    return out


def outcome_counts(rows):
    return dict(Counter(outcome_cat(r) for r in rows))


def selection(rows):
    """11-category distribution (tie episodes are counted separately and left out of the 11 categories)."""
    oc = outcome_counts(rows)
    cnt = [oc.get(k, 0) for k in CATS11]
    return {"counts": {k: oc.get(k, 0) for k in CATS11}, "tie_n": oc.get(TIE, 0),
            "entropy11": ST.entropy_normalized(cnt, 11), "entropy10_completed": ST.entropy_normalized(cnt[:10], 10)}


def condition_metrics(rows, tau=None):
    """Full metric block for one condition: counts, rates, Wilson intervals, selection, gate distances."""
    c = counts(rows)
    n = c["n"]
    m = dict(c)
    for k in ("completion", "success", "attempt", "non_contact_motion", "non_start", "no_contact"):
        m[f"{k}_rate"] = c[f"{k}_n"] / n if n else None
        m[f"{k}_wilson"] = ST.wilson(c[f"{k}_n"], n)
    if "commit_n" in c:
        m["commit_rate"] = c["commit_n"] / n
        m["commit_wilson"] = ST.wilson(c["commit_n"], n)
        ds = [r["gate_d"] for r in rows]
        m["gate_d"] = ST.five_number(ds)
        if tau is not None:
            m["gate_d_gt_tau_n"] = sum(d > tau for d in ds)
    m["selection"] = selection(rows)
    m["first_contact_counts"] = dict(Counter(str(r["first_contact"]) for r in rows))
    m["max_disp"] = ST.five_number([r["max_disp"] for r in rows])
    return m


def by_g(rows):
    d = {}
    for r in rows:
        if r["g"] in d:
            raise ValueError(f"duplicate g {r['g']} in {r.get('cond')}")
        d[r["g"]] = r
    return d


def paired(rows_a, rows_b, fn, gs=None):
    """McNemar on fn(record) between two conditions over the same initial states."""
    a, b = by_g(rows_a), by_g(rows_b)
    gs = sorted(a) if gs is None else gs
    if set(a) != set(b) or set(gs) - set(a):
        raise ValueError("conditions are not paired over the same initial states")
    p, b01, b10 = ST.mcnemar_exact([fn(a[g]) for g in gs], [fn(b[g]) for g in gs])
    return {"p": p, "b01": b01, "b10": b10, "n": len(gs)}


def check_integrity(rows, cond, expected_gs, cond_text=True, max_steps=None):
    """Problems list (empty = ok): duplicates, missing/extra g, condition text versus the spec rule, task/j consistent with g."""
    probs = []
    gs = [r["g"] for r in rows]
    dup = sorted(g for g, k in Counter(gs).items() if k > 1)
    if dup:
        probs.append(f"{cond}: duplicate g {dup[:5]}")
    miss, extra = set(expected_gs) - set(gs), set(gs) - set(expected_gs)
    if miss or extra:
        probs.append(f"{cond}: missing {len(miss)} extra {len(extra)}")
    for r in rows:
        k, j = S.split_g(r["g"])
        if (r["task"], r["j"]) != (k, j):
            probs.append(f"{cond}: g{r['g']} task/j {r['task']}/{r['j']} != {k}/{j}")
        if r.get("cond") != cond:
            probs.append(f"{cond}: g{r['g']} cond field {r.get('cond')}")
        if cond_text:
            t, mech, named = S.condition_text(cond, k, j)
            if r["text"] != t or r.get("mechanism", mech) != mech:
                probs.append(f"{cond}: g{r['g']} text {r['text']!r} != spec {t!r}")
            if named is not None and r.get("named_task") != named:
                probs.append(f"{cond}: g{r['g']} named_task {r.get('named_task')} != {named}")
        if r.get("scene_seed") is not None and r["scene_seed"] != S.scene_seed(r["g"]):
            probs.append(f"{cond}: g{r['g']} scene_seed {r['scene_seed']}")
        if max_steps is not None and r.get("max_steps", max_steps) != max_steps:
            probs.append(f"{cond}: g{r['g']} max_steps {r.get('max_steps')}")
        if "noaction" in r and bool(r["noaction"]) != noaction_flag(r):
            probs.append(f"{cond}: g{r['g']} stored noaction {r['noaction']} != recomputed")
        outcome_cat(r)
    return probs
