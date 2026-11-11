"""First-call gate: chunk distance, gate profiles and noise-floor calibration (spec v0).

Distance between two action chunks a, b of shape (T, 7):
  divide each action dimension by the reference standard deviation (dimensions whose std < std_min are excluded),
  take the L2 norm over the kept dimensions at each step, and average over the first `horizon` steps
  (horizon None = all steps).

Noise floor and tau (the commit threshold) come from one of two floor sources, and the profile records which:
  seed        - same observation and instruction, 10 different call seeds; for policies with sampling noise.
  settle_obs  - same instruction and seed, observations after settle steps 6..10 of the same initial state; for
                policies whose inference uses no randomness (their seed floor is identically 0, so tau would be 0).
tau = 95th percentile (numpy linear interpolation) of the floor distances.
"""
import hashlib
import json

import numpy as np

from . import spec as S

GATE = S.SPEC["gate"]
STD_MIN = GATE["std_min"]
FLOOR_SOURCES = tuple(GATE["floor_sources"])  # ("seed", "settle_obs")
SETTLE_OBS_STEPS = (6, 7, 8, 9, 10)


class GateProfile:
    def __init__(self, tau, std, dims_kept, horizon, floor_source="seed", name=None, raw=None):
        if floor_source not in FLOOR_SOURCES and floor_source != "fixed":
            raise ValueError(f"floor_source {floor_source}")
        self.tau = float(tau)
        self.std = list(std)
        self.dims_kept = [bool(x) for x in dims_kept]
        self.horizon = horizon
        self.floor_source = floor_source
        self.name = name
        self.raw = raw or {}

    @classmethod
    def from_json(cls, path, name=None):
        """Load a gate file. An explicit "tau" field (the threshold used) takes precedence over "tau_p95"."""
        with open(path) as f:
            d = json.load(f)
        src = d.get("floor_source", d.get("floor_method", "seed"))
        src = {"settle": "settle_obs"}.get(src, src)
        return cls(d.get("tau", d["tau_p95"]), d["std"], d["dims_kept"], d["horizon"], src, name or d.get("model"), d)

    def sha256(self):
        body = json.dumps({"tau": self.tau, "std": self.std, "dims_kept": self.dims_kept, "horizon": self.horizon,
                           "floor_source": self.floor_source}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def distance(self, a, b):
        return chunk_dist(a, b, self.std, self.dims_kept, self.horizon)

    def commit(self, chunk_cmd, chunk_empty):
        d = self.distance(chunk_cmd, chunk_empty)
        return d, bool(d > self.tau)


def chunk_dist(a, b, std, dims_kept, horizon=None):
    keep = np.asarray(dims_kept, dtype=bool)
    sd = np.where(keep, std, 1.0)
    a, b = np.asarray(a), np.asarray(b)
    if horizon:
        a, b = a[:horizon], b[:horizon]
    z = ((a - b) / sd)[:, keep]
    return float(np.mean(np.linalg.norm(z, axis=-1)))


def calibrate(refs, floor, floor_source, std_min=STD_MIN):
    """refs: (n_inits, 10 instructions, H, 7); floor: (m_inits, r, H, 7) - r = 10 seeds or 5 settle observations.

    Returns the profile fields (same names as the gate files of the paper's runs) plus floor_source.
    The arithmetic follows the calibration script of the paper's runs so that stored chunks reproduce the committed values bit for bit.
    """
    if floor_source not in FLOOR_SOURCES:
        raise ValueError(floor_source)
    refs, floor = np.asarray(refs), np.asarray(floor)
    std = refs.reshape(-1, refs.shape[-1]).std(0)
    keep = std >= std_min
    sd = np.where(keep, std, 1.0)

    def d(x, y):
        return float(np.mean(np.linalg.norm(((x - y) / sd)[:, keep], axis=-1)))
    r = floor.shape[1]
    fd = [d(floor[i, p], floor[i, q]) for i in range(len(floor)) for p in range(r) for q in range(p + 1, r)]
    tau = float(np.percentile(fd, GATE["percentile"]))
    nr = refs.shape[1]
    rp = [d(refs[i, p], refs[i, q]) for i in range(len(refs)) for p in range(nr) for q in range(p + 1, nr)]
    out = {"horizon": int(refs.shape[2]), "n_inits": len(refs), "n_floor_inits": len(floor), "floor_pairs": len(fd),
           "floor_median": float(np.median(fd)), "tau_p95": tau, "std": std.tolist(), "dims_kept": keep.tolist(),
           "ref_pairwise_median": float(np.median(rp)), "ref_pairwise_frac_le_tau": float(np.mean(np.array(rp) <= tau)),
           "floor_source": floor_source}
    if floor_source == "seed" and out["floor_median"] == 0.0:
        out["warning"] = "seed floor median is 0 (inference uses no randomness); recalibrate with floor_source=settle_obs"
    return out
