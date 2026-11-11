"""Wire protocol between the simulation worker and a policy adapter (spec v0).

Transport: ZMQ REQ/REP on 127.0.0.1, msgpack + msgpack_numpy, object-dtype arrays refused.
Request {"endpoint": str, "data": dict}; error reply {"error": str}.

Endpoints
  ping                          -> {"status": "ok", ...}
  kill                          -> {"status": "killed"} and the server exits
  infer {obs, text, seed, mask}   -> {"chunk": float32 (T, 7), "nan": bool}; mask = optional command-token masking
  tokens                          -> {"text": last text received, "prompt": what the model saw (or None), ...}
  meta (new in this kit)        -> adapter metadata, see META_REQUIRED
obs keys: video.image, video.wrist_image uint8 (1, 1, 256, 256, 3), already rotated 180 degrees by the worker;
state.x/y/z/roll/pitch/yaw float32 (1, 1, 1); state.gripper float32 (1, 1, 2).
chunk[:, :6] are end-effector deltas in environment space; chunk[:, 6] is g' in [0, 1] (1 = open direction).
An adapter whose model outputs environment-space gripper a in [-1, 1] (-1 = open) sends g' = (1 - a) / 2.
"""
import functools

import numpy as np

OBS_KEYS = {"video.image": ((1, 1, 256, 256, 3), np.uint8), "video.wrist_image": ((1, 1, 256, 256, 3), np.uint8),
            "state.x": ((1, 1, 1), np.float32), "state.y": ((1, 1, 1), np.float32), "state.z": ((1, 1, 1), np.float32),
            "state.roll": ((1, 1, 1), np.float32), "state.pitch": ((1, 1, 1), np.float32),
            "state.yaw": ((1, 1, 1), np.float32), "state.gripper": ((1, 1, 2), np.float32)}
ENDPOINTS = ("ping", "kill", "infer", "tokens", "meta")
META_REQUIRED = {
    "adapter": "adapter name",
    "checkpoint": "hub id or path",
    "checkpoint_revision": "revision or sha256 of the weights",
    "framework": "framework name and version (e.g. lerobot 0.6.2 @ commit)",
    "empty_handling": "true_empty (the empty string reaches the model) | framework_default (the framework substitutes text)",
    "prompt_of_empty": "the exact prompt the model sees for the empty instruction, or None if not exposed",
    "chunk_len": "model chunk length",
    "chunk_out": "steps returned per call (>= exec_steps and >= gate horizon)",
    "image_resolution": "resolution the model consumes",
    "norm_map": "normalization modes in use",
    "norm_map_installed_default": "normalization modes of the installed framework's default config",
    "norm_map_source": "ckpt | override",
    "seed_injection": "how the call seed reaches the sampler",
    "stochastic": "True if inference draws random numbers (decides the gate floor source)",
}
EMPTY_HANDLING = ("true_empty", "framework_default")
GRIP_TOL = 0.25  # beyond this the gripper column is not in the wire convention (an unconverted a in [-1, 1] reaches -1)


class MsgSerializer:
    """Same encoding as Isaac-GR00T gr00t/policy/server_client.MsgSerializer (object-dtype arrays refused)."""

    @staticmethod
    def _safe_encode(obj, chain=None):
        import msgpack_numpy as mnp
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError("refusing object-dtype ndarray")
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj, chain=None):
        import msgpack_numpy as mnp
        if isinstance(obj, dict):
            nd, kind = obj.get(b"nd", obj.get("nd")), obj.get(b"kind", obj.get("kind"))
            if nd and kind in (b"O", "O"):
                raise ValueError("refusing object-dtype ndarray payload")
        return mnp.decode(obj, chain=chain)

    @staticmethod
    def to_bytes(data):
        import msgpack
        return msgpack.packb(data, default=functools.partial(MsgSerializer._safe_encode, chain=lambda o: o))

    @staticmethod
    def from_bytes(data):
        import msgpack
        return msgpack.unpackb(data, object_hook=functools.partial(MsgSerializer._safe_decode, chain=lambda o: o), raw=False)


class PolicyClient:
    """Minimal REQ client. A timeout closes and recreates the socket (REQ sockets cannot resend after a lost reply)."""

    def __init__(self, port, host="127.0.0.1", timeout_ms=300000):
        import zmq
        self._zmq = zmq
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self._open()

    def _open(self):
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, endpoint, data=None):
        try:
            self.sock.send(MsgSerializer.to_bytes({"endpoint": endpoint, "data": data or {}}))
            return MsgSerializer.from_bytes(self.sock.recv())
        except self._zmq.Again:
            self.sock.close(0)
            self._open()
            raise TimeoutError(f"{endpoint} timed out after {self.timeout_ms} ms")

    def ping(self):
        try:
            return self.call("ping").get("status") == "ok"
        except TimeoutError:
            return False

    def infer(self, obs, text, seed, mask=False):
        r = self.call("infer", {"obs": obs, "text": text, "seed": int(seed), "mask": bool(mask)})
        if "error" in r:
            raise RuntimeError(r["error"])
        return np.asarray(r["chunk"]), bool(r["nan"])

    def close(self):
        self.sock.close(0)


def to_worker_chunk(actions):
    """Environment-space actions (T, 7), gripper a in [-1, 1] with -1 = open, to the wire convention g' = (1 - a) / 2."""
    a = np.asarray(actions, dtype=np.float32).copy()
    if a.ndim != 2 or a.shape[1] != 7:
        raise ValueError(f"actions shape {a.shape}, expected (T, 7)")
    a[:, 6] = (1.0 - a[:, 6]) / 2.0
    return a


def validate_obs(obs):
    probs = []
    for k, (shape, dt) in OBS_KEYS.items():
        if k not in obs:
            probs.append(f"missing {k}")
            continue
        a = np.asarray(obs[k])
        if a.shape != shape or a.dtype != dt:
            probs.append(f"{k}: {a.shape} {a.dtype}, expected {shape} {np.dtype(dt)}")
    extra = set(obs) - set(OBS_KEYS)
    if extra:
        probs.append(f"unexpected keys {sorted(extra)}")
    return probs


def validate_chunk(resp, min_steps):
    probs = []
    if "error" in resp:
        return [f"error reply: {resp['error']}"]
    ch = np.asarray(resp.get("chunk"))
    if ch.ndim != 2 or ch.shape[1] != 7:
        probs.append(f"chunk shape {ch.shape}, expected (T, 7)")
        return probs
    if ch.shape[0] < min_steps:
        probs.append(f"chunk has {ch.shape[0]} steps < {min_steps}")
    if ch.dtype != np.float32:
        probs.append(f"chunk dtype {ch.dtype}, expected float32")
    nan = bool(np.isnan(ch).any())
    if bool(resp.get("nan")) != nan:
        probs.append(f"nan flag {resp.get('nan')} but chunk nan={nan}")
    g = ch[:, 6]
    if not nan and (g.min() < -GRIP_TOL or g.max() > 1 + GRIP_TOL):
        probs.append(f"gripper column far outside [0, 1]: [{g.min():.3f}, {g.max():.3f}] (unconverted environment gripper? "
                     "wire convention g' = (1 - a) / 2)")
    return probs


def gripper_overshoot(resp):
    """Small excursions of g' beyond [0, 1] (continuous model outputs slightly beyond +-1); harmless because the worker
    binarizes the gripper by sign. Returns (min, max) if there is one, else None."""
    g = np.asarray(resp["chunk"])[:, 6]
    if np.isnan(g).any() or (g.min() >= 0 and g.max() <= 1):
        return None
    return float(g.min()), float(g.max())
