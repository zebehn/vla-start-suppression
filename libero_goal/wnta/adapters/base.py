"""Common adapter server: a model author implements Backend; the server handles the wire protocol.

    class MyBackend(Backend):
        def load(self, ckpt, device): ...
        def infer_env(self, flat_obs, text, seed): ...   # (T >= chunk_out, 7) environment-space actions, gripper -1 = open
        def meta(self): ...                              # protocol.META_REQUIRED fields
        def prompt_of(self, text, obs=None): ...         # optional

The server truncates to chunk_out, converts the gripper to the wire convention, sets the nan flag, answers ping, kill,
infer, tokens and meta, rejects mask=True unless the backend sets supports_mask, and refuses to start on a
GPU listed in BENCH_FORBIDDEN_CARDS (a site rule, see serve.py).
"""
import argparse
import os
import sys

import numpy as np

from .. import protocol as P


class Backend:
    name = "backend"
    chunk_out = 16
    supports_mask = False
    wire_gripper = False  # True if infer_env already returns the wire gripper g' (no conversion by the server)

    def load(self, ckpt, device, dtype=None, **opt):
        pass

    def infer_env(self, flat_obs, text, seed):
        raise NotImplementedError

    def meta(self):
        raise NotImplementedError

    def prompt_of(self, text, obs=None):
        """Exact prompt the model sees for text (obs: the observation of that call, for models whose prompt embeds the state)."""
        return None


class StubBackend(Backend):
    """No model. Output depends on seed (and text unless text_blind); used by the conformance self-test.

    Fault switches reproduce the failure modes the conformance test must catch:
      deterministic       - ignores the seed (as a regression head does)
      substitute_empty    - replaces "" by a framework sentence before the model (framework-default substitution of the empty instruction)
      norm_mismatch       - stored normalization differs from the installed default
      short_chunk         - returns fewer steps than chunk_out
      unseeded            - draws noise without the call seed (same seed twice gives different chunks)
    """
    name = "stub"

    def __init__(self, deterministic=False, substitute_empty=False, norm_mismatch=False, short_chunk=False,
                 unseeded=False, text_blind=False):
        self.deterministic, self.substitute_empty = deterministic, substitute_empty
        self.norm_mismatch, self.short_chunk, self.unseeded, self.text_blind = norm_mismatch, short_chunk, unseeded, text_blind
        self._rng = np.random.default_rng()

    def _text(self, text):
        return "perform the task" if (self.substitute_empty and text == "") else text

    def infer_env(self, flat_obs, text, seed):
        t = "" if self.text_blind else self._text(text)
        h = sum(ord(c) for c in t) % 997
        base = np.full((50, 7), (h - 498) / 1000.0, dtype=np.float32)
        base[:, 6] = -1.0
        if self.deterministic:
            noise = 0.0
        elif self.unseeded:
            noise = self._rng.normal(0, 0.05, size=(50, 7))
        else:
            noise = np.random.default_rng(int(seed)).normal(0, 0.05, size=(50, 7))
        a = (base + noise).astype(np.float32)
        a[:, 6] = np.clip(a[:, 6], -1, 1)
        return a[:8] if self.short_chunk else a

    def prompt_of(self, text, obs=None):
        return f"<img><img>{self._text(text)}"

    def meta(self):
        default = {"VISUAL": "IDENTITY", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
        used = dict(default, VISUAL="MEAN_STD") if self.norm_mismatch else dict(default)
        return {"adapter": self.name, "checkpoint": "stub", "checkpoint_revision": "none", "framework": "none",
                "empty_handling": "framework_default" if self.substitute_empty else "true_empty",
                "prompt_of_empty": self.prompt_of(""), "chunk_len": 50, "chunk_out": self.chunk_out,
                "image_resolution": [256, 256], "norm_map": used, "norm_map_installed_default": default,
                "norm_map_source": "ckpt", "seed_injection": "numpy default_rng(seed)",
                "stochastic": not self.deterministic}


class Server:
    def __init__(self, backend):
        self.be = backend
        self.last_text = None
        self.last_obs = None

    def handle(self, endpoint, data):
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running"}
        if endpoint == "meta":
            return self.be.meta()
        if endpoint == "tokens":
            return {"text": self.last_text, "prompt": self.be.prompt_of(self.last_text or "", self.last_obs), "backend": self.be.name}
        if endpoint == "infer":
            if data.get("mask") and not self.be.supports_mask:
                raise ValueError(f"mask (command-token masking) is not supported for {self.be.name}")
            a = np.asarray(self.be.infer_env(data["obs"], data["text"], data["seed"]), dtype=np.float32)
            self.last_text, self.last_obs = data["text"], data["obs"]
            if a.ndim != 2 or a.shape[1] != 7 or a.shape[0] < self.be.chunk_out:
                raise ValueError(f"backend returned {a.shape}, expected (>= {self.be.chunk_out}, 7)")
            ch = np.ascontiguousarray(a[: self.be.chunk_out]) if self.be.wire_gripper else P.to_worker_chunk(a[: self.be.chunk_out])
            return {"chunk": ch, "nan": bool(np.isnan(ch).any())}
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def serve(self, port, host="127.0.0.1"):
        import zmq
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://{host}:{port}")
        print(f"{self.be.name} server ready port={port} pid={os.getpid()} meta={self.be.meta()}", flush=True)
        running = True
        while running:
            msg = sock.recv()
            try:
                req = P.MsgSerializer.from_bytes(msg)
                ep = req.get("endpoint", "")
                if ep == "kill":
                    running, out = False, {"status": "killed"}
                else:
                    out = self.handle(ep, req.get("data", {}) or {})
                sock.send(P.MsgSerializer.to_bytes(out))
            except Exception as e:  # noqa: BLE001  (the error goes back to the client)
                print(f"Error in server: {e}", flush=True)
                sock.send(P.MsgSerializer.to_bytes({"error": str(e)}))
        sock.close(0)
        ctx.term()


def main(argv=None):
    ap = argparse.ArgumentParser(description="stub adapter server (conformance self-test)")
    ap.add_argument("--port", type=int, required=True)
    for f in ("deterministic", "substitute-empty", "norm-mismatch", "short-chunk", "unseeded", "text-blind"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(argv)
    be = StubBackend(a.deterministic, a.substitute_empty, a.norm_mismatch, a.short_chunk, a.unseeded, a.text_blind)
    Server(be).serve(a.port)


if __name__ == "__main__":
    sys.exit(main())
