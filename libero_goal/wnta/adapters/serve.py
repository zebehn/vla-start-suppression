"""Run an adapter server.

    python -m <package>.adapters.serve --adapter evo1 --ckpt <hub id or path> --port 5741 [--device cuda] [--opt vlm=<path>]
    python -m <package>.adapters.serve --adapter-module my_policy_adapter:MyPolicy --ckpt <path> --port 5741
    CPU (no GPU touched): CUDA_VISIBLE_DEVICES= python -m <package>.adapters.serve --adapter evo1 --ckpt ... --port 5799 --device cpu --dtype float32

Built-in adapters are listed in ADAPTERS. A new adapter does not need to be added to this file: --adapter-module takes
"<importable module>:<Backend subclass>" or "<path/to/file.py>:<Backend subclass>" and loads it at start-up.
Each adapter runs in its model's own environment (lerobot 0.6.2 for smolvla, fastwam, evo1, groot_n17; lerobot 0.4.5 for pi05; the
openvla-oft code and its transformers fork for oft; see env/ at the repository root).
The server refuses a GPU listed in BENCH_FORBIDDEN_CARDS (empty by default) and requires CUDA_VISIBLE_DEVICES to be
empty for --device cpu, so a CPU check cannot touch a GPU by accident.
"""
import argparse
import importlib
import importlib.util
import os
import sys
import time

ADAPTERS = {"smolvla": ("smolvla", "SmolVLA"), "fastwam": ("fastwam", "FastWAM"), "evo1": ("evo1", "Evo1"),
            "pi05": ("pi05", "Pi05"), "oft": ("oft", "OFT"), "groot_n17": ("groot_n17", "GrootN17")}


def make(name):
    mod, cls = ADAPTERS[name]
    return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)()


def make_from_module(spec):
    """Backend instance from "<module or file.py>:<ClassName>"."""
    from .base import Backend
    target, _, cls = spec.rpartition(":")
    if not target or not cls:
        raise ValueError(f"--adapter-module expects <module or file.py>:<Backend subclass>, got {spec!r}")
    if target.endswith(".py") or os.sep in target:
        name = os.path.splitext(os.path.basename(target))[0]
        ms = importlib.util.spec_from_file_location(name, target)
        mod = importlib.util.module_from_spec(ms)
        sys.modules[name] = mod
        ms.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    be = getattr(mod, cls)
    if not (isinstance(be, type) and issubclass(be, Backend)):
        raise TypeError(f"{spec}: {cls} is not a Backend subclass")
    return be()


def parse_opts(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, choices=sorted(ADAPTERS), help="a built-in adapter")
    ap.add_argument("--adapter-module", default=None, help="<module or file.py>:<Backend subclass> for your own adapter")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, help="override model dtype (CPU checks: float32)")
    ap.add_argument("--opt", action="append", help="adapter option key=value (repeatable)")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    if bool(a.adapter) == bool(a.adapter_module):
        ap.error("give exactly one of --adapter and --adapter-module")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    forbidden = {x for x in os.environ.get("BENCH_FORBIDDEN_CARDS", "").split(",") if x}
    if a.device == "cpu" and cvd != "":
        sys.exit("--device cpu requires CUDA_VISIBLE_DEVICES= (empty) so that no GPU is touched")
    if a.device != "cpu" and (not cvd or cvd in forbidden):
        sys.exit(f"set CUDA_VISIBLE_DEVICES to one allowed card (forbidden: {sorted(forbidden)})")
    from .base import Server
    from ._lerobot import warmup_obs
    be = make(a.adapter) if a.adapter else make_from_module(a.adapter_module)
    dtype = None
    if a.dtype:
        import torch
        dtype = getattr(torch, a.dtype)
    t0 = time.time()
    be.load(a.ckpt, a.device, dtype=dtype, **parse_opts(a.opt))
    print(f"{be.name} loaded in {time.time() - t0:.1f}s", flush=True)
    if not a.no_warmup:
        for i in range(2):
            be.infer_env(warmup_obs(), "", 0)
    Server(be).serve(a.port)


if __name__ == "__main__":
    main()
