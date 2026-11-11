"""Helpers shared by the LeRobot-based adapters (from the policy servers of the paper's runs)."""
import dataclasses
import json
import os

import numpy as np

STATE_KEYS = ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.gripper"]
IMG_KEYS = {"video.image": "observation.images.image", "video.wrist_image": "observation.images.image2"}


def flat_to_batch(flat, text, img_keys=IMG_KEYS):
    """Wire observation -> LeRobot batch (numpy): two images (1, 3, H, W) float32 in [0, 1], state (1, 8), task [text]."""
    out = {}
    for src, dst in img_keys.items():
        img = np.asarray(flat[src])
        img = img.reshape(img.shape[-3:])
        if img.dtype != np.uint8:
            raise TypeError(f"{src} dtype {img.dtype}, expected uint8")
        out[dst] = (np.ascontiguousarray(img.transpose(2, 0, 1))[None].astype(np.float32) / 255.0)
    state = np.concatenate([np.asarray(flat[k], dtype=np.float32).reshape(-1) for k in STATE_KEYS])
    if state.shape != (8,):
        raise ValueError(f"state dim {state.shape}, expected (8,)")
    out["observation.state"] = state[None]
    out["task"] = [text]
    return out


def to_torch(torch, b):
    return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in b.items()}


def resolve_ckpt(ckpt):
    """Local directory of a checkpoint given as a path or a hub id (uses the local cache when offline)."""
    if os.path.isdir(ckpt):
        return ckpt
    from huggingface_hub import snapshot_download
    return snapshot_download(ckpt)


def ckpt_norm_map(ckpt_dir):
    p = os.path.join(ckpt_dir, "policy_preprocessor.json")
    if not os.path.exists(p):
        return None
    for st in json.load(open(p)).get("steps", []):
        if "normaliz" in st.get("registry_name", "") and "norm_map" in st.get("config", {}):
            return {str(k): str(v) for k, v in st["config"]["norm_map"].items()}
    return None


def installed_default_norm(cfg):
    """Default normalization_mapping of the installed framework's config class (what the framework expects today)."""
    f = {x.name: x for x in dataclasses.fields(type(cfg))}.get("normalization_mapping")
    if f is None:
        return None
    d = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
    return {str(k): str(getattr(v, "value", v)) for k, v in d.items()}


def framework():
    from importlib.metadata import version
    import lerobot
    root = os.path.dirname(os.path.dirname(os.path.dirname(lerobot.__file__)))
    commit = None
    head = os.path.join(root, ".git", "HEAD")
    if os.path.exists(head):
        import subprocess
        try:
            commit = subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:  # noqa: BLE001
            commit = None
    return f"lerobot {version('lerobot')}" + (f" @ {commit}" if commit else "")


def warmup_obs():
    f = {k: np.zeros((1, 1, 256, 256, 3), np.uint8) for k in IMG_KEYS}
    for k in STATE_KEYS[:-1]:
        f[k] = np.zeros((1, 1, 1), np.float32)
    f["state.gripper"] = np.zeros((1, 1, 2), np.float32)
    return f
