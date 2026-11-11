"""EVO1 (zuoxingdong/evo1_libero) adapter, ported from the policy server of the paper's runs (lerobot 0.6.2 policies/evo1).

The instruction follows two image placeholders with no template, so "" leaves only the placeholders.
Seed: the flow-matching start noise is torch.rand from the global generator; torch.manual_seed(seed) before each call.
opt vlm=<path or hub id> pins the backbone (the reference run used OpenGVLab/InternVL3-1B-hf @ 014c058 from a local snapshot).
"""
import json
import re

from . import _lerobot as L
from .base import Backend

OUR_IMG_KEYS = ("observation.images.image", "observation.images.image2")


class Evo1(Backend):
    name = "evo1"
    chunk_out = 16

    def load(self, ckpt, device, dtype=None, vlm=None, norm_map=None, **opt):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        import lerobot.policies.evo1.configuration_evo1  # noqa: F401
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.evo1.modeling_evo1 import Evo1Policy
        self.torch, self.ckpt = torch, ckpt
        d = L.resolve_ckpt(ckpt)
        cfg = PreTrainedConfig.from_pretrained(d)
        cfg.device = device
        if vlm:
            cfg.vlm_model_name = vlm
        if dtype is not None:
            cfg.dtype = dtype
        vis = sorted(k for k, f in cfg.input_features.items() if str(getattr(f.type, "value", f.type)).upper() == "VISUAL")
        if not set(OUR_IMG_KEYS) <= set(vis):
            raise SystemExit(f"checkpoint visual keys {vis} do not contain {OUR_IMG_KEYS}")
        self.norm_default = L.installed_default_norm(cfg)
        self.norm_ckpt = L.ckpt_norm_map(d)
        self.norm_override = json.loads(norm_map) if norm_map else None
        self.policy = Evo1Policy.from_pretrained(d, config=cfg)
        self.policy.to(device).eval()
        ov = {"device_processor": {"device": device}}
        if self.norm_override:
            ov["normalizer_processor"] = {"norm_map": dict(self.norm_override)}
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=d, preprocessor_overrides=ov)

    def infer_env(self, flat, text, seed):
        torch = self.torch
        batch = self.pre(L.to_torch(torch, L.flat_to_batch(flat, text)))
        self.policy.reset()
        torch.manual_seed(int(seed))
        with torch.no_grad():
            act = self.policy.predict_action_chunk(batch)
        a = self.post(act)[0].float().cpu().numpy()
        if a.ndim != 2 or a.shape[1] != 7:
            raise ValueError(f"postprocessed action shape {a.shape}, expected (T, 7)")
        return a

    def prompt_of(self, text, obs=None):
        p = self.policy.model.embedder._build_multimodal_prompts([[1, 1]], [text])[0]
        return re.sub(r"(<IMG_CONTEXT>)+", lambda m: f"<IMG_CONTEXT>x{len(m.group(0)) // len('<IMG_CONTEXT>')}", p)

    def meta(self):
        c = self.policy.config
        return {"adapter": self.name, "checkpoint": self.ckpt, "checkpoint_revision": "515921f (reference run)",
                "framework": L.framework(), "backbone": c.vlm_model_name, "empty_handling": "true_empty",
                "prompt_of_empty": self.prompt_of(""), "chunk_len": int(c.chunk_size), "chunk_out": self.chunk_out,
                "image_resolution": list(c.image_resolution), "norm_map": self.norm_override or self.norm_ckpt,
                "norm_map_installed_default": self.norm_default, "norm_map_source": "override" if self.norm_override else "ckpt",
                "seed_injection": "torch.manual_seed(seed) before predict_action_chunk", "stochastic": True}
