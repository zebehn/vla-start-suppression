"""FastWAM (ZibinDong/fastwam_libero_uncond_2cam224) adapter, ported from the policy server of the paper's runs.

Normalization: the checkpoint (saved before the LeRobot integration) stores VISUAL MEAN_STD(0.5, 0.5), and the installed
framework maps images to [-1, 1] again inside the model, so as saved the video VAE receives [-3, 1] and original-instruction
success is 0/20. The adapter overrides VISUAL to IDENTITY and keeps the stored STATE/ACTION MIN_MAX (opt visual_identity=0
reproduces the saved configuration for diagnosis). Seed: batch["seed"] feeds the model's torch.Generator, and the global
generator is seeded too. Prompt: "A video recorded from a robot's point of view executing the following instruction: {task}".
"""
import os

from . import _lerobot as L
from .base import Backend

OVERRIDE = {"VISUAL": "IDENTITY", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}


class FastWAM(Backend):
    name = "fastwam"
    chunk_out = 16

    def load(self, ckpt, device, dtype=None, visual_identity="1", **opt):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        import lerobot.policies.fastwam.configuration_fastwam  # noqa: F401
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
        self.torch, self.ckpt = torch, ckpt
        d = L.resolve_ckpt(ckpt)
        cfg = PreTrainedConfig.from_pretrained(d)
        cfg.device = device
        cfg.tokenizer_model_id = os.path.join(d, "google", "umt5-xxl")  # the stored id "." fails outside the ckpt directory
        if dtype is not None:
            cfg.dtype = dtype  # CPU checks need float32 (time embedding is computed under a CUDA-only autocast block)
        self.norm_default = L.installed_default_norm(cfg)
        self.norm_ckpt = L.ckpt_norm_map(d)
        self.override = str(visual_identity) == "1"
        self.policy = FastWAMPolicy.from_pretrained(d, config=cfg, strict=False)
        self.policy.to(device).eval()
        ov = {"device_processor": {"device": device}}
        if self.override:
            ov["normalizer_processor"] = {"norm_map": dict(OVERRIDE)}
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=d, preprocessor_overrides=ov)

    def _batch(self, flat, text, seed):
        b = self.pre(L.to_torch(self.torch, L.flat_to_batch(flat, text)))
        b["seed"] = int(seed)
        return b

    def infer_env(self, flat, text, seed):
        torch = self.torch
        batch = self._batch(flat, text, seed)
        self.policy.reset()
        torch.manual_seed(int(seed))
        with torch.no_grad():
            act = self.policy.predict_action_chunk(batch)
        return self.post(act)[0].float().cpu().numpy()

    def prompt_of(self, text, obs=None):
        from lerobot.policies.fastwam.modeling_fastwam import _prompt_from_batch
        p = _prompt_from_batch(batch=self._batch(obs if obs is not None else L.warmup_obs(), text, 0), config=self.policy.config)
        return p[0] if isinstance(p, (list, tuple)) else p  # the framework returns one prompt per batch element

    def meta(self):
        c = self.policy.config
        return {"adapter": self.name, "checkpoint": self.ckpt, "checkpoint_revision": "a784645 (reference run)",
                "framework": L.framework(), "empty_handling": "true_empty", "prompt_of_empty": self.prompt_of(""),
                "chunk_len": int(c.action_horizon), "chunk_out": self.chunk_out, "image_resolution": [224, 224],
                "norm_map": dict(OVERRIDE) if self.override else self.norm_ckpt, "norm_map_ckpt": self.norm_ckpt,
                "norm_map_installed_default": self.norm_default, "norm_map_source": "override" if self.override else "ckpt",
                "norm_map_note": "VISUAL forced to IDENTITY: the stored MEAN_STD would be applied twice by the installed model "
                                 "(0/20 original-instruction success); STATE/ACTION keep the stored MIN_MAX statistics",
                "seed_injection": "batch['seed'] (model torch.Generator) and torch.manual_seed(seed)", "stochastic": True}
