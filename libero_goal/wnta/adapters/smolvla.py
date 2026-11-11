"""SmolVLA (HuggingFaceVLA/smolvla_libero) adapter, ported from the policy server of the paper's runs (lerobot 0.6.2).

Empty instruction: the preprocessor substitutes nothing; its newline step appends "\\n" (framework default kept).
Seed: torch.manual_seed(seed) right before predict_action_chunk (diffusion noise comes from the global generator).
"""
import numpy as np

from . import _lerobot as L
from .base import Backend


class SmolVLA(Backend):
    name = "smolvla"
    chunk_out = 16

    def load(self, ckpt, device, dtype=None, **opt):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.torch, self.ckpt = torch, ckpt
        d = L.resolve_ckpt(ckpt)
        cfg = PreTrainedConfig.from_pretrained(d)
        cfg.device = device
        self.norm_default = L.installed_default_norm(cfg)
        self.policy = SmolVLAPolicy.from_pretrained(d, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=d,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.norm_ckpt = L.ckpt_norm_map(d)
        self._tok = None

    def _batch(self, flat, text):
        return self.pre(L.to_torch(self.torch, L.flat_to_batch(flat, text)))

    def infer_env(self, flat, text, seed):
        torch = self.torch
        batch = self._batch(flat, text)
        self.policy.reset()
        torch.manual_seed(int(seed))
        with torch.no_grad():
            act = self.policy.predict_action_chunk(batch)
        return self.post(act)[0].float().cpu().numpy()

    def prompt_of(self, text, obs=None):
        b = self._batch(obs if obs is not None else L.warmup_obs(), text)
        ids = [k for k in b if "language" in k and k.endswith("tokens")][0]
        msk = [k for k in b if "language" in k and "mask" in k][0]
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.policy.config.vlm_model_name)
        i, m = b[ids][0].cpu(), b[msk][0].cpu().bool()
        return self._tok.decode(i[m])

    def meta(self):
        c = self.policy.config
        return {"adapter": self.name, "checkpoint": self.ckpt, "checkpoint_revision": self.ckpt.rstrip("/").split("/")[-1],
                "framework": L.framework(), "empty_handling": "true_empty", "prompt_of_empty": self.prompt_of(""),
                "chunk_len": int(c.chunk_size), "chunk_out": self.chunk_out, "image_resolution": list(getattr(c, "resize_imgs_with_padding", None) or []),
                "norm_map": self.norm_ckpt, "norm_map_installed_default": self.norm_default, "norm_map_source": "ckpt",
                "seed_injection": "torch.manual_seed(seed) before predict_action_chunk", "stochastic": True}
