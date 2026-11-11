"""pi0.5 (lerobot/pi05_libero_finetuned) adapter, ported from the policy server of the paper's runs (lerobot 0.4.5 environment).

Returns the whole 50-step chunk, as the reference run did. compile_model is turned off (first-call compilation exceeded
the worker timeout). Seed: torch.manual_seed(seed) before predict_action_chunk (diffusion noise is torch.normal).
"""
from . import _lerobot as L
from .base import Backend


class Pi05(Backend):
    name = "pi05"
    chunk_out = 50

    def load(self, ckpt, device, dtype=None, compile_model="0", **opt):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        import lerobot.policies.pi05.configuration_pi05  # noqa: F401
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        self.torch, self.ckpt = torch, ckpt
        d = L.resolve_ckpt(ckpt)
        cfg = PreTrainedConfig.from_pretrained(d)
        cfg.compile_model = str(compile_model) == "1"
        self.norm_default = L.installed_default_norm(cfg)
        self.norm_ckpt = L.ckpt_norm_map(d)
        self.policy = PI05Policy.from_pretrained(d, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=d,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self._tok = None

    def infer_env(self, flat, text, seed):
        torch = self.torch
        batch = self.pre(L.to_torch(torch, L.flat_to_batch(flat, text)))
        torch.manual_seed(int(seed))
        with torch.no_grad():
            act = self.policy.predict_action_chunk(batch)
        return self.post(act)[0].float().cpu().numpy()

    def prompt_of(self, text, obs=None):
        b = self.pre(L.to_torch(self.torch, L.flat_to_batch(obs if obs is not None else L.warmup_obs(), text)))
        ids = [k for k in b if "language" in k and k.endswith("tokens")]
        msk = [k for k in b if "language" in k and "mask" in k]
        if not ids or not msk:
            return None
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        i, m = b[ids[0]][0].cpu(), b[msk[0]][0].cpu().bool()
        return self._tok.decode(i[m])

    def meta(self):
        c = self.policy.config
        return {"adapter": self.name, "checkpoint": self.ckpt, "checkpoint_revision": "7152f13 (reference run)",
                "framework": L.framework(), "empty_handling": "true_empty", "prompt_of_empty": self.prompt_of(""), "prompt_note": "the prompt embeds the discretized state; prompt_of_empty uses a zero observation",
                "chunk_len": int(c.chunk_size), "chunk_out": self.chunk_out, "image_resolution": [224, 224],
                "norm_map": self.norm_ckpt, "norm_map_installed_default": self.norm_default, "norm_map_source": "ckpt",
                "seed_injection": "torch.manual_seed(seed) before predict_action_chunk", "stochastic": True}
