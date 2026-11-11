"""GR00T N1.7 (official, nvidia/gr00t17-lerobot-libero_goal-640) adapter, ported from the policy server of the paper's
runs (lerobot 0.6.2 environment with the `groot` extra; lerobot 0.4.5 lacks the groot_n1_7_pack_inputs_v1 step of the
checkpoint's preprocessor).

Observation: the two images go to observation.images.image and observation.images.wrist_image (the input names of this
checkpoint). Action: the postprocessor outputs the gripper in environment space (-1 = open); the server converts it to
the wire convention. The whole 16-step chunk is returned, as the reference run did.

Empty instruction: lerobot's prepare_n1_7_language_batch replaces the empty string by "Perform the task.". The adapter
turns this replacement off (option empty=true, the default), so that the empty string reaches the tokenizer; this is the
condition none/empty of the paper. empty=default keeps the framework's replacement.

Base model: config.json of the checkpoint names a base_model_path of the training site. If that path does not exist,
the adapter uses a local snapshot of nvidia/GR00T-N1.7-3B at the same revision (option base=<dir>, or the Hugging Face
cache) and stops if the revision differs.

Seed: torch.manual_seed(seed) right before predict_action_chunk.

This adapter was import-checked only; it was not executed on a GPU in this repository's checks.
"""
import os

from . import _lerobot as L
from .base import Backend

IMG_KEYS = {"video.image": "observation.images.image", "video.wrist_image": "observation.images.wrist_image"}
BASE_REPO = "nvidia/GR00T-N1.7-3B"
BASE_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"


def patch_language(keep_empty, record):
    """Wrap lerobot's N1.7 language preparation. With keep_empty the empty string is not replaced by the default
    sentence. record (a list) receives the strings that are fed to the model."""
    import lerobot.policies.groot.processor_groot as PG
    import lerobot.policies.groot.utils as GU

    orig = GU.prepare_n1_7_language_batch

    def wrapped(language, batch_size, *, formalize_language):
        if keep_empty:
            items = language if isinstance(language, (list, tuple)) else [language]
            items = list(items) if len(items) else [""]
            if len(items) == 1 and batch_size > 1:
                items = items * batch_size
            out = []
            for it in items:
                t = "" if it is None else str(it)
                if formalize_language:
                    t = "".join(ch for ch in t.lower() if ch.isalnum() or ch.isspace() or ch == "_")
                out.append(t)
        else:
            out = orig(language, batch_size, formalize_language=formalize_language)
        record[:] = list(out)
        return out

    GU.prepare_n1_7_language_batch = wrapped
    if hasattr(PG, "prepare_n1_7_language_batch"):
        PG.prepare_n1_7_language_batch = wrapped
    return orig


def resolve_base(configured, base=None):
    """Directory of the base model: the configured path if it exists, else base, else the Hugging Face snapshot of the
    recorded revision. The revision named by the checkpoint must be the recorded one."""
    if os.path.isdir(str(configured)):
        return str(configured)
    if os.path.basename(str(configured).rstrip("/")) != BASE_REVISION:
        raise ValueError(f"base_model_path {configured} does not name revision {BASE_REVISION} of {BASE_REPO}")
    if base:
        if not os.path.isdir(base):
            raise FileNotFoundError(f"base model directory {base} does not exist")
        return base
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE_REPO, revision=BASE_REVISION)


class GrootN17(Backend):
    name = "groot_n17"
    chunk_out = 16

    def load(self, ckpt, device, dtype=None, empty="true", base=None, **opt):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        if empty not in ("true", "default"):
            raise ValueError(f"option empty must be true or default, got {empty!r}")
        self.torch, self.ckpt = torch, ckpt
        self.keep_empty = empty == "true"
        self.lang_record = []
        patch_language(self.keep_empty, self.lang_record)
        d = L.resolve_ckpt(ckpt)
        cfg = PreTrainedConfig.from_pretrained(d)
        cfg.base_model_path = resolve_base(cfg.base_model_path, base)
        cfg.device = device
        self.norm_default = L.installed_default_norm(cfg)
        self.norm_ckpt = L.ckpt_norm_map(d)
        self.policy = GrootPolicy.from_pretrained(d, config=cfg)
        self.policy.config.device = device
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=d,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.chunk_out = int(self.policy.config.chunk_size)

    def infer_env(self, flat, text, seed):
        torch = self.torch
        batch = self.pre(L.to_torch(torch, L.flat_to_batch(flat, text, IMG_KEYS)))
        torch.manual_seed(int(seed))
        with torch.no_grad():
            act = self.policy.predict_action_chunk(batch)
        act = self.post(act)
        return act[0].float().cpu().numpy()

    def prompt_of(self, text, obs=None):
        """The language string that reaches the model for text (after the framework's formalization)."""
        self.pre(L.to_torch(self.torch, L.flat_to_batch(obs if obs is not None else L.warmup_obs(), text, IMG_KEYS)))
        return self.lang_record[0] if self.lang_record else None

    def meta(self):
        c = self.policy.config
        return {"adapter": self.name, "checkpoint": self.ckpt, "checkpoint_revision": "not recorded (model.safetensors sha256 71c8220d03c1d429fe90861869e82189e944aeef9a58aea13bc4f3140dfdb0ef in the reference run)",
                "framework": L.framework(), "empty_handling": "true_empty" if self.keep_empty else "framework_default",
                "prompt_of_empty": self.prompt_of(""), "chunk_len": int(c.chunk_size), "chunk_out": self.chunk_out,
                "image_resolution": None, "image_resolution_note": "set by the checkpoint preprocessor; not recorded", "norm_map": self.norm_ckpt, "norm_map_installed_default": self.norm_default,
                "norm_map_source": "ckpt", "seed_injection": "torch.manual_seed(seed) before predict_action_chunk",
                "stochastic": True}
