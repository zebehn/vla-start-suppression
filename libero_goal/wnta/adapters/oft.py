"""OpenVLA-OFT (moojink/openvla-7b-oft-finetuned-libero-goal) adapter, ported from the policy server of the paper's runs.

The model runs through the official openvla-oft code (commit e4287e9 in the reference run), which needs its own environment:
the transformers fork moojink/transformers-openvla-oft (4.40.1), torch 2.2.0, tensorflow 2.15.0 (image resizing only),
numpy 1.26.4. The checkout is passed with --opt oft_repo=<dir> (or the OFT_REPO environment variable).

- Checkpoint: a local directory, downloaded once with
  `huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-goal --revision c2d0f9f --local-dir <dir>`.
  The official get_vla rewrites config.json of the checkpoint directory and may replace its modeling_prismatic.py with the
  repository copy, so the adapter does not load from the shared hub cache.
- Device: the official code places the model on its module-level DEVICE (cuda:0 when CUDA is visible, else cpu); the
  --device argument must agree with it. The dtype argument is not used (the official loader uses bfloat16).
- Instruction: the official prompt "In: What action should the robot take to {instruction}?\\nOut:" (instruction lowercased).
  Under "" the template remains and only the instruction slot is empty.
- Seed: the L1 regression action head draws no random numbers, so the same input always gives the same chunk
  (stochastic False); the gate floor of this checkpoint in the paper is the settle_obs floor.
- Action: one inference predicts 8 steps; the whole (8, 7) chunk is returned. The gripper output is not normalized and is
  in [0, 1] with 1 = open, which is the wire convention g', so the server does not convert it (wire_gripper True).
"""
import hashlib
import os
import subprocess

import numpy as np

from .base import Backend

STATE_KEYS = ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.gripper"]
IMG_MAIN, IMG_WRIST = "video.image", "video.wrist_image"
PROMPT = "In: What action should the robot take to {}?\nOut:"
CHUNK_LEN = 8
IMAGE_RESOLUTION = [224, 224]
# meta fields that do not depend on the loaded model
META_STATIC = {"checkpoint_revision": "c2d0f9f (reference run)", "empty_handling": "true_empty",
               "prompt_of_empty": PROMPT.format(""), "chunk_len": CHUNK_LEN, "chunk_out": CHUNK_LEN,
               "image_resolution": IMAGE_RESOLUTION, "norm_map_source": "ckpt",
               "seed_injection": "torch.manual_seed(seed) before get_vla_action; the L1 regression head draws no random numbers",
               "stochastic": False, "gate_floor_source": "settle_obs"}


def _sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def _img(flat, key):
    img = np.asarray(flat[key])
    img = img.reshape(img.shape[-3:])
    if img.dtype != np.uint8:
        raise TypeError(f"{key} dtype {img.dtype}, expected uint8")
    return np.ascontiguousarray(img)


def _state(flat):
    s = np.concatenate([np.asarray(flat[k], dtype=np.float64).reshape(-1) for k in STATE_KEYS])
    if s.shape != (8,):
        raise ValueError(f"state dim {s.shape}, expected (8,)")
    return s


class OFT(Backend):
    name = "oft"
    chunk_out = CHUNK_LEN
    wire_gripper = True  # the model's gripper output is already g' in [0, 1] (1 = open)

    def load(self, ckpt, device, dtype=None, oft_repo=None, **opt):
        oft_repo = oft_repo or os.environ.get("OFT_REPO")
        if not oft_repo:
            raise SystemExit("pass --opt oft_repo=<openvla-oft checkout> or set OFT_REPO")
        if not os.path.isdir(ckpt):
            raise SystemExit("--ckpt must be a local checkpoint directory (the official loader rewrites files in it); download with "
                             "huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-goal --revision c2d0f9f --local-dir <dir>")
        import sys
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")  # TensorFlow only resizes images; keep it off the GPU
        import torch
        sys.path.insert(0, oft_repo)
        from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
        from experiments.robot.openvla_utils import (get_action_head, get_processor, get_proprio_projector, get_vla,
                                                     get_vla_action, resize_image_for_policy, OPENVLA_IMAGE_SIZE, DEVICE)
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
        if str(DEVICE).split(":")[0] != str(device).split(":")[0]:
            raise SystemExit(f"openvla-oft places the model on {DEVICE}, but --device is {device}")
        if NUM_ACTIONS_CHUNK != CHUNK_LEN:
            raise SystemExit(f"installed openvla-oft predicts {NUM_ACTIONS_CHUNK} steps, expected {CHUNK_LEN} (LIBERO constants)")
        self.torch, self.get_vla_action, self.resize = torch, get_vla_action, resize_image_for_policy
        self.ckpt, self.oft_repo, self.device, self.image_size = ckpt, oft_repo, DEVICE, OPENVLA_IMAGE_SIZE
        self.norm_type = getattr(ACTION_PROPRIO_NORMALIZATION_TYPE, "name", str(ACTION_PROPRIO_NORMALIZATION_TYPE))
        self.cfg = GenerateConfig(pretrained_checkpoint=ckpt, use_l1_regression=True, use_diffusion=False, use_film=False,
                                  num_images_in_input=2, use_proprio=True, center_crop=True, num_open_loop_steps=CHUNK_LEN,
                                  lora_rank=32, task_suite_name="libero_goal")
        if not torch.cuda.is_available():
            # CPU only: the action head and proprio projector .pt files hold CUDA tensors; map them to the CPU.
            import functools
            torch.load = functools.partial(torch.load, map_location="cpu")
        cwd = os.getcwd()
        os.chdir(oft_repo)  # the official evaluation runs from the repository root (get_vla synchronizes modeling files there)
        try:
            self.vla = get_vla(self.cfg)
        finally:
            os.chdir(cwd)
        self.processor = get_processor(self.cfg)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=PROPRIO_DIM)
        check_unnorm_key(self.cfg, self.vla)
        self.stats_sha = _sha12(os.path.join(ckpt, "dataset_statistics.json"))

    def infer_env(self, flat, text, seed):
        obs = {"full_image": self.resize(_img(flat, IMG_MAIN), self.image_size),
               "wrist_image": self.resize(_img(flat, IMG_WRIST), self.image_size),
               "state": _state(flat)}
        self.torch.manual_seed(int(seed))  # no effect on the L1 regression head
        acts = self.get_vla_action(self.cfg, self.vla, self.processor, obs, text, self.action_head, self.proprio_projector,
                                   use_film=False)
        a = np.asarray(acts, dtype=np.float32)
        if a.shape != (CHUNK_LEN, 7):
            raise ValueError(f"action chunk shape {a.shape}, expected ({CHUNK_LEN}, 7)")
        return a  # gripper already g' (1 = open)

    def prompt_of(self, text, obs=None):
        return PROMPT.format(text.lower())

    def framework(self):
        commit = None
        try:
            commit = subprocess.check_output(["git", "-C", self.oft_repo, "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:  # noqa: BLE001
            commit = None
        import transformers
        return "openvla-oft" + (f" @ {commit}" if commit else "") + f"; transformers {transformers.__version__}"

    def meta(self):
        norm = {"VISUAL": "PRISMATIC_PROCESSOR", "STATE": self.norm_type, "ACTION": self.norm_type}
        return dict(META_STATIC, adapter=self.name, checkpoint=self.ckpt, framework=self.framework(),
                    prompt_of_empty=self.prompt_of(""), norm_map=norm, norm_map_installed_default=dict(norm),
                    norm_map_note=f"STATE and ACTION statistics: dataset_statistics.json[{self.cfg.unnorm_key}] (sha256 {self.stats_sha}); "
                                  "the gripper action dimension is not normalized")
