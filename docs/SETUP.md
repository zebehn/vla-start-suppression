# Setup

The versions below are the ones used for the paper. Other versions were not tested. A different PyTorch or Transformers
version can change the action chunks of a policy and therefore the episode outcomes.

Checking the tables from the bundled records needs only Python 3.10 or later and NumPy (`README.md`). Everything else on
this page is for running episodes.

## Hardware and operating system

- Linux with an NVIDIA GPU. The paper's runs used one H200 and one A100. The simulators render with EGL on the GPU.
- The policy server and the simulator worker are separate processes in separate Python environments, because the
  policies and the simulators pin conflicting package versions. They communicate over a local port.

## Experiment A: LIBERO-Goal

### 1. Worker environment (simulator)

Python 3.10. Package versions: `env/worker_libero.txt`.

```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -e .
pip install robosuite==1.4.0 mujoco==3.9.0 numpy==1.26.4 bddl==1.0.1 opencv-python==4.6.0.66
pip install -e "<this repository>/libero_goal[wire]"
```

The versions of the remaining LIBERO requirements (robomimic, gym, egl-probe, imageio) were not recorded.

### 2. Policy environment (one per model family)

Install the model code from its upstream instructions at the recorded commit, compare the result with the file in
`env/`, and then install the kit into the same environment with `pip install -e "<this repository>/libero_goal[wire]"`.

| Checkpoint | Adapter name | Model code | Python | Versions |
|---|---|---|---|---|
| pi0.5 | `pi05` | LeRobot 0.4.5 | 3.10 | `env/lerobot045_pi05.txt` |
| GR00T N1.7 (official) | `groot_n17` | LeRobot 0.6.2 (commit 146d8474), extra `groot` | 3.12 | `env/lerobot062.txt` |
| SmolVLA | `smolvla` | LeRobot 0.6.2 (commit 146d8474), extras `smolvla,fastwam,evo1` | 3.12 | `env/lerobot062.txt` |
| Fast-WAM | `fastwam` | same environment as SmolVLA | 3.12 | `env/lerobot062.txt` |
| Evo-1 | `evo1` | same environment as SmolVLA | 3.12 | `env/lerobot062.txt` |
| OpenVLA-OFT | `oft` | openvla-oft at commit e4287e9 with the transformers fork `moojink/transformers-openvla-oft` at commit bc339d9 | 3.10 | `env/openvla_oft.txt` |

The `env/` files list the packages whose versions decide the results. They are not complete freeze files.

### 3. Checkpoints

| Checkpoint | Hub id | Revision |
|---|---|---|
| GR00T N1.7 (official) | `nvidia/gr00t17-lerobot-libero_goal-640` (base model `nvidia/GR00T-N1.7-3B`, revision 2fc962b) | not recorded; `model.safetensors` sha256 71c8220d03c1d429fe90861869e82189e944aeef9a58aea13bc4f3140dfdb0ef |
| pi0.5 | `lerobot/pi05_libero_finetuned` (now named `lerobot/pi05_libero_finetuned_v044`) | 7152f13 |
| SmolVLA | `HuggingFaceVLA/smolvla_libero` | 6721902 |
| Fast-WAM | `ZibinDong/fastwam_libero_uncond_2cam224` | a784645 |
| Evo-1 | `zuoxingdong/evo1_libero` (backbone `OpenGVLab/InternVL3-1B-hf`, revision 014c058) | 515921f |
| OpenVLA-OFT | `moojink/openvla-7b-oft-finetuned-libero-goal` | c2d0f9f |

The adapters accept a hub id or a local directory. Three exceptions:

- GR00T N1.7 (official): `config.json` of the checkpoint names a base model path of the training site. The adapter
  replaces it by a local snapshot of `nvidia/GR00T-N1.7-3B` at revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495, taken
  from the Hugging Face cache or from `--opt base=<dir>`, and stops if the checkpoint names another revision.
  LeRobot 0.4.5 cannot load this checkpoint (its preprocessor step `groot_n1_7_pack_inputs_v1` is missing there).

- Fast-WAM also needs `vae/` and `text_encoder/` of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (revision b8fff73) in the Hugging
  Face cache.
- OpenVLA-OFT must be a local directory, because the official loader rewrites files in it:
  `huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-goal --revision c2d0f9f --local-dir <dir>`.

GR00T N1.7 (ft) is a row of Table I for which this repository holds episode records but no adapter and no weights.

## Experiment B: GR-1 humanoid

Two Python environments from one Isaac-GR00T clone: its `.venv` for the policy server and its `robocasa_uv` environment
for the simulator worker.

| Component | Version |
|---|---|
| Isaac-GR00T | tag `n1.6.1-release`, commit `5dc80c4afd726b34faad1d8f7e007a13b34e4c88` |
| robocasa-gr1-tabletop-tasks | commit `4840e671596f93ca03651524b9f72ffb1aadfeff` (submodule of Isaac-GR00T) |
| robosuite | v1.5.1 (installed by the setup script) |
| Checkpoint | `nvidia/GR00T-N1.6-3B`, revision `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61` |

```
export GR1_ENV=<directory for the Isaac-GR00T clone> GR1_CKPT=<directory for the checkpoint>
git clone --branch n1.6.1-release --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "$GR1_ENV"
git -C "$GR1_ENV" rev-parse HEAD     # 5dc80c4afd726b34faad1d8f7e007a13b34e4c88
cd "$GR1_ENV" && uv sync                                                               # server environment (.venv)
bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh   # robocasa_uv environment
hf download nvidia/GR00T-N1.6-3B --revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 --local-dir "$GR1_CKPT"
( cd "$GR1_CKPT" && sha256sum *.safetensors *.json )   # compare with CKPT_SHA256 in gr1/gr1_common.py
```

The package versions inside the two environments are the ones pinned by the upstream setup at that commit; they were
not recorded separately. If the server environment needs a CUDA toolkit for compilation, set `CUDA_HOME` before running.

## Licenses of the checkpoints

Some checkpoints have non-commercial or model-specific terms (`THIRD_PARTY_NOTICES.md`). Read them before use.
