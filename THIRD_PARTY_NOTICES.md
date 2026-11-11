# Third-party models and software

This repository contains no model weights. The code loads the checkpoints below from their public hubs; each checkpoint
stays under its own license. The table records what the model cards and repositories state (checked between 2026-10-20
and 2026-11-03). It is not a legal assessment.

| Checkpoint | Used for | License as documented | Notes |
|---|---|---|---|
| `lerobot/pi05_libero_finetuned` (now `lerobot/pi05_libero_finetuned_v044`) | pi0.5, experiment A | gemma (model card); Gemma terms of use apply | Snapshot 7152f13. |
| `HuggingFaceVLA/smolvla_libero` | SmolVLA, experiment A | apache-2.0 (model card) | Snapshot 6721902. Backbone `HuggingFaceTB/SmolVLM2-500M-Instruct`, apache-2.0. |
| `ZibinDong/fastwam_libero_uncond_2cam224` | Fast-WAM, experiment A | weights: not stated in the model card or README | Revision a784645. Code (`yuantianyuan01/FastWAM`) uses the MIT text with a carve-out for third-party files. Bundled `Wan-AI/Wan2.2-TI2V-5B-Diffusers` and `google/umt5-xxl` are Apache-2.0. |
| `zuoxingdong/evo1_libero` | Evo-1, experiment A | `license: other`, `license_name: qwen` (model card) | Revision 515921f. The backbone's language component is Qwen2.5-0.5B (Apache-2.0). Evo-1 and InternVL3 code are MIT. |
| `moojink/openvla-7b-oft-finetuned-libero-goal` | OpenVLA-OFT, experiment A | `license: mit` (model card) | Revision c2d0f9f. The weights are a fine-tune of OpenVLA-7B, which its authors describe as derived from Llama 2 and subject to the Llama 2 Community License. Used for evaluation only, not redistributed. |
| `nvidia/gr00t17-lerobot-libero_goal-640` | GR00T N1.7 (official), experiment A | apache-2.0 (model card) | Revision not recorded. Loads the base model `nvidia/GR00T-N1.7-3B` (revision 2fc962b); read the license on its model card before use. |
| `nvidia/GR00T-N1.6-3B` | GR00T N1.6 (GR-1), experiment B | NVIDIA License, the NVIDIA One-Way Noncommercial License linked from the model card | Revision d0814e7. Used as released for evaluation, not redistributed. |

The episode records in `data/libero_goal/` also cover one checkpoint that the code here does not load: a fine-tune of
`nvidia/GR00T-N1.7-LIBERO` (NVIDIA Open Model License Agreement) that is not released.

Software used by the code (installed separately, not redistributed): LIBERO (MIT), robosuite (MIT; v1.4.0 for LIBERO,
v1.5.1 for the GR-1 runs), RoboCasa GR-1 tabletop tasks (`robocasa/robocasa-gr1-tabletop-tasks`, commit 4840e67, MIT),
MuJoCo (Apache-2.0), LeRobot (Apache-2.0), Isaac-GR00T (tag n1.6.1-release, commit 5dc80c4; it has an "NVIDIA License"
file whose section 3.3 limits use to non-commercial research), openvla-oft (MIT, commit e4287e9), the transformers fork
`moojink/transformers-openvla-oft` (Apache-2.0, commit bc339d9), TensorFlow (Apache-2.0), PyTorch (BSD-style), NumPy
(BSD-3-Clause). Check each project's license for the version you install.

`libero_goal/wnta/spec/init_states_v0.npz` holds LIBERO-Goal initial states exported from the LIBERO task files
(LIBERO, MIT). `libero_goal/wnta/env_libero.py` contains short helper functions from robosuite (MIT) and Isaac-GR00T
(Apache-2.0), marked in the file.
