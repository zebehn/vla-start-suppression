# VLA Policies Do Not Know When Not to Act: code and episode records

This repository accompanies the paper

> Minsu Jang, Jae-Woo Choi, Youngwoo Yoon, and Dohyung Kim. *VLA Policies Do Not Know When Not to Act: Unintended Task
> Execution and Start Suppression.* arXiv:2610.41609, 2026. https://arxiv.org/abs/2610.41609

The paper measures what vision-language-action (VLA) policies do when the instruction is empty, meaningless, or not
executable: whether the policy starts to act, and which task of the scene it completes. All results are from
simulation.

The repository currently covers two experiments of the paper. The other experiments are listed under "Coming soon".

| Experiment | Paper | Code | Episode records |
|---|---|---|---|
| A. LIBERO-Goal without a mechanism: seven checkpoints, six instruction conditions, 200 initial states | Table I | `libero_goal/` | `data/libero_goal/` |
| B. Humanoid: Fourier GR-1 with GR00T N1.6 on 120 RoboCasa GR-1 tabletop scenes, without a mechanism, with the instruction prefix, and with the first-call gate | Table II | `gr1/` | `data/gr1/` |

## Check the tables of the paper without a GPU

```
pip install numpy
python3 libero_goal/table1.py     # Table I from data/libero_goal/; the computed values of all 7 rows equal the paper
python3 gr1/table2.py             # Table II from data/gr1/; prints "cells equal to the paper: 80 of 80"
```

`table1.py` computes 46 of the 56 values of Table I from the bundled records. The Other-task and Imp. columns of
pi0.5, SmolVLA, Fast-WAM, Evo-1, and OpenVLA-OFT are not in the bundled records; the script prints the paper's values
for them, marked with "*". Running experiment A produces them (`docs/RUN.md`).

## Run the experiments

Both experiments need a GPU, the simulator, and the public checkpoints.

1. `docs/SETUP.md`: environments, simulator versions, checkpoints.
2. `docs/RUN.md`: commands for experiment A and experiment B, and how to compare the output with the paper.

## Contents

| Path | What it holds |
|---|---|
| `libero_goal/wnta/` | Evaluation kit for LIBERO-Goal: episode worker, adapters for six public checkpoints (GR00T N1.7 official, pi0.5, SmolVLA, Fast-WAM, Evo-1, OpenVLA-OFT), wire protocol, adapter conformance test, scorer |
| `libero_goal/table1.py` | Table I from the episode records or from new worker output |
| `gr1/` | GR-1 policy server, episode worker, runner, gate calibration, Table II script (`table2.py`), full analysis (`analyze_gr1.py`), scene list and instruction strings |
| `data/` | Episode records of both experiments (`data/README.md` describes the columns) |
| `env/` | Package versions of the environments used for the paper |
| `docs/` | Setup and run manual |

## Coming soon

Code and episode records for the other experiments of the paper will be added:

- Instruction prefix on LIBERO-Goal (a sentence that tells the policy to stay still without a task).
- Action-difference gate on LIBERO-Goal, including the gate calibration and the recalibration on held-out states.
- No-action fine-tuning (GR00T N1.7 and pi0.5, three seeds each).
- Perturbed LIBERO-Goal scenes (LIBERO-Plus: camera viewpoint, robot initial state, object layout).
- LIBERO-Object positive control.
- Seed replicates and chunk-execution settings.

## What is not included

- Model weights. The code loads public checkpoints from their hubs (`docs/SETUP.md`). Six of the seven checkpoints of
  Table I can be run with this repository. The seventh, GR00T N1.7 (ft), was fine-tuned for the paper and its weights
  are not released; its episode records are included.
- Raw per-step records and videos.

## License

MIT (`LICENSE`). Third-party checkpoints and software keep their own licenses (`THIRD_PARTY_NOTICES.md`).

## Citation

```bibtex
@misc{jang2026vla,
  title  = {{VLA} Policies Do Not Know When Not to Act: Unintended Task Execution and Start Suppression},
  author = {Jang, Minsu and Choi, Jae-Woo and Yoon, Youngwoo and Kim, Dohyung},
  year   = {2026},
  eprint = {2610.41609},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO}
}
```
