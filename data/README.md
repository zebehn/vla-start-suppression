# Episode records

One row per episode. All records are from simulation.

## `libero_goal/<checkpoint>.csv` (experiment A, Table I)

Seven files: conditions without a mechanism on the 200 shared initial states. `gr00t_n17_ft.csv` and
`gr00t_n17_official.csv` hold all six conditions (1,200 rows each). The other five files hold four conditions (800 rows
each): `none/original`, `none/empty`, `none/neutral`, `none/nonsense`. Their `none/other_task` and
`none/impossible_action` episodes are not bundled; running experiment A produces them (`docs/RUN.md`).

| Column | Meaning |
|---|---|
| `checkpoint` | `gr00t_n17_ft`, `gr00t_n17_official`, `pi05`, `smolvla`, `fastwam`, `evo1`, `openvla_oft` |
| `cond` | `none/original`, `none/empty`, `none/neutral`, `none/nonsense`, `none/other_task`, `none/impossible_action` |
| `g`, `task`, `j` | Initial-state index (50 x task + j), the task of the initial-state file (0-9), the index within the file (0-19) |
| `text` | The instruction string given to the policy |
| `named_task` | `none/other_task` only: the task whose instruction was given |
| `outcome`, `outcome_step` | The first task whose goal became true (`none` if none did) and the step at which it did |
| `first_contact` | The first object or fixture the gripper fingers touched (empty if none; the table is excluded) |
| `max_disp` | Largest end-effector displacement from its position after settling (m) |
| `non_start` | 1 if there was no completion, no contact, and `max_disp` < 0.05 |

## `gr1/` (experiment B, Table II)

- `gr1_tabletop_episodes.csv`: 1,200 rows, the seven conditions without a mechanism and the three prefix conditions on
  the 120 scenes.
- `gr1_gate_episodes.csv`: 1,080 rows, the gate conditions. `run` is `first` (H200, four conditions) or `repeat`
  (A100, five conditions; `gate/template_only` ran only in the repeat).
- `gr1_gate.json`: the gate calibration (threshold `tau_p95`, per-dimension standard deviations, calibration counts).
- `gr1_tabletop_summary.json`, `gr1_gate_summary.json`: output of `gr1/analyze_gr1.py`.

| Column | Meaning |
|---|---|
| `cond`, `mechanism` | Condition `<mechanism>/<instruction>` and its first part |
| `g`, `task`, `task_name`, `seed`, `scene_seed` | Episode index (300000 + 1000 x task + seed), task number (0-5) and name, scene seed within the task, environment seed |
| `gpu`, `gpu_type` | GPU that ran the episode and its type |
| `text` | String given to the policy |
| `gate_d`, `gate_commit`, `hold` | Gate table only: first-call distance, whether the gate let the policy act, whether the robot was held |
| `outcome`, `outcome_step` | First placement (`target`, `distractor_into_target`, `target_into_distractor`, `tie`, or `none`) and its step |
| `designated` | 1 if the environment's success check for the scene's task became true |
| `first_contact`, `first_contact_step` | First object, container, or fixture a hand touched, and the step |
| `max_hand_disp`, `torso_disp` | Larger of the two hand displacements and the torso displacement (m), maximum over the episode |
| `category` | `completion`, `attempt`, `ncmotion` (non-contact motion), or `nonstart` |
| `named_task` | `none/absent_object` only: the task whose instruction was given |
| `n_calls`, `nan_calls`, `steps` | Policy calls, calls with NaN actions, executed steps |
| `scene_sig` | Scene signature; equal across conditions for the same scene |

In the `none/impossible_action`, `none/absent_object`, and `prefix/original` rows, the step columns, the displacements,
and the call count are empty; their `category` was computed by the worker with the same rule.
