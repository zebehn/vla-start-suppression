"""LIBERO-Goal scene for spec v0 (LIBERO worker environment: robosuite 1.4.0, mujoco 3.9.0, LIBERO 8f1084e).

One environment is built from task 0's bddl; a task file's init state is loaded with set_init_state; the goal predicates
of all 10 tasks are evaluated every step; finger contacts are mapped to objects and fixtures (table excluded, no object
excluded). Observation and action conversions match the worker of the paper's runs.

Render device: robosuite's EGL renderer uses device 0 unless told otherwise, so the worker requires
MUJOCO_EGL_DEVICE_ID and CUDA_VISIBLE_DEVICES to be set to the same single card (protocol control "render device").
BENCH_FORBIDDEN_CARDS (comma list, e.g. "0") lets a site refuse cards.
"""
import math
import os

import numpy as np

from . import spec as S

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def check_render_device():
    egl, cvd = os.environ.get("MUJOCO_EGL_DEVICE_ID"), os.environ.get("CUDA_VISIBLE_DEVICES")
    forbidden = {x for x in os.environ.get("BENCH_FORBIDDEN_CARDS", "").split(",") if x}
    if not egl or not cvd or egl != cvd or "," in cvd:
        raise RuntimeError("set MUJOCO_EGL_DEVICE_ID and CUDA_VISIBLE_DEVICES to the same single card before creating envs")
    if egl in forbidden:
        raise RuntimeError(f"card {egl} is forbidden on this host (BENCH_FORBIDDEN_CARDS)")
    return int(egl)


# --- vendored helpers -------------------------------------------------------------------------------------------
# quat2axisangle: robosuite transform_utils (MIT), as copied in Isaac-GR00T gr00t/eval/sim/LIBERO/libero_env.py @ 9cce05c.
# normalize_gripper_action / invert_gripper_action: Isaac-GR00T (Apache-2.0) @ 9cce05c, same file.

def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


def process_obs(obs):
    """Wire observation (spec v0): both views flipped on both axes, state as (1, 1, d) float32."""
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"].copy())
    flat = {"video.image": obs["agentview_image"][::-1, ::-1], "video.wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1],
            "state.x": [xyz[0]], "state.y": [xyz[1]], "state.z": [xyz[2]], "state.roll": [rpy[0]], "state.pitch": [rpy[1]],
            "state.yaw": [rpy[2]], "state.gripper": obs["robot0_gripper_qpos"]}
    out = {}
    for k, v in flat.items():
        a = np.ascontiguousarray(np.asarray(v))
        a = a.astype(np.uint8) if k.startswith("video") else a.astype(np.float32)
        out[k] = a[None, None]
    return out


class GoalScene:
    def __init__(self, cam=256):
        self.card = check_render_device()
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        from libero.libero.utils import get_libero_path
        suite = benchmark.get_benchmark_dict()[S.SCENE["suite"]]()
        self.suite = suite

        def bddl(k):
            t = suite.get_task(k)
            return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl(S.SCENE["env_bddl_task"]), camera_heights=cam, camera_widths=cam,
                                      ignore_done=True)
        self.env.seed(0)
        self.env.reset()
        self.e = self.env.env
        self.goals = {k: robosuite_parse_problem(bddl(k))["goal_state"] for k in range(S.N_TASKS)}
        self.languages = [suite.get_task(k).language for k in range(S.N_TASKS)]
        if self.languages != S.TASK_LANGUAGE:
            raise RuntimeError("installed LIBERO task language differs from spec v0")
        self._contact_maps()

    def init_states(self, k):
        return np.asarray(self.suite.get_task_init_states(k))

    def _contact_maps(self):
        m = self.e.sim.model
        g = self.e.robots[0].gripper
        names = g.important_geoms["left_finger"] + g.important_geoms["right_finger"]
        self.finger_geoms = {m.geom_name2id(n) for n in names}
        roots = {k: self.e.obj_body_id[k] for k in list(self.e.objects_dict) + list(self.e.fixtures_dict)}
        self.geom2obj = {}
        for gid in range(m.ngeom):
            b = m.geom_bodyid[gid]
            chain = set()
            while b > 0:
                chain.add(b)
                b = m.body_parentid[b]
            for k, rb in roots.items():
                if rb in chain:
                    self.geom2obj[gid] = k
                    break

    def goal_true(self, k):
        return all(bool(self.e._eval_predicate(s)) for s in self.goals[k])

    def goals_true(self):
        return [k for k in range(S.N_TASKS) if self.goal_true(k)]

    def finger_contacts(self):
        d = self.e.sim.data
        hit = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.finger_geoms and g2 in self.geom2obj:
                hit.add(self.geom2obj[g2])
            elif g2 in self.finger_geoms and g1 in self.geom2obj:
                hit.add(self.geom2obj[g1])
        return hit

    def reset_to(self, init_state, settle_steps=S.SETTLE_STEPS, keep=()):
        """reset, load the init state, take settle_steps no-op steps (gripper open). Returns the last observation and,
        if keep is given, the observations after those settle steps (used by the settle_obs gate floor)."""
        self.env.reset()
        obs = self.env.set_init_state(init_state)
        kept = {}
        for st in range(1, settle_steps + 1):
            obs, _, _, _ = self.env.step(list(S.SCENE["settle_action"]))
            if st in keep:
                kept[st] = obs
        return (obs, kept) if keep else obs

    def step_wire(self, act7, binarize=True):
        """One wire action (gripper g' in [0, 1]) -> environment step."""
        v = np.asarray(act7, dtype=np.float64).copy()
        v = normalize_gripper_action(v, binarize=binarize)
        v = invert_gripper_action(v)
        obs, _, _, _ = self.env.step(v)
        return obs

    def step_env(self, act7):
        obs, _, _, _ = self.env.step(list(act7))
        return obs

    def close(self):
        self.env.close()
