"""Episode loop logic with a fake scene and a fake policy (no simulator)."""
import unittest

import numpy as np

from tests._util import BENCH  # noqa: F401  (puts the package on the path)

from wnta import spec as S  # noqa: E402
from wnta import gate as GT  # noqa: E402


class FakeScene:
    card = 9
    languages = S.TASK_LANGUAGE

    def __init__(self, complete_at=None, contact_at=None):
        self.t, self.complete_at, self.contact_at, self.pos = 0, complete_at, contact_at, np.zeros(3)
        self.seeds = []

    def reset_to(self, state):
        self.seeds.append(int(np.random.get_state()[1][0]))
        self.t, self.pos = 0, np.zeros(3)
        return self._obs()

    def _obs(self):
        return {"robot0_eef_pos": self.pos.copy()}

    def goals_true(self):
        return [8] if (self.complete_at is not None and self.t >= self.complete_at) else []

    def finger_contacts(self):
        return {"plate_1"} if (self.contact_at is not None and self.t >= self.contact_at) else set()

    def step_wire(self, a):
        self.t += 1
        self.pos = self.pos + np.asarray(a[:3]) * 0.01
        return self._obs()

    def step_env(self, a):
        self.t += 1
        return self._obs()


class FakeClient:
    def __init__(self, empty_same=False):
        self.calls, self.empty_same = [], empty_same

    def infer(self, obs, text, seed):
        self.calls.append((text, seed))
        base = 0.0 if (text == "" or self.empty_same) else 1.0
        ch = np.full((16, 7), base, np.float32)
        ch[:, 6] = 1.0
        return ch, False


class WorkerLoop(unittest.TestCase):
    def setUp(self):
        import wnta.env_libero as EL
        EL.process_obs = lambda obs: obs  # the fake observation goes through unchanged
        from wnta import worker as W
        W.process_obs = lambda obs: obs
        self.W = W
        self.gate = GT.GateProfile(0.5, [1.0] * 7, [True] * 7, 16, "seed")

    def test_seeds_text_and_completion(self):
        env, cl = FakeScene(complete_at=20, contact_at=10), FakeClient()
        r = self.W.run_episode(env, cl, "none/original", 8, 3, None)
        g = S.gidx(8, 3)
        self.assertEqual(r["outcome"], 8)
        self.assertEqual(r["outcome_step"], 20)
        self.assertEqual(r["first_contact"], "plate_1")
        self.assertEqual(r["first_contact_step"], 10)
        self.assertEqual([s for _, s in cl.calls], [S.call_seed(g, i) for i in range(3)])  # calls at t = 0, 8, 16
        self.assertEqual(cl.calls[0][0], S.TASK_LANGUAGE[8])
        np.random.seed(S.scene_seed(g))
        self.assertEqual(env.seeds[0], int(np.random.get_state()[1][0]))

    def test_gate_hold_and_commit(self):
        env, cl = FakeScene(), FakeClient(empty_same=True)
        r = self.W.run_episode(env, cl, "gate/neutral", 0, 0, None, gate=self.gate)
        self.assertEqual((r["gate_commit"], r["hold"], r["n_calls"], r["steps"], r["outcome"]), (False, True, 2, 300, "none"))
        self.assertTrue(r["noaction"])
        env, cl = FakeScene(), FakeClient()
        r = self.W.run_episode(env, cl, "gate/original", 0, 0, None, gate=self.gate)
        self.assertTrue(r["gate_commit"])
        self.assertEqual(cl.calls[1], ("", S.call_seed(0, 0)))  # reference call: empty text, same seed
        self.assertEqual(r["n_calls"], 2 + (300 // 8 + (1 if 300 % 8 else 0)) - 1)

    def test_q_without_gate_refused(self):
        with self.assertRaises(ValueError):
            self.W.run_episode(FakeScene(), FakeClient(), "gate/empty", 0, 0, None)


if __name__ == "__main__":
    unittest.main()
