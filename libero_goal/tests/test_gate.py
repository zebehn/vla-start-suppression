import unittest

import numpy as np

from wnta import gate as GT


class Calibration(unittest.TestCase):

    def test_deterministic_policy_needs_settle_floor(self):
        rng = np.random.default_rng(5)
        refs = rng.normal(size=(20, 10, 16, 7)).astype(np.float32)
        same = np.repeat(rng.normal(size=(5, 1, 16, 7)), 10, axis=1)  # identical across seeds
        out = GT.calibrate(refs, same, "seed")
        self.assertEqual(out["tau_p95"], 0.0)
        self.assertIn("warning", out)
        obs = same[:, :5] + rng.normal(scale=0.01, size=(5, 5, 16, 7))  # small observation differences
        out2 = GT.calibrate(refs, obs, "settle_obs")
        self.assertGreater(out2["tau_p95"], 0.0)
        self.assertEqual(out2["floor_pairs"], 5 * 10)
        self.assertEqual(out2["floor_source"], "settle_obs")

    def test_profile_records_source(self):
        p = GT.GateProfile(0.5, [1] * 7, [True] * 7, 8, "settle_obs")
        self.assertNotEqual(p.sha256(), GT.GateProfile(0.5, [1] * 7, [True] * 7, 8, "seed").sha256())


if __name__ == "__main__":
    unittest.main()
