"""OpenVLA-OFT adapter without the model: class attributes, static meta fields, prompt template, wire conversion, gate profile."""
import unittest

import numpy as np

from tests._util import BENCH  # noqa: F401

from wnta import protocol as P
from wnta.adapters import base
from wnta.adapters import oft
from wnta.gate import GateProfile


class FakeOFT(oft.OFT):
    """No model: returns a fixed 8-step chunk with the gripper already in the wire convention."""

    def infer_env(self, flat, text, seed):
        a = np.zeros((8, 7), np.float32)
        a[:, 6] = 0.97
        return a


class OFTAdapter(unittest.TestCase):
    def test_import_needs_no_torch(self):
        self.assertFalse(hasattr(oft, "torch"))  # torch is imported inside load only
        self.assertEqual(oft.OFT.name, "oft")
        self.assertEqual(oft.OFT.chunk_out, 8)
        self.assertTrue(oft.OFT.wire_gripper)
        self.assertFalse(oft.OFT.supports_mask)
        self.assertTrue(issubclass(oft.OFT, base.Backend))

    def test_static_meta(self):
        m = oft.META_STATIC
        self.assertIs(m["stochastic"], False)
        self.assertEqual(m["empty_handling"], "true_empty")
        self.assertIn(m["empty_handling"], P.EMPTY_HANDLING)
        self.assertEqual(m["prompt_of_empty"], "In: What action should the robot take to ?\nOut:")
        self.assertEqual(m["chunk_len"], 8)
        self.assertEqual(m["chunk_out"], 8)
        self.assertEqual(m["image_resolution"], [224, 224])
        self.assertEqual(m["gate_floor_source"], "settle_obs")
        # the fields filled at load time complete the required set
        loaded = {"adapter", "checkpoint", "framework", "norm_map", "norm_map_installed_default"}
        self.assertEqual(set(P.META_REQUIRED) - set(m) - loaded, set())

    def test_prompt_template(self):
        be = oft.OFT()
        self.assertEqual(be.prompt_of(""), oft.META_STATIC["prompt_of_empty"])
        self.assertEqual(be.prompt_of("Put the bowl on the plate"),
                         "In: What action should the robot take to put the bowl on the plate?\nOut:")

    def test_server_passes_gripper_unchanged(self):
        r = base.Server(FakeOFT()).handle("infer", {"obs": {}, "text": "", "seed": 0})
        self.assertEqual(r["chunk"].shape, (8, 7))
        self.assertTrue(np.allclose(r["chunk"][:, 6], 0.97))
        self.assertEqual(P.validate_chunk(r, 8), [])

    def test_mask_rejected(self):
        with self.assertRaises(ValueError):
            base.Server(FakeOFT()).handle("infer", {"obs": {}, "text": "", "seed": 0, "mask": True})

if __name__ == "__main__":
    unittest.main()
