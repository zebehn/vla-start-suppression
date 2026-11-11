"""Adapter layer without models: registry, lazy imports, server conversion (numpy only)."""
import unittest

import numpy as np

from tests._util import BENCH  # noqa: F401

from wnta import protocol as P
from wnta.adapters import base, serve


class Wire(base.Backend):
    name, chunk_out, wire_gripper = "wire", 8, True

    def infer_env(self, flat, text, seed):
        a = np.zeros((8, 7), np.float32)
        a[:, 6] = 0.25
        return a

    def meta(self):
        return {}


class Env(base.Backend):
    name, chunk_out = "env", 16

    def infer_env(self, flat, text, seed):
        a = np.zeros((50, 7), np.float64)
        a[:, 6] = -1.0  # open in environment space
        return a

    def meta(self):
        return {}


class AdapterLayer(unittest.TestCase):
    def test_registry_imports_without_frameworks(self):
        # module import must not need lerobot or torch (they are imported inside load)
        for name in serve.ADAPTERS:
            be = serve.make(name)
            self.assertEqual(be.name, name)
        self.assertEqual(serve.make("pi05").chunk_out, 50)

    def test_server_conversion(self):
        obs = {}
        r = base.Server(Env()).handle("infer", {"obs": obs, "text": "", "seed": 0})
        self.assertEqual(r["chunk"].shape, (16, 7))
        self.assertEqual(r["chunk"].dtype, np.float32)
        self.assertTrue(np.all(r["chunk"][:, 6] == 1.0))  # a = -1 (open) -> g' = 1
        r = base.Server(Wire()).handle("infer", {"obs": obs, "text": "", "seed": 0})
        self.assertTrue(np.all(r["chunk"][:, 6] == 0.25))  # already wire convention, untouched
        self.assertEqual(P.validate_chunk(r, 8), [])

    def test_short_chunk_is_an_error(self):
        class Short(Env):
            def infer_env(self, flat, text, seed):
                return np.zeros((8, 7))
        with self.assertRaises(ValueError):
            base.Server(Short()).handle("infer", {"obs": {}, "text": "", "seed": 0})

if __name__ == "__main__":
    unittest.main()
