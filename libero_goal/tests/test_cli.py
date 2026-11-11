"""Command line: score on the bundled sample and on an empty directory, and loading an adapter from a user module (numpy only)."""
import csv
import json
import os
import tempfile
import unittest

from tests._util import BENCH

from wnta import __main__ as CLI
from wnta.adapters import serve

SAMPLE = os.path.join(BENCH, "examples", "sample_episodes")


class Score(unittest.TestCase):
    def test_empty_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(CLI.main(["score", d]), 2)
            self.assertEqual(CLI.main(["score", os.path.join(d, "missing")]), 2)

    def test_sample_equals_episode_table(self):
        """The scorer applied to the bundled Evo-1 sample gives the counts of data/libero_goal/evo1.csv."""
        table = os.path.join(BENCH, "..", "data", "libero_goal", "evo1.csv")
        want = {}
        with open(table, newline="") as f:
            for r in csv.DictReader(f):
                w = want.setdefault(r["cond"], {"n": 0, "completion_n": 0, "success_n": 0, "non_start_n": 0})
                w["n"] += 1
                w["completion_n"] += r["outcome"] != "none"
                w["success_n"] += r["outcome"] == r["task"]
                w["non_start_n"] += int(r["non_start"])
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "scores.json")
            self.assertEqual(CLI.main(["score", SAMPLE, "--out", out]), 0)
            with open(out) as f:
                got = json.load(f)
        self.assertEqual(got["problems"], [])
        for c in ("none/original", "none/empty"):
            for k, v in want[c].items():
                self.assertEqual(got["conditions"][c][k], v, (c, k))


class AdapterModule(unittest.TestCase):
    def test_load_backend_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "my_adapter.py")
            with open(p, "w") as f:
                f.write("from wnta.adapters.base import Backend\n"
                        "class MyPolicy(Backend):\n    name = 'mypolicy'\n    chunk_out = 8\n"
                        "    def infer_env(self, flat_obs, text, seed):\n        import numpy as np\n        return np.zeros((8, 7), np.float32)\n"
                        "    def meta(self):\n        return {}\n")
            be = serve.make_from_module(p + ":MyPolicy")
            self.assertEqual((be.name, be.chunk_out), ("mypolicy", 8))
            with self.assertRaises(AttributeError):
                serve.make_from_module(p + ":Missing")
            with self.assertRaises(ValueError):
                serve.make_from_module(p)


if __name__ == "__main__":
    unittest.main()
