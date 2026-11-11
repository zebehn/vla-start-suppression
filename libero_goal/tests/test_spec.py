import json
import os
import unittest

from tests._util import BENCH

from wnta import spec as S


class InitStates(unittest.TestCase):
    def test_sha_and_language(self):
        with open(os.path.join(S.SPEC_DIR, "init_states_v0.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["libero_commit"], S.SCENE["libero_commit"])
        self.assertEqual([t["language"] for t in meta["tasks"]], S.TASK_LANGUAGE)
        arr = S.load_init_states()  # raises on sha mismatch
        self.assertEqual(arr.shape, (10, 50, 79))

    def test_shard_is_condition_free_and_partitions(self):
        gs = S.core_inits()
        parts = [S.shard(gs, i, 2) for i in range(2)]
        self.assertEqual(sorted(parts[0] + parts[1]), gs)
        self.assertTrue(all(g % 50 % 2 == 0 for g in parts[0]))


if __name__ == "__main__":
    unittest.main()
