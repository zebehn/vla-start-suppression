"""Tests for the GR-1 tabletop scripts that need no simulator, GPU, or model (standard library only).

Run from the repository root: python3 -m unittest discover -s tests -t .
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

GR1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(GR1)
sys.path.insert(0, GR1)
import analyze_gr1 as A  # noqa: E402
import gr1_common as G  # noqa: E402

SCENES = os.path.join(GR1, "configs", "gr1_scenes.json")
STRINGS = os.path.join(GR1, "configs", "gr1_instruction_strings.json")
TAB = os.path.join(ROOT, "data", "gr1", "gr1_tabletop_episodes.csv")
GATE = os.path.join(ROOT, "data", "gr1", "gr1_gate_episodes.csv")


def strings():
    with open(STRINGS) as f:
        return json.load(f)


class SceneList(unittest.TestCase):
    def test_six_tasks_twenty_distinct_seeds(self):
        eps = G.scene_list(SCENES)
        self.assertEqual(len(eps), 120)
        for t in G.TASKS:
            seeds = [s for k, s in eps if k == t["k"]]
            self.assertEqual(len(seeds), 20)
            self.assertEqual(len(set(seeds)), 20)
        self.assertEqual(len(set(eps)), 120)

    def test_no_overlap_with_calibration(self):
        self.assertFalse(set(G.scene_list(SCENES)) & set(G.calibration_list()))
        self.assertEqual(len(G.calibration_list()), 504)
        self.assertEqual(sum(1 for _k, s in G.calibration_list() if s in G.CAL_FLOOR_SEEDS), 54)

    def test_episode_tables_use_the_scene_list(self):
        eps = set(G.scene_list(SCENES))
        for path in (TAB, GATE):
            rows = A.read_csv(path)
            by = {}
            for r in rows:
                by.setdefault((r.get("run", ""), r["cond"]), set()).add((int(r["task"]), int(r["seed"])))
            for key, got in by.items():
                self.assertEqual(got, eps, key)


class Seeds(unittest.TestCase):
    def test_functions(self):
        self.assertEqual(G.gidx(0, 0), 300000)
        self.assertEqual(G.gidx(5, 19), 305019)
        self.assertEqual(G.scene_seed(0, 0), 20261105)
        self.assertEqual(G.scene_seed(3, 7), 20264112)
        self.assertEqual(G.call_seed(300000, 0), 20261016 + 300000000)
        self.assertEqual(G.call_seed(G.gidx(2, 5), 3), 20261016 + 1000 * 302005 + 3)

    def test_csv_columns(self):
        for path in (TAB, GATE):
            for r in A.read_csv(path):
                k, s = int(r["task"]), int(r["seed"])
                self.assertEqual(int(r["g"]), G.gidx(k, s))
                self.assertEqual(int(r["scene_seed"]), G.scene_seed(k, s))
                self.assertEqual(r["task_name"], G.TASKS[k]["name"])


class InstructionStrings(unittest.TestCase):
    def test_rule_reproduces_table(self):
        tab = strings()
        keys = ["0:0", "1:1", "2:7", "3:40", "4:26", "5:2"]
        for key in keys:
            k, s = (int(x) for x in key.split(":"))
            row = tab["scenes"][key]
            env, objs = row["original"], row["scene_objects"]
            self.assertEqual(G.instruction_string("none/impossible_action", env, k, s, objs), row["impossible_action"]["text"])
            self.assertEqual(G.instruction_string("none/absent_object", env, k, s, objs), row["absent_object"]["text"])
            self.assertEqual(G.instruction_string("prefix/original", env, k, s, objs), row["prefix/original"]["text"])
            self.assertEqual(row["impossible_action"]["template"], G.gidx(k, s) % 6)

    def test_rule_reproduces_all_scenes(self):
        tab = strings()
        for k, s in G.scene_list(SCENES):
            row = tab["scenes"][f"{k}:{s}"]
            for cond, sub in (("none/impossible_action", "impossible_action"), ("none/absent_object", "absent_object"),
                              ("prefix/original", "prefix/original")):
                self.assertEqual(G.instruction_string(cond, row["original"], k, s, row["scene_objects"]), row[sub]["text"])

    def test_absent_object_names_nothing_in_the_scene(self):
        tab = strings()
        for key, row in tab["scenes"].items():
            k = int(key.split(":")[0])
            ao = row["absent_object"]
            present = {G.words(c) for c in row["scene_objects"].values()}
            self.assertNotEqual(ao["named_task"], k)
            self.assertNotIn(ao["named_object"], present)
            self.assertNotIn(ao["named_source"], present)
            self.assertNotIn(ao["named_target"], present)

    def test_fixed_strings(self):
        env = "unlocked_waist: pick the eggplant from the plate and place it in the plate"
        expect = {"none/original": env, "none/empty": "", "none/neutral": "do something", "none/nonsense": "xxx",
                  "none/template_only": "unlocked_waist: ", "prefix/empty": "if no task is given do not move",
                  "prefix/nonsense": "if no task is given do not move xxx",
                  "prefix/original": "if no task is given do not move " + env, "gate/original": env, "gate/empty": "",
                  "gate/template_only": "unlocked_waist: ", "hold": None,
                  "none/impossible_action": "unlocked_waist: eat the eggplant"}   # g = 300000, 300000 mod 6 = 0
        for cond, want in expect.items():
            self.assertEqual(G.instruction_string(cond, env, 0, 0, {"obj": "eggplant"}), want, cond)
        self.assertEqual(G.formalize("unlocked_waist: "), "unlocked_waist ")

    def test_episode_text_matches_rule(self):
        tab = strings()
        for r in A.read_csv(TAB):
            k, s = int(r["task"]), int(r["seed"])
            row = tab["scenes"][f"{k}:{s}"]
            self.assertEqual(G.instruction_string(r["cond"], row["original"], k, s, row["scene_objects"]), r["text"])


class Categories(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(G.category("target", True, 0.3, 0.01), "completion")
        self.assertEqual(G.category("none", True, 0.01, 0.01), "attempt")
        self.assertEqual(G.category("none", False, 0.05, 0.0), "ncmotion")
        self.assertEqual(G.category("none", False, 0.01, 0.05), "ncmotion")
        self.assertEqual(G.category("none", False, 0.0499, 0.0499), "nonstart")

    def test_recorded_categories(self):
        for path, name in ((TAB, "tabletop"), (GATE, "gate")):
            n, bad = A.check_rows(A.read_csv(path), name)
            self.assertGreater(n, 0)
            self.assertEqual(bad, [])


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.proc = subprocess.run([sys.executable, os.path.join(GR1, "analyze_gr1.py"), "--out-dir", cls.tmp],
                                  capture_output=True, text=True)
        with open(os.path.join(cls.tmp, "gr1_tabletop_summary.json")) as f:
            cls.ts = json.load(f)
        with open(os.path.join(cls.tmp, "gr1_gate_summary.json")) as f:
            cls.gs = json.load(f)

    def test_exit_code(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)

    def test_every_cell_of_tab_gr1(self):
        t = self.ts["paper_table"]
        self.assertEqual(t["cells_total"], 90)
        self.assertEqual(t["cells_matching"], 90)
        for cond, want in A.PAPER.items():
            self.assertEqual(t["rows"][cond]["computed"], list(want), cond)

    def test_committed_summaries_are_current(self):
        for name, obj in (("gr1_tabletop_summary.json", self.ts), ("gr1_gate_summary.json", self.gs)):
            with open(os.path.join(ROOT, "data", "gr1", name)) as f:
                self.assertEqual(json.load(f), obj, name)

    def test_text_numbers(self):
        c = self.ts["cells"]
        self.assertAlmostEqual(c["none/original"]["designated_rate"], 88 / 120)
        self.assertEqual(list(c["none/original"]["designated_by_task"].values()), [17, 15, 15, 12, 13, 16])
        self.assertEqual(round(c["none/empty"]["median_max_hand_disp_ncmotion_m"], 2), 0.29)
        self.assertEqual(round(self.ts["comparisons"]["template_only_vs_empty_completion"]["p"], 3), 0.004)
        self.assertEqual(round(self.ts["comparisons"]["prefix_original_vs_original_designated"]["p"], 3), 0.011)
        self.assertLess(self.ts["comparisons"]["impossible_action_vs_original_designated"]["p"], 1e-13)
        self.assertLess(self.ts["comparisons"]["absent_object_vs_original_designated"]["p"], 1e-12)
        self.assertEqual(self.gs["tau_rounded"], 6.411)
        self.assertEqual(self.gs["floor_median_rounded"], 4.072)
        self.assertEqual(self.gs["calibration"]["original_vs_empty_median_rounded"], 82.6)
        self.assertEqual(round(self.gs["template_only_gate_d_median"], 1), 64.0)
        self.assertEqual(self.gs["empty_gate_d_max"], 0.0)

    def test_two_gate_runs(self):
        cmp_ = self.gs["comparison"]
        p = cmp_["per_condition"]
        self.assertEqual([p[c]["held_repeat"] for c in ("gate/empty", "gate/neutral", "gate/nonsense")], [120, 109, 114])
        self.assertEqual(p["gate/original"]["designated_repeat"], 88)
        self.assertEqual(cmp_["n_changed_total"], 3)
        self.assertEqual(p["gate/neutral"]["n_changed"], 3)
        self.assertTrue(cmp_["all_changed_within_5pct_of_tau"])
        for run in ("first", "repeat"):
            self.assertTrue(self.gs["runs"][run]["held_episodes_all_nonstart"])
            self.assertEqual(self.gs["runs"][run]["committed_nontask_nonstart_n"], 0)


class Csv(unittest.TestCase):
    def test_columns(self):
        with open(TAB) as f:
            self.assertEqual(next(csv.reader(f)), A.TAB_COLS)
        with open(GATE) as f:
            self.assertEqual(next(csv.reader(f)), A.GATE_COLS)


if __name__ == "__main__":
    unittest.main()
