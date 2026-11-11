import statistics
import unittest

import numpy as np

import tests._util  # noqa: F401  (adds the kit directory to sys.path)

from wnta import score as SC
from wnta import stats as ST


def rec(outcome, task=0, contacts=None, disp=0.0, **kw):
    r = {"outcome": outcome, "task": task, "tie_tasks": kw.pop("tie_tasks", None), "contact_first_step": contacts or {},
         "max_disp": disp, "n_calls": 1, "nan_calls": 0, "first_contact": None}
    r.update(kw)
    return r


class Thresholds(unittest.TestCase):
    def test_integer_boundaries(self):
        # 0.925 - 0.10 > 0.825 in floating point, so exactly 165/200 would fail a float comparison
        self.assertGreater(0.925 - 0.10, 0.825)
        self.assertTrue(ST.at_least(165, 200, 825, 1000))
        self.assertTrue(ST.at_least(180, 200, 90))
        self.assertFalse(ST.at_least(179, 200, 90))
        self.assertTrue(ST.below(39, 200, 20))
        self.assertFalse(ST.below(40, 200, 20))

    def test_median_even_n(self):
        xs = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(ST.median(xs), 2.5)  # not the upper order statistic 3.0

    def test_quantile_matches_numpy(self):
        rng = np.random.default_rng(0)
        for n in (2, 7, 200):
            xs = list(rng.normal(size=n))
            for i, q in ((2, 0.10), (5, 0.25), (10, 0.5), (19, 0.95)):
                self.assertAlmostEqual(ST.quantile(xs, i, 20), float(np.percentile(xs, q * 100)), places=12)

    def test_wilson(self):
        lo, hi = ST.wilson(190, 200)
        self.assertAlmostEqual(lo, 0.9104, places=3)
        self.assertLess(hi, 1.0)


class Outcomes(unittest.TestCase):
    def test_task_zero_is_a_completion(self):
        r = rec(0, task=0)
        self.assertTrue(SC.completed(r))
        self.assertTrue(SC.designated_success(r))
        self.assertFalse(SC.non_start(r))
        self.assertFalse(bool(r["outcome"]))  # the trap: 0 is falsy

    def test_none_tie_and_types(self):
        self.assertFalse(SC.completed(rec("none")))
        self.assertTrue(SC.no_contact(rec("none")))
        self.assertFalse(SC.no_contact(rec("none", contacts={"plate_1": 12})))
        t = rec("tie", task=3, tie_tasks=[3, 8])
        self.assertTrue(SC.designated_success(t))
        self.assertTrue(SC.completed_task(t, 8))
        self.assertFalse(SC.completed_task(t, 1))
        with self.assertRaises(ValueError):
            SC.outcome_cat(rec("0"))
        with self.assertRaises(TypeError):
            SC.outcome_cat(rec(True))

    def test_four_categories(self):
        self.assertEqual(SC.category(rec("none", disp=0.049)), "non_start")
        self.assertEqual(SC.category(rec("none", disp=0.05)), "non_contact_motion")
        self.assertEqual(SC.category(rec("none", disp=0.01, contacts={"plate_1": 3})), "attempt")
        self.assertEqual(SC.category(rec(4, disp=0.01)), "completion")
        self.assertTrue(SC.no_contact(rec("none", disp=0.3)))
        self.assertFalse(SC.non_start(rec("none", disp=0.3)))
        self.assertTrue(SC.noaction_flag(rec("none", disp=0.049)))
        self.assertFalse(SC.noaction_flag(rec("none", disp=0.01, contacts={"plate_1": 3})))

    def test_condition_metrics(self):
        rows = [rec(0, j=0, g=0, gate_commit=True, gate_d=2.0), rec("none", g=1, gate_commit=False, gate_d=0.1),
                rec(8, g=2, gate_commit=True, gate_d=1.5), rec("none", g=3, contacts={"plate_1": 4}, gate_commit=False, gate_d=0.2)]
        m = SC.condition_metrics(rows, tau=1.0)
        self.assertEqual((m["completion_n"], m["success_n"], m["no_contact_n"], m["commit_n"], m["gate_d_gt_tau_n"]), (2, 1, 1, 2, 2))
        self.assertEqual((m["attempt_n"], m["non_contact_motion_n"], m["non_start_n"]), (1, 0, 1))
        self.assertEqual(m["selection"]["counts"]["0"], 1)
        self.assertEqual(m["gate_d"]["median"], statistics.median([2.0, 0.1, 1.5, 0.2]))


if __name__ == "__main__":
    unittest.main()
