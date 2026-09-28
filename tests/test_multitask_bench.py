"""Phase 5 benchmark files must be well formed and keep held-out tasks out of training.
Needs processed_data/phase5/ (built by `python -m typed_decisions.multitask_bench`); skipped if absent.
"""
import os
import unittest
from collections import Counter

from typed_decisions.multitask_bench import OUT, TRAIN_TASKS, HELDOUT_TASKS, load


@unittest.skipUnless(os.path.isdir(OUT), "needs processed_data/phase5")
class TestMultitaskBench(unittest.TestCase):
    def test_items_well_formed(self):
        for name in ["train", "dev"] + [f"test_{t}" for t in HELDOUT_TASKS]:
            for it in load(name):
                self.assertIn(it.kind, ("choice", "noul", "score"), name)
                self.assertTrue(it.state.strip() and it.instruction.strip(), name)
                self.assertGreaterEqual(len(it.options), 2, name)
                self.assertTrue(all(o.strip() for o in it.options), name)
                self.assertTrue(0 <= it.label < len(it.options), name)
                if it.kind == "noul":
                    self.assertEqual(it.options, ["Yes", "No"], name)

    def test_heldout_tasks_never_in_training(self):
        train_tasks = {it.task for s in ("train", "dev") for it in load(s)}
        self.assertEqual(train_tasks, set(TRAIN_TASKS))
        self.assertFalse(train_tasks & set(HELDOUT_TASKS))
        train_states = {it.state for it in load("train")}
        for t in HELDOUT_TASKS:
            self.assertFalse(train_states & {it.state for it in load(f"test_{t}")}, t)

    def test_labels_not_degenerate(self):
        for t in TRAIN_TASKS:
            c = Counter(it.label for it in load("train") if it.task == t)
            self.assertLess(max(c.values()) / sum(c.values()), 0.75, t)  # no single answer dominates



@unittest.skipUnless(os.path.isdir("processed_data/phase6"), "needs processed_data/phase6")
class TestPhase6Mixes(unittest.TestCase):
    def test_no_heldout_item_in_any_mix(self):
        held = {(it.state.strip().lower(), tuple(o.strip().lower() for o in it.options))
                for t in HELDOUT_TASKS for it in load(f"test_{t}")}
        for mix in ("far", "all"):
            train = load("train", f"processed_data/phase6/{mix}")
            self.assertFalse({t for t in HELDOUT_TASKS} & {it.task for it in train}, mix)
            same = [it for it in train if (it.state.strip().lower(), tuple(o.strip().lower() for o in it.options)) in held]
            self.assertEqual(same, [], mix)


if __name__ == "__main__":
    unittest.main()
