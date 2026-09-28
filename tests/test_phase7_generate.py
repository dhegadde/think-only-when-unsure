import unittest

import torch

from typed_decisions.multitask_bench import DecisionItem
from typed_decisions.phase7_generate import answer_probs, subsample, user_turn


class TestPhase7Generate(unittest.TestCase):
    def test_subsample_fixed_sorted_and_capped(self):
        items = list(range(500))
        a = subsample(items, n=200, seed=0)
        self.assertEqual(a, subsample(items, n=200, seed=0))
        self.assertEqual(len(a), 200)
        self.assertEqual(a, sorted(a))
        self.assertEqual(subsample(list(range(100)), n=200), list(range(100)))

    def test_user_turn_letters_options_in_order(self):
        it = DecisionItem(task="copa", kind="choice", state="S", instruction="Why?", options=["x", "y"], label=1)
        u = user_turn(it)
        self.assertIn("A. x\nB. y", u)
        self.assertTrue(u.endswith("Reply with only the letter of the correct option."))

    def test_answer_probs_merges_token_variants(self):
        logits = torch.full((10,), -10.0)
        logits[1] = 2.0  # 'A'
        logits[2] = 2.0  # ' A'
        logits[3] = 2.0  # 'B' only
        p = answer_probs(logits, [[1, 2], [3]])
        self.assertAlmostEqual(sum(p), 1.0, places=5)
        self.assertAlmostEqual(p[0], 2 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
