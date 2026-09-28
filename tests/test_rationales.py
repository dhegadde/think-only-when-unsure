import unittest

from typed_decisions.multitask_bench import DecisionItem
from typed_decisions.phase5_multitask import rationale_row
from typed_decisions.rationales import RATIONALE_TASKS, clean, key, select, teacher_prompt


class CharTok:
    """Stand-in tokenizer: one token per character, bos=1, eos=2."""
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": ([1] if add_special_tokens else []) + [ord(c) for c in text]}


def item(task="mnli", label=2, state="Premise: a\nHypothesis: b"):
    return DecisionItem(task=task, kind="choice", state=state, instruction="How?", options=["yes", "maybe", "no"],
                        label=label)


class TestRationales(unittest.TestCase):
    def test_row_puts_correct_answer_before_rationale_and_loss_span_is_rationale_only(self):
        tok = CharTok()
        row, start = rationale_row(item(label=2), "because b.", tok)
        text = "".join(chr(t) for t in row[1:-1])
        self.assertTrue(text.endswith("Answer: no\nWhy: because b."))
        self.assertEqual("".join(chr(t) for t in row[start:-1]), " because b.")
        self.assertEqual(row[-1], tok.eos_token_id)

    def test_prompt_names_the_gold_option(self):
        self.assertIn("Correct answer: maybe", teacher_prompt(item(label=1)))

    def test_key_is_stable_and_label_independent(self):
        self.assertEqual(key(item(label=0)), key(item(label=2)))
        self.assertNotEqual(key(item(state="x")), key(item(state="y")))

    def test_select_only_rationale_tasks_capped_and_deterministic(self):
        train = [item(task=t, state=f"{t}{i}") for t in ("mnli", "wsc", "sst2") for i in range(5)]
        a, b = select(train, per_task=3, seed=0), select(train, per_task=3, seed=0)
        self.assertEqual([key(x) for x in a], [key(x) for x in b])
        self.assertEqual(len(a), 6)
        self.assertTrue(all(x.task in RATIONALE_TASKS for x in a))

    def test_clean_caps_sentences(self):
        self.assertEqual(clean("  One.  Two! Three? Four. "), "One. Two! Three?")


if __name__ == "__main__":
    unittest.main()
