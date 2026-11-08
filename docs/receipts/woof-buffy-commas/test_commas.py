"""The rebuilt pass keeps every stretch within Woof's clause limit, changes no word, and follows Buffy's comma rules."""
import json
import unittest
from pathlib import Path

from commas import CLAUSE_MAX, LIMIT, MARK, preprocess

STRESS = Path(__file__).with_name("stress-en.jsonl")


def words(text):
    return [w.strip(".,") for w in text.split()]


class Commas(unittest.TestCase):
    def test_stress_corpus_fits_with_the_same_words(self):
        for line in STRESS.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            with self.subTest(text=text[:40]):
                out = preprocess(text)
                self.assertTrue(any(len(s) > CLAUSE_MAX for s in MARK.split(text)))
                self.assertLessEqual(max(map(len, MARK.split(out))), LIMIT + 1)  # + the mark that ends a stretch
                self.assertEqual(words(text), words(out))

    def test_comma_before_the_joining_word_nearest_the_middle(self):
        text = ("The new train line will connect the airport with the city center in about twenty minutes and should "
                "take thousands of cars off the main road.")
        self.assertEqual(preprocess(text), text.replace(" and should", ", and should"))

    def test_spoken_numbers_and_relatives_stay_whole(self):
        number = ("Exactly two hundred and forty three thousand eight hundred and twelve people came out walking "
                  "along that winding road beside the river nearly every single weekend")
        self.assertNotIn(",", preprocess(number).split("people")[0])
        relative = ("They finally reached the small and very quiet valley town in the hills of the north for which "
                    "the old family had been searching across many long and difficult years.")
        self.assertIn(", for which", preprocess(relative))

    def test_short_lines_end_in_punctuation(self):
        self.assertEqual(preprocess("Harbor Report\n\n  The fleet is back.\n\"Done\"\n"),
                         "Harbor Report. The fleet is back. \"Done.\"")


if __name__ == "__main__":
    unittest.main()
