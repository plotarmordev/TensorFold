"""The rebuilt pass keeps every stretch within Woof's clause limit, changes no word, and follows Buffy's comma rules."""
import json
import re
import unittest
from pathlib import Path

from commas import CLAUSE_MAX, LIMIT, MARK, preprocess

STRESS = Path(__file__).with_name("stress-en.jsonl")


def words(text):
    return re.findall(r"[\w'-]+", text)


class Commas(unittest.TestCase):
    def test_stress_corpus_fits_with_the_same_words(self):
        for line in STRESS.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            with self.subTest(text=text[:40]):
                units = preprocess(text)
                self.assertTrue(any(len(s) > CLAUSE_MAX for s in MARK.split(text)))
                stretches = [s for unit in units for s in MARK.split(unit)]
                self.assertLessEqual(max(map(len, stretches)), LIMIT + 1)  # + the mark that ends a stretch
                self.assertEqual(words(text), words(" ".join(units)))

    def test_comma_before_the_joining_word_nearest_the_middle(self):
        text = ("The new train line will connect the airport with the city center in about twenty minutes and should "
                "take thousands of cars off the main road.")
        self.assertEqual(preprocess(text), [text.replace(" and should", ", and should")])

    def test_spoken_amounts_and_relatives_stay_whole(self):
        number = ("Exactly two hundred and forty three thousand eight hundred and twelve people came out walking "
                  "along that winding road beside the river nearly every single weekend")
        self.assertNotIn(",", preprocess(number)[0].split("people")[0])
        money = ("The final bill came to two hundred and forty three dollars and fifty cents including delivery "
                 "charges for the furniture we ordered last week.")
        self.assertNotIn(",", preprocess(money)[0].split("cents")[0])
        relative = ("They finally reached the small and very quiet valley town in the hills of the north for which "
                    "the old family had been searching across many long and difficult years.")
        self.assertIn(", for which", preprocess(relative)[0])

    def test_each_line_is_its_own_unit_ending_in_punctuation(self):
        self.assertEqual(preprocess("Harbor Report\n\n  The fleet is back.\n\"Done\"\n(Kitchen Notes)\nStir."),
                         ["Harbor Report.", "The fleet is back.", "\"Done.\"", "(Kitchen Notes.)", "Stir."])


if __name__ == "__main__":
    unittest.main()
