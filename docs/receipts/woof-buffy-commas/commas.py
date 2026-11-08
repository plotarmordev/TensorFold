"""Buffy's text pass for Woof, rebuilt from its README's rules: each non-blank line is its own unit ending in
punctuation, and every stretch between punctuation marks over LIMIT gets a comma, so Woof's splitter never falls back
to cutting before a conjunction.

    python3 commas.py CORPUS OUT    # JSON lines with id and text; a line a unit, ID.1, ID.2... when there are several
"""
import argparse
import json
import re
from pathlib import Path

CLAUSE_MAX = 120          # Woof's textsplit.py, clause_max in zig/src/families/woof/text.zig
LIMIT = CLAUSE_MAX - 10
MIN_PART = 15             # no added comma leaves a part shorter than this
# Woof's sentence and clause marks: a stretch ends after one of them
MARK = re.compile(r"(?<=[.!?…,;:])\s+|(?<=—)\s*|(?<=--)\s*")
LINE_END = tuple(".!?…,;:—-")
CLOSERS = "\"'”’)]"
JOINERS = {"and", "but", "or", "nor", "yet", "which", "who", "whom", "whose", "when", "where", "while", "because",
           "although", "though", "whereas", "unless", "until", "if", "whether"}
PREPOSITIONS = {"as", "for", "with", "from", "to", "in", "on", "at", "by", "into", "through", "after", "before",
                "without", "during", "about", "against", "between", "than", "like"}
RELATIVES = {"which", "whom", "whose"}
NUMBERS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
    seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
    point oh""".split())
MONEY = {"dollar", "dollars", "cent", "cents", "euro", "euros", "pound", "pounds", "pence", "penny"}
AMOUNT = NUMBERS | MONEY
NO_COMMA_AFTER = PREPOSITIONS | {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its",
                                 "our", "their", "some", "any", "no", "every", "each", "of"}


def end_line(line):
    """`line` ending in punctuation (a closing quote or bracket may follow it)."""
    core = line.rstrip(CLOSERS)
    return line if core.endswith(LINE_END) else core + "." + line[len(core):]


def cut(s):
    """The space in `s` a comma goes before: a joining word nearest the middle, else a preposition, else any word."""
    words = [(m.start(), re.sub(r"\W", "", m.group()).lower()) for m in re.finditer(r"\S+", s)]
    best = None
    for (_, prev), (at, word), (_, after) in zip(words, words[1:], words[2:] + [(0, "")]):
        space = at - 1
        if prev in NO_COMMA_AFTER or min(space, len(s) - space) < MIN_PART:
            continue
        if word in NUMBERS and prev in AMOUNT | {"and"} or word in MONEY and prev in NUMBERS \
                or word == "and" and prev in AMOUNT and after in NUMBERS:
            continue  # inside a spoken amount: "two hundred and forty three dollars and fifty cents"
        if word in JOINERS and not (word in RELATIVES and prev in PREPOSITIONS):
            penalty = 0
        elif word in PREPOSITIONS:
            penalty = 0 if after in RELATIVES else len(s) / 4  # ", for which" keeps its preposition
        else:
            penalty = len(s)
        cost = abs(space - len(s) / 2) + penalty
        if best is None or cost < best[0]:
            best = (cost, space)
    return best and best[1]


def fit(s):
    """`s` with commas added until every part is within LIMIT."""
    at = cut(s) if len(s) > LIMIT else None
    return s if at is None else fit(s[:at]) + ", " + fit(s[at + 1:])


def fit_stretches(line):
    out, pos = [], 0
    for m in MARK.finditer(line):
        out += [fit(line[pos:m.start()]), m.group()]
        pos = m.end()
    return "".join(out) + fit(line[pos:])


def preprocess(text):
    """One unit a non-blank line, each spoken in its own engine call so a line break always ends a piece."""
    return [fit_stretches(end_line(line)) for line in (" ".join(raw.split()) for raw in text.splitlines()) if line]


def long_stretches(text):
    """Stretches Woof's splitter can only cut before a conjunction: those over CLAUSE_MAX."""
    return sum(len(s) > CLAUSE_MAX for s in MARK.split(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    lines = []
    for line in args.corpus.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        units = preprocess(item["text"])
        print(f"{item['id']}: {long_stretches(item['text'])} -> {sum(map(long_stretches, units))} over {CLAUSE_MAX}")
        for k, unit in enumerate(units, 1):
            unit_id = item["id"] if len(units) == 1 else f"{item['id']}.{k}"
            lines.append(json.dumps({**item, "id": unit_id, "text": unit}, ensure_ascii=False))
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
