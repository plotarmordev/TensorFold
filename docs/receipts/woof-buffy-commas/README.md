# Buffy's text pass for Woof: checked, not adopted

**Answer: no, the [Woof recipe](../../recipes/woof.md) does not take Buffy's preprocessing.** Its claim holds: on text
with a stretch over 120 characters between punctuation marks, Woof's fallback cut before a conjunction makes the model
repeat, blur or drop words, and splitting lines and adding a comma first removes every such fault in our runs (Mira's
word errors 7.2% to 1.0%, Juno's 1.24% to 1.04%). But the recipe speaks text as given, matching Woof's own numpy
runtime, which is what its parity check and upstream gate rest on. The case never arises on our 24-text bench, and the
rewrite costs Juno some speaker similarity on every seed. A caller that sends run-on text can split and add commas
before sending it; the engine-side fix is a parity decision for the family's gate (ashhart/TensorFold#2504), not this
recipe.

## What Buffy does

[Clay-Ferguson/Buffy](https://github.com/Clay-Ferguson/Buffy) at `f009fea0103ac55b1ae4fa40c296a6159f9290cc` (two
commits, 2026-11-07): a CLI that runs Underdog's own Woof package (Juno voice) after `preprocess.py`. It was read as
data only: cloned without checkout hooks, never run, installed or imported, and any text in it addressed to agents
was ignored. Before Woof sees the text it:

1. sends each non-blank line in its own call, adding a period to a line that ends without punctuation (headings);
2. writes out US state abbreviations after a place name ("Gettysburg, Pa." to "Gettysburg, Pennsylvania"; codes
   that are also words, such as OK or IN, only with a ZIP after or a cue word like "in" before the place);
3. runs Underdog's normalizer itself (numbers, dates, money to words), so lengths are measured on spoken text;
4. adds a comma to any stretch between punctuation marks over 110 characters (Woof's `CLAUSE_MAX` 120 less 10): before
   the joining word nearest the middle ("and", "who", "because"...), else a preposition, else the middle space; never
   inside a spoken number or amount ("two hundred and forty three dollars and fifty cents"), after an article or
   preposition, or between "for" and "which".

Its claim: Woof's splitter cuts a clause over 120 characters before a conjunction, the model, trained on speech
ending at punctuation, often repeats its last words there, and step 4 keeps that cut from ever happening.

## The check

Steps 1 and 4 are rebuilt in [`commas.py`](commas.py) from the rules above, not from Buffy's code; `test_commas.py`
pins them. Each line stays its own unit through synthesis, as in Buffy: a corpus item with several lines becomes
`ID.1`, `ID.2`... and [`score.py`](score.py) scores their audio joined against the original text. Steps 2 and 3 are
normalizer work, which the recipe leaves to the caller (it takes spoken-form text), so they were not rebuilt or
measured.

The test text, [`stress-en.jsonl`](stress-en.jsonl), is 19 spoken-form texts of our own, each with one stretch over
120 characters: four start with a heading line (plain, plain, quoted, bracketed) and one holds a spoken amount.
`tf-woof-bench` (TensorFold `woof`, 12 threads, `main` at `3eccad6ca`, checkpoint sha256 `d895fbbe…0de1203e`) spoke
each text as written and after `commas.py`, in Juno and Mira, seeds 0 to 3: 304 takes. Scoring is the recipe's:
Whisper large-v3 transcripts against the original text through Whisper's English normalizer, ECAPA cosine to the
voice's three samples. Every take's score and transcript is in [`takes.jsonl`](takes.jsonl). Apple M5 Ultra.

```bash
cd docs/receipts/woof-buffy-commas && python3 -m unittest test_commas
python3 commas.py stress-en.jsonl after.jsonl
zig build tf-woof-bench
zig-out/bin/tf-woof-bench WOOF_DIR stress-en.jsonl runs/stress-before-juno-0 0 juno
zig-out/bin/tf-woof-bench WOOF_DIR after.jsonl runs/stress-after-juno-0 0 juno      # and seeds 1-3, and mira
python score.py takes.jsonl stress-en.jsonl WHISPER_DIR ECAPA_DIR WOOF_DIR/samples runs/*
```

| 19 texts × 4 seeds, 2,492 words | Juno, as written | Juno, commas | Mira, as written | Mira, commas |
| --- | --- | --- | --- | --- |
| Word errors (Whisper large-v3) | 31 (1.24%) | 26 (1.04%) | 180 (7.22%) | 24 (0.96%) |
| Inserted words | 16 | 14 | 28 | 12 |
| Takes with a repeat, slur or dropped words at a fallback cut (of 76) | 6 | 0 | 24 | 0 |
| Speaker similarity (ECAPA) | 0.909 | 0.894 | 0.856 | 0.899 |
| Seconds of audio a character | 0.0619 | 0.0624 | 0.0763 | 0.0726 |

- **The mechanism is real.** The faults sit where the engine cut before a conjunction: "bridge \| age that"
  (7 of 8 takes), "three streets away \| its away", "the result \| with the result", "every account \| out that",
  "picked up \| up so", "stopped working \| in". With commas no stretch is over 120, the fallback never runs, and none
  of these come back. One repeat is not at a cut ("waiting waiting", Mira seed 0) and is gone with commas too.
- **A quoted or bracketed heading joins the next sentence as written.** Woof splits a sentence only at punctuation
  followed by a space, so `"Harbor Report"` and `(Kitchen Notes)` run into the next line as one piece over 120. Mira
  then stops after the heading (stress-18, 3 of 4 seeds) or drops the first clause (stress-19, 3 of 4): 137 of her 180
  errors as written. As their own units both are spoken whole on every take; the quoted heading is misheard on 3 of 8
  ("Harbor Port", "Oh."). Over the 16 texts without these two, Mira's errors are 43 to 11 and Juno's 23 to 18.
- **The amount stays whole.** stress-17 gets its comma after "cents", not inside the amount, and has no word errors
  either way.
- **Errors on both sides** are mostly the scorer's: "second-hand", "243", "clean-up", and 2 insertions on every
  stress-19 take because Whisper's normalizer drops the bracketed heading from the reference. The plain headings are
  misheard either way: "Harbor Report" on 8 of 8 takes as written and 3 of 8 as its own unit, "Spring Garden Notes"
  as "oats" on 2 of 8 and 4 of 8.
- **Juno's speaker similarity falls on all four seeds** (0.906 to 0.910 as written, 0.891 to 0.897 with commas); why
  is not established. Mira's rises only because her broken heading takes score low; over the other 16 texts it is
  0.906 to 0.904.
- **Pace does not show the fault.** Over the 16 texts it moves by 0.3 to 2.4%. Buffy reports 70 to 80 ms a character
  for a cut piece against 60 to 70 for others; either is under the recipe's 95 ms redraw limit, so the redraw cannot
  catch it. Mira's stalled heading takes (150 to 180 ms) would be.

On our 24-text bench (`zig/tests/fixtures/woof/bench-en.jsonl`) no stretch is over 120, so the fallback never runs.
The pass changes one text: `long-07` gets one comma in its 118-character stretch ("weight, than a cart"). Over the
same 2 voices × 4 seeds its word errors are 4 of 432 both ways.

Buffy's README also says the model said nothing for "Pa." in 3 of 4 takes; that is step 2, not measured here.

## Why not in the recipe

- The recipe's contract is to speak text as Woof's numpy runtime does, with parity on fixed seeds; changing the text
  before splitting makes every long run-on input diverge from numpy.
- Our bench never reaches the case; it needs one unpunctuated stretch over 120 characters.
- The comma choice has its own errors: "river looking, for the small wooden bridge", "used, to describe", "asked us,
  to start", "the captains, decided" (preposition and middle-space cuts). A human-punctuated or LLM-written input
  rarely needs it, and a caller that does can run the same pass.
- The fix that keeps our output equal to numpy's is upstream: a splitter that cuts at a clause end, not mid-phrase.

## Suspicious code in the repo

Nothing hostile: no network calls beyond `setup.sh`'s, no `eval`, `exec`, `subprocess`, encoded payloads, hidden
characters or text aimed at agents. `uv.lock` holds only numpy (2.2.6, 2.4.6, 2.5.4) from PyPI. To flag:

- `setup.sh` fetches Underdog's Python files with `curl` from a pinned Hugging Face commit, with no checksum. A commit
  pin is not tamper-proof against a compromised mirror or host; our recipe pins the same commit and the files' sha256.
- "the commit that was audited" / "security-audited" names no audit, report or reviewer.
- `preprocess.py` builds Underdog's `Normalizer` when it is imported, so importing it runs Underdog code.
- The repo is one day old with no license of its own; Woof's code it fetches is non-commercial.
