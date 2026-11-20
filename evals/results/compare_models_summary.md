# Writer model comparison

**Task:** a whole lesson from `evals/fixtures/short_winter.pptx` (5 slides), the same request and
the same gate answers (Y everywhere), run without a keyboard by `evals/run_agent.py`. Only the
writer changes. Fixed for every run: the free profile's checkers (`google/gemma-4-31b-it:free`,
`google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3.5-super-vl:free`) and small models, and one
judge, `google/gemini-3.8-flash`, scoring by [../rubric.md](../rubric.md). Coverage is measured
against one fixed list of 31 key points (`key_points_short_winter.json`). `baseline_chat` is one
prompt to Sonnet 5.5 asking for lesson.json directly, without the agent. Run on 20.11.2026.

Means per model (every run is in `compare_models.csv`; failed runs count in `runs` but have no
scores):

<!-- table -->
| model | runs | duration_sec | calls | screens | relevance | accuracy | completeness | format | judge_avg | coverage_pct | open_findings | cost_usd | theoretical_cost_usd | schema_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baidu/ernie-5.5-flash-preview:free | 2 | 1378.50 | 18.50 | 7.00 | 5.00 | 5.00 | 5.00 | 3.50 | 4.62 | 95.15 | 26.00 | 0.00 | 2.06 | 2/2 |
| nvidia/nemotron-3.5-super-vl:free | 2 | 320.00 | 5.00 | 0.00 | - | - | - | - | - | - | - | 0.00 | 0.35 | 0/2 |
| anthropic/claude-sonnet-5.5 | 2 | 1283.50 | 27.50 | 8.50 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 98.40 | 6.50 | 0.89 | 1.82 | 2/2 |
| baseline_chat | 1 | 47.00 | 1.00 | 6.00 | 4.00 | 5.00 | 5.00 | 2.00 | 4.00 | 100.00 | - | 0.09 | 0.09 | 0/1 |
| google/gemini-3.8-flash | 1 | 778.00 | 23.00 | 5.00 | 5.00 | 5.00 | 4.00 | 5.00 | 4.75 | 93.50 | 12.00 | 0.26 | 1.08 | 1/1 |
<!-- /table -->

## Conclusion

- **The judge barely separates the agent's lessons.** Every lesson the agent finished scored 4.5-5
  on relevance, accuracy and completeness. With a 5-slide source the judge (one model, one call)
  does not see the differences the checkers see; it should not be read as "the free writer is as
  good as Sonnet".
- **Where the writers do differ:** format and the review loop. Sonnet's lessons scored 5 on format
  and ended with 1 and 12 open findings; ernie (free) scored 4 and 3 on format and ended with 14
  and 38 open findings (the checkers kept finding problems its rewrites did not fix). Gemini Flash
  ended with 12, in one run.
- **Reliability is the real cost of the free writers.** ernie finished both runs (one was cut short
  by a network error during fixing, before the fix-failure handling was added that night).
  nemotron-3.5-super-vl failed both regular runs: a 120 s timeout in planning, then screens that
  ignored the planned diagram and illustration even after the error was sent back. With a single
  writer and no fallback, as here, one bad reply ends the lesson; in normal use the agent falls back
  to the next writer.
- **Time and money:** Gemini Flash was the fastest full run (13 min, $0.26); Sonnet took 18-25 min
  for $0.80-0.98 (with free checkers); ernie took 21-24 min for $0. A whole balanced-profile lesson
  (Sonnet writer and the paid checkers) is estimated at about $1.7-2.0 (see `cost_summary.md`).
- **The agent versus one prompt:** the one-prompt baseline is fast and cheap (47 s, $0.09) and its
  content scored well, but its JSON is not valid against the player's schema (format 2) and nothing
  checked it. The agent's value is a lesson that is valid, reviewed, and approved at every gate, at
  10-20 times the time.
- **Caveats:** two runs per writer (one for Gemini Flash, because of the night's budget); the free
  checkers hit 429s in some runs (noted per row), so open-finding counts are not fully comparable;
  the judge is a Google model, so the Gemini Flash writer row is judged by its own family.
