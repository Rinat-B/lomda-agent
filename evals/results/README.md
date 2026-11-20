# Eval results

All evals run on the fixed source `evals/fixtures/short_winter.pptx` (5 slides on winter driving)
and run the real agent graph without a keyboard through `evals/run_agent.py` (scripted gate
answers, default Y; `write_files` answered N, the run saves its own files to `runs/`). Every run
is tagged in LangSmith with its experiment name. Results below are from 19-20.11.2026.

```powershell
.venv\Scripts\python -m evals.run_agent --experiment try1             # one run, any models
.venv\Scripts\python -m evals.compare_models                          # eval 1
.venv\Scripts\python -m evals.cost_eval                               # eval 2 (after eval 1)
.venv\Scripts\python -m evals.scope_eval                              # eval 3
$env:PROFILE="balanced"; .venv\Scripts\python -m pytest -m slow       # checker acceptance
.venv\Scripts\python -m evals.budget                                  # spending since a start point
```

## 1. Writer comparison (`compare_models.py`)

The same lesson by each writer; checkers, small model and judge fixed (free checkers, judge
`google/gemini-3.8-flash` by [../rubric.md](../rubric.md)). Files: `compare_models.csv`,
[compare_models_summary.md](compare_models_summary.md) (table and conclusion), `runs/compare-*`.

| writer | runs (ok) | judge avg | format | coverage | open findings | minutes | cost USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baidu/ernie-5.5-flash-preview:free | 2 (2) | 4.62 | 3.5 | 95% | 26 | 23 | 0 |
| nvidia/nemotron-3.5-super-vl:free | 3 (1; the third was an extra run) | 4.50 | 3 | 97% | 13 | 29 | 0 |
| anthropic/claude-sonnet-5.5 | 2 (2) | 5.00 | 5 | 98% | 6.5 | 21 | 0.89 |
| google/gemini-3.8-flash | 1 (1) | 4.75 | 5 | 94% | 12 | 13 | 0.26 |
| baseline_chat (one prompt to Sonnet) | 1 | 4.00 | 2 (schema invalid) | 100% | - | 0.8 | 0.09 |

The judge saturates near 5 for every lesson the agent finished; the differences are in format,
open findings after the review loop, reliability and time. See the summary for the full reading.

## 2. Cost per task (`cost_eval.py`)

Actual cost from OpenRouter's `usage.cost`; theoretical cost of a free call = its tokens at the
price of the balanced model of the same role (`prices.json`). Files: `cost_eval.csv`,
[cost_summary.md](cost_summary.md).

| task | free (actual / theoretical) | paid (actual) |
| --- | --- | --- |
| topic map | $0 / $0.013 | $0.010 (Sonnet) |
| structure plan | $0 / $0.178 | $0.056 (Sonnet) |
| write 4 screens | $0 / $0.386 | $0.103 (Sonnet) |
| Hebrew check of a lesson | $0 / $0.043 | $0.022 (Gemini Flash) |
| whole lesson | $0 / $2.76 | $0.80 (Sonnet writer, free checkers); about $1.7 with paid checkers |

The free models' theoretical cost is high because they are reasoning models: ernie wrote about
400k completion tokens for a lesson. The most expensive steps are the checkers (evaluator), the
fix rounds and the writing.

## 3. Scope and leakage (`scope_eval.py`)

10 out-of-scope messages, 5 in-scope controls and 2 source files with hidden injections, through
`scope_check` and `/add`'s `injection_guard`, in both profiles. Files: `scope_eval.csv` (after the
fixes), `scope_eval_before.csv`.

| profile | refusal rate (out of scope) | false refusals (controls) | injections caught |
| --- | --- | --- | --- |
| free, before | 90% | 0% | 1/2 |
| balanced, before | 80% | 0% | 1/2 |
| free, after | 100% | 0% | 2/2 |
| balanced, after | 100% | 0% | 2/2 |

Fixed by principle: e-learning means teaching people, not machine learning or general knowledge;
a question about the material is one the material answers (not shopping or outside advice);
scope_check sees three excerpts instead of one; the Hebrew injection patterns allow synonyms,
words in between, either word order and text addressed to the model.

## Checker acceptance (`tests/test_checkers_acceptance.py`)

The checkers on two lessons with hand-listed problems (`tests/fixtures/known_bad_*.json`);
the bar is 80% of the listed problems, and in the balanced profile also questions 6 and 7 of the
first lesson. Files: `checker_acceptance_<profile>_<lesson>.txt`.

| profile | known_bad_lesson (19 problems) | known_bad_lesson_2 (7 problems) |
| --- | --- | --- |
| free | 17/19 (89%) | 6/7 (86%) |
| balanced, 3 screens per call | 18/19 (95%), question 7 missed: strict test fails | 6/7 (86%) |
| balanced, 2 screens per call (current) | 17/19 (89%), questions 1 and 7 missed: strict test fails | 6/7 (86%) |

## Other files

- `day6_real_run_report.md`: report.md of a real free-profile run (day 6)
- `writer_bench.csv`: the day-5 benchmark that set the free writer order
- `key_points_short_winter.json`: the fixed key points used for coverage in eval 1
- `prices.json`: the OpenRouter prices used for theoretical costs
