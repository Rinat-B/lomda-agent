# Day 10: a real free run with the new checks (22.11.2026)

`evals/run_agent.py --profile free` on `evals/fixtures/short_winter.pptx`, Y at every gate
(including the browser check), the first motif candidate, no unverified web fact accepted.

**The run was stopped and resumed.** The Claude Code session that ran it ended at
`build_illustrations` (19:44). Its checkpoint was in `free/checkpoints.sqlite`, and
`run_agent --resume` (21:02) continued from `build_illustrations`, without repeating the research,
planning or writing. `run.json` counts only the calls after the resume; the ones before are
estimated below.

## Compared with day 9

| | day 9 | day 10 |
| --- | --- | --- |
| screens | 7 | 5 (the planner made one screen per topic of the 5 slides) |
| model calls | 41 | 28 after the resume + about 14 before (scope 1, type 1, research 5, topics 1, plan 1, motif 2, writing 2), about 42 |
| work time | about 55 minutes | about 39 minutes (10 before the stop, 29 after) |
| cost | $0.035 (5 searches) | $0.021 (3 searches) |
| findings | 48: 29 fixed, 19 open | 27: 26 fixed, 1 open |
| coverage of the source's key points | 96% | 100% (30/30) |
| weak-distractor findings | 10 open | 0 (the new definition, part C) |
| illustrations | 3 built, 3 passed Edge | 2 built (the planner chose 2), 2 passed the stricter Edge check |
| illustration logic check | did not exist | 2 pages checked, 1 finding (screen 3), rebuilt in round 1 and verified |
| diagram check against the source | did not exist | 1 diagram (screen 4), no finding; the code check found 2 questions with one branch, fixed |
| motif | none | 2 candidates (ernie: car + gauge; nemotron: car + shield-check), the first chosen, in the header |

Fewer screens make the numbers not directly comparable; the drop in open findings comes mostly
from the single-answer checker no longer marking every rule-breaking distractor as weak.

**The one open finding** (screen 4, more than one right answer): approaching a curve with one
lamp post visible, so visibility is under 100 m but not clearly under 50 m; the checker found
that "stop on the shoulder" is forbidden in a curve, and that the scenario does not settle
whether stopping is required. It stays for the owner at gate 3.

**The illustration logic finding was a false alarm.** On screen 3 the checker called the page's
reaction to a wrong move ("the skid continues, try again") a decision that differs from the
rule; it is a simulation result. The page was rebuilt anyway and the re-check passed. This is
the kind of false alarm the acceptance test still shows (3 of 10 findings, see
`../illustration_logic_free.txt`).

**Research.** 3 searches, 0 facts verified, 1 unverified, 6 rejected (sites outside the source
hierarchy). No web fact entered the lesson.

**The checkers were all of the writer's family.** The gemma checkers answered 429 all evening,
so every check ran on nemotron-3.5-super-vl, the same family as one of the writers; report.md
says so.

## Screenshots (`free/screenshots/`, 390 px, the "after" picture once animations ended)

- `lesson-first-screen.png`: the motif (car and gauge) and the title in the header
- `s3-before/after.png`: skid practice; the learner chooses the skid type (rear or front) and a
  steering action; a wrong action gets a hint ("note the type of skid and try again"), not the answer
- `s4-before/after.png`: fog simulator; the learner chooses how many lamp posts they see, the page
  shows the estimated visibility (under 100 m / under 50 m, the source's thresholds) and the
  learner picks an action

The lesson: `free/lesson/index.html`; the owner's report: `free/report.md`.
