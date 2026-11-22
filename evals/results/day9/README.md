# Day 9: web research and interactive illustrations (real runs, 21-22.11.2026)

Both runs: `evals/run_agent.py` on `evals/fixtures/short_winter.pptx`, Y at every gate
(including the browser check), no unverified web fact accepted.

## Free profile (`free/`)

| | |
| --- | --- |
| screens | 7 |
| model calls | 41: research 7 (5 searches + 2 small), key points 1, planning 5, writing 2, illustrations 3, checkers 19, fixes 4 |
| cost | $0.035 (the 5 web searches; every model was free) |
| time | about 55 minutes of work (the computer slept during one verification, so the wall time was 13 hours) |
| review | 48 findings, 29 fixed in 2 rounds, 19 open; coverage of the source's key points 96% |
| research | 14 findings: 0 verified, 12 unverified, 2 rejected (unofficial sites) |
| illustrations | 3 built, 3 passed the Edge check at 390 and 1280 px |

**Why nothing was verified, and what changed.** 10 of the 12 unverified facts came from PDF
files on gov.il and cbs.gov.il, which the first version did not read; gov.il HTML pages are
behind bot protection, for httpx and for an automated browser alike; ynet and walla answer
HTTP 418. After the run the agent reads PDFs, matches quotes also on letters and digits only
(PDF text extraction glues Hebrew words to numbers and moves punctuation), renders
JavaScript-built pages in Edge, and reports bot-protection pages as such. Re-checking the same
12 quotes with these fixes (no model calls): **7 verified**, 5 still unverified (words
reordered by the PDF extraction, the news sites' 418, one quote not on its page).

Screenshots (390 px, before and after using the controls): `free/screenshots/`.

- `s2`: stopping distance: choose the vehicle and the road, set the gap with a slider, press
  "braking" and watch whether the car stops in time.
- `s4`: skidding: choose gas, steering left or right; the car turns back or not, with feedback.
- `s6`: fog: set the fog density, count the visible lamp posts, choose an action, get feedback.

The playable lesson with the built illustrations: `free/lesson/index.html`; the owner's report:
`free/report.md`.

## Sonnet 5.5 as the writer (`sonnet/`)

Stopped by the $1 spending guard during the second fix round (the run had spent just over $1:
research, planning and writing with Sonnet, 3 illustrations, one fix round). The run state was
in memory, so the lesson was lost; two of the three illustration pages were recovered from the
LangSmith traces of the run and checked with the same code.

- `sonnet_1`: fog: a road in perspective with numbered lamp posts, a fog slider, the learner
  picks the visibility range (radio buttons) and gets an explanation.
- `sonnet_2`: following distance: vehicle, road and gap, then "the car ahead brakes suddenly"
  with a stopping line.

Both pass the Edge check (after a fix to the check itself: a group of radio buttons is one Tab
stop, not one per button). Screenshots: `sonnet/screenshots/`.

## Free versus Sonnet illustrations

Both models built real interactions (the learner acts, the page answers) that pass the
mechanical checks. Sonnet's are visually richer (perspective drawing, numbered posts, a stop
line) and use more varied controls; the free model's are plainer but clear. The checks are
mechanical only: Sonnet's fog page shows more posts as the fog gets denser in its "after"
screenshot, which looks like inverted logic that no check catches, and its distance page
states a derived number (one second at 50 km/h is about 13.9 m) that the source does not give.
Teaching correctness of an illustration still needs the owner's look at gate 3.
