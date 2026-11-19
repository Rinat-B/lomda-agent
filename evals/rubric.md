# Judge rubric

One fixed judge model scores every lesson in an eval (`evals/judge.py`, default: the
balanced checker, `google/gemini-3.8-flash`, a family different from every compared writer).
The judge gets the lesson (as the checkers see it, without source ids) and the source
excerpts, and returns a score from 1 to 5 and a short reason for each criterion.

A lesson that is not valid against `agent/assets/lesson.schema.json` can still be judged on
content, but **format** is then at most 2.

## 1. Relevance (`relevance`)

Is this a lesson on the requested subject, for the employees it is meant for, written as
the requested lesson type (training: explains why, warm tone, may enrich; compliance:
formal, close to the document's wording, no enrichment)?

| Score | Description |
| --- | --- |
| 5 | Every screen serves the request and the audience; the tone and structure match the lesson type exactly. |
| 4 | On subject and audience throughout; one screen or block drifts (an aside, a slightly wrong tone). |
| 3 | Mostly on subject, but several blocks are generic, off-audience, or written in the wrong style for the type. |
| 2 | Large parts are generic or about something else; the lesson type is barely recognisable. |
| 1 | Not a lesson on the requested subject, or written for the wrong audience altogether. |

## 2. Factual accuracy (`accuracy`)

Is every claim supported by the source excerpts? Invented numbers, comparisons, reasons,
people or procedures count against the lesson, even when they sound plausible.

| Score | Description |
| --- | --- |
| 5 | Every claim, number and rule matches the source; nothing is invented. |
| 4 | One small unsupported detail (an added reason or a loose paraphrase) that does not change a rule. |
| 3 | Two or three unsupported claims, or one that changes how a rule would be applied. |
| 2 | Several invented facts or a wrong rule (a wrong number, threshold or step). |
| 1 | Many invented or contradicted facts; the lesson cannot be trusted. |

## 3. Completeness (`completeness`)

Are all learning objectives covered, and do the source's key points (rules, thresholds,
obligations, procedure steps) appear in the lesson?

| Score | Description |
| --- | --- |
| 5 | Every key point of the source is taught; every objective has a screen and a question. |
| 4 | One minor point is missing or only mentioned in passing. |
| 3 | Two or three key points are missing, or one objective has no question. |
| 2 | A whole topic of the source is missing, or many key points are absent. |
| 1 | Only a small part of the source is covered. |

## 4. Format and rules (`format`)

Schema-valid JSON, correct Hebrew, and the lesson rules: questions ask for a decision in a
situation (not recall), each has an alternate on the same objective, exactly one correct
option with plausible distractors, feedback after a mistake never reveals the answer, each
fact is taught once (no repetition across blocks or screens), and did-you-know holds only
extra facts.

| Score | Description |
| --- | --- |
| 5 | Valid; clean Hebrew; every question is a real decision with one defensible answer and plausible distractors; no repetition; feedback never reveals the answer. |
| 4 | Valid; one or two small slips (a typo, one weak distractor, one repeated sentence). |
| 3 | Valid, but several rule breaks (recall questions, repeated facts, an alternate on another objective, awkward Hebrew). |
| 2 | Invalid against the schema, or many rule breaks (answers revealed, several questions with two right answers). |
| 1 | Broken output: not a usable lesson. |

`judge_avg` in the CSV files is the mean of the four scores.
