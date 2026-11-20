# Cost per task

Free row: run `compare-ernie-5.5-flash-preview-free-2`; paid row: run `compare-claude-sonnet-5.5-1` (from compare_models.py). Theoretical cost of a free call = its tokens at the price of the balanced model of the same role (prices in `prices.json`).

| task | model | prompt tokens | completion tokens | actual USD | theoretical USD |
| --- | --- | --- | --- | --- | --- |
| topic_map | baidu/ernie-5.5-flash-preview:free | 1296 | 1004 | 0.0 | 0.012632 |
| structure_plan | baidu/ernie-5.5-flash-preview:free | 2086 | 17431 | 0.0 | 0.178482 |
| write_4_screens | baidu/ernie-5.5-flash-preview:free | 4147 | 37747 | 0.0 | 0.385764 |
| full_lesson | baidu/ernie-5.5-flash-preview:free (whole run) | 150681 | 399790 | 0.0 | 2.75767 |
| topic_map | anthropic/claude-sonnet-5.5 | 1751 | 635 | 0.009852 | 0.009852 |
| structure_plan | anthropic/claude-sonnet-5.5 | 2894 | 5069 | 0.056478 | 0.056478 |
| write_4_screens | anthropic/claude-sonnet-5.5 | 5106 | 9322 | 0.103432 | 0.103432 |
| full_lesson | anthropic/claude-sonnet-5.5 (whole run) | 166842 | 294218 | 0.798472 | 1.676737 |
| hebrew_check | google/gemma-4-31b-it:free | 10081 | 9459 | 0.0 | 0.043032 |
| hebrew_check | google/gemini-3.8-flash | 10179 | 3962 | 0.022492 | 0.022492 |

## A whole lesson per profile

- **free profile**: $0.0 actual; at balanced prices it would be $2.75767.
- **Sonnet writer** (checkers and small model on the free lists, as in compare_models): $0.798472 actual; with the balanced checkers and small model it would be about $1.676737.

Most expensive steps of a whole lesson (theoretical, both runs together): evaluator $1.6840, fix_screens $1.2955, write_lesson $1.1389.
