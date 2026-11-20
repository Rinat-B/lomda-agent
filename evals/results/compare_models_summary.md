# Writer model comparison

Means per model (see compare_models.csv for every run).

<!-- table -->
| model | runs | duration_sec | calls | screens | relevance | accuracy | completeness | format | judge_avg | coverage_pct | open_findings | cost_usd | theoretical_cost_usd | schema_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baidu/ernie-5.5-flash-preview:free | 2 | 1378.50 | 18.50 | 7.00 | 5.00 | 5.00 | 5.00 | 3.50 | 4.62 | 95.15 | 26.00 | 0.00 | 2.06 | 2/2 |
| nvidia/nemotron-3.5-super-vl:free | 2 | 320.00 | 5.00 | 0.00 | - | - | - | - | - | - | 0.00 | 0.00 | 0.35 | 0/2 |
| anthropic/claude-sonnet-5.5 | 2 | 1283.50 | 27.50 | 8.50 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 98.40 | 6.50 | 0.89 | 1.82 | 2/2 |
| baseline_chat | 1 | 47.00 | 1.00 | 6.00 | 4.00 | 5.00 | 5.00 | 2.00 | 4.00 | 100.00 | - | 0.09 | 0.09 | 0/1 |
<!-- /table -->
