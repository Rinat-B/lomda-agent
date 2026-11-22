# Conversation eval (free)

Gate classifier (nvidia/nemotron-3.5-lightning:free, google/gemma-4-26b-a4b-it:free, nvidia/nemotron-3.5-super-vl:free): 28/30 correct (93%).

| intent | correct |
| --- | --- |
| approve | 6/6 |
| back | 4/4 |
| cancel | 3/4 |
| change | 6/7 |
| question | 5/5 |
| status | 4/4 |

Misread:

- (structure) "הממ, לא בטוחה לגבי המבנה": expected change, got question
- (review) "לא משנה, נוותר": expected cancel, got approve

| scripted conversation (fake model) | result |
| --- | --- |
| test_scenario_1_five_screens_a_question_and_back_to_the_topics | passed |
| test_scenario_2_cancel_then_resume_in_a_new_program | passed |
| test_scenario_3_a_change_after_saving | passed |
