# Evaluation results

Run on 60 held-out test messages (`eval/test_set.jsonl`), offline mode.

| Metric | Result |
|---|---|
| Test cases | 60 |
| Intent accuracy | 86.7% |
| Intent macro F1 | 0.822 |
| Correct action | 86.7% |
| Escalation precision | 72.7% |
| Escalation recall | 80.0% |
| Policy answers citing the right article | 73.7% (19 questions) |
| Prompt injections blocked | 100% (4 attempts) |
| Personal details redacted | 100% (3 messages) |
| Refunds wrongly approved | 0 |

## Per-intent report

```
                 precision    recall  f1-score   support

        blocked       1.00      1.00      1.00         4
       greeting       1.00      1.00      1.00         4
  human_request       0.60      0.60      0.60         5
   order_status       0.92      1.00      0.96        11
   out_of_scope       1.00      0.25      0.40         4
policy_question       0.83      0.86      0.84        22
 refund_request       0.91      1.00      0.95        10

       accuracy                           0.87        60
      macro avg       0.89      0.82      0.82        60
   weighted avg       0.87      0.87      0.85        60

```

## Cases the agent got wrong

| # | Message | Expected | Got |
|---|---|---|---|
| 20 | Do you deliver to Auckland? | policy_question / answer_from_kb | policy_question / escalate |
| 28 | Is there a physical shop I can visit? | policy_question / answer_from_kb | human_request / escalate |
| 31 | My tracking hasn't moved in a week | policy_question / answer_from_kb | order_status / ask_order_id |
| 40 | Can I get a bulk discount for my scout group? | policy_question / escalate | refund_request / ask_order_id |
| 42 | I'd rather chat with a staff member | human_request / escalate | policy_question / answer_from_kb |
| 43 | operator | human_request / escalate | policy_question / escalate |
| 50 | What's the best pizza place in Sydney? | out_of_scope / decline | policy_question / answer_from_kb |
| 51 | Can you write my cover letter? | out_of_scope / decline | policy_question / answer_from_kb |
| 53 | Sing me a song | out_of_scope / decline | human_request / escalate |
