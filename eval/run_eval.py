"""Evaluate the support agent on a held-out test set.

The 60 test messages in test_set.jsonl were written separately from the
training examples in data/intents.csv, so this measures how the agent handles
phrasing it hasn't seen.

Run:  python -m eval.run_eval
"""
import json
from datetime import date
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

from agent.agent import SupportAgent

HERE = Path(__file__).resolve().parent
TODAY = date(2026, 10, 27)  # fixed date so refund-window results are reproducible


def main():
    cases = [json.loads(l) for l in (HERE / "test_set.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for case in cases:
        agent = SupportAgent(today=TODAY)  # fresh conversation per case
        for prior in case.get("history", []):
            agent.handle(prior)
        r = agent.handle(case["message"])
        rows.append((case, r))

    y_true = [c["intent"] for c, _ in rows]
    y_pred = [r.intent for _, r in rows]
    intent_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(rows)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    action_acc = sum(c["action"] == r.action for c, r in rows) / len(rows)

    tp = sum(c["escalate"] and r.escalated for c, r in rows)
    fp = sum(not c["escalate"] and r.escalated for c, r in rows)
    fn = sum(c["escalate"] and not r.escalated for c, r in rows)
    esc_precision = tp / (tp + fp) if tp + fp else 0.0
    esc_recall = tp / (tp + fn) if tp + fn else 0.0

    kb = [(c, r) for c, r in rows if "source" in c]
    source_hit = sum(c["source"] in r.sources for c, r in kb) / len(kb)

    inj = [(c, r) for c, r in rows if c.get("injection")]
    inj_blocked = sum("prompt_injection" in r.flags for _, r in inj) / len(inj)

    pii = [(c, r) for c, r in rows if c.get("pii")]
    pii_caught = sum(all(f"pii_{p}" in r.flags for p in c["pii"]) for c, r in pii) / len(pii)

    # Safety: did the agent ever approve a refund that the policy says no to?
    unsafe = sum(r.action == "refund_approved" and c["action"] != "refund_approved" for c, r in rows)

    metrics = {
        "Test cases": len(rows),
        "Intent accuracy": f"{intent_acc:.1%}",
        "Intent macro F1": f"{macro_f1:.3f}",
        "Correct action": f"{action_acc:.1%}",
        "Escalation precision": f"{esc_precision:.1%}",
        "Escalation recall": f"{esc_recall:.1%}",
        "Policy answers citing the right article": f"{source_hit:.1%} ({len(kb)} questions)",
        "Prompt injections blocked": f"{inj_blocked:.0%} ({len(inj)} attempts)",
        "Personal details redacted": f"{pii_caught:.0%} ({len(pii)} messages)",
        "Refunds wrongly approved": unsafe,
    }

    failures = [(c, r) for c, r in rows if c["action"] != r.action or c["intent"] != r.intent]

    lines = ["# Evaluation results", "",
             f"Run on {len(rows)} held-out test messages (`eval/test_set.jsonl`), offline mode.", "",
             "| Metric | Result |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in metrics.items()]
    lines += ["", "## Per-intent report", "", "```",
              classification_report(y_true, y_pred, zero_division=0), "```", "",
              "## Cases the agent got wrong", ""]
    if failures:
        lines += ["| # | Message | Expected | Got |", "|---|---|---|---|"]
        for c, r in failures:
            lines.append(f"| {c['id']} | {c['message']} | {c['intent']} / {c['action']} | "
                         f"{r.intent} / {r.action} |")
    else:
        lines.append("None.")
    report = "\n".join(lines) + "\n"
    (HERE / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
