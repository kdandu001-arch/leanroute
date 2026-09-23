"""Answer simple decisions with Laya only — no LLM call, no cost."""
from leanroute import Leanroute, choice, level, yes_no

lr = Leanroute(api_url="http://localhost:8000")

print("spam probability:", lr.check("FREE crypto signals!!! link in bio", "Is this spam?"))

a = lr.decide("We were billed twice for March. Refund today or we cancel.", {
    "team":   choice("Which team should handle this?", ["billing", "technical", "sales", "other"]),
    "urgent": yes_no("Is there time pressure or a deadline?"),
    "anger":  level("How frustrated is the customer?", ["calm", "annoyed", "furious"]),
})
for k, v in a.items():
    print(k, v)
