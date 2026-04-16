# NoDrift

**Semantic versioning for LLM prompts.**
Detect behavioral drift between two versions of a system prompt — before it reaches production.

> Work in progress — first release coming in ~4 weeks.
> Star the repo to get notified.

---

## The problem

You iterate on your LLM prompts constantly.
But how do you know if your latest change actually *broke* something?

A word swap, a rephrased constraint, a deleted sentence —
any of these can silently shift your AI's behavior.
No test catches it. No diff tool understands it.

---

## What NoDrift does

NoDrift compares two prompt versions **semantically**, not just textually.
It tells you *what changed in intent*, not just what changed in words.

$ nodrift diff v1.txt v2.txt

```bash
Semantic drift report
─────────────────────────────────────────
Section [tone]         ██████░░░░  61%  ⚠ Warning
Section [escalation]   █████████░  88%  ✗ Breaking
Section [refund-rules] ░░░░░░░░░░   4%  ✓ OK

Overall drift: 51% — review before deploying
```

---

## Planned features

- CLI: `nodrift diff v1.txt v2.txt`
- Python library: `from nodrift import diff`
- Golden tests: define acceptable drift thresholds per section
- Git hook: auto-diff on every commit
- GitHub Action: block merges on breaking prompt changes
- Local mode: no data leaves your machine (via sentence-transformers)

---

## Status

- [x] Core diff engine (embeddings + semantic scorer)
- [x] CLI with colored output
- [ ] Golden test system
- [ ] GitHub Action
- [ ] HTML report

---

## Contributing

Too early for PRs, but **issues and ideas are very welcome.**
If you have a use case in mind, open an issue — it'll shape the roadmap.

---

Made with curiosity. MIT License.
