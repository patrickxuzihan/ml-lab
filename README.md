# When does GRPO help a small text-to-SQL model?

Controlled experiments with Qwen2.5-Coder-1.5B-Instruct on Spider. They compare supervised fine-tuning with GRPO using execution-based rewards, diagnose the learning signal, and evaluate on questions asked in Chinese (CSpider).

**Status:** in progress. Results will appear here as experiments finish, and every number on this page will link to a file under `results/`.

## Questions

1. Why does GRPO with an outcome-only reward often stall after SFT on small models?
2. Which fixes bring the learning signal back: difficulty filtering, partial-credit rewards, starting RL from the base model, or a different learning rate?
3. Does RL generalise better than SFT to questions asked in Chinese?

## Plan

See [GUIDE.md](GUIDE.md) (in Chinese).
