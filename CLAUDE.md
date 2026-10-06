# CLAUDE.md

Instructions for Claude Code working in this repository.

## What this repo is

`llm-lab` is Zihan Xu's portfolio of hands-on LLM projects. Each project lives in its own folder and is self-contained, with its own README, requirements, scripts, tests and results.

Current project: `text2sql-grpo/`, a set of controlled experiments on when GRPO helps a ~1.5B model write SQL. The plan, experiment matrix, phases and acceptance criteria are in `text2sql-grpo/GUIDE.md` (written in Chinese). Read it before any task in that folder. Work on one phase at a time. Don't change the plan on your own; propose changes in the PR description.

## Where code runs

- Cloud sessions have no GPU (4 vCPU, 16 GB RAM, 30 GB disk, Ubuntu x86_64). Never start full training or full evaluation runs in a session. The repo owner runs GPU jobs on a rented machine and pushes the results.
- Everything that doesn't need a GPU must be testable on CPU: data loading, prompt building, SQL execution, metrics, rewards, configs, analysis and plotting. The core modules (`data`, `sql_exec`, `rewards`, `metrics`) must not import torch.
- GPU code (vLLM generation, training) must import cleanly on CPU. It must also have a `--smoke` mode that runs end to end on CPU with a tiny model and a few examples in under about 5 minutes.
- Tests never depend on downloaded datasets; they use the small SQLite fixtures under `tests/fixtures/`. Spider, CSpider and BIRD are downloaded by `scripts/` on the GPU machine.
- If a download fails because the network allowlist blocks a host, report it instead of working around it.
- Target Python 3.10–3.12 so the code stays compatible with vLLM.

## Rules

1. Never invent numbers. Every number in a README, note or PR must come from a file under `results/`. If that file doesn't exist yet, write TODO.
2. Never commit datasets, model weights, checkpoints, `wandb/` folders or files over about 5 MB. Commit only small artifacts: configs, `metrics.json`, `predictions.jsonl`, training curves exported as CSV, and figures.
3. Every run writes `results/<run_id>/` with the full effective config, git commit hash, seed, package versions, GPU type and metrics. Run IDs look like `20261020_q15b_grpo-r2_s0`.
4. All experiments share one evaluation protocol (GUIDE.md §4.1): the prompt template, schema serialization, decoding settings and metric code. Changing it invalidates earlier results. If a change is unavoidable, record it in `notes/` and re-run the affected evaluations.
5. Model-generated SQL is untrusted. Execute it read-only (SQLite `mode=ro`), with a time limit and a cap on the number of rows returned.
6. TRL, vLLM and PEFT APIs change often. Check the docs for the installed versions before using them, and pin versions once a setup works.
7. Keep core logic in small, pure, unit-tested functions, with thin adapters for TRL and vLLM.
8. Before opening a PR, make sure `ruff check`, `ruff format --check` and `pytest` all pass.

## Pull requests

- One phase, or one clearly scoped task, per branch and PR.
- The description covers what changed and checks off the phase's acceptance criteria from GUIDE.md one by one. If the next step needs a GPU, it also gives the exact commands the owner should run there, ready to copy.
- End every PR description with a `学习要点` section in Chinese: 3–6 bullets that explain the key concepts and design choices, and what the owner should be able to explain in an interview.
- When the owner pushes results, sanity-check them before building on them. Flag anything suspicious, such as gold-as-prediction accuracy well below 100%, baselines far from published references, or training curves that don't move.

## Language

Code, comments, commit messages and READMEs are in English. Notes under `notes/` and the `学习要点` sections are in Chinese.
