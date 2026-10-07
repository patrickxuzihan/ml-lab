# CLAUDE.md

Instructions for Claude Code working in this repository.

## What this repo is

`llm-lab` is Zihan Xu's portfolio of hands-on LLM projects. Each project lives in its own folder and is self-contained, with its own README, requirements, scripts, tests and results.

Current project: `frontdesk-slm/`, a set of controlled experiments on what RAG, SFT and RL each fix when a ~2B model works as the front desk of an institution (a hospital triage desk and a law-firm intake desk). The plan, experiment matrix, phases and acceptance criteria are in `frontdesk-slm/GUIDE.md` (written in Chinese). Read it before any task in that folder. Work on one phase at a time. Don't change the plan on your own; propose changes in the PR description.

## Where code runs

- Cloud sessions have no GPU (4 vCPU, 16 GB RAM, 30 GB disk, Ubuntu x86_64). Never start full training or full evaluation runs in a session.
- GPU jobs run on the owner's Google Colab Pro account, and the owner pushes the results. Notebooks are thin launchers: they clone the repo, install requirements and call scripts, and all logic lives in the repo. Colab sessions can disconnect, so long jobs save checkpoints to Google Drive and resume from the latest one.
- Deployment tests run on the owner's server: 2 vCPU, 4 GB RAM, no GPU, in mainland China. It can't reach Hugging Face, so models are downloaded from ModelScope. Code that runs there must fit in 4 GB of RAM and must not need torch.
- The owner's own computer runs nothing.
- Everything that doesn't need a GPU must be testable on CPU: data loading and validation, retrieval, prompt building, output parsing, scoring, rewards, metrics, configs, analysis and plotting. The core modules (`data`, `retrieval`, `prompts`, `actions`, `scoring`, `metrics`) must not import torch.
- GPU code (vLLM generation, training) must import cleanly on CPU. It must also have a `--smoke` mode that runs end to end on CPU with a tiny model and a few examples in under about 5 minutes.
- Tests never depend on downloaded datasets; they use the small fixtures under `tests/fixtures/`. Public datasets are downloaded by `scripts/`.
- If a download fails because the network allowlist blocks a host, report it instead of working around it.
- Target Python 3.10–3.13. Cloud sessions run 3.13; check the Python version on Colab and on the server before pinning dependencies.

## Rules

1. Never invent numbers. Every number in a README, note or PR must come from a file under `results/`. If that file doesn't exist yet, write TODO.
2. Never commit downloaded datasets, model weights (including GGUF files), checkpoints, `wandb/` folders or files over about 5 MB. For public datasets, commit only the IDs of sampled examples and the scripts that produce them. Synthetic data we write ourselves (institution profiles, knowledge bases, questions and escalation cases under `synthetic/`) is committed, with a dataset card.
3. Every run writes `results/<run_id>/` with the full effective config, git commit hash, seed, package versions, hardware (GPU type, or CPU model for deployment tests) and metrics. Run IDs look like `20261120_q35-2b_rr-asym_hosp_s0`.
4. All experiments share one evaluation protocol (GUIDE.md §4.3): the prompt template, action schema, retrieval settings, decoding settings and scoring code. Changing it invalidates earlier results. If a change is unavoidable, record it in `notes/` and re-run the affected evaluations.
5. Model output is untrusted. Parse it strictly against the action schema and never execute or `eval` it. The demo server is public: run it as a non-root user, protect the demo with a password and serve only synthetic data.
6. The assistant only routes visitors, answers institution facts from the knowledge base and escalates to a person. It never gives medical diagnoses or legal opinions and never promises outcomes. Prompts, data and the demo must stay within this scope.
7. Synthetic data must say that it is synthetic, record how it was made, and pass the owner's spot check before it is used for evaluation.
8. TRL, vLLM, PEFT and llama.cpp APIs change often. Check the docs for the installed versions before using them, and pin versions once a setup works.
9. Keep core logic in small, pure, unit-tested functions, with thin adapters for TRL, vLLM and llama.cpp.
10. Before opening a PR, make sure `ruff check`, `ruff format --check` and `pytest` all pass.

## Pull requests

- One phase, or one clearly scoped task, per branch and PR.
- The description covers what changed and checks off the phase's acceptance criteria from GUIDE.md one by one. If the next step needs Colab or the server, it also gives the exact commands the owner should run there, ready to copy.
- End every PR description with a `学习要点` section in Chinese: 3–6 bullets that explain the key concepts and design choices, and what the owner should be able to explain in an interview.
- When the owner pushes results, sanity-check them before building on them. Flag anything suspicious, such as gold-as-prediction scores below 100%, results above the ideal-retrieval upper bound, or training curves that don't move.

## Language

Code, comments, commit messages and READMEs are in English. Notes under `notes/` and the `学习要点` sections are in Chinese.
