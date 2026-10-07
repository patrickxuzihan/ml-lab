# Knowledge or behavior? What RAG, SFT and RL each fix in a small front-desk model

Controlled experiments with a ~2B open model (Qwen3.5-2B, or Qwen3-1.7B if the tooling isn't ready) working as the front desk of a hospital (triage desk) and a law firm (intake desk). The assistant routes visitors, answers questions about the institution from its knowledge base, and escalates emergencies and anything it can't handle to a person. It never gives diagnoses or legal opinions, and every institution in the project is fictional.

**Status:** planning. Results will appear here as experiments finish, and every number on this page will link to a file under `results/`.

## Questions

1. **Knowledge.** How much of the institution-specific knowledge gap does RAG close, and can SFT or RL replace it? Which methods give outdated answers after the institution's information changes?
2. **Behavior.** How far does SFT get on routing, emergency escalation and escalating when unsure? Does RL with a cost-sensitive reward cut dangerous errors further, and how many extra escalations does that cost?
3. **Combination and transfer.** Are RAG and RL complementary? Does the learned behavior carry over to an institution the model has never seen, given only its knowledge base?
4. **Deployment.** After quantization, what are the latency and memory use on a 2-core, 4 GB CPU server, and do the learned behaviors survive?

## Plan

See [GUIDE.md](GUIDE.md) (in Chinese).
