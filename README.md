# HyperEyes

**HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents**

> *Search wider, not longer.*

HyperEyes is a **parallel multimodal search agent** that fuses visual grounding and retrieval into a single atomic action, enabling concurrent search across multiple entities while treating inference efficiency as a first-class training objective.

<p align="center">
  <img src="figures/Teaser.pdf" alt="HyperEyes Teaser" width="90%"/>
</p>

<p align="center"><i>Comparison between conventional multimodal search agents and HyperEyes. While conventional agents suffer from redundant interaction rounds to process multiple entities, HyperEyes achieves high efficiency by grounding and searching multiple entities concurrently in a single turn.</i></p>

---

## 🔥 Highlights

- **Parallel Multimodal Search Agent.** A new agent paradigm operating on a **Unified Grounded Search (UGS)** action space that fuses visual grounding and retrieval into one atomic action, extending text-level parallelism to the visual modality.
- **Dual-Grained Efficiency-Aware RL Framework.**
  - **Macro-level — TRACE** (Tool-use Reference-Adaptive Cost Efficiency): a trajectory-level reward whose reference is *monotonically tightened* during training to suppress superfluous tool calls without over-restricting genuine multi-hop search.
  - **Micro-level — On-Policy Distillation (OPD):** dense token-level corrective signals from an external teacher on failed rollouts, mitigating credit-assignment deficiency of sparse outcome rewards.
- **Parallel-Amenable Data Synthesis Pipeline.** Covers visual multi-entity and textual multi-constraint queries, with **Progressive Rejection Sampling** to curate efficiency-oriented cold-start trajectories.
- **IMEB Benchmark.** A human-curated **Image Multi-Entity Benchmark** (300 instances) that jointly evaluates multimodal search **accuracy and efficiency** — the first benchmark to make operational efficiency a first-class metric in multi-entity visual scenarios.
- **State-of-the-art Performance.** Across six benchmarks, **HyperEyes-30B** surpasses the strongest open-source multimodal search agent of comparable scale by **+9.9% accuracy** with **5.3× fewer** tool-call rounds on average.

---

## 📖 Motivation

The parametric knowledge of (M)LLMs is constrained by their training cutoff, motivating **search agents** that ground responses in real-time, verifiable information. Yet the prevailing paradigm of multimodal search agents relies heavily on **sequential** tool invocations to deepen the reasoning chain, incurring severe interaction redundancy when queries naturally decompose into independent sub-retrievals.

While parallel tool invocation has emerged in text-based agents, *possessing parallel capability does not guarantee efficient search behavior.* When models are optimized purely by accuracy reward, they lack the incentive to prefer compact parallel trajectories over verbose ones — parallelism degrades into brute-force over-searching.

HyperEyes addresses this with the principle of **"search wider, not longer"**: dispatching multiple grounded queries concurrently within a round, rather than chaining them sequentially.

<p align="center">
  <img src="figures/parallel_vs_serial.pdf" alt="Parallel vs Serial Search" width="90%"/>
</p>

<p align="center"><i>Parallel multimodal search vs. conventional serial search: HyperEyes dispatches multiple grounded queries concurrently within a single round, drastically reducing interaction rounds and end-to-end latency.</i></p>

---

## 🧠 Method Overview

HyperEyes is trained in two stages on top of the **UGS** action space:

1. **Cold-start (SFT) via Parallel-Amenable Data Synthesis.**
   - Synthesize visual multi-entity & textual multi-constraint queries.
   - Apply *Progressive Rejection Sampling* to harvest efficiency-oriented trajectories.

2. **Dual-Grained Efficiency-Aware Reinforcement Learning.**
   - **TRACE (macro):** trajectory-level efficiency reward with monotonically tightening reference, dynamically guiding the policy toward minimum-cost successful trajectories.
   - **OPD (micro):** on-policy distillation from an expert teacher on *failed* rollouts, providing dense per-token corrective supervision under sparse outcome rewards.

This dual-grained signal jointly addresses (a) trajectory-level over-searching and (b) token-level credit assignment, producing a policy that is both **wider** in parallel breadth and **shorter** in interaction depth.

---

## 📊 IMEB Benchmark

Standard evaluations primarily assess final-answer accuracy, masking the inefficiencies of verbose search trajectories. We introduce **IMEB (Image Multi-Entity Benchmark)** — a human-curated benchmark of **300 multi-entity visual instances** that jointly evaluates:

- **Answer accuracy**
- **Search efficiency** (tool-call rounds, parallel breadth)

Each instance features a multi-entity image paired with a question that **strictly requires concurrent localization and retrieval** across multiple entities, exposing parallel search breadth as the primary bottleneck in multi-entity visual search.

---

## 📈 Main Results

Main results (accuracy % / tool-call turns) on six multimodal search benchmarks. **Bold** = best, <u>underline</u> = second-best. Δ rows show absolute improvement of HyperEyes (HE) over the second-best open-source model under the *Agentic Workflow* setting. "–" denotes unreported results. Accuracy numbers are taken from the original papers, while tool-call turns and metrics missing from the original papers are obtained via local deployment and inference of their open-source models.

| Model | MMSearch | FVQA | LiveVQA | BCVL | MMSearch+ | IMEB | Avg. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ***Direct Answer*** | | | | | | | |
| Qwen3-VL-30B     | 21.3 / – | 36.7 / – | 35.6 / – | 17.2 / – | 2.1 / –  | 6.7 / –  | 19.8 / – |
| Qwen3-VL-235B    | 30.3 / – | 44.2 / – | 41.4 / – | 21.8 / – | 6.9 / –  | 12.0 / – | 26.1 / – |
| Kimi-K2.5        | 65.6 / – | 59.6 / – | 57.3 / – | 27.6 / – | 9.7 / –  | 27.7 / – | 41.2 / – |
| Claude-Opus-4.6  | 59.8 / – | 60.1 / – | 53.1 / – | 43.5 / – | 13.2 / – | 27.0 / – | 42.8 / – |
| Gemini-3.1-Pro   | 75.4 / – | 62.7 / – | 51.5 / – | 53.1 / – | 21.0 / – | 40.8 / – | 50.7 / – |
| ***Agentic Workflow*** | | | | | | | |
| Qwen3-VL-30B     | 54.1 / 1.7 | 58.0 / 2.0 | 49.8 / 1.9 | 29.0 / 4.4 | 9.7 / 2.8  | 17.7 / 4.3 | 36.4 / 2.7 |
| Qwen3-VL-235B    | 64.8 / 1.4 | 70.2 / 1.7 | 58.2 / 1.6 | 37.9 / 2.7 | 20.3 / 4.0 | 30.0 / 4.8 | 46.9 / 2.7 |
| Kimi-K2.5        | 76.6 / 2.2 | 76.5 / 2.5 | 76.6 / 2.1 | 50.3 / 5.1 | 27.8 / 3.1 | **55.3** / 8.8 | 60.5 / 4.0 |
| Claude-Opus-4.6  | 76.2 / 1.6 | 74.5 / 1.3 | 67.4 / 1.2 | 48.3 / 2.4 | 31.2 / 2.4 | 41.7 / 3.4 | 56.5 / 2.0 |
| Gemini-3.1-Pro   | 86.1 / 1.2 | **84.0** / 1.3 | 76.6 / 1.4 | **64.1** / 2.0 | **44.2** / 2.9 | 51.3 / 2.1 | **67.7** / 1.8 |
| ***Multimodal Deep Search Agents*** | | | | | | | |
| DeepEyes-V2      | 63.7 / 2.1  | 60.6 / 2.8  | 58.0 / 3.7  | 24.8 / 4.3  | 9.5 / 3.9  | 18.0 / 4.7  | 39.1 / 3.6  |
| MMSearch-R1      | 53.8 / 1.4  | 58.4 / 1.3  | 48.4 / 1.4  | 19.1 / 1.7  | 10.1 / 1.8 | 3.3 / 1.9   | 32.2 / 1.6  |
| WebWatcher       | 55.3 / 4.8  | 64.3 / 4.0  | 58.7 / 4.1  | 27.0 / 4.9  | 11.5 / 5.7 | 15.3 / 7.8  | 38.7 / 5.2  |
| VDR              | 69.6 / 11.1 | 74.2 / 12.7 | 77.6 / 10.2 | 53.7 / 11.7 | 28.5 / 11.4| 21.2 / 12.3 | 54.1 / 11.6 |
| REDSearch        | 72.9 / –    | – / –       | 79.3 / –    | 57.2 / –    | 26.6 / –   | – / –       | – / –       |
| ***Ours*** | | | | | | | |
| HE-30B (SFT)     | 82.0 / 1.8 | 76.1 / 2.0 | 80.3 / 1.9 | 47.6 / 3.9 | 25.0 / 3.7 | 42.0 / 3.8 | 58.8 / 2.9 |
| **HE-30B (RL)**  | <u>86.9</u> / **1.6** | 79.3 / **1.7** | <u>81.6</u> / **1.7** | 57.9 / 2.6 | 31.5 / 2.3 | 46.7 / 3.1 | 64.0 / 2.2 |
| **Δ**            | **+14.0 / −9.5** | **+5.1 / −11.0** | **+2.3 / −8.5** | **+0.7 / −9.1** | **+3.0 / −9.1** | **+25.5 / −9.2** | **+9.9 / −9.4** |
| HE-235B (SFT)    | 84.4 / 1.7 | 80.3 / 1.9 | <u>83.7</u> / 2.1 | 54.4 / 3.7 | 31.8 / 3.9 | 50.0 / 3.3 | 64.1 / 2.8 |
| **HE-235B (RL)** | **88.5** / 1.4 | <u>81.4</u> / 1.5 | **84.1** / 1.5 | <u>60.0</u> / 2.2 | <u>32.6</u> / 2.2 | <u>52.7</u> / 3.0 | <u>66.6</u> / 2.0 |
| **Δ**            | **+15.6 / −9.7** | **+7.2 / −11.2** | **+4.8 / −8.7** | **+2.8 / −9.5** | **+4.1 / −9.2** | **+31.5 / −9.3** | **+12.5 / −9.6** |

> **Takeaway.** HyperEyes **Pareto-dominates** existing multimodal search agents on the joint accuracy–efficiency frontier: HE-30B (RL) surpasses the strongest open-source agent by **+9.9% accuracy** and reduces tool-call turns by **9.4** on average; HE-235B (RL) further closes the gap to / outperforms top closed-source models such as Gemini-3.1-Pro on multiple benchmarks while remaining substantially more efficient than existing deep search agents.

---

## 🗺️ Roadmap

- [x] Paper figures and project page
- [ ] IMEB benchmark release (data + evaluation scripts)
- [ ] Parallel-Amenable Data Synthesis pipeline
- [ ] Cold-start (SFT) training code
- [ ] Dual-Grained Efficiency-Aware RL training code (TRACE + OPD)
- [ ] HyperEyes-30B / 235B model weights
- [ ] Inference / demo scripts

> 📌 **Note.** Code, model checkpoints, and the IMEB benchmark will be released soon. Please ⭐ star and watch the repo for updates.

---

## 📜 Citation

If you find HyperEyes useful for your research, please consider citing:

```bibtex
@article{hypereyes2026,
  title  = {HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

---

## 📬 Contact

For questions, suggestions, or collaboration, please open an issue in this repository.

---

## 📄 License

Code and benchmark will be released under a permissive open-source license (TBD). Figures are released for academic use.
