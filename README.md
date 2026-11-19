# Task-Oriented Sparsification of Dynamic 3D Scene Graphs (Phase A)

Phase A experimental repository for **Task-Oriented Sparsification of Dynamic 3D Scene Graphs under Cognitive Bandwidth Constraints**.

Instead of transmitting all observed 3D scene objects ($30$–$50$ entities) to a wearable sensory display, the proposed sparsifier filters and stabilizes a compact $k$-node subgraph ($k \in \{3, 5, 7, 10, 15\}$) in real time using:
1. **Dynamic Collision Cone & TTC ($R_v$):** Reachable forward motion cone based on relative 3D velocity $\vec{v}_{rel}$ and vertical overlap filtering.
2. **Asymmetric Safety Score ($S_v$):**
   $$S_v = \alpha \cdot R_v \cdot (1 + \gamma \cdot U_v \cdot \mathbb{I}(R_v > \tau_{haz})) + \beta \cdot Q_v \cdot C_v$$
   where perception uncertainty $U_v = 1 - Q_v$ boosts priority for uncertain hazards inside the collision cone rather than suppressing them.
3. **Temporal Hysteresis:** Persistence memory and hysteresis bonus to suppress frame-to-frame flicker ($F_t$).

## Quick Start

```bash
pip install -r requirements.txt
python run_benchmark.py      # Multi-seed Pareto benchmark across k in {3, 5, 7, 10, 15}
python run_ablations.py      # Noise stress test, ablation study, and k=5 error analysis
python simulate_viewer.py    # Generates interactive HTML5 3D + Top-Down viewer (results/simulation.html)
```

## Phase A Benchmark Results (5 Seeds, Medium Noise, 42 Objects)

| Method | Budget $k=5$ Hazard Recall | $k=5$ CVaR-5% Recall | $k=5$ Flicker $F_t$ ($\downarrow$) | Latency (ms/frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Random** | $26.6\% \pm 1.7\%$ | $0.0\%$ | $0.845 \pm 0.009$ | $0.011$ |
| **Semantic** | $59.6\% \pm 9.9\%$ | $11.6\%$ | $0.180 \pm 0.036$ | $0.036$ |
| **Distance** | $65.3\% \pm 5.3\%$ | $16.6\%$ | $0.258 \pm 0.017$ | $0.034$ |
| **Proposed (Ours)** | **$87.2\% \pm 6.8\%$** | **$49.9\%$** | **$0.180 \pm 0.015$** | **$0.551$** |

At $k=7$, **Proposed** reaches **$97.1\% \pm 2.9\%$** Hazard Recall (vs. $79.3\% \pm 5.1\%$ for Distance) while running at $< 0.6\text{ ms/frame}$ ($> 1,500\text{ Hz}$).

![Phase A Pareto Plot](results/benchmark_pareto.png)

## Key Verification & Ablation Findings (`run_ablations.py`)

1. **Noise Robustness ($k=5$):** Proposed consistently outperforms Distance by $>20\%$ Hazard Recall and $\sim 3\times$ CVaR-5% across `clean` ($86.0\%$ vs. $62.0\%$), `medium` ($87.2\%$ vs. $65.3\%$), and `heavy` ($88.6\%$ vs. $68.4\%$) noise levels.
2. **Hysteresis Impact (H3):** Disabling hysteresis (`h=0`) increases Jaccard flicker $F_t$ by **$+57\%$** ($0.180 \to 0.283$ under medium noise; $0.337 \to 0.419$ under heavy noise) with negligible impact on recall.
3. **Asymmetric Risk vs. False Positives Trade-off:** Under `heavy` noise, setting $\gamma = 1.5$ inadvertently boosts low-confidence false-positive ghost detections into the top-$k$ slots (reducing recall from $90.4\%$ at $\gamma=0$ to $88.6\%$). Addressing FP-aware uncertainty weighting is a primary target for Phase B.
4. **Budget Overflow Analysis ($k=5$):** Across $600$ frames, $18.3\%$ of frames contain $>5$ simultaneous hazards, accounting for **$37.4\%$ of all missed hazards**—providing empirical motivation for hierarchical **Graph Coarsening (super-nodes)** in Phase B.
