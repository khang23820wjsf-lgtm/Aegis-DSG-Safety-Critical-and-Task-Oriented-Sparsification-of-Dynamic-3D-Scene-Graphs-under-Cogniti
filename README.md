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
python run_benchmark.py
```

- Preview a 3D + Top-Down snapshot: `python preview_dsg.py`
- Generate interactive HTML5 simulation: `python simulate_viewer.py` (opens `results/simulation.html`)

## Phase A Benchmark Results (5 Seeds, Medium Noise, 42 Objects)

| Method | Budget $k=5$ Hazard Recall | $k=5$ CVaR-5% Recall | $k=5$ Flicker $F_t$ ($\downarrow$) | Latency (ms/frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Random** | $26.6\% \pm 1.7\%$ | $0.0\%$ | $0.845 \pm 0.009$ | $0.011$ |
| **Semantic** | $59.6\% \pm 9.9\%$ | $11.6\%$ | $0.180 \pm 0.036$ | $0.036$ |
| **Distance** | $65.3\% \pm 5.3\%$ | $16.6\%$ | $0.258 \pm 0.017$ | $0.034$ |
| **Proposed (Ours)** | **$87.2\% \pm 6.8\%$** | **$49.9\%$** | **$0.180 \pm 0.015$** | **$0.551$** |

At $k=7$, **Proposed** reaches **$97.1\% \pm 2.9\%$** Hazard Recall (vs. $79.3\% \pm 5.1\%$ for Distance) while running at $< 0.6\text{ ms/frame}$ ($> 1,500\text{ Hz}$).

![Phase A Pareto Plot](results/benchmark_pareto.png)
