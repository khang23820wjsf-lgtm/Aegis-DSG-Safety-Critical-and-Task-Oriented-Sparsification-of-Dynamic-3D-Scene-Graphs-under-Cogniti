# Aegis-DSG: Safety-Critical & Task-Oriented Sparsification of Dynamic 3D Scene Graphs under Cognitive Bandwidth Constraints

**Aegis-Core v2.0** is a real-time (~337 Hz on CPU) downstream sparsification engine for Dynamic 3D Scene Graphs (DSGs) built for wearable assistive displays, AR glasses, and bandwidth-constrained mobile robots.

Instead of flooding a user or planner with $60$–$100+$ raw 3D scene objects, **Aegis-Core v2.0** (`src/aegis_core.py`) compresses the scene into a stable, safety-preserving subgraph ($|V'| \le k$, e.g. $k=5$) through a unified **4-layer pipeline**:

1. **Layer 1 — Temporal Belief Filter & Object Permanence:** Tracks existence probability $p_t(v)$ via EMA to reject 1-frame false-positive ghosts while retaining briefly occluded hazards in the front hemisphere.
2. **Layer 2 — Continuous 3D Kinematic Risk Field:** Combines a near-body hemisphere safety clearance ($\sigma_{\text{near}} = 0.50\text{ m}$ to eliminate side-swipe blind spots during sharp turns), a forward 3D Time-to-Collision (TTC) cone, close-range manipulation reach ($2.20\text{ m}$) for intrinsic hazards, and dual-tier asymmetric uncertainty weighting.
3. **Layer 3 — Free-Space-Preserving Coarsening & Non-Coarsenable Atomic Hazard Invariant:** Uses per-class **Union-Find** clustering ($\text{gap} < 0.70\text{ m}$) to collapse dense static clutter into compact super-nodes (e.g., `cabinet/wardrobe_group(x15)`), while enforcing a **hard mathematical invariant** that intrinsic sharp/hot/human hazards (`knife`, `scissors`, `stove`, `stair_edge`, `pedestrian`) are **never merged or semantically erased**.
4. **Layer 4 — Dynamic-Budget Monotone Submodular Selection ($1 - 1/e$) + Hysteresis:** Maximizes a safety-dominant facility-location objective with $(1 - 1/e)$ approximation guarantee, early stopping when marginal utility $< \tau_{\min}$, and temporal hysteresis to suppress flicker ($F_t$).

![Meta Aria Digital Twin Aegis-Core v2.0 Showcase](results/aria_adt_aegis_v2_showcase.png)
*Real egocentric frame from Meta Project Aria Digital Twin (`Apartment_release_golden_skeleton_seq100`): 89 visible 3D objects compressed into 5 clean entities in `2.9 ms`, collapsing 15 cabinets into `cabinet/wardrobe_group(x15)` while preserving the small butcher knife as `[!] ATOMIC: KNIFE | 0.5m | R=0.92`.*

---

## Positioning vs. Hydra, D-Lite, and Clio

| System | Role | Strengths | Limitations in Safety-Critical Egocentric Compression |
| :--- | :--- | :--- | :--- |
| **Hydra** (MIT SPARK) | 3D Scene Graph Construction | Builds full SLAM/Mesh/Places/Rooms/Objects DSG | Upstream map builder; outputs $80$–$500+$ nodes without bandwidth compression |
| **D-Lite** (ICRA 2024) | Topological DSG Sparsification | Preserves global shortest-path distances across rooms | Static geometry only; ignores velocity/TTC and drops small hazards (`35.8%` coverage) |
| **Clio** (RA-L 2024) | Task-Driven IB Compression | Open-vocabulary task grouping via CLIP + Information Bottleneck | Safety-agnostic (`19.6%` coverage), high flicker (`0.682`), heavy $O(N^3)$ latency (`~873 ms`) |
| **Aegis-Core v2.0 (Ours)** | **Real-Time Safety-Critical Sparsifier** | **Kinematic risk, atomic hazard invariant, temporal stability, `3.0 ms` (~337 Hz)** | **Complementary plug-in downstream of Hydra; lightweight task semantics (no heavy VLM yet)** |

---

## Real-World Benchmark: 11 Diverse Meta Project Aria Digital Twin (ADT) Sequences

Evaluated on **11 real human egocentric sequences** from **Meta Project Aria Digital Twin (ICCV 2023)** spanning 6 real activity types (`golden_skeleton`, `clean`, `meal`, `work`, `decoration`, `multiskeleton_party`, `multiuser_clean`), comparing `Distance`, `D-Lite (True Dijkstra)`, `Clio (True aIB)`, and `Aegis-Core v2.0` at $k=5$:

### Grand Aggregate Across All 11 Real Meta Aria Sequences (Mean $\pm$ 95% CI, $k=5$)

| Method | Hazard Coverage ($\uparrow$) | CVaR-5% Worst Frames ($\uparrow$) | Jaccard Flicker $F_t$ ($\downarrow$) | Latency (ms/frame $\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| **Distance** | $18.2\% \pm 10.0\%$ | $2.3\% \pm 4.5\%$ | $0.125 \pm 0.032$ | `0.12 ms` |
| **D-Lite (True Dijkstra)** | $35.8\% \pm 9.8\%$ | $6.7\% \pm 5.7\%$ | $0.199 \pm 0.037$ | `21.96 ms` |
| **Clio (True aIB)** | $19.6\% \pm 7.1\%$ | $4.5\% \pm 6.0\%$ | $0.682 \pm 0.053$* | `872.55 ms` |
| **Aegis-Core v2.0 (Ours)** | **$88.5\% \pm 3.0\%$** | **$48.0\% \pm 12.5\%$** | **$0.133 \pm 0.026$** | **`2.97 ms` (~337 Hz)** |

*\*Note: Because exact $O(N^3)$ `Clio (aIB)` averages `872.6 ms/frame` (up to `2.27 s/frame` on dense party scenes), `Clio` was evaluated at 8× temporal subsampling (`frames[::8]`) across the 11-sequence sweep; even on dense consecutive frames (`golden_skeleton`), `Clio` exhibits `F_t = 0.619`.*

### Per-Sequence Breakdown (11/11 Wins — Zero Cherry-Picking)

| Meta Aria ADT Sequence (Activity) | Avg Obs | Distance | D-Lite | Clio | **Aegis-Core v2.0** | Flicker $F_t$ | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `golden_skeleton_seq100_sample` | 85 | 26.8% | 45.2% | 30.0% | **93.9%** | 0.023 | 2.98 ms |
| `clean_seq150` | 73 | 8.0% | 32.4% | 12.6% | **88.9%** | 0.140 | 2.71 ms |
| `clean_seq131` | 84 | 10.3% | 9.4% | 32.8% | **97.5%** | 0.108 | 3.51 ms |
| `meal_seq142` | 54 | 0.2% | 19.1% | 9.3% | **84.0%** | 0.140 | 1.73 ms |
| `meal_seq146` | 81 | 12.7% | 24.8% | 17.1% | **92.4%** | 0.175 | 3.38 ms |
| `work_seq139` | 61 | 13.9% | 30.2% | 11.7% | **91.6%** | 0.117 | 2.02 ms |
| `work_seq106` | 70 | 15.0% | 36.7% | 6.2% | **86.9%** | 0.143 | 2.68 ms |
| `decoration_seq135` | 24 | 2.1% | 27.4% | 26.7% | **81.9%** | 0.112 | 0.79 ms |
| `decoration_seq140` | 44 | 61.4% | 62.6% | 39.9% | **86.1%** | 0.176 | 2.13 ms |
| `multiskeleton_party_seq127` | 81 | 22.7% | 43.5% | 3.3% | **82.0%** | 0.169 | 4.52 ms |
| `multiuser_clean_seq120` | 76 | 27.3% | 62.0% | 26.1% | **87.9%** | 0.157 | 6.16 ms |

![11-Sequence Real Meta Aria ADT Benchmark](results/aria_11seq_benchmark.png)

> **Honest Failure-Mode Note (`decoration_seq135`):** On 8/120 frames where the wearer stands nearly stationary ($v \approx 0$) looking away, `Aegis-Core v2.0` early-stops at `1/5` entities via `tau_min = 0.15` to save cognitive bandwidth, missing two static non-atomic furniture pieces at `1.7–2.4 m` (0 atomic hazards missed). This highlights the explicit trade-off between `tau_min` bandwidth conservation and stationary proximity conservatism.

---

## Quick Start

```bash
pip install -r requirements.txt
python -m unittest tests/test_aegis_core.py -v   # Core unit tests (6/6 invariants: belief filter, atomic knife, Union-Find, dynamic budget)
python run_aria_adt_benchmark.py                 # Real Meta Aria ADT single-sequence benchmark & showcase visual
python run_aria_10seq_benchmark.py               # 11-sequence Meta Aria ADT multi-activity benchmark
python simulate_aegis_core.py                    # Interactive Aegis-Core v2.0 simulator -> results/aegis_core_sim.html
python run_prong3_benchmark.py                   # Phase B Grand Benchmark: True D-Lite, Clio (aIB), GNN-GAT, Submodular, Aegis
python run_mujoco_3dssg.py                       # Real 3DSSG Scanned Rooms x MuJoCo 4.0 Rigid-Body Physics
python run_real_photo_aegis.py                   # Real RGB Room Photo -> YOLOv8 + MiDaS 3D Depth -> Aegis-Core
python run_benchmark.py                          # Phase A: Multi-seed Pareto benchmark across k in {3, 5, 7, 10, 15}
python run_ablations.py                          # Phase A: Noise stress test, ablation study, and k=5 error analysis
python run_scenarios.py                          # Phase A: 4-environment benchmark (corridor, blind corner, lobby, 3D vertical)
```

---

## Phase A Experimental Setup & Hyperparameters

| Parameter | Value | Description |
| :--- | :---: | :--- |
| Scene & Horizon | $42$ objects, $120$ frames ($dt = 0.1\text{s}$) | 3D corridor ($4\text{m} \times 24\text{m}$), 5 seeds (`42..46`) |
| Cognitive Budgets ($k$) | $\{3, 5, 7, 10, 15\}$ | Max active entities retained per frame |
| Collision Cone | $\Delta t = 3.2\text{s}$, $\pm 35^\circ$ | Reachable forward cone & TTC window |
| $\alpha, \beta, \gamma, \tau_{haz}$ | $1.0,\ 0.20,\ 1.50,\ 0.30$ | Safety, semantic, asymmetric uncertainty, hazard threshold |
| Hysteresis | bonus $= 0.18$, $\theta_{out} = 0.12$, persist $= 2$ | Temporal anti-flicker stabilization |

## EXP 0: Main Benchmark Results ($k=5$, Medium Noise, 5 Seeds)

| Method | Budget $k=5$ Hazard Recall ($\uparrow$) | $k=5$ CVaR-5% Recall ($\uparrow$) | $k=5$ Flicker $F_t$ ($\downarrow$) | Latency (ms/frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Random** | $26.6\% \pm 1.7\%$ | $0.0\%$ | $0.845 \pm 0.009$ | $0.011$ |
| **Semantic** | $59.6\% \pm 9.9\%$ | $11.6\%$ | $0.180 \pm 0.036$ | $0.036$ |
| **Distance** | $65.3\% \pm 5.3\%$ | $16.6\%$ | $0.258 \pm 0.017$ | $0.034$ |
| **Proposed (Ours)** | **$87.2\% \pm 6.8\%$** | **$49.9\%$** | **$0.180 \pm 0.015$** | **$0.551$** |

At $k=7$, **Proposed** reaches **$97.1\% \pm 2.9\%$** Hazard Recall (vs. $79.3\% \pm 5.1\%$ for Distance) while running at $< 0.6\text{ ms/frame}$ ($> 1,500\text{ Hz}$).

![Phase A Pareto Plot](results/benchmark_pareto.png)

## EXP 1: Noise Stress Test ($k=5$, 5 Seeds)

| Noise Level | Method | Hazard Recall ($\uparrow$) | CVaR-5% ($\uparrow$) | Flicker $F_t$ ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: |
| `clean` | Distance | $62.0\%$ | $16.6\%$ | $0.132$ |
| `clean` | **Proposed** | **$86.0\%$** | **$57.1\%$** | **$0.090$** |
| `medium` | Distance | $65.3\%$ | $16.6\%$ | $0.258$ |
| `medium` | **Proposed** | **$87.2\%$** | **$49.9\%$** | **$0.180$** |
| `heavy` | Distance | $68.4\%$ | $18.1\%$ | $0.441$ |
| `heavy` | **Proposed** | **$88.6\%$** | **$51.2\%$** | **$0.337$** |

## EXP 2: Ablation Study ($k=5$, 5 Seeds)

| Noise | Variant | Hazard Recall ($\uparrow$) | CVaR-5% ($\uparrow$) | Flicker $F_t$ ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: |
| `medium` | Full Proposed | $87.2\%$ | $49.9\%$ | $0.180$ |
| `medium` | No Asymmetric ($\gamma = 0$) | $87.8\%$ | $48.3\%$ | $0.157$ |
| `medium` | No Hysteresis ($h = 0$) | $87.0\%$ | $53.5\%$ | $0.283$ |
| `medium` | No Semantic ($\beta = 0$) | $87.9\%$ | $54.2\%$ | $0.183$ |
| `heavy` | Full Proposed | $88.6\%$ | $51.2\%$ | $0.337$ |
| `heavy` | No Asymmetric ($\gamma = 0$) | $90.4\%$ | $56.5\%$ | $0.297$ |
| `heavy` | No Hysteresis ($h = 0$) | $89.2\%$ | $49.2\%$ | $0.419$ |
| `heavy` | No Semantic ($\beta = 0$) | $88.7\%$ | $50.9\%$ | $0.340$ |

## EXP 3: Error Analysis at $k=5$ (Medium Noise, 600 Frames)

| Metric | Value | Notes |
| :--- | :---: | :--- |
| Frames with $>5$ simultaneous hazards (Overflow) | **$110 / 600$ ($18.3\%$)** | Impossible to retain all without coarsening |
| Total missed hazards | $409 / 2357$ ($17.4\%$) | Across 5 seeds |
| Unavoidable misses (Budget Overflow $> k$) | **$153$ ($37.4\%$)** | Directly motivates **Graph Coarsening** in Phase B |
| Scoring / Ranking misses ($\le k$) | $256$ ($62.6\%$) | Top missed classes: `chair` (197), `table` (85), `pedestrian` (77) |

## EXP 4: Multi-Scenario Benchmark ($k=5$, Medium Noise, 5 Seeds)

Includes Personal Safety Bubble ($2.2\text{m}$) patch to eliminate side-swipe blind spots during sharp evasive steering:

| Environment | Distance (Recall / CVaR-5%) | Semantic (Recall / CVaR-5%) | **Proposed (Recall / CVaR-5%)** | Flicker $F_t$ (Prop vs. Dist) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Straight Corridor** | $65.8\%$ / $20.5\%$ | $59.0\%$ / $13.6\%$ | **$87.6\% \pm 7.4\%$ / $55.2\%$** | **$0.198$** vs. $0.271$ |
| **2. 90° Blind Corner** | $64.4\%$ / $3.6\%$ | $65.2\%$ / $12.9\%$ | **$83.7\% \pm 4.9\%$ / $38.3\%$** | **$0.247$** vs. $0.265$ |
| **3. Crowded Lobby** | $70.3\%$ / $5.3\%$ | $58.7\%$ / $0.0\%$ | **$96.0\% \pm 2.3\%$ / $62.4\%$** | **$0.221$** vs. $0.304$ |
| **4. 3D Vertical Hazards** | $61.5\%$ / $15.4\%$ | $64.6\%$ / $27.8\%$ | **$88.9\% \pm 7.2\%$ / $63.0\%$** | **$0.189$** vs. $0.262$ |

![Multi-Scenario Benchmark](results/scenarios_benchmark.png)

## Live 60 FPS Closed-Loop Infinite Demo (`results/infinite_demo.html`)

Side-by-side procedural infinite-world comparison at $k=5$ (`medium` noise, $1007$ frames): **Proposed (Aegis-DSG)** reaches **$99.8\%$ Hazard Recall**, lower flicker ($0.358$ vs. $0.365$), and travels **$71\text{m}$ vs. $57\text{m}$** (fewer collisions per meter), whereas **Distance** wastes slots on overhead ceiling lights (`Đèn trần`).

![Closed-Loop Infinite Demo](results/closed_loop_demo.png)

## Phase B: Real-Photo 3D Perception, True Baselines, Principled Optimization & MuJoCo Physics

### 1. Real-World Indoor Photo Pipeline (`run_real_photo_aegis.py`)
Real RGB room photo $\to$ **YOLOv8** + **MiDaS 3D Depth** $\to$ NetworkX 3D Scene Graph $\to$ **Aegis-v2** with dynamic budget ($|V'| \le k,\ S \ge 0.20$) and proactive spatial clustering (`chair_group(x4)` Super-Node):

![Real Photo Aegis-v2](results/real_photo_aegis_v2.png)

### 2. Grand Benchmark: True MIT SPARK Baselines vs. Aegis Family ($k=5$, Unseen Seeds `100..104`)

| Environment | Distance | D-Lite (Dijkstra) | Clio (True aIB) | GNN-Aegis (GAT) | Submod-Aegis ($1-1/e$) | **Aegis-v2 (Phase B)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Straight Corridor** | $63.9\%$ ($10.6\%$) | $57.3\%$ ($12.1\%$) | $36.7\%$ ($0.0\%$) | $83.5\%$ ($30.5\%$) | $94.8\%$ ($50.7\%$) | **$100.0\%$ (CVaR $99.4\%$)** |
| **2. 90° Blind Corner** | $64.7\%$ ($8.7\%$) | $59.9\%$ ($7.2\%$) | $48.5\%$ ($0.0\%$) | $82.5\%$ ($38.0\%$) | $92.5\%$ ($48.4\%$) | **$98.2\%$ (CVaR $78.9\%$)** |
| **3. Crowded Lobby** | $66.6\%$ ($3.7\%$) | $64.4\%$ ($5.7\%$) | $43.0\%$ ($0.0\%$) | $87.0\%$ ($34.1\%$) | $90.8\%$ ($36.7\%$) | **$99.0\%$ (CVaR $81.4\%$)** |
| **4. 3D Vertical Hazards** | $65.5\%$ ($19.0\%$) | $62.4\%$ ($19.6\%$) | $48.8\%$ ($2.3\%$) | $88.3\%$ ($56.8\%$) | $89.6\%$ ($43.0\%$) | **$99.9\%$ (CVaR $97.9\%$)** |
| **Flicker $F_t$ ($\downarrow$)** | `0.248–0.284` | `0.321–0.353` | `0.610–0.706` | `0.190–0.261` | `0.202–0.286` | **`0.161–0.238`** |
| **Latency (ms/frame $\downarrow$)** | `0.03 ms` | `1.07–2.11 ms` | `9.56–28.89 ms` | `0.39–0.44 ms` | `0.78–1.03 ms` | **`0.41–0.50 ms`** |

### 3. Real 3DSSG Scanned Rooms $\times$ MuJoCo 4.0 Rigid-Body Physics (`run_mujoco_3dssg.py`)
Evaluated across $10$ real scanned indoor rooms from **3DSSG (Wald et al., CVPR 2020)** compiled into solid 3D rigid-body worlds in **MuJoCo 4.0** ($200\text{ Hz}$ C++ contact solver, $k=5$, medium noise):

| Method | Success Rate | Real Physical Collisions ($\downarrow$) | SPL ($\uparrow$) | Hazard Coverage ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: |
| **D-Lite\*** | $50.0\%$ | $1.70$ | $0.489$ | $77.6\%$ |
| **Clio\*** | $50.0\%$ | $0.90$ | $0.499$ | $73.0\%$ |
| **Distance** | $50.0\%$ | $0.70$ | $0.499$ | $85.9\%$ |
| **Aegis-v2 (Ours)** | **$50.0\%$** | **$0.60$** | $0.493$ | **$100.0\%$** |

Interactive viewers included in `results/`: `phase_b_demo.html` (60 FPS Coarsening + DWA), `mujoco_3dssg_viewer.html` (360° 3D MuJoCo playback), and `space_to_graph.png` (Hydra ESDF $\to$ GVD $\to$ Rooms visualization).

## Key Takeaways for Phase B

1. **Hysteresis Validated (H3):** Disabling hysteresis increases Jaccard flicker $F_t$ by **$+57\%$** ($0.180 \to 0.283$ under medium noise; $0.337 \to 0.419$ under heavy noise) with negligible impact on recall.
2. **Asymmetric Risk vs. False Positives Trade-off:** Under `heavy` noise, $\gamma = 1.5$ inadvertently boosts low-confidence false-positive ghost detections into top-$k$ slots ($90.4\% \to 88.6\%$ recall). Phase B will introduce FP-aware uncertainty weighting.
3. **Graph Coarsening Necessity:** Because $37.4\%$ of all misses at $k=5$ stem from hard budget overflow, Phase B will implement hierarchical **Graph Coarsening (super-nodes)** and closed-loop **DWA** evaluation.
