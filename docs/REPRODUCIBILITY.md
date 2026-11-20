# Reproducibility Guide (`Aegis-DSG v3.0-frozen`)

This guide provides exact, end-to-end instructions to reproduce every table, unit test, and figure reported in the **Aegis-DSG** repository using the frozen codebase (`git checkout v3.0-frozen`).

---

## 1. Environment Setup

Tested on **Python 3.12** (Windows / Linux, CPU-only laptop — no GPU required).

```bash
# Core engine & unit tests (zero heavy ML dependencies)
pip install -r requirements.txt

# Full benchmark environment (Meta Aria ADT, MuJoCo 4.0, OpenCLIP ViT-B-32)
pip install mujoco==4.0.1 open-clip-torch opencv-python requests torch torchvision
```

---

## 2. Dataset Preparation

### A. Meta Project Aria Digital Twin (ADT, ICCV 2023)
1. Request and download `ADT_download_urls.json` from Meta's official Project Aria portal: https://www.projectaria.com/datasets/adt/
2. Place `ADT_download_urls.json` at `data/aria_adt/ADT_download_urls.json` (or set environment variable `ADT_URLS_JSON=/path/to/ADT_download_urls.json`).
3. Run `python scripts/benchmarks/run_aria_10seq_benchmark.py` — the script automatically downloads only the compact ground-truth + SLAM + Eye-Gaze archives (~25–35 MB each) for all 11 benchmark sequences (`golden_skeleton`, `clean`, `meal`, `work`, `decoration`, `multiskeleton_party`, `multiuser_clean`).

### B. 3DSSG Scanned Indoor Rooms (CVPR 2020)
1. Download the official `3DSSG` JSON annotations (`objects.json`, `relationships.json`) from https://3dssg.github.io/ into `data/3dssg/3DSSG/`.

---

## 3. Exact Commands to Reproduce Every Table & Figure

| Paper Table / Figure | Exact Reproduction Command | Output Artifacts |
| :--- | :--- | :--- |
| **Core Invariants (13/13 Unit Tests)** | `python -m unittest tests/test_aegis_core.py -v` | Terminal verification (`< 0.1s`) |
| **Table 1 & Fig. 2 (11 Real Meta Aria Sequences)** | `python scripts/benchmarks/run_aria_10seq_benchmark.py` | `results/aria_11seq_benchmark.png`, `results/aria_11seq_results.json` |
| **Table 2 & Fig. 3 (5 Breakthroughs v3.0 Master Suite)** | `python scripts/benchmarks/run_v3_master_benchmark.py` | `results/aegis_v3_master_benchmark.png`, `results/aegis_v3_master_results.json` |
| **Table 3 & Fig. 4 (Fair Baselines + Leave-One-Out Ablations)** | `python scripts/benchmarks/run_paper_ablations_and_fair_baselines.py` | `results/paper_ablation_loo.png`, `results/paper_ablations_fair_baselines.json` |
| **Table 4 (10 Real 3DSSG Rooms × MuJoCo 4.0)** | `python scripts/benchmarks/run_mujoco_3dssg.py` | Terminal closed-loop table (`k=5`, seeds `42..51`) |
| **Hero Visual (Meta Aria Butcher Knife Showcase)** | `python scripts/benchmarks/run_aria_adt_benchmark.py` | `results/aria_adt_aegis_v2_showcase.png` |
| **Human-in-the-Loop Cognitive Load Study** | `python user_study/build_user_study.py` then `python user_study/analyze_study.py` | `user_study/aegis_human_eval.html`, `user_study/human_study_results.png` |
| **Supplementary Synthetic 4-Scenario Benchmark** | `python scripts/benchmarks/run_prong3_benchmark.py` | Terminal table (unseen seeds `100..104`) |

---

## 4. Determinism & Seed Protocol
- Synthetic benchmarks use fixed NumPy seeds (`42..46` for development/ablations; strictly unseen `100..104` for held-out evaluation).
- Real 3DSSG × MuJoCo 4.0 physics episodes use fixed seeds `42..51` across rooms `0..9`.
- Split Conformal Safety Calibration uses a strict sequence-level split: **5 calibration sequences** (`golden_skeleton_seq100`, `clean_seq150`, `clean_seq131`, `meal_seq142`, `meal_seq146`) $\to$ **6 completely unseen test sequences** (`work_seq139`, `work_seq106`, `decoration_seq135`, `decoration_seq140`, `multiskeleton_party_seq127`, `multiuser_clean_seq120`).
