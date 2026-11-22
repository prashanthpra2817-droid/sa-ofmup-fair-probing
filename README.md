# Fair Probing for Multi-Agent Bandits, Made Robust to Stale Probes

**Korra Prashanth Nayak** (Roll No. 25210075) · CS 620: Incentives in Machine Learning · IIT Gandhinagar

This repository contains a from-scratch reproduction of **OFMUP** (Online Fair
Multi-agent UCB with Probing) from *Fair Algorithms with Probing for Multi-Agent
Multi-Armed Bandits* (Xu, Liu, Mattei & Zheng, AAAI-26), and **SA-OFMUP**, a
staleness-aware extension in which a probed reward may be out of date by the
time an agent reaches the arm.

The full write-up is in [`report/report.pdf`](report/report.pdf).

## SA-OFMUP in brief

- **Staleness model.** With probability ρ a probed arm goes stale, and an agent
  assigned to it receives a fresh draw instead of the probed value. ρ = 0 is
  the original model.
- **Objective.** Each probe value R is replaced by its conditional expectation
  (1 − ρ)·R + ρ·μ, which is exact for the product-of-expected-utilities NSW
  objective.
- **Learning ρ.** ρ̂ = Σ(X − Y) / Σ(X − μ̂), computed from probe–pull pairs.
- **Lower confidence bound.** The planner uses (Σ(X − Y) − c√n) / Σ(X − μ̂),
  which prevents a noisy early estimate from switching probing off for good.
- **Counting fix.** A pull from a just-probed arm is not recorded as a second
  sample.
- At ρ = 0, SA-OFMUP reduces exactly to OFMUP with the counting fix.

## Main results (setting (a): M = 12 agents, A = 8 arms, Bernoulli rewards, 3 seeds)

| Experiment | Non-Probing | OFMUP | SA-OFMUP (LCB, final) |
|---|---|---|---|
| ρ = 0, T = 3000 (reproduction) | 299.7 | 203.6 | — |
| ρ = 0.4, T = 1500 | 144.6 | 123.6 | 117.9 |
| ρ = 0.8, T = 1500 | 123.7 | 128.9 | 116.2 |
| ρ = 0.8, T = 3000 | 192.2 | 208.5 | 179.1 |

Values are final cumulative regret (lower is better). Per-seed values, the
other settings and all caveats are given in the report.

## Repository layout

```
code/
  sa_ofmup.py          simulator, NSW solver, Xu et al.'s Algorithms 1-2, baselines, SA-OFMUP
  final_one_cell.py    all experiments in one script / one notebook cell (resumable)
  run_experiments.py   reproduction of Figure 1 and the staleness sweep (command line)
  stale_chunks.py      resumable staleness sweep used for the report (setting a)
  repro_long.py        setting (a) at T = 3000, rho = 0
  long_check.py        setting (a) at T = 3000, rho = 0.8
  make_figures.py      rebuilds the report figures from the saved runs
results/               saved runs (CSV, per-seed .npz, logs) and figures
report/                LaTeX source of the report (AAAI format), tables, figures, PDF
```

## How to run

Requires Python 3 with NumPy and Matplotlib:

```bash
pip install -r requirements.txt
```

Run all commands from the repository root.

```bash
# Everything in one go (also works pasted into a single Jupyter/Colab cell).
# Set QUICK = False at the top of the file for the full experiments.
python code/final_one_cell.py

# Or step by step, as used for the report:
python code/run_experiments.py repro --T 1500 --seeds 3   # Figure 1, four settings
python code/stale_chunks.py run && python code/stale_chunks.py aggregate
python code/repro_long.py                                  # rho = 0,   T = 3000
python code/long_check.py                                  # rho = 0.8, T = 3000
python code/make_figures.py                                # report figures
```

Finished runs are saved under `results/`, so an interrupted run resumes where it
stopped. On one CPU core the full set of experiments takes several hours.

## Building the report

```bash
cd report
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

`aaai2026.sty` and `aaai2026.bst` are stand-ins that reproduce the AAAI layout.
Replace them with the official files from the AAAI author kit for the
camera-ready format; no change to `report.tex` is needed.

## Reference

T. Xu, J. Liu, N. Mattei and Z. Zheng. *Fair Algorithms with Probing for
Multi-Agent Multi-Armed Bandits.* AAAI-26, pp. 27332–27340.
Full version: arXiv:2506.14988.
