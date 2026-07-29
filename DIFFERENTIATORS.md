---
title: "Differentiators: 2025-26 trends we adopt, competitors don't"
project: nexus-energy
type: planning
status: active
last_verified: 2026-04-26
tags:
  - differentiators
  - speed
  - innovation
---

# Differentiators: 2025–26 trends we adopt, competitors don't

The headline is speed — faster than PyPSA / GenX / Calliope / oemof /
SpineOpt at equal feature coverage. This file tracks the specific
techniques that give us that edge, where they come from in the
literature, and which `ROADMAP.md` phase lands them.

Classification:
- **(a) Production-ready + not in competitors** — ship, differentiator.
- **(b) Research-grade but tractable in ≤ 6 months** — medium bet.
- **(c) Research-only today** — park; don't put on critical path.

## (a) Ship these — differentiator tier

### §1. GPU LP with dual recovery [Phase 10]

- cuPDLP-C integrated into HiGHS as `solver=pdlp`; HiGHS Newsletter
  25.0 (May 2025).
- NVIDIA cuOpt 26.04 (Apr 2026): full LP + MILP on GPU; GAMS
  integration Sept 2025.
- Realistic speedup on large energy LPs: **5–30×** (treat NVIDIA's
  "5000×" as marketing).
- Caveat: first-order methods give **noisy duals**. We add a crossover
  / single-IPM-polish pass to recover prices for LMP / shadow-price
  workflows. No open energy library does this first-class.

**Land as:** `nexus-opt` solver selector with `gpu=True` + mandatory
polish for price-reporting modes. Marketing line: "GPU LP with
production-quality duals."

### §4. Tight-by-default UC [Phase 2]

- 3-bin formulation (up / startup / shutdown, Morales-España et al.
  2013): tighter LP relaxation than 2-bin.
- Perspective cuts for quadratic fuel costs (Frangioni & Gentile);
  separable on the fly.
- Polynomial-size convex-hull min-up / min-down (Gentile et al.,
  refined Tahanan et al. 2024, OR).

**Why it beats competitors:** GenX exposes 3-bin but docs are weak and
perspective cuts aren't standard; PyPSA ships a naïve 2-bin; Calliope
/ oemof are worse. Making these the default — with zero user-visible
API change — gives single-digit to 10× MIP speedups on realistic UC
instances.

### §5. LDS-aware time aggregation + ML feature clustering [Phase 7]

- Inter/intra-period storage superposition (Kotzur 2018 lineage).
- Minimum-representative-hours-for-LDS (arXiv 2512.00892, Dec 2025) —
  explicit linkage formulation.
- ML-feature clustering: autoencoder embedding of (VRE_CF, load,
  shadow-price-proxy) before k-medoids — better rep periods than raw
  time-series clustering.

**Why it beats competitors:** GenX / Calliope / SpineOpt have
representative-period support but the LDS linkage is imperfect;
ML-feature clustering isn't in any open energy library. Error-bound
reporting ("your TDR error is ≤ 2 % of true optimum") is
unique to us.

### §3a. Regularized + adaptive-oracle Benders [Phase 8]

- Regularized (level-bundle) Benders — Pecci & Jenkins arXiv
  2403.02559, updated Jan 2025. GenX already has this; **matching is
  table stakes**.
- Adaptive-oracle stabilized Benders — Mazzi et al. C&OR 2024. Cross-
  scenario cut reuse; **not in GenX**. This is where we beat GenX.

### §6b. Rust / PyO3 constraint-assembly hot path [Phase 10]

- linopy (xarray-vectorised) is the Python ceiling today (~ 4–6×
  Pyomo). No one has gone below Python.
- We write a PyO3 Rust kernel that streams CSC sparse constraints
  directly into HiGHS — **3–5× linopy** on large model builds.
- No open energy library does this. linopy's authors have not
  indicated plans for a Rust layer as of 2026-Q1.

### §7a. Ray scenario runner [Phase 12]

- Pyomo has async solver managers but no polished Ray integration.
- JuMP has no Ray story.
- We ship a turnkey scenario-parallel harness, identical interface
  local vs multi-node.

## (b) Medium bets — research-to-product

### §2. GNN UC warm-start [Phase 11]

- "Stable Variable Fixation via GNN for Accelerated UC" (2025); "Neural
  Two-Stage Stochastic UC" (arXiv 2507.09503).
- Trained **per-system** from N historical solves; predicts
  on/off schedule to warm-start the MILP.
- Lowest integration risk (no solver internals; just warm-start hints).
  Highest ROI in rolling-UC / MPC workflows.

**Risk:** distribution shift if the system evolves; we ship an online
confidence score + auto-fallback to cold start.

### §3b. SDDiP [Phase 9, stretch]

- Zou, Ahmed, Sun 2019 — Lagrangian cuts on binary state for
  multi-stage stochastic MILP.
- Research repos exist (`leoschleier/sddip`, Julia `SDDP.jl`).
- **No open energy library ships this.** We're either the first or
  very early.

### §6a. Differentiable dispatch [Phase 12]

- `cvxpylayers` + `diffcp` are mature for convex problems.
- Degleris / El Gamal / Rajagopal 2024 (SSRN 5169721) — SGD via
  implicit diff for bilevel grid expansion. Real energy application.
- We expose gradients from dispatch LPs → downstream losses; lets
  users learn demand elasticity, storage bid curves, market-maker
  params from real-world data.

**Caveat:** honest scope is **parameter learning**, not "end-to-end
differentiable capacity expansion" (that's still research).

## (c) Park — not on critical path

### §1c. QMC-SDDP
Discussed on Julia Discourse 2024; not in `SDDP.jl`'s trunk.
Implementation barrier not worth the risk.

### §2c. RL branching inside the solver
Ecole + learn2branch are brittle across problem distributions and
need solver-internal hooks. Skip; our ML focus is warm-start.

### §6c. MLIR / LLVM codegen for LP build
No ecosystem; Modular's Mojo isn't there yet. Revisit in 2027 if the
LP-modeling-via-MLIR story materialises.

### §9. Quantum UC
QAOA / quantum annealing (arXiv 2502.15917, Joule 2024) — all < 100
units, nothing production. Add a `nexus_energy.experimental.qaoa`
stub so we can claim capability, but don't put it on the roadmap
path.

## What we explicitly do NOT claim

- **"5000× faster than anyone"** — that's NVIDIA marketing. Our honest
  numbers are 8× / 10× / 14× / 61× against specific competitors on
  specific LPs (see `FLAGSHIP_COMPARISON.md`, `GENX_COMPARISON.md`).
- **"End-to-end differentiable grid expansion"** — bilevel MIP
  gradients remain research. Our diff story is parameter learning
  on the inner LP.
- **"Quantum-accelerated energy planning"** — not production in 2026.
- **"AI-optimised solver"** — we have ML-warm-start for UC; that is a
  specific and delimited thing, not a general claim.

## Sources

- cuPDLP-C: https://github.com/COPT-Public/cuPDLP-C
- cuPDLPx (arXiv 2507.14051): https://arxiv.org/html/2507.14051
- NVIDIA cuOpt: https://github.com/NVIDIA/cuopt
- HiGHS + cuOpt: https://blogs.ed.ac.uk/mathematics/2025/03/18/highs-and-nvidia-cuopt
- PDLP (arXiv 2501.07018): https://arxiv.org/abs/2501.07018
- Regularized Benders (Pecci / Jenkins, arXiv 2403.02559): https://arxiv.org/abs/2403.02559
- Stabilized Benders adaptive oracles (C&OR 2024): https://www.sciencedirect.com/science/article/pii/S0305054824001370
- New MINLP Formulations for UC (OR 2024): https://pubsonline.informs.org/doi/10.1287/opre.2023.2435
- Convex Hull Pricing for UC (MDPI Energies 2024): https://www.mdpi.com/1996-1073/17/19/4851
- Min representative hours for LDS (arXiv 2512.00892): https://arxiv.org/html/2512.00892
- Stable Variable Fixation via GNN for UC: https://www.researchgate.net/publication/390960604
- Neural Two-Stage Stochastic UC (arXiv 2507.09503): https://arxiv.org/pdf/2507.09503
- SDDiP (Zou / Ahmed / Sun 2019), repo `leoschleier/sddip`
- linopy benchmarks: https://linopy.readthedocs.io/en/latest/benchmark.html
- cvxpylayers: https://github.com/cvxpy/cvxpylayers
- Bilevel Electricity Grid Expansion (Degleris et al.): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5169721
- highs-js: https://github.com/lovasoa/highs-js
