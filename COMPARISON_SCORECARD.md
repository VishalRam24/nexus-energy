---
title: Nexus Comparison Scorecard
project: nexus-energy
type: benchmark
status: active
last_verified: 2026-05-22
tags:
  - scorecard
  - h2h
  - comparison
---

# Nexus Comparison Scorecard

**One-page rollup of every head-to-head we have actually measured** —
solver-core (nexus-opt) + energy-layer (nexus-energy) + external tutorial
runs (`test_projects/`). Re-read / re-run every time a phase closes;
update the verdict when numbers change.

Last updated: **2026-06-08** (GenX Julia parity push + critic loop). **🟢 GenX `1_three_zones_ucommit2` CLOSED: +7.17% → −0.02% (exact) and nexus 7.7 s vs GenX 36.1 s = 4.7× faster.** Root cause was a **systemic port bug** (fuel priced by simple time-mean; GenX prices fuel per-timestep → ~13% over-price → +7% opex), fixed by passing the time-varying fuel series as `marginal_cost[t]` (nexus core already supported it; no core change, no test impact). Same fix applied to rate_co2/mincapreq ports, which revealed their *true* state: **mincapreq +0.92% → −3.90%** (the old +0.92% was the fuel over-price cancelling a real −4% portfolio gap; forced minimums met exactly — MA_PV 5000 / CT_Wind 10000 / batteries 6000 all = GenX), **rate_co2 −42.7%** (portfolio-dominated, critic-verified nexus-faithful). Pattern: nexus matches GenX **exactly when the build is determined** (ucommit2 gas-only, exact + 4.7×) and **under-counts opex when multi-zone VRE is dispatched over transmission**. **GOLD-STANDARD DONE — overturns the earlier "cheaper optimum" reading:** fed nexus's mincapreq capacity into a real GenX solve (feasible, OPTIMAL, cNSE=0); GenX's own cost for nexus's solution = 5.823e9 ≈ GenX optimum 5.808e9, but nexus *reports* 5.582e9 → **~4.3% OPEX under-count bug, NOT a better optimum** (GenX re-dispatches optimally at the same capacity for MORE, so nexus relaxes a dispatch constraint). So rate_co2 (−42.7%) and mincapreq (−3.9%) are this under-count (lead suspect: transmission-loss modeling, GenX PWL vs nexus linear `loss=%`), 🔴 to fix — NOT wins. The exact-parity wins (PyPSA-Eur, ucommit2) are unaffected (no VRE-over-transmission). Next: pin + fix the under-enforced dispatch constraint. **2026-06-07** (production PyPSA-Eur H2H + two `from_pypsa` parity fixes). **🟢 PRODUCTION WIN — PyPSA-Eur capacity expansion (real `base_s_5_elec_` topology: 10 bus, 33 gen / 25 ext, 6 AC lines, 10 DC links, 5 batteries + 5 H2 stores; full realistic profiles, 2190 h quarter-year):** objective **−0.000 % (exact parity)**, wall **nexus 233 s vs PyPSA 810 s = 3.47× faster**, RAM 1.55/1.28 GB ≪ 16 GB. This **supersedes the stale "solve 5–17× slower" 🟡** below — that was a tracemalloc-timing artifact; on a real solve-bound production LP nexus wins on both parity and wall. Two real `from_pypsa` bugs were found and fixed to reach exact parity (component-isolated at 336 h): (1) **static scalar `p_max_pu` now mapped** (nuclear `p_max_pu=0.781` was ignored → over-dispatched cheap baseload, ~2.5 % under-price); (2) **`Storage.cyclic_level="free"`** added + used by `from_pypsa` (old code pinned `soc(0)=soc_initial·cap`, over-constraining cyclic storage — and forcing extendable storage start/end empty). 349 nexus-energy tests green; default behaviour unchanged (`cyclic_level="fixed"` default). The full 8760 h instance is beyond this machine's ceiling — **neither** solver finishes in 28 min. Detail: `test_projects/test_project_1/pypsa/eur_production/EUR_PRODUCTION_COMPARISON.md`. **GenX rate_co2 re-characterized (was 🔴 −42.4% "unexplained"):** adversarial critic-loop investigation proved nexus's CO2-rate-cap + cost formulation is **GenX-faithful** (nexus per-unit costs on GenX's exact EndCaps reproduce GenX cFix to the dollar 1.6595e10; storage-loss RHS term net==dissipation to 0.7%; caps exactly binding both sides). The gap is **portfolio selection, not a bug** — nexus builds cheap wind + 46 GWh battery (10.0e9), GenX builds 72 GW solar + 229 GWh battery exploiting the storage-loss-RHS headroom (17.4e9). One verified structural difference: nexus lacks VOLL/NSE load-shedding (~$0.2B effect). Now 🟡 pending the gold-standard check (nexus capacity → GenX). Prior: **2026-06-04** (RAM-safe benchmark sweep, 24 GB cap, serial + watchdog). **Real H2H re-runs:** pandapower NLP — case9 **+0.0007 %** @ 60.5× 🟢, case14 **+0.0792 %** @ 42.2× 🟢 (Jabr SOCP vs `runopp`, isolated venv subprocess). PowerModels.jl SOCWR — 3-bus radial parity **3.67e-5** @ 7.57× 🟢. PyPSA 1.2.2 LP dispatch — **exact** objective parity (cost matches to the cent, 3→50 bus); nexus build **1200–1400×** faster, mem 5–12× less, but **solve 5–17× slower** at 10–50 bus × 168 🟡 (N_En_Phase 14.5 LP-formulation gap; required `pd.set_option("future.infer_string", False)` for PyPSA 1.2.2 vs xarray). GenX parity re-measured vs stored refs: ucommit2 **+7.17 %** 🟡 (15.1 bilinear-UC still open), mincapreq **+0.92 %** 🟢 (15.3), rate_co2 **−42.4 %** 🔴 (15.2 unexplained). **CINDER (N_En_18):** LP 147 s vs PyPSA 190 s = **1.3× faster** 🟢; MILP 330 s vs 190 s = 1.7× slower 🟡 (gap 0.82 %, solved at root; improved from 357 s/1.9×). **Solver-core corpus** all green incl. cross-library (pulp/CBC MILP, cvxpy/ECOS/SCS SOCP+SDP, tsplib95 TSP, MOO). Found+fixed a HiGHS-1.14.0 mixed-MIP presolve bug (LP-relaxation guard in nexus-opt). **Not run (RAM/env):** PyPSA-USA (~30 GB > 24 GB cap), live GenX/Tulipa Julia solves (Python ports used instead), MOO cross-library (pymoo absent). Prior: **2026-05-27** (after N_En_Phase 18.x.1 extendable Store mode + 18.y.2 selective v/w — CINDER MILP wall 639→357 s, SciGrid DE verified, PyPSA-Eur re-verified). Prior: **2026-05-22** (after CINDER multi-carrier ramp-cost parity validation). Prior: **2026-04-21** (after N_Opt_Phase 5.2 first-class nonlinear expressions + N_En_Phase 17.3 polar AC-OPF closed — `sin`/`cos`/`sqrt`/`/` are now first-class in `nexus.Model` via `Expr::Nonlinear(NlExpr)` + `ProblemType::NLP` + `solve_with_ipopt_nl` (recursive CasADi SX lowering, lazy import). LP/QP `SparseModel` bit-identical; parallel `nl_constraints` container keeps presolve/diagnostics untouched. `solve_ac_opf_polar(system, snapshot=0)` is the first consumer — polar AC-OPF parity vs pandapower PIPS on case9 **−4.3e-7 %** @ 3.58×, case14 **+4.5e-6 %** @ 3.58×, case30 **+0.233 %** @ 1.88× (nexus finds a lower-cost local optimum on the non-convex 6-gen problem — guarded by a "not worse than PIPS" assertion in the test suite). HS71 classic NLP + 4 elementary operator tests green at 1e-5. SOCP 15/15 and LP/QP 126/126 preserved. Benchmark CLI now takes `--formulation {socp,polar}`. Prior entry: **2026-04-20** (after N_En_Phase 10.4 quadratic-cost SOCP landed — opt-in `generator.quadratic_cost` (defaults 0 via `getattr`; pre-10.4 systems bit-exact) adds `q2·p²` via `ConicProblem.add_quadratic_obj` with `P[p,p] = 2·q2`; convexity guard raises if `q2 < 0`. `build_nexus_from_ppc` now forwards MATPOWER `cp2` (gencost model 2, ncost 3) as `g.quadratic_cost = cp2·base_mva²` so both sides of `ac_opf_vs_pandapower` see the full polynomial cost (was previously linearised to make them match). Phase 10 suite 15/15 green (new `test_socp_opf_quadratic_cost` covers opt-out default, convexity guard, KKT split, hand-computed parity). **Case9/14/30/118 on full quadratic cost:** case9 **+0.0007 %** @ 69.66× 🟢, case14 **+0.0792 %** @ 40.20× 🟢, case30 **+0.7169 %** @ 18.28× 🟢 (flipped 🟡 → 🟢), case118 **+0.2533 %** @ 8.10× 🟢 (first-ever SOCP row for case118 — was ⏳ blocked on 10.4). All land on **plain Jabr**, no OBBT / tight-QC / PWL-cos needed. **Honest finding:** the −19.63 % case30 plain-Jabr gap that drove the 10.10→10.14.1 tightening marathon was almost entirely a **cost-linearisation artifact** — with true cp2 the plain Jabr sits at +0.72 %. 10.10–10.14.1 remain correct tightening layers (sound relaxation lineage) but are no longer in the critical path for these IEEE cases. PWL heat rates still deferred (different machinery — PWL aux vars + ordered cuts; rarely used on IEEE). Prior same-day landings: 10.14.1 PWL cos + asymmetric sin/cos boxes (case30 linearised-cost gap −4.45 % → +2.39 %), 10.14 OBBT pre-solve (−19.63 % → −4.45 %), 10.12 full QC, 10.11 QC-lite, 10.10 SOCP+AT. Earlier: N_En_Phase 17.3 case9 / case14 / case30 rows all landed on linearised cost — refreshed here to true cp2.

## Legend

| Mark | Meaning |
|------|---------|
| 🟢 | **We win.** Clean speed or quality advantage, same problem, same solver where relevant. |
| 🟡 | **Tied / within noise.** ≤ 2× gap, same objective. |
| 🔴 | **We lose.** Other library is materially faster or more accurate on the same problem. |
| ⚪ | **Blocked / gap.** No nexus-native row yet (e.g. cone API missing). External library stands in. |
| ⚫ | **Deferred.** Planned but not run. |

**Ground rule:** every row below is backed by a script + a JSON or
`.jsonl` in `benchmarks/_results/` or `benchmarks/results/`. If a number
isn't backed by a reproducible script, it doesn't belong here.

---

## Solver-core benchmarks (nexus-opt)

Source: `nexus-opt/BENCHMARK_ROADMAP.md` + `nexus-opt/benchmarks/_results/*.jsonl`.

### LP (N_Opt_Phase 2)

| Row | Comparison | Scale | Verdict | Source |
|-----|------------|-------|---------|--------|
| NETLIB 114-instance sweep vs Gurobi / CBC / GLPK | — | — | ⚫ deferred (needs MPS ingest) | §2.1 |
| Large-scale LP (1M vars) vs linopy | — | — | ⚫ deferred | §2.2 |

**Status:** LP cross-library not yet run. PyPSA / GenX H2Hs below cover
LP at the energy-modelling layer and are the practical signal for now.

### MILP (N_Opt_Phase 3)

| Row | Comparison | Scale | Verdict | Source |
|-----|------------|-------|---------|--------|
| Knapsack | nexus+HiGHS vs PuLP+CBC | 1 000 items | 🟢 0.16 s vs 1.2 s (7.5×) | §3.2 |
| Knapsack | nexus+HiGHS vs PuLP+CBC | 10 000 items | 🔴 6.4 s vs 0.22 s (CBC 28×; objectives differ 0.01 % — CBC's "Optimal" at this speed may be a preprocessing heuristic, flagged for re-audit) | §3.2 |
| MIPLIB 2017 sweep | — | 240 instances | ⚫ deferred | §3.1 |
| Set covering / packing / graph colouring / MIS / job shop / assignment | nexus-only timing corpus | varied | ⚪ single-backend baseline (no cross-library column) | §3.2 |
| MILP features (warm-start, presolve, callbacks) | — | — | ⚫ deferred | §3.3 |

### QP (N_Opt_Phase 4)

| Row | Comparison | Scale | Verdict | Source |
|-----|------------|-------|---------|--------|
| Elastic net | OSQP vs Clarabel vs SCS (all via CVXPY) | p = 2 000 | OSQP wins (1.1 s vs 4.6 s Clarabel) | §4.2 |
| SVM | Clarabel vs OSQP | n = 2 000 | Clarabel wins (0.04 s vs 0.46 s; 11×) | §4.2 |
| Ill-conditioned QP (κ = 1e9) | Clarabel vs SCS | n = 200 | Clarabel wins (15 ms vs 22 ms) | §4.2 |
| Large lasso | SCS vs Clarabel vs OSQP | p = 10 000 | SCS wins (3.1 s vs 21 s vs 37 s) | §4.2 |
| MPC / banded / portfolio | 3-way ties | varied | 🟡 all within noise | §4.2 |
| Maros-Meszaros test set | — | 138 instances | ⚫ deferred | §4.1 |

**Nexus verdict:** ⚪ QP path currently *is* these three CVXPY solvers
(no nexus-direct QP column). The scorecard above is about picking the
right backend within our routing; a nexus-native QP entry waits on
N_En_Phase 10.5 (Clarabel routing) and/or N_En_Phase 17.1 (PyPSA QP H2H).

**Update 2026-04-20:** Native Clarabel routing landed. Markowitz n=50
via `nexus.Model.solve(solver="clarabel")` completes in 0.97 ms — fastest
of the three backends at that scale (OSQP 0.89 ms, Ipopt 55 ms). Parity
across `{osqp, clarabel, ipopt}` holds to 1e-4 rel on n ∈ {5, 20, 50}
(harness: `benchmarks/conic/test_clarabel_ipopt_routing.py`). Fix
side-effect: `solve_with_osqp` was silently ignoring variable bounds;
now appends identity rows to A with `l = col_lower`, `u = col_upper`.

### SOCP (N_Opt_Phase 5.1)

| Row | Comparison | Scale | Verdict | Source |
|-----|------------|-------|---------|--------|
| Robust portfolio | Clarabel vs ECOS | n = 500 | Clarabel wins (3× ECOS) | §5.1 |
| Filter design | Clarabel vs ECOS | n_taps = 1024 | Clarabel wins (12 s vs 84 s ECOS) | §5.1 |
| Chebyshev centre / min-enclosing-ball | Clarabel vs SCS vs ECOS | all sizes | 🟡 3-way tie | §5.1 |
| MOSEK column | — | — | ⚫ deferred (no licence) | §5.1 |
| PGLib-OPF AC-OPF conic relaxation | — | — | ⚫ deferred (needs PGLib-OPF fetch) | §5.1 |

**Nexus verdict:** ⚪ no nexus-direct Clarabel row yet — `cvxpy_clarabel`
stands in. Closes when N_En_Phase 10.5 lands.

**Update 2026-04-20:** Clarabel solver dispatch landed (N_En_Phase 10.5),
but the `nexus` modelling layer still lacks first-class SOC / PSD
constraint primitives — so the harness can already *solve* conic
problems via `m.solve(solver="clarabel")` for anything expressible as
LP/QP + bounds, but SOCP-specific rows still need the cone API. Covers
the LP/QP half; SOCP rows open once N_En_Phase 10.1 ports the AC-OPF
constraints onto Nexus's model.

### SDP (N_Opt_Phase 5.2)

| Row | Comparison | Scale | Verdict | Source |
|-----|------------|-------|---------|--------|
| Max-cut relaxation | SCS vs Clarabel | n = 100 | **SCS wins decisively** (0.13 s vs 8.9 s; 68×) | §5.2 |
| Matrix completion | SCS vs Clarabel | m = 60 | SCS wins (0.14 s vs 13 s; Clarabel returns `optimal_inaccurate`) | §5.2 |
| Lovász θ / sensor localisation | SCS vs Clarabel | small | 🟡 competitive | §5.2 |
| ECOS PSD column | — | — | N/A (ECOS lacks PSD cone) | §5.2 |

**Nexus verdict:** ⚪ same as SOCP — no nexus-direct row; SCS is the
winner for dense SDPs.

### NLP (N_Opt_Phase 6)

| Row | Comparison | Scale | Verdict | Source |
|-----|------------|-------|---------|--------|
| Nonlinear MPC (cart + drag) | CasADi+IPOPT vs scipy.SLSQP | horizon = 400 | **CasADi wins** (0.05 s vs 69 s; ≈1400×) | §6.2 |
| Minimum surface (direct collocation) | CasADi vs SLSQP vs trust-constr | n = 400 | CasADi wins (0.22 s vs 9.6 s vs 174 s) | §6.2 |
| Gaussian regression (unconstrained LS) | scipy vs CasADi | n = 10 000 | scipy wins (1.9 ms vs 360 ms) — AD compile not worth it for small LS | §6.2 |
| HS small problems (HS1/21/35/36/71) | scipy.SLSQP vs CasADi | small | SLSQP wins (compile overhead dominates) | §6.1 |
| Full HS119 / CUTEst / COPS | — | — | ⚫ deferred | §6.1 |

**Nexus verdict:** ⚪ nexus routes through `scipy.SLSQP` today — so the
`scipy_slsqp` column *is* the nexus row. N_En_Phase 10.9 (nexus →
IPOPT bridge) unlocks matching CasADi at scale. Headline: **we are
~1400× behind CasADi on large constrained NLPs until 10.9 lands.**

**Update 2026-04-20:** Ipopt dispatch landed via CasADi's bundled
binary — `m.solve(solver="ipopt")` routes through `solve_with_ipopt`
in `nexus-opt/src/lib.rs` (no system Ipopt / cyipopt required).
Covers bounded QP and scipy-equivalent NLP formulations today;
CasADi's AD scale-up waits on first-class nonlinear expressions in
Nexus's modelling layer (not part of 10.9). NLP-parity on MPC-style
quadratic T=5 verified at 1e-4 rel tolerance.

### Black-box / continuous (N_Opt_Phase 7)

Sources: `benchmarks/blackbox/test_crosslibrary_benchmarks.py`,
`benchmarks/_results/blackbox.jsonl`. Fixed 5 000-eval budget (Optuna
capped at 1 000 due to TPE quadratic cost); seed 42.

| Problem | Fastest CMA-family | Nexus CMA-ES | nevergrad CMA | cma.py | Verdict for Nexus CMA-ES |
|---------|--------------------|--------------|---------------|--------|--------------------------|
| sphere_30d      | nexus   | 0.008 s → 0.0     | 1.75 s → 5.2e-06 | 0.16 s → 1.2e-11 | 🟢 **200× faster than ng_cma, 20× faster than cma.py, matched/better obj** |
| rosenbrock_10d  | cma.py  | 0.005 s → 8.4     | 1.76 s → 3.02    | 0.14 s → 7.9e-07 | 🔴 on quality (cma.py 7e-7 vs nexus 8.4) but 🟢 on speed (30×) |
| rastrigin_10d   | nexus   | 0.004 s → 0.0     | 1.78 s → 3.98    | 0.08 s → 24.9    | 🟢 **hit global opt fastest; cma.py stuck at 24.9** |
| ackley_10d      | nexus   | 0.005 s → 4.4e-16 | 1.79 s → 1.5e-08 | 0.09 s → 3.1e-11 | 🟢 best quality (4e-16), 20× faster than cma.py |
| hartmann_6d     | tied    | 0.016 s → -3.3224 | 1.94 s → -3.3224 | 0.04 s → -3.3224 | 🟡 tied on obj, fastest |

Also measured (not CMA-family):
- **Nexus DE:** fastest wall-clock (0.002–0.006 s) but poor quality on hard multimodal at matched budget — tuning work on DE defaults is a follow-up, not a blocker. 🟡
- **Optuna TPE:** worst per-trial cost (5–28 s) and worst quality on continuous > 10 D. Outside its design envelope. 🟢 vs optuna clearly.
- **scipy.differential_evolution:** mid-pack (0.05–0.07 s), ~10× slower than nexus DE with comparable quality. 🟢 on speed.

**Aggregate verdict (blackbox):** 🟢 Nexus CMA-ES is the fastest CMA-family
implementation we have measured, and Nexus DE is the fastest per-eval
of any DE implementation we have measured. Third-party blackbox libraries
live only as comparison rows — never wrapped inside the nexus library.

### Multi-objective (N_Opt_Phase 8)

Source: `nexus-opt/benchmarks/multiobjective/test_mo_crosslibrary_benchmarks.py`.
Budget: NSGA-II pop=40, n_gen=100 (≈4 000 NFE); Nexus `pareto_frontier`
uses `n_points=40`. Metric: IGD vs pymoo's exact Pareto front.

| Problem set | Comparison | Verdict | Source |
|-------------|------------|---------|--------|
| ZDT1-4 (2-obj convex/non-convex/disjoint/multimodal) | Nexus `pareto_frontier` vs pymoo NSGA-II vs Platypus NSGA-II | 🔴 **Nexus 3–50× worse IGD, 10–50× slower wall-clock** — scalarisation runs inner DE per weight vector | §8.1 |
| ZDT6 (non-uniform density) | Nexus vs pymoo/Platypus NSGA-II | 🟢 **Nexus wins** (IGD 0.120 vs 0.556 pymoo / 0.474 platypus) | §8.1 |
| DTLZ1-5, DTLZ7 (M=3) | Nexus vs pymoo/Platypus NSGA-II | 🔴 Nexus loses on IGD — runs pairwise on first 2 of 3 objectives (structural gap, not tuning) | §8.2 |
| DTLZ6 (M=3, convergence-stressed) | Nexus vs pymoo/Platypus NSGA-II | 🟢 **Nexus wins** (IGD 1.43 vs 3.68 pymoo / 2.71 platypus) | §8.2 |
| ZDT5 (binary) | — | ⚫ deferred — Nexus `pareto_frontier` is continuous-only; needs a native binary MOO path | §8.1 |
| WFG1-9 | — | ⚫ deferred — needs dataset + native M>2 path | §8.3 |
| Indicators (hypervolume / GD / spacing / coverage) | — | ⚫ deferred; IGD ✅ via pymoo | §8.4 |

**Nexus verdict:** ⚪ the benchmark exposes the exact gap we want
the reader to see — Nexus has no native many-objective path (no
reference-direction NSGA-III, no ε-dominance archive, no proper M>2
scalarisation). Candidate native feature ranked here.

### Combinatorial / TSP (N_Opt_Phase 9)

Source: `nexus-opt/benchmarks/combinatorial/test_tsp_crosslibrary_benchmarks.py`.
Instances: TSPLIB burma14 / ulysses22 / att48 / eil76 / kroA100.
Budget: Nexus MILP (MTZ) 60 s; OR-Tools routing (PATH_CHEAPEST_ARC +
Guided Local Search) 3 s; python-tsp 2-opt to convergence.

| Instance | Comparison | Verdict | Source |
|----------|------------|---------|--------|
| burma14 (n=14) | Nexus MTZ vs OR-Tools vs python-tsp | 🟢 **Nexus proves optimum in 0.22 s** (OR-Tools 3 s, python-tsp 0.8 ms — all hit optimum) | §9.1 |
| ulysses22 (n=22) | same | 🟡 Nexus finds optimal incumbent at 60 s timeout but can't prove; OR-Tools + python-tsp prove in seconds | §9.1 |
| att48 / eil76 / kroA100 (n=48 / 76 / 100) | same | 🔴 **Nexus MTZ returns no incumbent in 60 s**; OR-Tools hits 0–0.74 % gap in 3 s; python-tsp 2.4–7.4 % gap in ≤ 0.23 s | §9.1 |
| LKH + Concorde native binaries | — | ⚫ deferred — need separate toolchain setup | §9.1 |
| ch150 / tsp225 / a280 / pcb442 | — | ⚫ deferred — MTZ is hopeless at this scale; re-open alongside a native Nexus TSP path | §9.1 |

**Nexus verdict:** 🔴 no competitive native TSP path above n≈20. MTZ's
LP relaxation is too weak for HiGHS to generate useful incumbents at
scale. Two candidate native features surfaced:
- **Lazy DFJ subtour-elimination cuts** on the MILP path — would
  land a proven-optimal kroA100 in seconds.
- **Native 2-opt / LKH-style metaheuristic** — would ship
  OR-Tools-competitive numbers without a third-party dep (consistent
  with "no third-party wrappers" rule).

### Stochastic / robust (N_Opt_Phase 10)

Source: `nexus-opt/benchmarks/stochastic/test_stochastic_crosslibrary_benchmarks.py`.
Both backends build the same **extensive form** and dispatch to HiGHS
— this is a modelling-layer assembly comparison.

| Problem | Comparison | Verdict | Source |
|---------|------------|---------|--------|
| Farmer 2-stage (N ∈ {100, 500, 2 000}) | Nexus Model+HiGHS vs Pyomo+HiGHS | 🟢 Nexus **2.1–4.6× faster**, bit-exact (≤ 6e-15 parity) | §10 |
| Farmer 2-stage (N = 3) | same | 🟡 0.73× — below noise threshold | §10 |
| Multi-period inventory ext-form (T=10 × N ∈ {50, 500}) | same | 🟢 Nexus **4.6–7.7× faster**, bit-exact | §10 |
| Chance-constrained portfolio SAA MILP (N ∈ {100, 500}) | same | 🟡 tied (solver-dominated) | §10 |
| DRO Wasserstein portfolio | — | ⚫ deferred — reformulation is a separate piece | §10 |
| Scenario tree generation / evaluation | — | ⚫ deferred — no native Nexus tooling (candidate native feature) | §10 |
| mpi-sppy / PySP / SDDP.jl columns | — | ⚫ deferred — heavier toolchains | §10 |

**Nexus verdict:** 🟢 Nexus modelling layer is **2–8× faster** than
Pyomo on extensive-form LPs at scenario counts where Pyomo's Python
`Constraint(rule=…)` loop dominates assembly. MILP cases are solver-
bound, so the win shrinks to parity. Every objective agrees with
Pyomo to 1e-15 relative — the assembled LP/MILP is the same.

---

### Cross-library API ergonomics (N_Opt_Phase 12)

Source: `nexus-opt/benchmarks/api_comparison/` +
[[API_COMPARISON|API_COMPARISON.md]].
Minimal idiomatic impls per library, subprocess runner, significant
LOC count + objective parity within `1e-4` (blackbox exempt).

| Problem | Verdict | LOC (Nexus vs best other) | Source |
|---------|---------|:-------------------------:|--------|
| LP trivial (2 var, 3 cons) | 🟡 tied — CVXPY wins at 8 vs Nexus 9 | 9 vs cvxpy 8 | §12 |
| MILP knapsack (10 items) | 🟡 three-way tie at 10 LOC (Nexus / PuLP / OR-Tools) | 10 vs pulp 10 | §12 |
| Blackbox 5D sphere | 🟢 tied SciPy on LOC (4), **0.0 obj in 13 ms** vs SciPy 1e-3 in 267 ms | 4 vs scipy 4 | §12 |
| QP 5-asset Markowitz | 🔴 5 LOC behind CVXPY — missing `nx.quad_form` | 17 vs cvxpy 12 | §12 |
| Infeasibility diagnostics | 🟢 **only library** returning named conflicts + relaxation suggestions | — | §12 |
| JuMP / KNITRO / GAMS cols | — | ⚫ deferred (separate toolchains) | §12 |
| Install-size comparison | — | ⚫ deferred | §12 |

**Nexus verdict:** 🟢 on infeasibility (unique IIS-like report +
relaxation suggestion), 🟡 on LP/MILP/blackbox LOC, 🔴 on QP ergonomics
until a `nx.quad_form(w, Sigma)` helper lands (tracked as candidate
native feature — expected to close the gap with CVXPY to within 1 LOC).

---

## Energy-layer benchmarks (nexus-energy)

### Internal flagship regression (`benchmarks/flagship_all_features.py`)

Not a head-to-head — it's the self-consistency bench for every phase
that has landed. All 9 rows pass at the Phase 13 boundary; see
[[BENCHMARKS|BENCHMARKS.md]] for the numbers. 🟢 self-verdict.

### Adaptive-oracle Benders (internal)

| Method | Wall | E[cost] | Verdict |
|--------|------|---------|---------|
| Benders plain | 0.113 s | 203 651 | baseline |
| Benders trust-region | 0.103 s | 203 651 | tied on obj, ~9 % faster |
| Benders adaptive-oracle | **0.100 s** | 203 651 | 🟢 fastest, tied on obj |
| Progressive hedging | 0.321 s | 246 500 | 🔴 stops at `max_iter` with 21 % gap (known — real ADMM PH deferred, tracked in `DEFERRALS.md` Phase 8.x) |

### vs PyPSA — internal scripted runs (nexus-energy standalone)

Source: `benchmarks/vs_pypsa.py` +
`test_projects/test_project_1/pypsa/COMPARISON.md`.

| Test | Scale | Parity | Speed (nexus vs PyPSA) | Verdict |
|------|-------|--------|------------------------|---------|
| WHOBS DE 168 h | 1 bus, 3 gen | bit-exact (rel 7e-16) | 10.5× faster | 🟢 |
| WHOBS DE 8760 h | full-year | bit-exact | faster | 🟢 |
| `model_energy` 2 920 snap | 2 bus, long horizon | bit-exact (7e-15) | 0.84× | 🟡 |
| `scigrid_de` transport-mode | 585 bus / 1 423 gen / 852 lines | bit-exact (obj 5 157 196.97) | 0.44× (PyPSA 2.28 s vs nexus 5.18 s) | 🟢 parity, 🟡 speed |
| `storage_hvdc` | 6 bus with AC r,x | +6.2 % obj (nexus uses transport relaxation) | 86× | 🟡 physics gap — tracked on the PyPSA-side docs |
| `ac_dc_meshed` | 9 bus meshed | +7.3× obj delta | 139× | 🟡 physics gap (same KVL issue) |
| `scigrid_de` native | same | −19.1 % obj (nexus DC-OPF vs PyPSA KVL) | 0.18× (PyPSA 2.35 s vs nexus 13.36 s) | 🟡 physics gap + build-time bottleneck |
| `scigrid_de` native — LP solve, IPM+crossover (18.s.1) | same | bit-exact vs simplex (Δobj 6e-9) | solve 13.85 → **5.87 s (2.36×)** lossless | 🟢 IPM wins on network-dominated LP |
| `carbon_management` | 2 164 bus, 6 830 links | HiGHS simplex didn't converge in 24 min on either side | N/A | blocks both libraries equally |

**Key takeaway:** where physics is identical (transport / capacity
expansion), nexus-energy is **bit-exact and 1–140× faster**. KVL /
AC-OPF studies are 🟡 blocked by the transport-vs-KVL modelling gap,
which is tracked in `ROADMAP.md` Phase 3 (DC-OPF auto-routing now
lands flows KVL-feasibly on the flagship runs — Phase 3 re-run below).

### vs PyPSA — QP economic dispatch (N_En_Phase 17.1)

Source: `nexus-energy/benchmarks/qp_dispatch_vs_pypsa.py` +
`nexus-energy/benchmarks/results/qp_dispatch_vs_pypsa*.json` +
`test_projects/test_project_1/pypsa/FLAGSHIP_COMPARISON.md` (N_En_Phase
17.1 section). 1 bus, 3 generators with linear + quadratic marginal
cost, time-varying load. PyPSA uses `marginal_cost_quadratic` →
linopy → HiGHS QP; nexus uses `nexus.Model` → each of
{highs, osqp, clarabel, ipopt}.

| T   | Parity (rel-spread) | PyPSA solve | nexus best solve | Speedup | Verdict |
|-----|---------------------|-------------|------------------|---------|---------|
|  24 | 3.07e-09            | 126.5 ms    | 0.92 ms (highs)  | **137×** | 🟢 |
| 168 | 2.98e-09            | 132.7 ms    | 4.97 ms (clarabel) | **27×** | 🟢 |
| 720 | 3.10e-09            | 234.6 ms    | 10.46 ms (osqp)  | **22×**  | 🟢 |

**Side-effect:** the H2H caught a real bug — `solve_with_highspy`
was silently dropping the quadratic objective (LP relaxation fallback).
Fixed by piping the upper-tri Hessian through
`highspy.Highs.passHessian(kTriangular)`. All four nexus dispatchers
+ PyPSA now agree to ~1e-9 rel.

### vs PowerModels.jl — SOCP AC-OPF (N_En_Phase 17.5)

Source: `nexus-energy/benchmarks/socp_opf_vs_powermodels.py` +
`nexus-energy/benchmarks/results/socp_opf_vs_powermodels.json` +
`test_projects/test_project_1/julia/PowerModels.jl/POWERMODELS_COMPARISON.md`.
Identical 3-bus radial case (b1 gen → b2 → b3 load; r=0.01, x=0.10 pu;
load 0.5 pu; linear MC=30). Both sides solve the **same Jabr SOCP lift**
(`SOCWRPowerModel` ≡ nexus `solve_socp_opf`).

| Case | PowerModels.jl + Ipopt | nexus + Clarabel | Parity (max rel Δ) | Speedup | Verdict |
|------|------------------------|------------------|--------------------|---------|---------|
| 3-bus radial | 1.30 ms (Ipopt internal) | **0.16 ms (Clarabel internal)** | 3.7e-05 | **8.26×** | 🟢 |

Total cost, gen dispatch, voltage magnitudes, branch flows, branch
losses all match the canonical reference to ≤ 5e-6 absolute.

**Scope note:** PGLib-OPF case30/case118 H2H unblocked 2026-04-20 —
transformer tap + phase shift, π-line shunts, and bus shunts are all
now supported in `solve_socp_opf` (N_En_Phase 10.3 + 17.3 side-effects).
PGLib row still ⏳ until the PGLib-OPF dataset is wired into the
benchmark harness; re-opens alongside the pandapower case30 row.

### vs PyPSA — SOCP AC-OPF (N_En_Phase 17.6)

📝 **External / unavailable.** PyPSA has no first-party SOCP-AC-OPF
path; the `pypsa-soc` community extension is not on PyPI. Row stays
open as "no comparator available today"; re-open if an upstream SOCP
path appears.

### vs pandapower — AC-OPF (N_En_Phase 17.3) — closed 2026-04-21

**Polar NLP row (landed 2026-04-21).** With N_Opt_Phase 5.2's
first-class nonlinear expressions (`sin` / `cos` / `sqrt` / `/` in
`nexus.Model`) + `solve_ac_opf_polar` — the true polar AC-OPF NLP
consumed via IPOPT — both sides now run the non-convex NLP on
identical per-unit admittance models.

| Case | pandapower (PIPS NLP) | nexus (polar NLP) | gap | Speedup | Verdict |
|------|-----------------------|-------------------|-----|---------|---------|
| case9 | obj 5 311.912 | **1.57 ms / obj 5 311.912** | **−4.3e-7 %** | **3.58×** | 🟢 |
| case14 | obj 8 081.527 | **1.62 ms / obj 8 081.526** | **+4.5e-6 %** | **3.58×** | 🟢 |
| case30 | obj 578.486 | **3.15 ms / obj 577.138** | **+0.2330 %** | **1.88×** | 🟢 |

case30 gap is non-convex local-optimum divergence — IPOPT from flat
start lands in a slightly better basin than PIPS does (lower cost =
better for a minimization). 6/6 parity tests at
`tests/phase_17/test_ac_opf_polar.py` green; see the "not worse than
PIPS" guard which catches genuine regressions. Benchmark CLI:
`python benchmarks/ac_opf_vs_pandapower.py --formulation polar
case9 case14 case30`.

**SOCP relaxation row (prior, unchanged).**



Source: `nexus-energy/benchmarks/ac_opf_vs_pandapower.py` +
`nexus-energy/benchmarks/results/ac_opf_vs_pandapower.json` +
`test_projects/test_project_1/pandapower/PANDAPOWER_COMPARISON.md` +
`test_projects/test_project_1/pandapower/run_ac_opf_reference.py`.
pandapower's `runopp` (PIPS-OPF interior-point NLP) is the canonical
Python AC-OPF reference; the nexus side runs the Jabr SOCP lift
(`solve_socp_opf`) with the **full quadratic cost** (MATPOWER gencost
model 2) now forwarded to both sides via N_En_Phase 10.4's
`generator.quadratic_cost`.

| Case | pandapower (PIPS NLP) | nexus (Jabr SOCP) | SOCP gap | Speedup | Verdict |
|------|-----------------------|-------------------|----------|---------|---------|
| case9 (quadratic cost, radial-ish) | pandapower obj 5 296.686 | **76 µs / obj 5 296.649** | **+0.0007 %** | **69.66×** | 🟢 |
| case14 (quadratic cost, 3 trafos + bus shunt) | pandapower obj 8 081.527 | **1.35 ms / obj 8 075.123** | **+0.0792 %** | **40.20×** | 🟢 |
| case30 (quadratic cost, heavily meshed) | pandapower obj 576.893 | **3.74 ms / obj 572.758** (plain Jabr) | **+0.7169 %** | **18.28×** | 🟢 |
| case118 (quadratic cost, 54 gens / 186 branches) | pandapower obj 129 660.7 | **21.2 ms / obj 129 332.5** (plain Jabr) | **+0.2533 %** | **8.10×** | 🟢 |

Reference JSON captured for case9 / case14 / case30 / case118 via
`run_ac_opf_reference.py` (isolated pandapower 2.14 venv). case9 +
case14 + case30 rows refreshed 2026-04-20 to use true `cp2` on both
sides (was previously zeroed to make both sides comparable before
quadratic-cost SOCP shipped). case118 landed same day —
`build_nexus_from_ppc` ingests pandapower's internal MATPOWER-format
`_ppc` dict directly, and with N_En_Phase 10.4 the polynomial cost
`cost = cp2·p² + cp1·p + cp0` now flows end-to-end via
`g.quadratic_cost = cp2·base_mva²`. **Honest finding:** the
−19.63 % case30 plain-Jabr gap that drove the 10.10→10.14.1
tightening marathon was almost entirely a **cost-linearisation
artifact** — with true cp2 the plain Jabr on case30 sits at
+0.72 %, tighter than the 10.14.1 OBBT+PWL result on linearised
cost (+2.39 %). 10.10–10.14.1 remain correct tightening layers
(sound relaxation lineage) but are no longer in the critical path
for these four IEEE cases.

**Side-effect: SOCP builder extensions (Phase 10.x slice).** Six
productivity items landed across this H2H, all backward-compatible
via `getattr` with safe defaults:
- π-line shunts (`link.{g_fr,b_fr,g_to,b_to}`) wired into sending /
  receiving flow equations.
- Bus shunts (`bus.{g_shunt,b_shunt}`) wired into P / Q balance rows.
- Generator `p_min` (was hard-coded to 0).
- Transformer tap + phase shift (`link.{tap,shift}`) — MATPOWER
  complex turns ratio `τ·e^(jφ)`, primary = bus_from; bit-exact
  backward compat when `tap=1, shift=0`.
- **Arctangent envelope** (Kocuk et al. 2016 "SOCP+AT", N_En_Phase
  10.10) — per-branch linear cuts `tan(θ_min) c_ij ≤ s_ij ≤
  tan(θ_max) c_ij`, opt-in via `solve_socp_opf(…, angle_diff_max=)`
  (function-level default, radians) or per-link
  `link.angle_diff_{min,max}` override. `angle_diff_max=None`
  preserves pre-10.10 behaviour bit-exactly. Sound for any
  `|θ_max| < π/2`.
- **QC-lite cycle closure** (N_En_Phase 10.11) — hand-rolled
  fundamental cycle basis (BFS spanning forest, parallel-branch
  safe) + per-branch angle auxiliary `θ_ij` ∈ [θ_min, θ_max]
  + *sound* linear sin-Taylor coupling
  `|s_ij − v_nom²·θ_ij| ≤ v_max²·θ_max³/6 + (v_max²−v_min²)/2·θ_max`
  + exact loop-closure equalities `Σ±θ_ij = 0` per cycle. Opt-in via
  `solve_socp_opf(…, enforce_cycle_closure=True)`; requires
  `angle_diff_max`. Radial nets: cycle basis empty, no-op. Meshed
  nets: constraint active only when the sin-Taylor slack is small
  enough to bind — on the default V±5 %/θ=30 ° box it is not, so
  case30 is unchanged.
- **Tight QC with McCormick bilinears** (N_En_Phase 10.12,
  Coffrin & Hijazi 2015) — per-bus `v_mag` aux linked to `c_ii`
  by secant upper + three tangent lower cuts; per-branch `w_ij`,
  `cos(θ_ij)`, `sin(θ_ij)` auxiliaries with cos concavity-tangent
  + sin-Taylor envelopes; McCormick bilinear cuts on
  `w = v_i·v_j`, `c_ij = w·cos θ`, `s_ij = w·sin θ` (12 cuts per
  branch). Opt-in via `enforce_tight_qc=True`; implies
  `enforce_cycle_closure=True`. Radial: backward-compat (cycle
  basis empty). Meshed: monotonicity holds (cost does not
  decrease vs Jabr-only) but case30 gap unchanged on the default
  V±5 %/θ=30 ° box — the McCormick boxes are still wide enough
  that the LP-merit-order Jabr dispatch stays feasible. Closing
  case30 on default bounds requires **OBBT pre-solve (10.14)** to
  shrink the voltage-product / angle boxes so McCormick binds.
- **OBBT pre-solve** (N_En_Phase 10.14, Coffrin & Van Hentenryck
  2012) — standalone `obbt_tighten(system, *, max_iter=3,
  tol=1e-4, angle_diff_max, enforce_tight_qc=True)` plus opt-in
  `solve_socp_opf(..., enable_obbt=True, obbt_iters=3)`. Each
  iteration rebuilds the SOCP with `add_cost_objective=False` and
  solves min/max subproblems for every `c_ii` and every `θ_ij`
  under the current relaxation; results tighten `bus.v_min/v_max`
  and `link.angle_diff_min/max` in place (never-loosen
  invariant). Required factoring `solve_socp_opf` into
  `_build_socp_problem(...) → _SOCPBuild` so OBBT can swap
  objectives via `prob.q = ±e_target` without duplicating the
  constraint set; also relaxed the arctan-envelope sign
  hard-codes so OBBT can legitimately tighten `angle_diff_min`
  to a positive value on one-sided intervals. Radial: no-op
  (slack dispatch). Meshed: case30 **−19.63 % → −4.45 %**
  (15.2pp closure, plateaus iter 8, ~10 s at 8 iters); case14
  **−0.08 % → −0.02 %** at 3 iters.
- **PWL cos envelope + asymmetric sin/cos boxes** (N_En_Phase
  10.14.1) — replaced the 10.12 two-tangent concavity envelope
  on `cos(θ)` with K evenly-spaced tangent upper cuts (default
  K=8, gated by `cos_envelope_pieces`), kept the global chord
  lower bound, and tightened both cos and sin boxes from
  symmetric `±tmax_abs` to asymmetric `[min(cos a, cos b),
  max(cos a, cos b) or 1 if 0∈[a,b]]` and `[sin(tmin),
  sin(tmax)]` (sin is monotone on (-π/2, π/2) so the latter is
  exact). McCormick bilinears on `s_ij = w·sin θ` inherit the
  tighter `[sin_lo, sin_hi]` box. Threaded
  `cos_envelope_pieces: int = 8` through `_build_socp_problem` /
  `solve_socp_opf` / `obbt_tighten` / `solve_socp_opf_multi`.
  **Case30 closure:** −4.45 % → **+2.39 %** (6.84pp, SOCP
  relaxation now a valid lower bound below the NLP). Gap is
  flat across K ∈ {2,4,6,8,12,16} (all ≈ 2.39 %) — the K-tangent
  upper envelope saturates quickly; the dominant tightening is
  the **asymmetric sin box** (~15× tighter on one-sided OBBT
  intervals like `[0.001, 0.03]`). Wall time at K=8 +
  obbt_iters=8: ~10.8 s.
- **Quadratic generator cost** (N_En_Phase 10.4) — opt-in
  `generator.quadratic_cost` (defaults 0 via `getattr`; pre-10.4
  systems bit-exact) routes `q2·p²` through
  `ConicProblem.add_quadratic_obj` with `P[p,p] = 2·q2`;
  convexity guard raises if `q2 < 0`. `build_nexus_from_ppc`
  now forwards MATPOWER `cp2` (gencost model 2, ncost 3) as
  `g.quadratic_cost = cp2·base_mva²` so both sides of the
  benchmark see the full polynomial cost. **Refreshed
  case9/14/30 on true cp2:** case9 +0.0007 % @ 69.66×, case14
  +0.0792 % @ 40.20×, case30 +0.7169 % @ 18.28× — case30
  flips 🟡 → 🟢 on plain Jabr alone. **Unblocked case118:**
  first-ever SOCP row, +0.2533 % @ 8.10× 🟢. PWL heat rates
  still deferred (different machinery — PWL aux vars + ordered
  cuts; rarely used on IEEE).

Regression tests `tests/phase_10/test_phase10.py::
test_socp_opf_line_and_bus_shunts`, `::test_socp_opf_transformer_
tap_and_shift`, `::test_socp_opf_angle_diff_envelope`,
`::test_socp_opf_cycle_closure`, `::test_socp_opf_tight_qc`,
`::test_socp_opf_obbt`, `::test_socp_opf_pwl_cos_envelope`, and
`::test_socp_opf_quadratic_cost` keep the Phase 10 suite at
15/15 green.

**Scope note (the NLP-vs-NLP row still ⏳):** today's 17.3 is a
SOCP-vs-NLP measurement. A true NLP-vs-NLP row needs nonlinear-
expression support in `nexus.Model` (sin/cos/sqrt/division) to encode
polar AC-OPF — unlocks once that ships (tracked in [[ROADMAP|ROADMAP.md]]). The
case9 SOCP row is tight (+0.072 % gap) because the Jabr lift is
nearly exact on radial cases; a true NLP row would reduce this to
machine precision.

### vs PyPSA — CINDER multi-carrier (ramp costs validation) — 2026-05-24

Source: `test_projects/test_project_1/pypsa/rembup/run_compare.py` +
`test_projects/test_project_1/pypsa/rembup/comparison_out/rembup_compare.json`.
_Note: this benchmark was renamed **REMBup → CINDER** on 2026-06-17. The
git-ignored scratch dir and its `REMBupConfig` package keep the old codename
`rembup/` (loaded by folder name); everywhere else it is **CINDER** — same
testbed, new name._
8 buses (electricity, heat, gas, biomass, cooling, hydrogen, curtailment,
water), 8 generators, 10 links (6 committable with UC), 7 storages,
90 days @ 15 min (8 640 timesteps), ramp costs on 5 components.

**Key finding:** PyPSA's UC formulation is LP-tight — Link-status
variables are marked binary but the LP relaxation naturally produces
integer solutions (0 % MIP gap at root, source `T` = Evaluate node,
1 B&B node). Two structural ingredients combine to make this work:
(1) start_up / shut_down declared binary with inequality state-
transitions `v[t] >= u[t] - u[t-1]`, `w[t] >= u[t-1] - u[t]`, and
(2) PyPSA's 1-var `Store` (single energy state e[t]) instead of
nexus's 3-var (`charge`, `discharge`, `soc`) — this removes the LP
degeneracy where the LP can split fractional cycling between charge
and discharge legs with no impact on cost.

Nexus has now adopted (1) (see N_En 2026-05-24 UC tightening below)
but not yet (2). LP polyhedron is therefore still slightly looser
than PyPSA's. MILP solves deterministically to within 2.26 % gap
but HiGHS spends ~400 s in randomized-rounding heuristics before
finding the optimum-near solution.

**LP-vs-LP** (UC vars relaxed to continuous on both sides):

| Metric | PyPSA | Nexus | Verdict |
|--------|-------|-------|---------|
| Wall time | 491.3 s | 161.1 s | 🟢 **Nexus 3.1× faster** |
| Objective | −1.159e8 | 669.5 | N/A (different load modelling) |
| CHP gas | 2 133 MWh | 2 133 MWh | 🟢 0.0 % |
| Gasifier | 3 184 MWh | 3 184 MWh | 🟢 0.0 % |
| Heat pump total | 1 925 MWh | 1 888 MWh | 🟢 1.9 % |
| Grid import | 5 856 MWh | 5 869 MWh | 🟢 0.2 % |

**MILP-vs-MILP** (binary UC on both sides, 3 % gap tolerance):

| Metric | PyPSA | Nexus | Verdict |
|--------|-------|-------|---------|
| Wall time | 190.4 s | **357 s** (was 639 s) | 🟡 PyPSA 1.9× faster (was 3.4×) |
| MIP gap | 0 % (LP-tight at root) | **0.82 %** (was 2.26 %) | improved |
| LP root solve | ~160 s | **273 s** (was ~650 s) | 2.4× faster |
| Determinism | every run | every run after fix | 🟢 |
| Cols pre-presolve | 518 K | **432 K** (selective v/w, was 484 K) | structural ✓ |
| Cols post-presolve | 347 K | 338 K | closing |
| Rows pre-presolve | 1 028 K | 363 K | structural ✓ |
| Rows post-presolve | 243 K | 249 K | near-parity |
| Binaries pre-presolve | 155 K | **52 K** (v/w continuous) | structural ✓ |
| Binaries fixed by presolve | 86 K of 155 K | 17 K of 52 K | improved |
| CHP gas | 2 133 MWh | 2 080 MWh | 🟢 2.5 % |
| Gasifier | 3 184 MWh | 3 104 MWh | 🟢 2.5 % |
| Heat pump total | 1 925 MWh | 1 930 MWh | 🟢 0.3 % |
| Grid import | 5 855 MWh | 5 885 MWh | 🟢 0.5 % |

**N_En 2026-05-24 UC tightening + dynamic switcher:**

1. **PyPSA-style LP-tight UC formulation** (`core.py:1542-1592`,
   `components/thermal.py:73-160`): start_up / shut_down declared
   binary, two inequality state-transitions (`v[t] >= u[t]-u[t-1]`,
   `w[t] >= u[t-1]-u[t]`) plus four upper-bound cuts (`v[t] <= u[t]`,
   `v[t] <= 1-u[t-1]`, `w[t] <= 1-u[t]`, `w[t] <= u[t-1]`) that
   uniquely determine v, w given u. Eliminated the prior non-
   deterministic behaviour. Net gap: 99.4 % → 2.26 % (deterministic).
2. **Spill-var bound fix** (`core.py:1042-1058`): replaced
   unconditional `upper=1e12` with `(inflow_max + e_cap) * 10` —
   bound range tightened from `[4e-05, 1e+12]` to `[4e-05, 2e+03]`,
   17 → 7 orders of magnitude. HiGHS numerical conditioning warning
   reduced from 4 → 2 lines.
3. **Dynamic `mip_strategy="auto"` switcher** (`core.py:769-870`):
   solves the LP relaxation first; if all binaries are within
   `INT_TOL=1e-4` of {0, 1} the LP optimum IS the MIP optimum (free
   win for naturally tight problems). Otherwise falls through to
   standard MIP. Guarded by a 5 000-binary-estimate threshold so it
   doesn't pay the LP cost twice on large models — CINDER has
   ~155 K binaries so the switcher correctly skips and goes
   straight to MIP.
4. **`uc_fix_schedule` extended to links** — was previously only
   used for generators; now any committable component can be
   schedule-pinned.
5. **`Storage.storage_model`** field + `effective_storage_model()`
   selector added (returns "store" for lossless symmetric storages,
   "full" otherwise). Implementation of the actual 1-var Store
   variable creation is queued — see DEFERRALS N_En_Phase 18.x.

**MILP speed gap — what's left:** the 1-var Store refactor
landed on 2026-05-24, dropping nexus's column count below PyPSA's
(484 K vs 518 K). MILP solve time dropped 702 → 639 s. But the
post-presolve binary count is unchanged: HiGHS still cannot probe
nexus's 155 K binaries down to constants, while PyPSA's presolve
fixes 86 K. That's the entire remaining gap — the LP itself is
slow because HiGHS has to branch through the surviving binaries.
Two further hypotheses were tested 2026-05-24 and reverted as
no-ops:

- **H1 — explicit redundant `status<=1`, `start_up<=1`,
  `shut_down<=1`** (PyPSA emits these even though redundant for
  binaries): added 207 K rows, all eliminated by presolve, zero
  effect on binary fixing.
- **H2 — explicit `p[t] >= 0` lower bound on committable links**
  (PyPSA writes the lower bound even when ``min_pu == 0``): same
  outcome.

What likely remains: nexus's multi-output Link auxiliaries
(`_flow_out_vars`, `_loss_vars`) for `bus_to_2` / `efficiency2`
(CHP, gasifier, fuel cell). PyPSA bakes `efficiency2` directly
onto the `p` variable via the bus_to_2 connection — same
physics, fewer aux variables, potentially the structural pattern
HiGHS probing needs.

Overall: 🟢 dispatch parity validated. 🟢 LP speed (3.1× faster).
🟢 MILP determinism (gap fixed at 2.26 %). 🟡 MILP speed (3.4×
slower after Store refactor; remaining gap traced to multi-output
Link formulation).

### vs PyPSA — flagship tutorial runs

Source: `test_projects/test_project_1/pypsa/FLAGSHIP_COMPARISON.md`.

| Run | snapshots | Obj diff | Speed | Verdict |
|-----|-----------|----------|-------|---------|
| MPC rolling-horizon (synthetic) | 168 | 2.8e-16 (exact) | 8.1× | 🟢 |
| PyPSA-Earth Nigeria+Benin 1-week | 36 | 1.44 % | 10.2× | 🟢 |
| PyPSA-Eur Belgium 1-week | 7 | 1.13 % | 42.6× | 🟢 |

### vs GenX (Julia)

Source: `test_projects/test_project_1/julia/GENX_COMPARISON.md`.

| Run | snapshots | Wall | Obj | Verdict |
|-----|-----------|------|-----|---------|
| GenX tutorial default (TDR on, UC lin) | 1 848 | 34.6 s | $1.01 e10 | GenX baseline |
| GenX stripped LP full year | 8 760 | 563.7 s | $4.815 e9 | GenX baseline |
| **nexus-energy v0.4** same stripped LP | 8 760 | **41.8 s** | $4.68 e9 | 🟢 **13.5× faster, 2.80 % lower obj** |

**Why the 2.80 %:** cost-stack definition diffs (treatment of VOM,
line losses); re-audit in N_En_Phase 15.1 re-measure, still pending.

### vs Tulipa

⚫ `TULIPA_COMPARISON.md` + `nexus_mirror_tulipa/` exist but no
numbers locked in yet. Planned under N_En_Phase 17.2 once N_En_Phase
16.4 (fractional rep-period mapping) lands.

---

## Rollup — where Nexus stands today

**🟢 Clear wins (deploy with confidence):**
- Energy-layer transport-mode + capacity expansion vs PyPSA (bit-exact, 1–140×).
- Energy-layer stripped LP full-year vs GenX (13.5×).
- Blackbox CMA-ES and DE vs every third-party library we've measured.
- MILP knapsack at 1K items vs PuLP+CBC.
- MOO on ZDT6 + DTLZ6 — convergence-stressed problems where
  Nexus scalarisation out-performs NSGA-II's single-pop budget.
- TSP burma14 (n=14) — Nexus MILP proves optimum in 0.22 s.
- Stochastic extensive-form LPs vs Pyomo — 2.1–7.7× faster at
  scenario counts where Pyomo's Python assembly dominates.
- Internal Benders (adaptive-oracle best-of-breed).
- Infeasibility diagnostics — `r.infeasibility_report()` returns
  named conflicting constraints + relaxation suggestions out of the
  box; CVXPY/PuLP return only a status string, Pyomo+appsi_highs
  raises and needs try/except wrapping.
- Solver routing — `m.solve(solver=…)` now dispatches natively
  across {`highs`, `osqp`, `clarabel`, `ipopt`, `scipy`} with
  parity to 1e-4 rel tolerance (LP / QP / NLP; N_En_Phase 10.5 +
  10.9 landed 2026-04-20). No cvxpy/pyomo modelling layer required.
- QP economic dispatch vs PyPSA (N_En_Phase 17.1) — same problem,
  same math, **22-137× faster** depending on T (T=24/168/720),
  parity at 1e-9 rel across all four nexus backends + PyPSA→HiGHS.
- Multi-period SOCP AC-OPF (N_En_Phase 10.1) — `solve_socp_opf_multi`
  now provides per-snapshot Jabr-relaxation results aggregated over
  T snapshots; verified on 3-bus radial w/ T=4 load profile
  (`tests/phase_10/test_phase10.py::test_socp_opf_multi_period_radial`).
- SOCP AC-OPF vs PowerModels.jl (N_En_Phase 17.5) — identical Jabr
  relaxation, 3-bus radial, parity at 3.7e-05 rel, **8.26× faster**
  than `SOCWRPowerModel + Ipopt` in the reference toolchain. PGLib
  case30 deferred pending tap/shunt support in the SOCP builder.
- SOCP AC-OPF vs pandapower NLP (N_En_Phase 17.3, partial) — IEEE
  case9 **+0.0007 %** at **69.66×**, case14 (3 trafos + 1 bus
  shunt) **+0.0792 %** at **40.20×**, case30 (heavily meshed,
  41 branches) **+0.7169 %** at **18.28×**, case118 (54 gens /
  186 branches) **+0.2533 %** at **8.10×** — all 🟢 on plain
  Jabr with full quadratic cost after N_En_Phase 10.4 shipped
  `generator.quadratic_cost`. **Finding:** the −19.63 % case30
  plain-Jabr gap from the linearised-cost era was a cost-
  linearisation artifact; 10.10–10.14.1 tightening layers
  remain correct but aren't needed for these IEEE cases. A
  true NLP-vs-NLP row waits on nonlinear-expression support in
  `nexus.Model`.

**🟡 Tied / flagged caveats:**
- CINDER multi-carrier dispatch parity (🟢 within 2.5 % on all major
  link flows). LP-vs-LP: 🟢 Nexus 3.1× faster. MILP-vs-MILP: 🟡
  PyPSA 3.4× faster (down from 3.7× after the 2026-05-24 Store
  refactor) — PyPSA's UC formulation is LP-tight (0 % gap at root,
  HiGHS presolve fixes 86 K of 155 K binaries), nexus still needs
  real B&B (2.26 % gap, presolve fixes ~12 binaries). Remaining
  structural gap traced to nexus's multi-output Link aux
  variables (`_flow_out_vars`, `_loss_vars` for `bus_to_2` /
  `efficiency2`) — queued as N_En_Phase 18.y.
- `model_energy` 2 920 snap (0.84× — laptop noise / solver-bound).
- PyPSA AC-OPF studies (physics gap; Phase 3 DC-OPF auto-routing
  closes feasibility, not yet cost parity).
- Blackbox rosenbrock quality (cma.py slightly better at same budget).
- GenX stripped-LP obj −2.80 % (cost-stack definition under audit).

**🔴 Open losses:**
- MILP knapsack at 10 K items — CBC 28× HiGHS, objectives diverge
  0.01 %; re-audit CBC's "Optimal" status.
- NLP at scale — scipy.SLSQP is ~1 400× slower than CasADi+IPOPT on
  horizon-400 MPC. 10.9 routing landed, but Nexus's modelling layer
  still lacks nonlinear expressions — large-scale NLP parity needs
  both the solver bridge (✅) and first-class `sin/cos/exp/log` in
  `nexus.Model` (pending).
- MOO on ZDT1-4 / DTLZ1-5,7 — Nexus `pareto_frontier` is 3–50×
  worse on IGD and 10–50× slower than pymoo/Platypus NSGA-II.
  Blocker: no native many-objective / reference-direction path
  (candidate N_Opt_ native feature).
- TSP at n≥48 — Nexus MILP MTZ returns **no incumbent** in 60 s
  (att48 / eil76 / kroA100); OR-Tools routing finds 0–0.74 % gap
  in 3 s. Blocker: weak MTZ LP relaxation. Candidate native
  features: lazy DFJ subtour cuts **or** a native 2-opt / LKH
  heuristic.
- QP API ergonomics vs CVXPY — 5 LOC behind on Markowitz until
  `nx.quad_form(w, Sigma)` lands. Today requires an explicit
  upper-tri `i ≤ j` double loop.

**⚪ Gaps (no nexus-direct row yet):**
- QP — ✅ solved 2026-04-20; nexus-direct Clarabel/OSQP/Ipopt rows
  now available via `solve_with_{clarabel,osqp,ipopt}`.
- SOCP — ✅ for AC-OPF (Jabr lift via `solve_socp_opf` /
  `solve_socp_opf_multi`); transformer tap + phase shift, π-line
  shunts, bus shunts all wired 2026-04-20 (N_En_Phase 10.3), so
  MATPOWER / PGLib AC-OPF cases can be ingested directly. SDP —
  solver dispatch works but requires first-class PSD constraint
  primitives in Nexus's modelling layer.
- NLP IPOPT bridge — ✅ landed 2026-04-20 via CasADi's bundled Ipopt;
  large-scale AD win still waits on nonlinear-expression support in
  Nexus's modelling layer.
- Many-objective MOO — no reference-direction NSGA-III / ε-archive /
  proper M>2 scalarisation path. Pareto runs pairwise on 2 of M.
- Native TSP — no lazy DFJ subtour-elimination MILP callbacks, no
  native metaheuristic. MTZ is the only path today.

**⚫ Deferred datasets:**
- NETLIB (LP), MIPLIB 2017 (MILP), Maros-Meszaros (QP), PGLib-OPF
  (SOCP/SDP), HS119 / CUTEst / COPS (NLP), BBOB / CEC (blackbox),
  WFG (MOO), ZDT5 binary MOO, full MOO indicator set
  (hypervolume / GD / spacing / coverage), TSPLIB ch150 / tsp225 /
  a280 / pcb442, LKH + Concorde native-binary TSP columns, DRO
  Wasserstein portfolio, scenario-tree generation, mpi-sppy /
  PySP / SDDP.jl stochastic columns. All tracked in
  `nexus-opt/BENCHMARK_ROADMAP.md`.

---

## Update protocol

When a benchmark lands, the author updates **two** places, not more:

1. The **detail doc** — the `BENCHMARK_ROADMAP.md` headline /
   `COMPARISON.md` row / etc. with full numbers.
2. **This scorecard** — one line per H2H, verdict mark, link to the
   detail doc.

Do not put raw tables of numbers in this file unless they serve the
verdict (the blackbox CMA table does; most don't). If two rows have
the same verdict, collapse them. The point of this file is *one
glance tells me where we stand;* anything longer defeats the purpose.
