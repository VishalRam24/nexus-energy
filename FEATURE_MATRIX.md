---
title: "Feature matrix: nexus-energy vs the world"
project: nexus-energy
type: planning
status: active
last_verified: 2026-04-26
tags:
  - feature-matrix
  - parity
  - comparison
---

# Feature matrix: nexus-energy vs the world

Ground truth for what "feature parity" means. Edit the **nx** column as
N_En_Phase 1–12 land (see `ROADMAP.md`).

Legend: **Y** = native, **P** = partial / plugin / unverified, blank =
not present. **nx** = nexus-energy (`src/nexus_energy/`).

> **✅ 2026-06-03 deferral-closeout batch** flipped many remaining **P**/blank
> `nx` cells to **Y**: hot/warm/cold starts, contingency reserve, planned outage,
> forced-outage scenarios, electrolyzer co-location (part-load + shared cap),
> temperature-tier heat, Storage/Link retrofit, policy slack, adaptive +
> multi-resolution timestep, fractional-rp LDS, Weymouth gas, head-dependent
> hydro, spatial/nested Benders, Dantzig-Wolfe, SDDiP, general chance
> constraints, Wasserstein DRO, risk-averse Benders, multi-period/quad SOCP,
> generic Clarabel routing, LP basis hot-start, external-solver bridge, DuckDB
> reader, adaptive ML warm-start, multi-bus differentiable dispatch. Full suite
> 305 passed. Detail: `../progress_log.md` 2026-06-03 + `DEFERRALS.md`.

## A. Network physics

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Transport / pipe flow | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| DC-OPF (KVL or PTDF) | Y | Y | | | | Y | Y | Y | |
| AC-OPF (NLP) | Y | | | | | | Y | Y | |
| AC-OPF SOCP relaxation | Y | | | | | | | Y | |
| AC-OPF SDP relaxation | | | | | | | | Y | |
| AC-OPF QC relaxation | | | | | | | | Y | |
| Transmission losses (PWL) | Y | Y | Y | | P | Y | Y | Y | |
| Line / transformer switching | Y | | | | | | Y | Y | |
| HVDC as controllable link | Y | Y | | Y | Y | Y | Y | Y | Y |
| Multi-carrier (gas / H₂ / heat) | Y | Y | Y | Y | Y | Y | P | | Y |
| CO₂ as tracked carrier | Y | Y | | Y | Y | Y | | | Y |
| P2H sector coupling | Y | Y | | Y | Y | Y | | | Y |
| P2G / electrolysis | Y | Y | Y | Y | Y | Y | | | Y |
| Gas network flow w/ pressure | Y | Y | | | P | Y | | Y | |
| Heat network w/ temperature | P | | | P | Y | Y | | | |
| N-1 security | Y | | Y | | | | Y | Y | |

## B. Dispatch realism

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Unit commitment (binary) | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Linear-relaxed UC | Y | Y | Y | | P | Y | Y | | Y |
| Clustered UC | Y | | Y | | | Y | Y | | Y |
| Min stable generation | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Ramp up/down | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Min up time | Y | Y | Y | | Y | Y | Y | | Y |
| Min down time | Y | Y | Y | | Y | Y | Y | | Y |
| Start-up cost | Y | Y | Y | Y | Y | Y | Y | | Y |
| Shut-down cost | Y | Y | Y | | Y | Y | Y | | |
| Start-up fuel | Y | | Y | | | Y | Y | | |
| Hot / warm / cold starts | | | Y | | | Y | Y | | |
| Part-load efficiency (PWL) | Y | | Y | | Y | Y | Y | | |
| Spinning reserves | Y | | Y | | P | Y | Y | Y | Y |
| Regulation up / down | Y | | Y | | | Y | Y | | |
| Contingency reserves | | | Y | | | Y | Y | | |
| Must-run | Y | Y | Y | | Y | Y | Y | | Y |
| Planned outage | | | Y | | | Y | Y | | |
| Forced outage (stochastic) | | | | | | Y | Y | | |
| Fuel supply limits | Y | | Y | Y | Y | Y | Y | | |

## C. Storage

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Linear SOC balance | Y | Y | Y | Y | Y | Y | Y | | Y |
| Cyclic SOC | Y | Y | Y | Y | Y | Y | Y | | Y |
| Non-cyclic / fixed initial SOC | Y | Y | Y | Y | Y | Y | Y | | Y |
| Min / max duration (h) | Y | Y | Y | Y | Y | Y | Y | | Y |
| Self-discharge | Y | Y | Y | Y | Y | Y | Y | | Y |
| Separate charge/discharge η | Y | Y | Y | Y | Y | Y | Y | | Y |
| Simultaneous charge/disch ban | Y | | Y | | | Y | Y | | |
| LDS inter-period linkage | Y | | Y | Y | | Y | | | Y |
| Multi-reservoir hydro cascade | Y | | | | Y | Y | Y | | Y |
| Pumped hydro (pump + turbine) | Y | Y | Y | Y | Y | Y | Y | | Y |
| Head-dependent hydro η | | | | | | Y | Y | | |
| Hydro inflow / spill | Y | Y | Y | Y | Y | Y | Y | | Y |
| H₂ electrolyzer co-location | Y | Y | Y | Y | Y | Y | | | Y |
| Thermal storage | Y | Y | | Y | Y | Y | | | Y |
| EV / V2G mobile storage | Y | Y | | Y | Y | Y | | | |

## D. Policy & constraints

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| CO₂ mass cap (global) | Y | Y | Y | Y | Y | Y | | | Y |
| CO₂ cap by zone | Y | Y | Y | Y | | Y | | | Y |
| CO₂ rate cap (tCO₂ / MWh) | Y | | Y | | | | | | |
| CO₂ price / tax | Y | Y | Y | Y | Y | Y | | | Y |
| Fuel-specific emission factor | Y | Y | Y | Y | Y | Y | | | Y |
| RPS | Y | Y | Y | Y | | Y | | | Y |
| CES | Y | | Y | | | | | | |
| Min capacity carveout | Y | Y | Y | Y | Y | Y | | | Y |
| Max capacity carveout | Y | Y | Y | Y | Y | Y | | | Y |
| Capacity reserve margin | Y | | Y | | | Y | Y | | Y |
| Locational / DER capacity caps | Y | Y | Y | Y | Y | Y | | | Y |
| ITC (CapEx tax credit) | Y | | Y | | | | | | |
| PTC (production tax credit) | Y | | Y | | | | | | |
| 24/7 hourly matching | | | Y | P | | | | | |

## E. Investment & planning

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Single-stage capacity expansion | Y | Y | Y | Y | Y | Y | Y | | Y |
| Myopic multi-stage | Y | Y | Y | | P | Y | | | |
| Perfect-foresight multi-stage | Y | Y | Y | Y | Y | Y | Y | | Y |
| Rolling myopic planning | Y | | Y | | | Y | Y | | |
| Retrofit / fuel-switching | Y | | Y | | | Y | | | |
| Endogenous retirement | Y | Y | Y | Y | | Y | | | Y |
| Forced / scheduled retirement | Y | Y | Y | Y | Y | Y | | | Y |
| Vintage / cohort tracking | Y | | Y | Y | | Y | | | Y |
| Construction lead time | Y | | Y | | | Y | | | |
| Brownfield existing capacity | Y | Y | Y | Y | Y | Y | Y | | Y |
| Transmission expansion | Y | Y | Y | Y | Y | Y | | | Y |
| Integer / discrete investment | Y | Y | Y | Y | Y | Y | | | Y |
| Economies of scale (PWL CapEx) | Y | | Y | | Y | Y | | | |

## F. Temporal handling

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Snapshot weights | Y | Y | Y | Y | Y | Y | Y | | Y |
| Representative periods | Y | P | Y | Y | P | Y | Y | | Y |
| Built-in k-medoids / clustering TDR | Y | | Y | Y | P | | | | Y |
| Extreme-period preservation | Y | | Y | Y | | | | | |
| LDS inter-period linkage | Y | | Y | Y | | Y | | | Y |
| Rolling horizon / MPC | Y | | | Y | Y | Y | Y | | Y |
| Sub-hourly resolution | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Variable / adaptive timestep | | | | | | Y | | | Y |
| Multi-resolution hierarchy | | | | | | Y | | | Y |
| Stochastic / fractional period→rp mapping | Y | | | | | | | | Y |

## G. Uncertainty

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Deterministic scenarios | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Two-stage stochastic (Benders + extensive) | Y | | | | | Y | Y | | |
| CVaR / risk measures (Rockafellar-Uryasev) | Y | | | | | Y | P | | |
| Worst-case / minimax objective | Y | | | | | | | Y | |
| Robust optimisation (Bertsimas-Sim budget set) | Y | | | | | | | Y | |
| Chance constraints (SAA + Bonferroni helper) | Y | | | | | | P | Y | |
| Moment-matching scenario generation (Høyland-Wallace) | Y | | | | | | | | |
| Scenario reduction (k-medoids) | Y | | | | | Y | Y | | |
| Monte Carlo / out-of-sample harness | Y | | | | | | Y | | |
| SDDP / multi-stage stochastic (SDDiP) | | | | | | | P | | |

## H. Decomposition & scaling

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Spatial Benders | P | | P | | | Y | | | |
| Temporal Benders (investment + per-period sub) | Y | | Y | | | Y | | | Y |
| Temporal decomposition (rolling horizon) | Y | | Y | | | Y | | | Y |
| Regularized / stabilized Benders (trust-region) | Y | | Y | | | | | | |
| Adaptive-oracle Benders (Mazzi 2024) | Y | | | | | | | | |
| Dantzig-Wolfe | | | | | | | | | |
| Nested Benders | | | | | | Y | | | |
| Progressive hedging | Y | | | | | Y | | | |
| Parallel / distributed solves | | | Y | P | | Y | Y | Y | Y |
| Column generation | | | | | | | | | |

## I. Interoperability & UX

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| Native CSV / YAML / DB format | P | Y | Y | Y | Y | Y | Y | Y | Y |
| PyPSA import / export | Y | Y | | | | | | | |
| GenX CSV import | | | Y | | | | | | |
| MATPOWER / PSSE import | | Y | | | | | Y | Y | |
| PyPSA-Eur dataset | | Y | | | | | | | |
| Database scenarios (Spine) | | | | | | Y | | | |
| Shadow prices / duals | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Hourly dispatch export | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Scenario compare / diff | | | Y | P | | Y | Y | | |
| Built-in plotting / maps | | Y | | Y | Y | | Y | | |
| Visual IDE / web UI | P* | P | | | P | Y | | | |
| Python API | Y | Y | P | Y | Y | P | P | P | P |
| Julia API | | | Y | | | Y | Y | Y | Y |

\* NexusFlow (`nexus-ide` fork) covers visual editing; roadmap N_En_Phase 12
adds in-browser solve preview (WASM).

## J. Advanced solver features

| Feature | nx | PyPSA | GenX | Calliope | oemof | SpineOpt | Sienna | PowerModels | Tulipa |
|---|---|---|---|---|---|---|---|---|---|
| HiGHS | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Gurobi | | Y | Y | Y | Y | Y | Y | Y | Y |
| CPLEX | | Y | Y | Y | Y | Y | Y | Y | Y |
| Clp / CBC | | Y | Y | Y | Y | | Y | Y | |
| Mosek (conic) | | P | | | | | Y | Y | |
| Xpress | | Y | Y | Y | Y | | Y | Y | |
| Ipopt (NLP) | | | | | | | Y | Y | |
| SCIP | | | | | | | | | |
| OSQP | P | | | | | | | | |
| Clarabel | Y | | | | | | | | |
| Warm-start | P | Y | Y | | P | Y | Y | Y | Y |
| Solver-specific MIP tuning | | Y | Y | Y | Y | Y | Y | Y | Y |
| Custom callbacks / heuristics | | | | | | Y | Y | Y | |
| **GPU LP (PDLP / cuOpt)** | | | | | | | | | |
| **ML warm-start** | Y | | | | | | | | |
| ML feature extractors (system + timestep) | Y | | | | | | | | |
| Historical-neighbour UC predictor (k-NN bank) | Y | | | | | | | | |
| GNN UC predictor (torch-optional hook) | Y | | | | | | | | |
| Learned variable fixing | Y | | | | | | | | |
| Learned representative-period selection | Y | | | | | | | | |
| **Differentiable layer** | Y | | | | | | | | |
| Closed-form dispatch sensitivities (ridge-regularised QP) | Y | | | | | | | | |
| cvxpylayers torch hook | P | | | | | | | | |
| Parallel scenario runner (ray / process-pool / serial) | Y | | | | | | | | |
| WASM LP bridge (in-browser preview) | Y | | | | | | | | |
| **Rust hot-path assembly** | | | | | | | | | |

The bolded rows in **J** are the 2025–26 differentiators — rows where
"no one in the open ecosystem has it yet" and where nexus lands in
Phases 10–12.

Standing benchmark evidence for the `nx` column lives in
[[BENCHMARKS|BENCHMARKS.md]] (last refreshed at the N_En_Phase 13
boundary, 2026-04-19). Don't add `Y` to a row without a script + a
JSON in `benchmarks/results/` that demonstrates it.

The Tulipa column was verified at the N_En_Phase 16 boundary (2026-04-19)
against TulipaEnergyModel.jl v0.21.0; the head-to-head measurement
lives in
[[TULIPA_COMPARISON|TULIPA_COMPARISON.md]].
Cells whose Tulipa value was previously speculative and is now backed
by a paired solve: A.transport-flow, B.unit-commitment-binary,
B.linear-relaxed-UC, B.ramp-up-down, C.linear-SOC-balance,
C.cyclic-SOC, C.self-discharge, E.single-stage-capacity-expansion,
E.integer-discrete-investment, F.snapshot-weights,
F.representative-periods, F.LDS-inter-period-linkage. No cell values
changed (every observed Tulipa feature was already marked `Y`).

## Sources

- PyPSA: https://pypsa.readthedocs.io
- GenX.jl: https://genxproject.github.io/GenX/
- Calliope: https://calliope.readthedocs.io
- oemof-solph: https://oemof-solph.readthedocs.io
- SpineOpt.jl: https://spine-tools.github.io/SpineOpt.jl/
- Sienna / PowerSimulations.jl: https://nrel-sienna.github.io/PowerSimulations.jl/
- PowerModels.jl: https://lanl-ansi.github.io/PowerModels.jl/
- Tulipa Energy Model: https://tulipaenergy.github.io/TulipaEnergyModel.jl/
