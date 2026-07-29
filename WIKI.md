# Nexus-Energy — Comprehensive Library Wiki & User Guide

Welcome to the **Nexus-Energy** Wiki. This document is a complete, project-agnostic, studyable reference for the Nexus Energy modeling library (`nexus-energy`). It outlines the library's high-level components, solver controls, temporal engine, robust/stochastic optimization suite, AC-OPF relaxations, ML-guided acceleration, differentiable economic dispatch layers, and the auto-calibration / MPC operations stack.

---

## 1. Abstraction Model & Parity Status

Nexus-Energy is a high-level, 15-sector energy system optimization library. It abstracts away the complex mathematical programming of grid physics, temporal linkages, and policy rules, letting you model by placing modular components on a topological network graph.

```
                  ┌──────────────────────┐
                  │     EnergySystem     │
                  └──────────┬───────────┘
         ┌───────────────────┼───────────────────┐
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│       Bus       │ │   Generator     │ │     Storage     │
│ (Electricity,   │ │ (Solar, Gas,    │ │ (Battery, Pump, │
│  H2, Heat, CO2) │ │  Hydro, MustRun)│ │  Thermal, V2G)  │
└────────┬────────┘ └─────────────────┘ └─────────────────┘
         │
         ├───────────────────┐
┌────────▼────────┐ ┌────────▼────────┐
│      Load       │ │      Link       │
│  (Demand side)  │ │ (Line, HP, HVDC,│
└─────────────────┘ │  Electrolyzer)  │
                    └─────────────────┘
```

### Competitor Feature Parity
Nexus-Energy is designed to achieve full feature parity with, and exceed the speed of, major industry libraries: **PyPSA, GenX, Calliope, oemof, SpineOpt, Sienna, PowerModels, and Tulipa**.

- **Network Physics**: N-1 security, DC-OPF, HVDC, polar AC-OPF, and SOCP conic relaxations.
- **Dispatch Realism**: Tighter-by-default Unit Commitment (UC), startup/shutdown costs, ramping, must-run rules, and spinning/regulation reserves.
- **Storage Complexity**: Self-discharge, cyclic SOC, separate charge/discharge efficiency, simultaneous charge/discharge bans, cascade hydro, and mobile EV/V2G.
- **Policy Rules**: Zonal/global/pooled-group CO2 caps, RPS, Clean Energy Standards (CES), CapEx/production tax credits (ITC/PTC), and hourly matching.

---

## 2. Core Modeling Syntax Guide

### Basic Economic Dispatch
```python
import nexus_energy as ne

# 1. Initialize the system
sys = ne.EnergySystem("My_System")

# 2. Add Buses (carriers map to default units like MWh, tCO2)
elec = sys.add_bus("elec", carrier="electricity")

# 3. Add Generators
# Generators automatically inherit carrier properties
sys.add_generator("pv", bus=elec, capacity=500.0, marginal_cost=0.0)
sys.add_generator("gas", bus=elec, capacity=200.0, marginal_cost=50.0)

# 4. Add Loads (Demand side)
sys.add_load("demand", bus=elec, amount=300.0)

# 5. Optimize the dispatch
result = sys.optimise()

if result.status == "optimal":
    print(f"Total Dispatch Cost: ${result.total_cost}")
    print(f"PV Generation: {result.generator_dispatch['pv'][0]} MW")
    print(f"Gas Generation: {result.generator_dispatch['gas'][0]} MW")
```

### Unit Commitment (MILP)
For thermal generators, unit commitment incorporates start-up costs, ramping constraints, and minimum up/down time limits. Nexus-Energy automatically compiles these into a highly tight **3-bin formulation**, minimizing solver branch-and-bound nodes under the hood.

```python
sys = ne.EnergySystem("Unit_Commitment_Demo")
elec = sys.add_bus("elec")

# Add a thermal unit with integer/binary unit commitment active
sys.add_generator(
    "coal_plant", 
    bus=elec, 
    capacity=100.0,
    min_stable_generation=20.0,    # Minimum stable load (P_min)
    marginal_cost=30.0,
    startup_cost=5000.0,            # Start-up cost
    shutdown_cost=1000.0,           # Shut-down cost
    min_up_time=8,                  # Minimum uptime (hours)
    min_down_time=6,                # Minimum downtime (hours)
    ramp_limit_up=40.0,             # MW/hour
    ramp_limit_down=40.0,           # MW/hour
    unit_commitment=True            # Activates binary commitment tracking
)

sys.add_load("demand", bus=elec, amount=[10, 50, 90, 80, 20, 10, 80, 90])
result = sys.optimise()
```

**Formulation notes (LP-tight 3-bin).** For committable Links and non-clustered committable Generators, `start_up`/`shut_down` indicators are declared **binary** and the state transition is compiled as two inequalities (`v[t] ≥ u[t] − u[t−1]`, `w[t] ≥ u[t−1] − u[t]`) plus four upper-bound cuts (`v[t] ≤ u[t]`, `v[t] ≤ 1 − u[t−1]`, `w[t] ≤ 1 − u[t]`, `w[t] ≤ u[t−1]`) — PyPSA-style LP-tight; the `v + w ≤ 1` mutex is implied and no longer a separate row. Clustered generators keep the equality transition form.

**Multi-state (hot/warm/cold) starts.** `Generator.start_up_segments` accepts a list of `(min_off_timesteps, startup_cost)` tuples (ascending `min_off_timesteps`, non-decreasing costs, first entry `0`). When set it overrides the flat `startup_cost` with the tight Morales-España (2013) start-type formulation. Binary UC only — clustered UC falls back to the flat cost.

**Link UC indicator economy.** Committable links create the startup indicator `v[t]` only when `startup_cost > 0` or `min_up_time > 1`, and the shutdown indicator `w[t]` only when `shutdown_cost > 0` or `min_down_time > 1` — fewer binaries when the costs/durations don't need them.

### Multi-Stage Capacity Expansion (Planning)
Examines investment decisions over decades, balancing capital expenditures (CapEx) for building new plants against operational expenditures (OpEx).

```python
# Create a multi-decade greenfield/brownfield investment model
planning_sys = ne.MultiStageSystem("Grid_Expansion_Plan")

# Add a candidate investment generator
planning_sys.add_candidate_generator(
    "new_solar",
    bus=elec,
    max_capacity_limit=2000.0,      # Maximum physical build limit
    capital_cost=80000.0,           # CapEx ($/MW-year)
    marginal_cost=0.0
)

# Optimize investment decision
planning_result = planning_sys.solve_expansion()
```

---

## 3. Storage Complexity

Nexus-Energy models advanced storage physics with highly specific parameters to prevent simultaneous charge/discharge behavior and represent real-world losses.

```python
sys.add_storage(
    "lithium_battery",
    bus=elec,
    power_capacity=100.0,           # Max charge/discharge rate (MW)
    energy_capacity=400.0,          # Storage volume (MWh) -> 4-hour battery
    charge_efficiency=0.92,         # Charge efficiency (92%)
    discharge_efficiency=0.92,      # Discharge efficiency (92%)
    self_discharge_rate=0.001,      # Self-discharge fraction per hour
    cyclic_state_of_charge=True,    # Final SOC must equal initial SOC
    ban_simultaneous_charge=True    # Enforce strict binary lock
)
```

### Store vs Full formulation

`Storage.storage_model` selects the LP variable shape: `"auto"` (default) | `"store"` | `"full"`. The `"store"` model compiles a **single energy-state variable `e[t]`** per timestep (PyPSA Store-style) instead of the 3-variable (`charge`, `discharge`, `soc`) formulation — a smaller LP polyhedron that helps HiGHS find integer-feasible vertices at root in models with committable links. `Storage.effective_storage_model()` resolves `"auto"`: a storage qualifies for `"store"` only when round-trip efficiency is ≈ 1, charge/discharge capacities are symmetric, and none of `inflow`, `spill_to`, `ramp_cost`, `no_simultaneous`, `availability`, `long_duration`, or non-zero `marginal_cost`/`marginal_cost_charge` is in use; otherwise it returns `"full"`. Result extraction derives `charge[t]`/`discharge[t]` from the signed `Δe/dt`, so downstream consumers see the same arrays either way.

**Extendable storages** are also supported in store mode: `cap_power_var`/`cap_energy_var` are created with SOC bounded by `soc_min·cap_energy ≤ e[t] ≤ soc_max·cap_energy` and the power rate by `cap_power·dt`.

### Cyclic SOC level

`cyclic=True` enforces `soc(0) == soc(T)`. The new `cyclic_level` field controls how the boundary anchors the *level*:
- `"fixed"` (default, historical) — additionally pins `soc(0) = soc_initial·capacity`.
- `"free"` — only continuity is enforced; the optimiser picks the cheapest cyclic level. This is PyPSA's `cyclic_state_of_charge` convention, and `from_pypsa` sets it automatically (see §14).

### Hydro inflow & spill

Storages with `inflow` get a spill variable with a small default `spill_cost` (1e-3 $/MWh) so the LP prefers keeping water when indifferent. The spill upper bound is derived from `inflow_max + energy_capacity` (clamped ≥ 1.0, ×10 slack) rather than an unconditional 1e12, avoiding HiGHS "excessively large column bounds" warnings.

### Decomposition / boundary fields (advanced)

The 18.P2 family of fields exists for rolling-horizon stitching and certified temporal decomposition (§7); all default off and are optimum-preserving when unset: `soc_initial_free` (+ `soc_initial_free_min/max` envelope bounds), `soc_terminal_min`/`soc_terminal_max`, `soc_start_cost`/`soc_terminal_cost` (±λ Lagrangian boundary prices), `ramp_t0_reference` (price the t=0 ramp against the previous window's net flow instead of zero), and the proximal V-terms `soc_start_v_cost` / `soc_terminal_v_rebate` / `net_terminal_v_rebate`.

---

## 4. Link Components & Flow Formulations

Links transport or convert energy between two buses (`model_type="transport"`, default) or carry network physics (`"dc_opf"` / `"ptdf"` / `"switched"`).

### Signed vs forward/reverse flow variables

`Link.effective_link_model()` resolves the LP variable shape per link:

- `'signed'` — PyPSA-style symmetric bidirectional transport: a single `f[t] ∈ [−cap, cap]` variable per timestep, no fwd/rev mutex row. Halves the per-line variable count for transmission lines.
- `'fwd_rev'` — one non-negative flow variable (plus a reverse variable and a `fwd + rev ≤ cap` mutex when `bidirectional`).
- `'dc_opf'` / `'ptdf'` — physics modes, always signed, handled by the network builders.

Eligibility for `'signed'` is conservative: any feature that applies a coefficient to the *absolute* flow magnitude (loss, `marginal_cost`, `ramp_cost`, ramp limits, multi-output `efficiency2/3`, CO2 output, UC/switching, linepack) keeps `fwd_rev` so the LP stays linear. **Cost-aware extendable rule:** extendable bidirectional links also stay `fwd_rev` — signed flow under a capacity *variable* needs a two-sided coupling (+1 row) where fwd+rev needs a single mutex, and it measures slower. Signed is reserved for fixed-capacity links where it is strictly −1 var / −1 row. The optimum is bit-identical under either shape.

`optimise(link_formulation="auto"|"fwd_rev"|"signed")` (default `"auto"`, behaviour-preserving) pins the shape for A/B measurement. Measured on a fixed-cap transport LP: −19 % vars/constraints, −17 % wall; extendable cases show no regression.

---

## 5. Solving: `optimise()` Options & LP Method Tuning

`EnergySystem.optimise()` exposes the full lossless solver-control surface:

| Kwarg | Default | Meaning |
|---|---|---|
| `lp_backend` | `None` | LP engine: `auto\|simplex\|ipm\|ipm_fast\|pdlp\|gpu`. `None` resolves: explicit arg > `.nexus_solver.json` sidecar (written by the tuner) > built-in default `"ipm_fast"`. Pure-LP only; MILP is never forced onto IPM. |
| `solver_method` | `None` | Raw HiGHS `solver` option; always wins over `lp_backend`. `"ipm"` auto-forces `run_crossover="on"` so the result stays a bit-exact vertex. |
| `run_crossover` | `None` | HiGHS crossover control `"on"/"off"/"choose"` (lossless — crossover recovers an exact vertex + duals). |
| `parallel` | `None` | HiGHS parallel `"on"/"off"/"choose"`; auto-set `"on"` when `threads > 1` (same algorithm, bit-exact). |
| `scale_cleanup` | `True` | Snap finite bounds/RHS with `0 < \|v\| < 1e-9` to exactly 0 (sub-tolerance ⇒ optimum unchanged). |
| `simplex_scale_strategy` | `None` | HiGHS matrix-equilibration strategy, int 0–5. |
| `eliminate_redundant` | `True` | Exact nexus-side pre-solve pass dropping provably non-binding rows (feasible region unchanged). |
| `link_formulation` | `"auto"` | See §4. |
| `ramp_cost_formulation` | `"split"` | `"signed"` (opt-in) replaces each ramp-cost up/down aux pair with ONE aux var `r ≥ ±Δ`; both forms price exactly `\|Δ\|` at optimum — identical optimum, one fewer column per component per step. |
| `mip_strategy` | `"auto"` | See below. |
| `uc_fix_schedule` | `None` | `{name: u-array}` pins `u[t]` equalities for committable **generators and links** (rolling-horizon / ML warm-start). |
| `warm_start` | `None` | Previous result (or raw value vector) forwarded to HiGHS `setSolution()`. |
| `basis` | `None` | Simplex basis hot-start. |

**`mip_strategy` values** (all accuracy-preserving):
- `"auto"` (default) — for MIPs with ≤ 5 000 estimated binaries, solve the LP relaxation first; if every binary lands within `1e-4` of {0, 1} the LP optimum *is* the MIP optimum and is returned directly (single LP, zero gap). Otherwise fall through to HiGHS MIP.
- `"mip_only"` — always go straight to the MIP solver.
- `"lp_first"` (opt-in) — force the LP-first path at any size, on a vertex-producing backend (an interior point would spuriously look fractional). Exact by the LP-relaxation theorem; the LP cost is paid twice when fractional, hence opt-in.
- `"fix_and_certify"` (opt-in) — `lp_first` plus: on a fractional vertex, fix every `u[t]` that *is* integral there, solve the residual MIP, and return it when its cost is within `gap` of the relaxation bound (a valid optimality certificate); falls back to the full MIP otherwise.

**`ipm_fast` caveat:** it returns an interior (non-vertex) point — degenerate duals/dispatch can differ from simplex, and basis warm-start is a no-op. Use `lp_backend="simplex"` when vertex duals or warm-start matter. A non-optimal `ipm_fast` result auto-falls-back to simplex.

### Auto-tuning the LP method (`solver_tuner`)

Static shape heuristics proved unreliable (an early ">=50K cols ⇒ ipm" rule regressed CINDER), so the tuner is **empirical**: it races `simplex` / `ipm` / `ipm_fast` on reduced-horizon proxies of your system, checks the winner is stable across two probe sizes (a clean simplex→IPM size-crossover counts as stable), checks objective parity within each size, and writes a `.nexus_solver.json` sidecar that `optimise()` reads automatically.

```python
from nexus_energy.solver_tuner import tune_solver, recommend_lp_method

res = tune_solver(system)        # races, prints report, writes sidecar
print(res.recommended)           # e.g. "ipm_fast"

# report-only / fine control:
res = recommend_lp_method(system, time_cap=120.0, threads=8,
                          need_duals=False, probe_hours=[365, 730])
```

Library-agnostic — works on any `from_pypsa` import. MILP systems short-circuit to `simplex` (IPM cannot solve branch-and-bound node LPs). `need_duals=True` disqualifies `ipm_fast`. The sidecar is advisory: an explicit `lp_backend=` always wins; delete the file to revert. Validated: PyPSA-Eur → `ipm_fast`, CINDER → `simplex`.

---

## 6. Advanced Temporal Engine

Running a full 8760-hour planning model is computationally heavy. The temporal engine uses advanced ML embeddings to cluster identical load/generation profiles into representative days.

```python
# Cluster a full year (8760 hours) into 10 representative days (240 hours total)
rep = ne.RepresentativePeriods(n_periods=10)

# Feature embedding: autoencoder clusters loading, VRE capacity factors,
# and proxy shadow prices before running k-medoids
aggregate_data = ne.aggregate_with_feature_embedding(
    system=sys,
    n_days=10,
    embedding_dim=4
)

# Solve system on aggregated timeline with inter-period storage linkages (LDS)
result = ne.apply_representative_days(sys, aggregate_data)
print(f"Reconstructed annual cost error is <= {result.error_bounds['TDR']}%")
```

**Adaptive / variable resolution.** `temporal.ResolutionPlan`, `adaptive_resolution_plan`, `multi_resolution_hierarchy`, and `apply_adaptive_resolution` build non-uniform timestep plans; `EnergySystem.set_snapshot_durations()` applies per-snapshot hour lengths directly.

**Certified reduction bounds.** `temporal.certify_reduction` returns a `CertifiedBound` — a provable two-sided bound on the cost error introduced by a temporal aggregation (see `certified_reduction_demo` for the worked example), turning "representative days look close" into a certificate.

---

## 7. Certified Temporal Decomposition

`temporal_certified.optimise_temporal_certified` solves a long-horizon system as K temporal blocks **with a global optimality certificate** — the same `(UB − LB)/UB ≤ gap` contract a monolithic MIP solver gives, at an order-of-magnitude lower wall time (CINDER-class MILP wall scales ~T^2.5).

```python
from nexus_energy.temporal_certified import optimise_temporal_certified

res = optimise_temporal_certified(
    system_factory,            # callable(t0, t1) -> EnergySystem over [t0, t1)
    total_steps=8760,
    n_blocks=8,
    gap=1e-2,
    lb_blocks=4,               # fewer, longer LB blocks => tighter bound
    lb_rounds=3,               # subgradient ascent rounds on boundary prices λ
    lb_workers=4,              # process-parallel LB blocks (factory must pickle)
    ub_boundary="both",        # "prices" | "floors" | "both" | "none"
    boundary_soc_guide=None,   # optional guide SOC trajectory => pin-dual λ harvest
    reachability_envelopes=True,
)
print(res)   # TemporalCertifiedResult(status='certified', objective=..., gap=...)
```

- **LB** — Lagrangian dual decomposition over interior SOC boundaries: each block is a *relaxation* (free interior start SOC via `Storage.soc_initial_free`, telescoping ±λ boundary prices via `soc_start_cost`/`soc_terminal_cost`), so Σ block dual bounds ≤ full optimum for arbitrary λ; `lb_rounds` refines λ by projected subgradient and the best round is reported.
- **UB** — sequential block stitching handing each block its predecessor's terminal SOC; `ub_boundary="prices"` prices terminal energy at −λ (payment subtracted back out, so the UB is the stitched trajectory's TRUE full-model cost), `"floors"` uses hard `soc_terminal_min` floors with unfloored retry. Boundary corrections make ramp and startup/shutdown costs across block edges exact.
- **Reachability envelopes** cap each LB block's free start/terminal SOC by what the full problem could physically have stored by that boundary (bound-tightening, never validity-affecting).

Result fields include `objective` (UB), `lower_bound`, `gap`, per-block objectives/bounds, `lambda_final`, and stitched full-horizon trajectories. Supported scope (guarded with `ValueError`): non-cyclic, non-LDS, non-extendable storages; committable links with min up/down ≤ 1; no hard ramp limits; no committable generators; uniform snapshot weights/durations. Measured on CINDER MILP: +0.91 % from parity at 58 s (6.1× faster than monolithic) / beats the incumbent at 87 s.

**Supporting plumbing on the core model:** `OptimisationResult.storage_soc_duals` (duals of the SOC-recursion equalities — the marginal value of stored energy, $/MWh; full-mode storages, LP solves) and `OptimisationResult.soc_fixed_duals` (duals of `soc_fixed` pin rows keyed `(name, t)` — the marginal cost of delivering a pinned SOC level). `Link.ramp_t0_reference` / `Storage.ramp_t0_reference` price the t=0 ramp against the previous window's flow in rolling/stitched solves.

---

## 8. Decomposition & Scaling

### Temporal & Spatial Benders Decomposition
For massive networks that exceed single-machine memory, the Benders Decomposer decouples the system. It compiles master models (investments) and solves per-period dispatch sub-problems in parallel. 
Unlike competitive frameworks, Nexus-Energy incorporates **stabilized, adaptive-oracle Benders** with cross-scenario cut reuse to prevent sub-gradient oscillation and reduce iterations by up to 60%.

```python
# Initialize parallel decomposition
decomposer = ne.BendersDecomposer(
    system=sys,
    stabilization="level-bundle",   # Regularization to prevent oscillation
    reuse_cuts=True                 # Accelerate multi-scenario runs
)

# Solve Benders master and sub-problems in parallel
result = decomposer.solve()
```

**Further decomposition drivers** (top-level exports): `solve_with_nested_benders`, `solve_with_dantzig_wolfe`, `solve_with_column_generation` (+ `SpatialBendersResult`), and Farkas feasibility cuts on `BendersDecomposer`. `rolling_horizon_solve` threads the LP simplex basis between windows for hot-started resolves. Note: a `StageProblem` class exists in both `decomposition` and `stochastic`; the top-level aliases are `NestedStageProblem` / `SDDiPStageProblem`.

---

## 9. Stochastic & Robust Optimization

Nexus-Energy provides a rigorous framework for modeling uncertainty in renewable generation and demand.

### Two-Stage Stochastic Programming
Optimize investments to hedge against an array of demand scenarios.
```python
# Define scenario paths
scenarios = [
    ne.Scenario(name="low_demand", demand_multiplier=0.8, probability=0.25),
    ne.Scenario(name="base_demand", demand_multiplier=1.0, probability=0.50),
    ne.Scenario(name="high_demand", demand_multiplier=1.3, probability=0.25),
]

# Solves extensive form in parallel
stochastic_result = ne.solve_stochastic(sys, scenarios)
```

### Robust Optimization (Bertsimas-Sim Budget Sets)
Protect the system against worst-case realization of solar/wind capacity factors within a specified uncertainty budget ($\Gamma$).

```python
# Protect system against worst-case solar dropouts
uncertainty_set = ne.BudgetUncertaintySet(
    target_technology="solar",
    deviation=0.30,                 # Maximum deviation (30% drop)
    gamma=2.0                       # Uncertainty budget
)

robust_result = ne.solve_robust(sys, uncertainty_set)
```

### Chance-Constrained Optimization
Guarantee that transmission lines do not overload with a specified probability threshold (e.g., $95\%$).

```python
# Constraint: Probability(Line Flow <= Limit) >= 0.95
cc = ne.ChanceConstraint(
    target_component="line_1",
    reliability_level=0.95
)

result = ne.solve_saa_chance_constrained(sys, constraints=[cc])
```

**Extended stochastic toolkit** (top-level exports): `solve_sddip` (multi-stage SDDiP), `solve_general_chance_constrained`, `solve_wasserstein_dro` (distributionally-robust over Wasserstein balls), `solve_risk_averse_benders`, `generate_forced_outage_scenarios`, and `reduce_scenarios_wasserstein` (optimal-transport scenario reduction).

---

## 10. AC-OPF Relaxations & Physics

For transmission modeling, linear transport models (copper plate) are insufficient. Nexus-Energy houses full AC physics.

### Conic SOCP AC-OPF with OBBT Tightening
The Second-Order Cone Programming (SOCP) Jabr relaxation provides a convex, globally optimal approximation of power flow. 
To tighten this relaxation on difficult networks, Nexus-Energy uses **Optimization-Based Bound Tightening (OBBT)** to shrink bounds on voltage magnitudes and phase angles, combined with **asymmetric sin boxes** for unmatched angular tightness.

```python
# Solve SOCP AC power flow with automated OBBT iterative interval tightening
socp_result = ne.solve_socp_opf(
    system=sys,
    obbt_iterations=5,              # Number of iterative shaves
    asymmetric_bounds=True          # Highly tight angular sin boxes
)
print(f"SOCP Duality Gap: {socp_result.duality_gap * 100}%")
```

**Conic extensions:** `solve_socp_opf_expansion` / `SOCPExpansionResult` (capacity-expansion AC-OPF on the Jabr relaxation), `add_weymouth_pipe` (gas-network Weymouth SOC relaxation), and `add_head_dependent_hydro` (head-dependent hydro production cones).

---

## 11. ML-Guided Solving & Warmstarts

To accelerate rolling-horizon MPC (Model Predictive Control) and real-time operations, Nexus-Energy incorporates neural classifiers that bypass MILP branches.

```python
# Warmstart UC schedules using a Graph Neural Network (GNN)
predictor = ne.UCWarmstartPredictor(
    method="gnn",                  # Uses torch-optional Graph Neural Network
    historical_database="runs/"
)

# Extract topological and loading features
sys_feat = ne.extract_system_features(sys)
t_feat = ne.extract_timestep_features(sys)

# Predict the optimal on/off binary schedules for thermal units
predicted_schedule = predictor.predict(sys_feat, t_feat)

# Solve MILP with warmstarted schedules
result = ne.warm_start_from_prediction(sys, predicted_schedule)
```

**Additions:** `AdaptiveThresholdController` / `solve_with_adaptive_warmstart` (self-tuning confidence threshold for prediction-fixing), `feature_embedding_periods` (embedding-space representative-period selection), and `RLVarFixer` / `solve_with_rl_search` (reinforcement-style variable-fixing search).

---

## 12. Differentiable Dispatch & Inverse Calibration

The dispatch problem is formulated as a differentiable layer: analytic `d dispatch / d parameter` Jacobians via implicit differentiation of the KKT system at the (ridge-regularised) optimum. Honest scope: **parameter learning on the inner LP/QP** — every layer uses a mandatory strict-convexity ridge that shifts dispatch vs. the true LP by O(ridge); disclose it in results. Pure numpy; torch is optional.

### Layer inventory

| Layer | Scope | Differentiable inputs |
|---|---|---|
| `solve_dispatch_with_sensitivities` | single-bus, single-period, closed form | mc, capacity, demand |
| `EconomicDispatchLayer` | stateful OO wrapper of the above | same |
| `MultiBusDispatchProblem` + `solve_multibus_dispatch_with_sensitivities` | multi-bus, multi-period transport QP (per-period solves) | mc, capacity, demand, line_limit |
| `StorageDispatchProblem` + `solve_storage_dispatch_with_sensitivities` | single-bus multi-period + one storage (stacked QP) | mc, capacity, demand, soc_init, **η_c, η_d** |
| `MultiBusStorageProblem` + `solve_multibus_storage_dispatch_with_sensitivities` | multi-bus + bus-attached storages, ONE stacked QP | mc, η_c, η_d, soc_init |
| `SmoothCommitmentLayer` / `smooth_commitment` / `fit_commitment_threshold` | sigmoid-smoothed commitment | threshold |
| `CapacityExpansionProblem` + `CapacityExpansionLayer` / `solve_capacity_expansion_with_sensitivities` / `fit_component_params` | differentiable capacity expansion (design gradients) | capex, mc, demand, … |
| `TorchDispatchLayer` | cvxpylayers hook for problems beyond the numpy paths | — |

### Multi-bus dispatch: availability, line bounds, selective Jacobians

`MultiBusDispatchProblem` carries two optional fields beyond the toy core:
- `availability` — `(G, T)` in [0, 1]: per-period derating `p[g,t] ≤ capacity[g]·availability[g,t]` (VRE capacity factors / outages). `d_dispatch_d_capacity` stays w.r.t. nameplate capacity (the factor is chained in).
- `line_min` — `(L,)` lower flow bounds; `None` keeps symmetric `−line_limit`, `0.0` entries make links unidirectional. `d_dispatch_d_linelimit` differentiates only the upper bound.

```python
sol = solve_multibus_dispatch_with_sensitivities(
    problem,
    jacobians=("mc",),    # selective blocks; () = forward-only solve
)
# sol.dispatch (G,T), sol.flows (L,T)
# sol.d_dispatch_d_mc (G·T, G), d_dispatch_d_capacity (G·T, G),
# d_dispatch_d_demand (G·T, B·T), d_dispatch_d_linelimit (G·T, L)
```

Skipped blocks come back as zeros — pass `()` for the price a sampling baseline pays per draw, or `("mc",)` when only cost-side gradients are chained (e.g. CO₂-price calibration).

### Storage efficiency Jacobians (η calibration)

`StorageDispatchSolution` exposes, beyond the bound/RHS blocks, the **constraint-matrix sensitivities** `d_dispatch_d_charge_eff` / `d_dispatch_d_discharge_eff` (shape `(G·T,)`), which require the equality duals recovered from the frozen-active-set KKT — these are what an auto-calibration loop fits from battery telemetry. Dispatch-only telemetry identifies only the η_c·η_d *product*; the SOC-trace rows `d_soc_d_charge_eff` / `d_soc_d_discharge_eff` (shape `(T,)`) split charge from discharge efficiency when observed SOC (BMS logs) is added to the residual.

### Multi-bus + storage (stacked, windowed)

`MultiBusStorageProblem` extends the multibus problem with `S` bus-attached storages (`sto_bus`, `charge_eff`, `discharge_eff`, `power_limit`, `soc_max`, `soc_init`). SOC couples the periods, so it is solved as **one stacked QP over all periods — keep the window small (representative days/weeks, T ≤ ~48)**: the dense stacked KKT is O(((G+L+3S)·T)³).

```python
from nexus_energy.diff import (MultiBusStorageProblem,
                               solve_multibus_storage_dispatch_with_sensitivities)
sol = solve_multibus_storage_dispatch_with_sensitivities(
    prob, jacobians=("mc", "eta", "soc_init"))
# sol.dispatch/flows/charge/discharge/soc;
# d_dispatch_d_mc (G·T,G), d_dispatch_d_charge_eff / _discharge_eff (G·T,S),
# d_dispatch_d_soc_init (G·T,S), d_soc_d_charge_eff / _discharge_eff (S·T,S)
```

### Calibrating against PyPSA networks (`diff_bridge`)

`nexus_energy.diff_bridge` composes the `from_pypsa` import with the multibus layer, so analytic Jacobians become available for *real* networks (PyPSA-Eur slices included):

```python
from nexus_energy.diff_bridge import (multibus_problem_from_system,
                                      d_dispatch_d_co2_price, fit_co2_price)

bridge = multibus_problem_from_system(system, co2_price=80.0, ridge=1e-2)
# bridge.problem is ready for solve_multibus_dispatch_with_sensitivities;
# bridge.emission, bridge.mc_base satisfy mc = mc_base + price·emission.

fit = fit_co2_price(system, observed_dispatch, price_bounds=(0.0, 1000.0))
print(fit.price, fit.converged, fit.n_solves)
```

Honest scope — the bridge **fails loudly, never silently approximates**: dispatch-only (every capacity fixed — extendable components raise), transport flow model only (DC-OPF links raise; re-import with `from_pypsa(n, line_model="transport")`), no storage, lossless/cost-free/single-output links, no UC / p_min / must_run / PWL heat rates. The CO₂ price enters via `mc_eff = mc + price·emission`, so `d dispatch/d price = d_dispatch_d_mc @ emission` (chain rule — `d_dispatch_d_co2_price`). `fit_co2_price` is two-stage: a coarse forward-only **bracket** (dispatch is piecewise-linear in price, so the loss has flat pieces a pure gradient method is born stuck on) followed by **safeguarded Gauss-Newton** with bisection fallback; returns a `CO2FitResult(price, history, n_solves, converged)`.

### Demand elasticity & component fitting

`fit_demand_elasticity` (Phase 12.3) recovers a demand-elasticity parameter from observed dispatch; `fit_component_params` (Phase 20) fits component design parameters through the capacity-expansion layer, returning a `ComponentFitResult`.

### Solver notes (internals)

The per-period QPs are solved by a **dual semismooth Newton** method (`_solve_period_qp_dual`): for the diagonal Hessian the primal is the closed-form clip `x(λ) = clip((Cᵀλ − q)/h, lb, ub)`, the dual residual is piecewise-affine in λ, and the Newton step uses an **exact kink line search** — `‖r(t)‖` is evaluated at every bound-crossing breakpoint plus each segment's closed-form interior minimiser, because backtracking can miss arbitrarily narrow acceptance windows between flat plateaus. λ is warm-started across periods, with a cold retry on stall and a `strict` residual guard that raises rather than return an infeasible point. The last-resort fallback for degenerate kink geometry is a POCS feasible point (`_alternating_projection_feasible`) followed by a textbook ratio-test primal active-set QP (`_solve_box_eq_qp_primal`) that never leaves the feasible set. These are private internals (`_`-prefixed) and may change.

---

## 13. MPC & Auto-Calibration

### Persistent warm-started resolves (`mpc.PersistentDispatchSession`)

Rolling-horizon LP resolves **without rebuilding the model**: `build()` solves once through the normal `optimise()` path while capturing the assembled nexus-opt Model, then loads it into a persistent HiGHS instance; `advance()` pushes parameter changes straight into HiGHS columns/rows and re-solves from the retained simplex basis — a window resolve costs a handful of iterations instead of a rebuild + cold solve.

```python
from nexus_energy.mpc import PersistentDispatchSession

sess = PersistentDispatchSession(system, threads=4)
base = sess.build()                       # normal optimise() + capture
res = sess.advance(
    demand={"elec": new_total_demand},    # bus -> (T,) TOTAL demand
    cf={"wind": new_cf},                  # gen -> (T,) availability in [0,1]
    mc={"gas": 62.0},                     # gen -> scalar or (T,) marginal cost
    soc_init={"battery": 120.0},          # storage -> start SOC in MWh
)
print(sess.n_resolves, sess.n_rebuilds, sess.last_iterations)
```

Honest scope (`build()` raises otherwise): **pure LP dispatch** — no committable / integer-investment / switchable / PWL-capex components, fixed capacities (extendable generators put availability into a `p ≤ cap_var·cf` matrix row this path cannot touch), and storage start-SOC carry-over requires `soc_initial_free=True` (the start SOC is then a pinnable column). Anything outside scope at `advance()` time falls back to a full rebuild — correct, just slower — and is counted in `n_rebuilds`. Plumbing: `OptimisationResult._balance_row_idx` records the (bus, t) → balance-row mapping at build time; the session drives `nexus_opt.PersistentHighs` (see the nexus-opt wiki).

### Auto-calibration (`autocal`)

`autocal.fit_params` generalises the scalar CO₂-price recovery to an m-dimensional parameter vector with a damped (scaled Levenberg-Marquardt) Gauss-Newton driver; `AutoCalibrator` wraps it into a moving-horizon tracker so a running MPC keeps its own model honest.

```python
from nexus_energy.autocal import fit_params, AutoCalibrator

report = fit_params(
    make_solution,                  # θ-dict -> SOLVED diff-layer solution
    observed,                       # telemetry, flattened vs the residual
    params={"eta_c": (0.95, 0.5, 1.0), "mc_gas": (50.0, 10.0, 200.0)},
    jacobian_fn=jac,                # (solution, θ) -> (len(r), m) analytic columns
    gate_threshold=1e-3,            # identifiability gate (relative column energy)
    max_rel_step=0.25,              # per-run trust region
)
print(report.changed(), report.frozen)   # frozen[name]=True => data-silent

cal = AutoCalibrator(solution_factory, jac, observed_fn, params,
                     update_every=4, noise_std=0.5, smooth=0.3)
rep = cal.step(window)              # None on skipped cycles
cal.disable(); cal.enable()         # master toggle
cal.lock_param("eta_c")             # operator override (unlock_param to release)
cal.set_param("mc_gas", 55.0, lock=True)
print(cal.believed)                 # current point beliefs
```

Two design rules make this trustworthy in operations: the **identifiability gate** freezes any parameter whose (scale-aware) Jacobian column carries no signal in the window — flagged `data_silent`, never nudged by noise — and the **per-cycle slew limit** (`max_rel_step`) bounds how far beliefs move per cycle, so one bad window cannot wreck the model. Noise-aware MHE controls: `noise_std` enables EMA smoothing (`smooth` gain) across windows and outright **rejection of outlier windows** whose post-fit loss exceeds `outlier_zscore²·(½nσ²)` (sensor fault / unmodeled event — beliefs stay put). `residual_fn` switches the fitted observable (dispatch shares, SOC traces, …). Any parameter chain-rulable from the exposed diff-layer blocks works: marginal costs, fuel price/efficiency via `mc = fuel/η`, CO₂ price via emissions, storage η via the efficiency blocks.

---

## 14. PyPSA Import Conventions (`from_pypsa`)

Two parity-critical conventions in the importer:

- **Static (scalar) `p_max_pu` is honoured.** When a generator has no `p_max_pu` time series, a constant capacity-factor array is synthesized from the static column (e.g. PyPSA-Eur nuclear `p_max_pu = 0.781`) so `p[t] ≤ capacity·p_max_pu` is enforced exactly as in PyPSA. (Previously constant de-ratings were dropped → over-dispatched baseload and under-priced systems.) Exports round-trip the factor back into `generators_t.p_max_pu`.
- **Cyclic storage level is free.** StorageUnits and Stores import with `cyclic_level="free"` — only continuity `soc(0) = soc(T)` is enforced and the optimiser picks the level, matching PyPSA `cyclic_state_of_charge` / `e_cyclic`. The historical fixed-level pin over-constrained (and for extendable storage, whose energy capacity starts at 0, forced start/end empty).

For the differentiable bridge (§12), import with `from_pypsa(n, line_model="transport")`.

---

## 15. Policy, Planning & I/O Additions

### Pooled multi-zone CO2 cap-group

```python
sys.set_co2_cap_group(
    [bus_a, bus_b, bus_c], limit=0.05,   # tCO2/MWh (is_rate=True default)
    is_rate=True,
    storage_losses_on_rhs=True,          # GenX CO2Cap=2 RHS term
    loss_accounting="net",               # "net" (legacy, bit-stable) |
)                                        # "dissipation" ((1/ηc−1)·ch + (1−ηd)·dis)
```

One constraint over a *set* of buses (GenX `Cap_Zone`) is tighter than independent per-bus caps — it forbids inter-zone emission averaging.

### Other API surface (verified, brief)

- **Reliability / operations:** `set_contingency_reserve()` (single-largest-unit N-1 reserve), `set_outage()`, `set_shared_capacity()` (shared converter ratings with optional mutex), `set_rps()` / `set_ces()` gain `slack_penalty` (soft-constraint pricing).
- **Components:** `Generator.start_up_segments` (§2), `Storage.retrofit_of` / `Link.retrofit_of` (retrofit/repower bounded by retiring host capacity), `Link.efficiency_segments` (concave part-load conversion curves).
- **Sector coupling:** `sectors.create_temperature_heat_network`.
- **Composability:** `components.composition.{Subsystem, CarrierMismatchError, FIDELITY_LEVELS}` — typed subsystem composition with carrier checking.
- **External solvers:** `external_solvers` module — LP-export bridge towards Gurobi/CPLEX/SCIP/Mosek/Xpress for comparison runs; `optimise()` itself rejects external solver names with a pointer (no third-party wrappers inside nexus).
- **Tabular I/O:** `io_tables` — DuckDB reader with pandas fallback.

---

## 16. Turnkey Speed Differentiators

What makes Nexus-Energy solve up to **30× faster** than competing libraries like PyPSA or GenX?

1. **GPU LP Dual Recovery**: Massively large linear models are solved using **cuPDLP-C** / **cuOpt** on NVIDIA GPUs, with a native dual-recovery IPM-polish pass to reconstruct exact shadow prices for market dispatch.
2. **Tighter-by-Default UC Formulations**: Unit commitment binary structures utilize advanced perspective cuts and 3-bin convex hulls natively, reducing branch-and-bound solving times by up to 10× without any user-visible API changes.
3. **Rust Constraint Assembly**: Linopy (vectorized Python) is the modeling speed limit in standard libraries. Nexus-Energy bypasses Python entirely by streaming CSC sparse constraints directly to the solver using its native PyO3 Rust assembly kernel, providing a **3-5× build speedup** on large models.
4. **ML Feature-Guided Clustering**: Rather than raw time-series clustering, Nexus-Energy embeds load, renewable capacity factors, and proxy prices in an autoencoder space before k-medoids, keeping representative day counts low while preserving the extreme load periods required to ensure system adequacy.
5. **Empirical LP-Method Selection**: the `solver_tuner` races simplex/IPM variants on reduced-horizon proxies of *your* problem and locks the winner in a sidecar — `ipm_fast` (IPM without crossover) is the lean default for well-conditioned expansion LPs (6× over PyPSA on Eur 2190h), while staircase MILPs keep warm-startable simplex.

---

## 17. Optimization Decision Matrix

Use the matrix below to select the optimal configuration for your energy systems models:

| Problem Domain | Recommended Choice | Rationale | Alternatives |
|---|---|---|---|
| **Real-time Operations / MPC** | `PersistentDispatchSession` + `ML Warmstart UC` | Hot-basis resolves cost a handful of simplex iterations; GNN/k-NN predictions bypass the MILP cold start. | Full rebuild + cold solve every window (too slow for RT). |
| **Transmission Planning** | `SOCP OPF w/ OBBT` | Conic relaxation is convex and globally optimal, providing a robust planning envelope. | Polar AC-OPF (non-convex; risk of local traps). |
| **Operations Analysis** | `Polar AC-OPF` | True non-linear, non-convex grid equations reflect real-world reactive power. | SOCP (relaxed bounds may violate strict voltage limits). |
| **Multi-Decade Grid Planning** | `Benders Decomposition` | Decouples regional/annual dimensions, keeping RAM requirements within bounds. | Extensive formulation (will crash on memory limits). |
| **Long-Horizon MILP Dispatch** | `optimise_temporal_certified` | K short blocks + a valid global LB certificate beat the ~T^2.5 monolithic wall. | Monolithic MIP (exact but order-of-magnitude slower). |
| **High Renewable Scenarios** | `ML-Embedded Aggregation` | Autoencoder clustering captures the joint correlation of wind/solar dropouts. | Raw k-medoids (ignores price/adequacy peaks). |
| **Model-Reality Drift** | `AutoCalibrator` | Identifiability-gated, slew-limited Gauss-Newton keeps believed parameters honest from telemetry. | Manual recalibration campaigns. |
