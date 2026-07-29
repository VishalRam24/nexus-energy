---
title: Parity ledger — nexus-energy vs peer libraries
project: nexus-energy
type: benchmark
status: active
last_verified: 2026-04-26
tags:
  - parity
  - ledger
  - numbers
---

# Parity ledger — nexus-energy vs peer libraries

Locked-format table. One row per (library, case). Columns are fixed
and in this order so a diff between phases is trivially readable:

```
library | case | phase | obj_nexus | obj_reference | Δobj | wall_nx | wall_ref | speedup | notes
```

- `obj_nexus`, `obj_reference`: LP / MILP objective reported by each
  solver stack (same $/units).
- `Δobj = (obj_nexus − obj_reference) / |obj_reference|` in %. Sign is
  **nx − ref**, so negative means nexus is cheaper.
- `wall_nx`, `wall_ref`: end-to-end wall-clock (build + solve) for each
  side of the paired run. Not comparable across rows — case sizes vary
  by three orders of magnitude.
- `speedup = wall_ref / wall_nx`.
- `notes`: "pass" / "informational" / open deferral with target phase.

Source scripts all land under
`test_projects/test_project_1/` and `nexus-energy/benchmarks/`;
every row must be reproducible from a named script listed in the
notes column.

---

## PyPSA

| case | phase | obj_nexus | obj_reference | Δobj | wall_nx | wall_ref | speedup | notes |
|---|---|---|---|---|---|---|---|---|
| ac_dc_meshed | 0 / 14 | 9.0008e6 | 9.0008e6 | −0.000 % | 0.004 s | 0.283 s | 80.1× | pass — `pypsa/native/run_ac_dc_meshed_native.py` |
| storage_hvdc | 0 / 14 | (see port) | (see port) | −3.640 % | 0.008 s | 0.536 s | 70.2× | informational — SOC boundary pin open; `pypsa/native/run_storage_hvdc_native.py` |
| model_energy | 14 | (see port) | (see port) | −0.000 % | 5.98 s | 5.01 s | 0.84× | pass (obj); wall-clock parity deferred to N_En_Phase 14.x (build-side slowdown) |
| cinder_multicarrier (LP) | ramp_cost | 669.53 | −1.159e8 * | N/A ** | 161.1 s | 491.3 s | 3.05× | pass — LP relaxed UC. Dispatch parity: CHP gas 0.0 %, gasifier 0.0 %, total HP 1.9 %, grid 0.2 %. 8-bus 10-link 7-store 90d@15min. Re-measured 2026-05-24 after UC tightening (no LP regression). \* PyPSA obj includes loads as neg-cost generators. \*\* Obj not comparable. `cinder/run_compare.py` |
| cinder_multicarrier (MILP) | ramp_cost | 685.01 | −1.159e8 * | N/A ** | 639.1 s | 190.4 s | 0.30× | 🟡 MILP binary UC. Dispatch parity <2.5 % on major links. Nexus 2.26 % gap (deterministic after 2026-05-24 LP-tight UC fix: binary v/w + inequality state-transition + 4 upper-bound cuts). PyPSA LP-tight (0 % gap at root, 1 node). 2026-05-24 storage refactor (1-var Store mode active for 5 of 7 CINDER storages — battery, biomass, ice, hydrogen, thermal_storage) saved 86 K cols and 63 s; sand_storage falls back to full mode for ramp_cost, groundwater for inflow. Remaining MILP gap traced to HiGHS presolve not probing nexus's 155 K binaries down to constants (PyPSA presolve fixes 86 K). Adding redundant binary upper bounds + p ≥ 0 link lower bounds was tested 2026-05-24 — presolve absorbed all 287 K extra rows without changing post-presolve binaries; reverted. \* \*\* Same as above. `cinder/run_compare.py --milp` |

## GenX

| case | phase | obj_nexus | obj_reference | Δobj | wall_nx | wall_ref | speedup | notes |
|---|---|---|---|---|---|---|---|---|
| three_zones_LP (8760 h)      | 5 / 14  | 4.68027e9  | 4.68027e9   | −0.000 %  | cached   | cached   | —     | pass — `julia/nexus_mirror/port_three_zones.py` |
| three_zones_default (TDR+UC) | 14      | 1.01136e10 | 1.02283e10  | −1.12 %  | 35.8 s   | 34.6 s   | 0.97× | informational — bilinear UC+extendable deferred to N_En_Phase 14.x; `port_three_zones_default.py` |
| three_zones_mincapreq        | 15      | 5.862e9    | 5.808e9     | +0.92 %  | 4.74 s   | 11.09 s  | 2.3×  | **pass** ≤ 5 % — `port_three_zones_mincapreq.py` |
| three_zones_ucommit2         | 15      | 4.9369e9   | 4.607e9     | +7.17 %  | 6.71 s   | 32.07 s  | 5.4×  | informational — **re-measured 2026-04-20 post-N_En_Phase 14.1** (linearised `unit_size·u[t] ≤ cap_var` engages in `core.py:1424-1449`, confirmed by debug); number unchanged → residual gap is a modelling difference with GenX's clustered UC formulation, not a 14.1-addressable bilinearity. Candidate follow-ups: (a) audit GenX's UCommit=2 start-up / min-gen accounting vs `build_three_bin_uc`, (b) check ramping-on-extendable treatment. `port_three_zones_ucommit2.py` |
| three_zones_rate_co2         | 15      | 1.0000e10  | 1.73658e10  | −42.42 % | 103.65 s | 10.33 s  | 0.10× | **open** — CO2 caps bind exactly per zone; slosses-RHS ruled out (re-solved GenX with StorageLosses=0 → only $1 B shift). LP compositions diverge (nexus CT_wind-heavy vs GenX CT_solar+battery-heavy). See [[DEFERRALS|DEFERRALS.md]] N_En_Phase 15 — target N_En_Phase 15.x |

## Tulipa (N_En_Phase 16 — TulipaEnergyModel.jl v0.21.0, HiGHS)

| case | phase | obj_nexus | obj_reference | Δobj | wall_nx | wall_ref | speedup | notes |
|---|---|---|---|---|---|---|---|---|
| Tiny (LP, integer invest)                  | 16 | 2.69238e5 | 2.69238e5 | −0.000 % | 0.045 s | 6.78 s | 241× | **pass** ≤ 0.5 % — `julia/nexus_mirror_tulipa/port_tulipa_tiny.py` |
| Storage (LP, fractional rp-mapping)        | 16 | 2.53888e3 | 2.54223e3 | −0.132 % | 0.019 s | 6.80 s | 358× | **pass** ≤ 5 % — fractional stochastic-rp LDS collapsed to per-rep cyclic under argmax chrono mapping (see [[DEFERRALS|DEFERRALS.md]] N_En_Phase 16.x); `port_tulipa_storage.py` |
| UC-ramping (MILP, UC + integer invest)     | 16.x | 2.97027e5 | 2.93075e5 | +1.348 % | 0.095 s | 8.43 s | 89× | **pass** ≤ 5 % — N_En_Phase 14.x linearised clustered-UC × cap_var (per-unit `p ≤ unit_size × u[t]` + `unit_size × u[t] ≤ cap_var`); N_En_Phase 16.x added `Generator.no_load_cost` for Tulipa's `units_on_cost`; `port_tulipa_uc_ramping.py` |

## Internal: ML warm-start speedup (N_En_Phase 11 / 11.x)

Not a peer head-to-head — this row measures cold-start vs warm-start
re-solve on the same nexus problem to validate the N_En_Phase 11 acceptance
bar from `ROADMAP.md`: "≥ 2× faster on the re-solve vs a cold start".
Single-bench reproducer: `benchmarks/phase11_rolling_uc.py` (3 zones,
48 h horizon, 52 weekly windows, 4-window burn-in; 39 binary
committable thermals per window). N_En_Phase 11.x added a global
`max_fix_fraction=0.75` cap on pinned cells plus a retry-on-infeasible
driver (`solve_with_warm_retry`) that halves the cap and retries.

| predictor | phase | wall_cold (med) | wall_warm (med) | speedup | Δobj_max | infeasible | notes |
|---|---|---|---|---|---|---|---|
| HistoricalNeighborPredictor (k-NN) | 11.x | 1.366 s | 0.242 s | **5.66×** | +0.91 % | 0 / 48 | **pass** ≥ 2× — threshold 0.8, k_sys=8, k_step=5, max_fix_fraction=0.75 (1404 / 1872 pinned). N_En_Phase 11.x landing: pre-cap bench had 12.0× headline but 17.9 % drift + 6/48 infeasible |
| MeritOrderPredictor (rule-based)   | 11.x | 1.366 s | 0.329 s | 4.15× | +0.60 % | 0 / 52 | **pass** ≥ 2× — threshold 0.99, max_fix_fraction=0.75. No training required; low-risk default |

## Calliope / oemof / SpineOpt / Sienna / PowerModels

| library | case | phase | obj_nexus | obj_reference | Δobj | wall_nx | wall_ref | speedup | notes |
|---|---|---|---|---|---|---|---|---|---|
| Calliope    | — | — | — | — | — | — | — | — | not yet ported |
| oemof       | — | — | — | — | — | — | — | — | not yet ported |
| SpineOpt    | — | — | — | — | — | — | — | — | not yet ported |
| Sienna      | — | — | — | — | — | — | — | — | not yet ported |
| PowerModels | — | — | — | — | — | — | — | — | not yet ported |

---

## Regression budget

≤ 10 % on any row already listed as "pass". Any row flipping from
pass → informational → open must be logged in `DEFERRALS.md` under
the phase that caused the regression.

## Reproducing a row

1. Ensure the GenX / PyPSA reference has been solved once and its
   result cached (GenX: `julia/nexus_mirror/out/genx_refs/*.json`;
   PyPSA: the ports re-solve in-process each time).
2. Run the paired nexus port (script named in the row's `notes`).
   Each port writes its own JSON including `genx_reference_obj`,
   `genx_reference_wall_seconds`, `rel_obj_delta_vs_genx`, and
   `speedup_vs_genx` (or the PyPSA analogues).
3. Or run the whole standing set via
   `test_projects/test_project_1/test_native_parity.py`.
