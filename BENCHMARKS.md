---
title: nexus-energy — standing benchmark numbers
project: nexus-energy
type: benchmark
status: active
last_verified: 2026-04-26
tags:
  - benchmarks
  - performance
---

# nexus-energy — standing benchmark numbers

Consolidated index for the running benchmark suite. Each row points at
the script + the persisted JSON under `benchmarks/results/`. Re-run at
every phase boundary; budget = 10 % regression on wall-clock.

Numbers below were captured at the **Phase 13** boundary
(2026-04-19, library v0.2.0, Darwin arm64 / Python 3.12.13, HiGHS via
nexus-opt). All wall-clock figures are single-run; treat ≤ 2× as noise
on a laptop.

## Flagship — every phase, one script

`benchmarks/flagship_all_features.py` →
`benchmarks/results/flagship_all_features.json`

Single integration bench that exercises every phase that landed in the
library (status must be `optimal` or `max_iter` for the run to pass):

| # | Row                  | Phase  | Wall    | Cost          | Notes |
|---|----------------------|--------|---------|---------------|-------|
| 1 | `baseline_lp`        | P0     |   5.4ms | 201 913.02    | T=168 LP dispatch |
| 2 | `dc_opf_loop`        | P3     |   0.3ms |   7 400.00    | KVL on 3-bus loop, bottleneck `l13@40` |
| 3 | `unit_commitment`    | P2     |  37.5ms | 892 456.42    | 3 binaries, T=168 |
| 4 | `storage_repdays`    | P5/P7  |   2.0ms | 824 589.18    | 3 representative days, 40 MW / 160 MWh storage |
| 5 | `benders`            | P8     |   4.1ms | 504 000.00    | 6 scenarios, converged in 2 iters |
| 6 | `stochastic_cvar`    | P9     |   2.7ms | 888 000.00    | CVaR α=0.1, 4 scenarios |
| 7 | `ml_warmstart`       | P11    |  21.5ms | 160 881.23    | cold=17 ms, warm=5 ms → **3.58× speed-up**, Δcost = 0 |
| 8 | `rolling_horizon`    | P10    |   0.7ms | 140 360.25    | T=48, 12-step windows, HiGHS, warm-start on |
| 9 | `differentiable`     | P12    |   0.2ms |   3 700.00    | analytic vs FD grad error ≤ 2.21e-11 |

All 9 rows pass. Run yourself:

```sh
.venv/bin/python benchmarks/flagship_all_features.py
```

## Decomposition — adaptive-oracle Benders vs plain / trust-region / PH

`benchmarks/decomposition/adaptive_oracle_bench.py --json benchmarks/results/phase13_adaptive_oracle.json`

20 moment-matched scenarios, T=24, two-stage capacity expansion
(`solar`, `peaker` extendable):

| Method               | Status   | Iter | Sub-LP | Wall (s) | E[cost]     |
|----------------------|----------|------|--------|----------|-------------|
| `benders_plain`      | optimal  |   10 |    200 |    0.113 | 203 651.48  |
| `benders_trust_region`| optimal |   10 |    200 |    0.103 | 203 651.48  |
| `benders_adaptive`   | optimal  |   10 |    200 |    0.100 | 203 651.48  |
| `progressive_hedging`| max_iter |   30 |    600 |    0.321 | 246 500.40  |

Cross-method spread on Benders variants is 0 % — same incumbent in 10
iterations on this scenario tree. PH stops at `max_iter` (30) with a
21 % gap, which is the expected behaviour for the trust-region-shrink
analogue used in lieu of a full quadratic proximal — flagged in
`DEFERRALS.md` (Phase 8.x: real ADMM PH).

## vs PyPSA (nexus-only baseline)

`benchmarks/vs_pypsa.py --json benchmarks/results/phase13_vs_pypsa_nexus_only.json`

PyPSA is **not installed in this venv** so the head-to-head columns are
captured separately under `test_projects/test_project_1/pypsa/`. The
table below is the nexus-only column at the Phase 13 boundary, used to
spot regressions in the LP-build + HiGHS solve path:

| n_buses | T    | Build (ms) | Solve (ms) | Mem (MB) | Cost           |
|---------|------|------------|------------|----------|----------------|
|       3 |   24 |        0.1 |      147.4 |     0.8  |     175 746.88 |
|      10 |  168 |        0.3 |    3 019.4 |     0.9  |   3 406 145.56 |
|      30 |  168 |        0.7 |    9 150.9 |     2.5  |  10 141 302.88 |
|      50 |  168 |        1.2 |   17 927.5 |     4.3  |  16 805 919.60 |

The 8760 h `--quick` skip case still runs in under a minute on the
laptop; re-add it before any release re-run.

## External head-to-heads (rerun deferred)

Two long-form comparisons live under `test_projects/test_project_1/`
and are the source of truth for the cross-framework story. They are
**not re-run on every phase boundary** — see `DEFERRALS.md` Phase 13
for the exact rerun command + environment requirements:

- `test_projects/test_project_1/pypsa/FLAGSHIP_COMPARISON.md` —
  PyPSA-MPC / PyPSA-Earth / PyPSA-Eur tutorials (needs micromamba).
- `test_projects/test_project_1/julia/GENX_COMPARISON.md` — GenX
  `1_three_zones` tutorial (needs Julia + GenX env).

## Re-run cookbook

```sh
# Flagship — must pass cleanly before tagging a release.
.venv/bin/python benchmarks/flagship_all_features.py

# Decomposition — rerun whenever Phase 8 code changes.
.venv/bin/python benchmarks/decomposition/adaptive_oracle_bench.py \
    --json benchmarks/results/phase13_adaptive_oracle.json

# vs PyPSA (nexus-only) — rerun on LP-build / solver-pipeline changes.
.venv/bin/python benchmarks/vs_pypsa.py --quick \
    --json benchmarks/results/phase13_vs_pypsa_nexus_only.json
```

Persist results JSON next to the script so phase-to-phase deltas can
be diffed without re-running. Don't claim a number in any
`*_COMPARISON.md` that isn't backed by a script + a JSON in
`benchmarks/results/`.
