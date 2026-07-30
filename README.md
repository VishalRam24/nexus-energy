<div align="center">

# nexus-energy

**Energy-system optimisation in Python — that also answers backwards.**

Build and optimise an energy-system digital twin the way you do today, at exact
parity with PyPSA and GenX. Then ask the question they can't: *which of my
inputs is wrong?*

[![License: MIT](https://img.shields.io/badge/License-MIT-14b8a6.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nexus-energy.svg?color=14b8a6)](https://pypi.org/project/nexus-energy/)
[![Python](https://img.shields.io/badge/python-3.9%2B-4a5573.svg)](pyproject.toml)
[![Solver core](https://img.shields.io/badge/core-nexus--opt%20(Rust)-1c2337.svg)](https://github.com/VishalRam24/nexus-opt)

[Website](https://vishalram24.github.io/nexus/) ·
[User guide](WIKI.md) ·
[Benchmarks](COMPARISON_SCORECARD.md) ·
[Solver core](https://github.com/VishalRam24/nexus-opt)

</div>

---

## Who this is for

Engineers and researchers who build energy-system models in Python — capacity
expansion, economic dispatch, unit commitment, optimal power flow — and who need
the model to agree with a system that actually exists.

## Why it exists

Every production energy tool solves the problem **forward**: given these costs
and efficiencies, here is the optimal plan. That is well served already.

The expensive daily question is the **reverse**. Your twin's output does not
match the real system — which input is wrong, and by how much? Today that is
answered by brute force: perturb a parameter, re-solve, repeat. Morris, Sobol
and MGA sweeps cost hours to days, scale with every parameter, and return a
confident number even when the data cannot identify it.

nexus-energy differentiates through the optimisation itself, so the answer comes
from the model's own structure — in a handful of solves, and it says so when the
data is silent.

## Install

```bash
pip install nexus-energy
```

This pulls in [`nexus-opt`](https://github.com/VishalRam24/nexus-opt), the Rust
solver core, automatically. No separate step, no Rust toolchain. Python 3.9+.

## Sixty seconds

```python
import nexus_energy as ne

sys = ne.EnergySystem("my_system")
elec = sys.add_bus("elec", carrier="electricity")

sys.add_generator("solar", bus=elec, capacity=500, marginal_cost=0)
sys.add_generator("gas",   bus=elec, capacity=200, marginal_cost=50)
sys.add_load("demand", bus=elec, amount=300)

result = sys.optimise()
print(result.status, result.total_cost)   # optimal 0.0
print(result.generator_dispatch)          # {'solar': [300.], 'gas': [0.]}
```

And the part that is new — recovering a hidden CO₂ price from observed dispatch:

```python
from nexus_energy.pypsa_compat import from_pypsa
from nexus_energy.diff_bridge import fit_co2_price

system = from_pypsa(network, line_model="transport")
fit = fit_co2_price(system, observed_dispatch)

print(fit.price, fit.n_solves, fit.converged)
```

`fit_co2_price` runs in two stages — a coarse forward-only bracket (dispatch is
piecewise-linear in price, so the loss has flat pieces a pure gradient method is
stuck on) followed by safeguarded Gauss–Newton with a bisection fallback.

## What it covers

| Area | Capability |
|---|---|
| **Network physics** | DC-OPF, polar AC-OPF, SOCP conic relaxations, HVDC, N-1 security |
| **Dispatch** | Unit commitment (tighter formulation by default), start-up/shutdown costs, ramping, must-run, spinning and regulation reserves |
| **Storage** | Self-discharge, cyclic SoC, asymmetric charge/discharge efficiency, simultaneous charge/discharge bans, cascade hydro, EV and V2G |
| **Policy** | Zonal/global/pooled CO₂ caps, RPS, clean energy standards, ITC/PTC, hourly matching |
| **Temporal** | Time-series aggregation, representative periods, rolling horizon |
| **At scale** | Benders and temporal decomposition, stochastic scenarios, CVaR, robust optimisation |
| **Operations** | Model predictive control, auto-calibration from telemetry |
| **Differentiable** | Analytic KKT gradients, design gradients (∂cost/∂capacity), inverse calibration |

### The component library

223 components across 15 sectors, each at several **fidelity** levels — fidelity
being how much physics the component carries. The same LFP cell is available as:

| Level | Model | What it adds |
|---|---|---|
| `F0a` | Round-trip efficiency curve | One lookup vs C-rate. Cheapest. |
| `F1a` | State of charge | Energy in, energy out, SoC over time |
| `F1b` | + thermal | Cell temperature moves the efficiency |
| `F1c` | + degradation | Capacity fade over cycles |
| `F2a` | Equivalent circuit (1RC) | Real voltage dynamics |
| `F2d` | Single-particle model | Electrochemical detail |

Pick the cheapest level that still answers the question — **per component, not
per study**. A capacity screen can run everything at F0 while the one asset
under investigation runs at F2.

Sector split: thermal 41 · power electronics 32 · batteries 26 · solar 18 ·
hydrogen 17 · conventional 13 · hydro & marine 12 · biomass 11 · carbon capture
11 · gas systems 9 · thermoelectric 8 · desalination 7 · wind 6 · mechanical
storage 6 · geothermal 6.

F0–F2 are built for all 223. F3–F6 (distributed physics, AI surrogates, PINNs)
are scaffolded but not yet implemented.

## Coming from PyPSA

`from_pypsa(network)` converts a PyPSA `Network` into an `EnergySystem`. It is a
one-way adapter that reads the network's dataframes — **the library itself does
not depend on PyPSA and never calls it to solve**. It exists so a benchmark can
put the identical network in front of both solvers, and so an existing PyPSA
workflow has an on-ramp.

```python
from nexus_energy.pypsa_compat import from_pypsa

system = from_pypsa(n)                            # AC lines auto-route to DC-OPF
system = from_pypsa(n, line_model="transport")    # required by the diff layer
```

## Benchmarks

Full detail in [`COMPARISON_SCORECARD.md`](COMPARISON_SCORECARD.md); stored
results under [`benchmarks/results/`](benchmarks/results), so every number can be
re-run rather than taken on trust.

| Case | Objective vs reference | Wall clock |
|---|---|---|
| PyPSA-Eur capacity expansion (10 bus, 2190 h, real profiles) | **−0.000 %** exact parity | 233 s vs 810 s — **3.47× faster** |
| GenX `1_three_zones_ucommit2` | **−0.02 %** exact | 7.7 s vs 36.1 s — **4.7× faster** |
| pandapower AC-OPF case9 / case14 | +0.0007 % / +0.0792 % | 60.5× / 42.2× |
| PowerModels.jl SOCWR, 3-bus radial | 3.67e-5 | 7.57× |
| CINDER LP | parity | 147 s vs 190 s — 1.3× faster |
| CINDER MILP | MIP gap 0.82 % | 330 s vs 190 s — **1.7× slower** |

Both solvers see the identical network — that is what `from_pypsa` is for. Two
real bugs in this library were found *because* the inputs were held identical
(a static scalar `p_max_pu` being ignored, and cyclic storage being over-pinned).

### Known open problem

Two GenX cases (`rate_co2` at −42.7 %, `mincapreq` at −3.90 %) look like large
wins and are **not**. Feeding the capacity nexus chose back into a real GenX
solve showed GenX's own cost for the nexus solution (5.823e9) essentially
matches GenX's optimum (5.808e9), while nexus *reports* 5.582e9 — a **~4.3 %
OPEX under-count bug here**, not a cheaper optimum. Lead suspect is
transmission-loss modelling (GenX piecewise-linear vs a linear `loss=%`).

The pattern: exact match when the build is determined, opex under-count when
multi-zone renewables are dispatched over transmission. The exact-parity rows
above are unaffected — none of them involve that case.

## What it does not claim

- **Not dynamic or EMT simulation.** No transients, no swing equation. Time is
  snapshots and rolling horizons coupled by algebraic constraints — a different
  category from Simulink and Modelica.
- **No integer (UC-MILP) differentiability.** Future work, not a claim.
- **Calibration is LP/QP class.** A small ridge term, required to make the
  gradients well defined, shifts economics by roughly 1–7 percentage points and
  is disclosed per result.
- **The speed headline is forward-only** — the calibration solve uses a denser path.

## Development

```bash
git clone https://github.com/VishalRam24/nexus-energy
cd nexus-energy
uv sync
uv run pytest
```

Benchmark scripts that compare against pandapower, PowerModels.jl or GenX expect
those reference installations outside this repo and skip automatically when they
are absent. Point them elsewhere with `NEXUS_PANDAPOWER_DIR` /
`NEXUS_POWERMODELS_DIR`.

Third-party solvers and frameworks appear only as benchmark comparison rows —
they are never wrapped inside the library.

## Licence

MIT — see [LICENSE](LICENSE).
