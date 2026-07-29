# nexus-energy

Energy system optimisation across 15 sectors, built on the
[nexus-opt](https://github.com/VishalRam24/nexus-opt) Rust solver core.

You describe the system as components on a network graph — buses, generators,
storage, loads, links — and the library assembles and solves the mathematical
program behind it.

```python
import nexus_energy as ne

sys = ne.EnergySystem("my_system")
elec = sys.add_bus("elec", carrier="electricity")

sys.add_generator("solar", bus=elec, capacity=500, marginal_cost=0)
sys.add_generator("gas",   bus=elec, capacity=200, marginal_cost=50)
sys.add_load("demand", bus=elec, amount=300)

result = sys.optimise()
```

## Install

```bash
pip install nexus-energy
```

This pulls in `nexus-opt` (the Rust solver core) automatically — you don't need
to install it separately, and no Rust toolchain is required.

## What it covers

**Network physics** — DC-OPF, polar AC-OPF, SOCP conic relaxations, HVDC, N-1 security.

**Dispatch realism** — unit commitment (tighter formulation by default), start-up
and shutdown costs, ramping, must-run rules, spinning and regulation reserves.

**Storage** — self-discharge, cyclic state of charge, asymmetric charge/discharge
efficiency, simultaneous charge/discharge bans, cascade hydro, EV and V2G.

**Policy** — zonal/global/pooled CO₂ caps, RPS, clean energy standards, ITC/PTC
tax credits, hourly matching.

**Beyond a single solve** — time-series aggregation and rolling horizon
(`temporal`), Benders and temporal decomposition (`decomposition`), scenarios /
CVaR / robust optimisation (`stochastic`), model predictive control (`mpc`), and
a differentiable dispatch layer with analytic KKT gradients (`diff`, `diff_bridge`).

**223 component templates** across 15 sectors live in `Energy_Components/`, each
with multiple fidelity levels (F0 empirical → F2 lumped-physics).

**PyPSA interop** — `from_pypsa(network)` converts a PyPSA `Network` into an
`EnergySystem`. Note that this is a one-way adapter that reads the network's
dataframes; the library itself does not depend on PyPSA and never calls it to solve.

See [`WIKI.md`](WIKI.md) for the full user guide.

## Benchmarks — and their honest state

Head-to-head results live in [`COMPARISON_SCORECARD.md`](COMPARISON_SCORECARD.md),
with the per-case detail in [`BENCHMARKS.md`](BENCHMARKS.md) and
[`benchmarks/PARITY_LEDGER.md`](benchmarks/PARITY_LEDGER.md).

**Verified wins.** These reproduce, and both sides see the identical network:

| Case | Objective vs reference | Speed |
|---|---|---|
| PyPSA-Eur capacity expansion (10 bus, 2190 h, real profiles) | −0.000 % (exact parity) | 233 s vs 810 s — **3.47× faster** |
| GenX `1_three_zones_ucommit2` | −0.02 % (exact) | 7.7 s vs 36.1 s — **4.7× faster** |
| pandapower AC-OPF case9 / case14 | +0.0007 % / +0.0792 % | 60.5× / 42.2× |
| PowerModels.jl SOCWR, 3-bus radial | 3.67e-5 | 7.57× |
| CINDER LP | parity | 147 s vs 190 s — 1.3× faster |

**Known open problem — read this before quoting numbers.** Two GenX cases
(`rate_co2` at −42.7 % and `mincapreq` at −3.90 %) look like large wins and are
**not**. A gold-standard check — feeding nexus's chosen capacity back into a real
GenX solve — showed GenX's own cost for nexus's solution (5.823e9) essentially
matches GenX's optimum (5.808e9), while nexus *reports* 5.582e9. That is a
**~4.3 % OPEX under-count bug in nexus**, not a cheaper optimum. Lead suspect is
transmission-loss modelling (GenX uses piecewise-linear, nexus a linear `loss=%`).
The pattern is that nexus matches exactly when the build is determined, and
under-counts opex when multi-zone VRE is dispatched over transmission. The
exact-parity rows above are unaffected — none of them involve VRE over transmission.

CINDER MILP is also slower than PyPSA (330 s vs 190 s, 1.7×, gap 0.82 %), improved
from an earlier 639 s but not yet at parity.

## Development

```bash
git clone https://github.com/VishalRam24/nexus-energy
cd nexus-energy
uv sync
uv run pytest
```

Benchmark scripts that compare against pandapower, PowerModels.jl or GenX expect
those reference installations outside this repo, and skip automatically when they
are absent. Point them elsewhere with `NEXUS_PANDAPOWER_DIR` /
`NEXUS_POWERMODELS_DIR`.

Third-party solvers and frameworks appear only as benchmark comparison rows —
they are never wrapped inside the library.

## Licence

MIT — see [LICENSE](LICENSE).
