# nexus-energy — User Guide

**Energy-system optimisation in Python — that also answers backwards.**

Build and optimise an energy-system digital twin the way you do today, at exact
parity with PyPSA and GenX. Then ask the question they can't: *which of my
inputs is wrong?*

`nexus-energy` is for engineers and researchers who build energy-system models
in Python — capacity expansion, economic dispatch, unit commitment, optimal
power flow — and who need the model to agree with a system that actually
exists.

> **Every code block in this guide has been executed against a clean install.**
> Where a printed result is shown, that is the actual output.

---

## Contents

**Getting started**

1. [What this library does](#1-what-this-library-does)
2. [Install and verify](#2-install-and-verify)
3. [Your first model](#3-your-first-model)
4. [Adding time](#4-adding-time)
5. [The five building blocks](#5-the-five-building-blocks)
6. [Reading results](#6-reading-results)

**Doing real work**

7. [Storage](#7-storage)
8. [Unit commitment](#8-unit-commitment)
9. [Capacity expansion](#9-capacity-expansion)
10. [Networks and transmission](#10-networks-and-transmission)
11. [Policy constraints](#11-policy-constraints)
12. [Coming from PyPSA](#12-coming-from-pypsa)
13. [Multi-stage planning](#13-multi-stage-planning)

**The part that is new**

14. [Differentiable dispatch and inverse calibration](#14-differentiable-dispatch-and-inverse-calibration)
15. [MPC and auto-calibration](#15-mpc-and-auto-calibration)

**Making it fast**

16. [Solver controls](#16-solver-controls)
17. [Temporal aggregation](#17-temporal-aggregation)
18. [Decomposition at scale](#18-decomposition-at-scale)

**Reference**

19. [Uncertainty — stochastic and robust](#19-uncertainty--stochastic-and-robust)
20. [AC power flow](#20-ac-power-flow)
21. [ML-guided solving](#21-ml-guided-solving)
22. [The component library](#22-the-component-library)
23. [Benchmarks](#23-benchmarks)
24. [Honest scope and known limits](#24-honest-scope-and-known-limits)
25. [Troubleshooting](#25-troubleshooting)
26. [API index](#26-api-index)

---

## 1. What this library does

Every production energy tool solves the problem **forward**: given these costs
and efficiencies, here is the optimal plan. That is well served already, and
`nexus-energy` does it too — at exact parity with PyPSA and GenX, several times
faster.

The expensive daily question is the **reverse**. Your twin's output does not
match the real system — which input is wrong, and by how much? Today that is
answered by brute force: perturb a parameter, re-solve, repeat. Morris, Sobol
and MGA sweeps cost hours to days, scale with every parameter, and return a
confident number even when the data cannot identify it.

`nexus-energy` differentiates through the optimisation itself, so the answer
comes from the model's own structure — in a handful of solves — and it says so
when the data is silent. That is [section 14](#14-differentiable-dispatch-and-inverse-calibration).

You model by placing components on a network graph:

```
                  ┌──────────────────────┐
                  │     EnergySystem     │
                  └──────────┬───────────┘
         ┌───────────────────┼───────────────────┐
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│       Bus       │ │    Generator    │ │     Storage     │
│  (electricity,  │ │  (solar, gas,   │ │ (battery, pump, │
│   H2, heat, CO2)│ │   hydro, …)     │ │  thermal, V2G)  │
└────────┬────────┘ └─────────────────┘ └─────────────────┘
         │
         ├───────────────────┐
┌────────▼────────┐ ┌────────▼────────┐
│      Load       │ │      Link       │
│  (demand side)  │ │ (line, HP, HVDC,│
└─────────────────┘ │  electrolyser)  │
                    └─────────────────┘
```

---

## 2. Install and verify

```bash
pip install nexus-energy
```

This pulls in [`nexus-opt`](https://github.com/VishalRam24/nexus-opt), the Rust
solver core, automatically. No separate step, no Rust toolchain. Python 3.9+.

Verify:

```python
import nexus_energy as ne

sys = ne.EnergySystem("check")
elec = sys.add_bus("elec")
sys.add_generator("g", bus=elec, capacity=100, marginal_cost=10)
sys.add_load("d", bus=elec, amount=50)
print(sys.optimise().total_cost)      # 500.0
```

### Development install

```bash
git clone https://github.com/VishalRam24/nexus-energy
cd nexus-energy
uv sync
uv run pytest
```

Benchmark scripts that compare against pandapower, PowerModels.jl or GenX
expect those reference installations outside this repo and skip automatically
when absent. Point them elsewhere with `NEXUS_PANDAPOWER_DIR` /
`NEXUS_POWERMODELS_DIR`.

---

## 3. Your first model

One bus, two generators, one load. Solar is free, gas costs 50/MWh, demand is
300 MW.

```python
import nexus_energy as ne

sys = ne.EnergySystem("my_system")
elec = sys.add_bus("elec", carrier="electricity")

sys.add_generator("solar", bus=elec, capacity=500, marginal_cost=0)
sys.add_generator("gas",   bus=elec, capacity=200, marginal_cost=50)
sys.add_load("demand", bus=elec, amount=300)

result = sys.optimise()
print(result.status, result.total_cost)
print(result.generator_dispatch)
```

```
optimal 0.0
{'solar': array([300.]), 'gas': array([0.])}
```

Solar covers the whole load, so the cost is zero and gas stays off. That is the
merit order, and it is the entire idea of economic dispatch.

Note that `generator_dispatch` values are **arrays**, not scalars — this model
has one timestep, so each array has length 1.

---

## 4. Adding time

Real models have profiles. `set_timesteps(n, dt)` sets the number of snapshots
and the hours each one spans; loads and availabilities then take arrays.

```python
import numpy as np
import nexus_energy as ne

sys = ne.EnergySystem("with_time")
elec = sys.add_bus("elec")
sys.set_timesteps(6, dt=1.0)

sys.add_generator("solar", bus=elec, capacity=400, marginal_cost=0,
                  carrier_factor=np.array([0, .2, .8, 1., .5, 0.]))
sys.add_generator("gas", bus=elec, capacity=500, marginal_cost=60)
sys.add_load("demand", bus=elec,
             amount=np.array([200, 250, 300, 350, 300, 220.]))

r = sys.optimise()
print(r.status, r.total_cost)
print("solar:", r.generator_dispatch["solar"])
print("gas  :", r.generator_dispatch["gas"])
print("price:", r.bus_shadow_prices["elec"])
```

```
optimal 41400.0
solar: [  0.  80. 300. 350. 200.   0.]
gas  : [200. 170.   0.   0. 100. 220.]
price: [60. 60.  0.  0. 60. 60.]
```

Three things to read out of that:

- **`carrier_factor`** is the per-timestep availability in `[0, 1]`. Solar's
  usable output is `capacity × carrier_factor[t]`, so at `t=3` its ceiling is
  400 MW and it serves the whole 350 MW load.
- **Solar is curtailed at `t=3`**: it could produce 400 but only 350 is needed.
- **`bus_shadow_prices`** is the marginal price of energy at that bus — the
  cost of one more MWh of demand. It is 60 when gas is on the margin and 0
  when free solar is, which is exactly the market-clearing price.

`dt` matters for anything integrated over time: storage state of charge, energy
costs, emissions. With `dt=0.25` each snapshot is 15 minutes.

---

## 5. The five building blocks

### Bus

A connection point. Everything attaches to a bus, and each bus enforces energy
balance at every timestep.

```python
elec = sys.add_bus("elec", carrier="electricity")
h2   = sys.add_bus("h2",   carrier="hydrogen")
```

The carrier sets default units (MWh for electricity, tCO₂ for carbon) and lets
the library check that you are not wiring hydrogen into an electricity balance.

### Generator

`add_generator(name, bus, capacity, marginal_cost=0.0, **kwargs)`

| Parameter | Unit | Meaning |
|---|---|---|
| `capacity` | MW | nameplate output |
| `marginal_cost` | $/MWh | variable cost of production |
| `capital_cost` | **$/MW/year** | annualised capex — only used when `extendable=True` |
| `carrier_factor` | array in `[0,1]` | per-timestep availability |
| `p_min` | MW | minimum stable output |
| `emission_factor` | tCO₂/MWh | for carbon pricing and caps |
| `extendable` | bool | make capacity a decision variable |
| `max_capacity`, `min_capacity` | MW | bounds when extendable |
| `committable` | bool | binary on/off unit commitment |
| `ramp_up`, `ramp_down` | MW/timestep | ramp limits |
| `min_up_time`, `min_down_time` | timesteps | dwell constraints |
| `startup_cost`, `shutdown_cost` | $ | per transition |
| `must_run` | bool | always on |
| `tech` | str | technology tag, used by policy carve-outs |

### Load

`add_load(name, bus, amount)` — `amount` is a scalar or a per-timestep array.
Load is inelastic by default: it must be served.

### Storage

`add_storage(name, bus, power_capacity, energy_capacity, **kwargs)` — see
[section 7](#7-storage).

### Link

`add_link(name, bus_from, bus_to, capacity, efficiency=1.0, **kwargs)`

A link moves or converts energy between two buses. A transmission line is a
link with `efficiency` slightly below 1; an electrolyser is a link from an
electricity bus to a hydrogen bus with `efficiency` around 0.7; a heat pump is
a link with `efficiency` above 1 (its coefficient of performance).

---

## 6. Reading results

`optimise()` returns an `OptimisationResult`:

| Field | Type | Contents |
|---|---|---|
| `status` | str | `"optimal"`, `"infeasible"`, … — **check this first** |
| `total_cost` | float | objective value |
| `solve_time` | float | seconds |
| `generator_dispatch` | `{name: array}` | MW per timestep |
| `storage_charge` / `storage_discharge` | `{name: array}` | MW per timestep |
| `storage_soc` | `{name: array}` | MWh per timestep |
| `link_flow` | `{name: array}` | MW per timestep |
| `bus_shadow_prices` | `{name: array}` | $/MWh — marginal price at each bus |
| `unit_status` | `{name: array}` | commitment `u[t]`; only when `committable=True` |
| `capacity_additions` | `{name: float}` | MW built; only for extendable components |
| `storage_soc_duals` | `{name: array}` | $/MWh marginal value of stored energy (LP solves, full-mode storages) |
| `cap_dual` | `{name: float}` | marginal operational value of capacity (Benders) |

Inspect the model itself before solving:

```python
print(sys.summary())
print(sys.n_buses, sys.n_components, sys.n_timesteps)   # properties, not methods
```

```
EnergySystem: sum
  Buses: 1
  Generators: 1
  Storages: 0
  Loads: 1
  Links: 0
  Timesteps: 1 (dt=1.0h)
```

---

## 7. Storage

```python
sys.add_storage(
    "battery",
    bus=elec,
    power_capacity=100.0,        # MW  — max charge/discharge rate
    energy_capacity=400.0,       # MWh — 4-hour battery
    efficiency_charge=0.92,
    efficiency_discharge=0.92,
    self_discharge=0.001,        # fraction lost per timestep
    soc_initial=0.5,             # fraction of energy_capacity
    cyclic=True,                 # soc(0) == soc(T)
)
```

Worked example — a battery shifting solar into the evening:

```python
import numpy as np, nexus_energy as ne

sys = ne.EnergySystem("storage_demo")
elec = sys.add_bus("elec")
sys.set_timesteps(8)

sys.add_generator("solar", bus=elec, capacity=400, marginal_cost=0,
                  carrier_factor=np.array([0, 0, .6, 1., 1., .4, 0, 0.]))
sys.add_generator("gas", bus=elec, capacity=400, marginal_cost=80)
sys.add_storage("battery", bus=elec, power_capacity=100.0, energy_capacity=400.0,
                efficiency_charge=0.92, efficiency_discharge=0.92,
                self_discharge=0.001, soc_initial=0.5, cyclic=True)
sys.add_load("demand", bus=elec, amount=np.full(8, 250.0))

r = sys.optimise()
print(r.status, r.total_cost)
print("soc:", np.round(r.storage_soc["battery"], 1))
```

```
optimal 74561.3
soc: [200.   91.1  80.1 172.1 263.9 200.6 200.4 200.2]
```

The battery discharges into the dark early hours, refills through the solar
peak, and returns to its starting level because `cyclic=True`.

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `efficiency_charge` / `efficiency_discharge` | 0.95 | one-way efficiencies |
| `self_discharge` | 0.0 | fraction lost per timestep |
| `soc_min`, `soc_max`, `soc_initial` | 0, 1, 0.5 | fractions of `energy_capacity` |
| `cyclic` | `True` | enforce `soc(0) == soc(T)` |
| `cyclic_level` | `"fixed"` | `"fixed"` also pins the level to `soc_initial`; `"free"` lets the optimiser choose it (PyPSA's convention) |
| `no_simultaneous` | `False` | binary lock preventing simultaneous charge and discharge |
| `marginal_cost` | 0.0 | $/MWh on the discharge leg |
| `extendable` | `False` | make power and energy capacity decision variables |
| `max_hours` / `min_hours` | `None` | bind energy capacity to a duration window |
| `inflow`, `spill_to` | `None` | hydro reservoirs and cascades |
| `pump_capacity` / `turbine_capacity` | `None` | asymmetric rates for pumped hydro |
| `long_duration` | `False` | inter-period SOC carry-over under representative periods |

> **`no_simultaneous` costs one binary per timestep per storage.** You only
> need it when round-trip efficiency is exactly 1 (otherwise losses already
> make simultaneous charge/discharge uneconomic) or when the physical
> constraint is required for its own sake. Leave it off for pure LP dispatch.

### Store vs full formulation

`storage_model` selects the LP variable shape: `"auto"` (default), `"store"` or
`"full"`. The `"store"` model compiles a single energy-state variable `e[t]` per
timestep instead of the three-variable `(charge, discharge, soc)` form — a
smaller polyhedron that helps HiGHS find integer-feasible vertices at the root
in models with committable links.

`"auto"` picks `"store"` only when round-trip efficiency is ≈ 1, charge and
discharge capacities are symmetric, and none of `inflow`, `spill_to`,
`ramp_cost`, `no_simultaneous`, `availability`, `long_duration` or a non-zero
marginal cost is in use. Result extraction derives `charge[t]` / `discharge[t]`
from the signed `Δe/dt`, so downstream code sees the same arrays either way.

---

## 8. Unit commitment

Thermal plants cannot be switched on and off freely. Setting `committable=True`
introduces binary on/off variables and turns the problem into a MILP.

```python
import numpy as np, nexus_energy as ne

sys = ne.EnergySystem("uc")
elec = sys.add_bus("elec")
sys.set_timesteps(8)

sys.add_generator(
    "coal", bus=elec, capacity=100.0,
    p_min=20.0,                # minimum stable output when on
    marginal_cost=30.0,
    startup_cost=5000.0,
    shutdown_cost=1000.0,
    min_up_time=3,             # timesteps
    min_down_time=2,
    ramp_up=40.0,              # MW per timestep
    ramp_down=40.0,
    committable=True,          # <- activates binary commitment
)
sys.add_generator("peaker", bus=elec, capacity=100.0, marginal_cost=200.0)
sys.add_load("demand", bus=elec, amount=np.array([10, 50, 90, 80, 20, 10, 80, 90.]))

r = sys.optimise()
print(r.status, r.total_cost)
print("u[t]:", r.unit_status["coal"])
```

```
optimal 49400.0
u[t]: [ 0.  1.  1.  1. -0.  0.  1.  1.]
```

Coal is off for the first cheap hour, runs for three (satisfying
`min_up_time=3`), shuts down through the trough, and restarts for the evening
peak — paying `startup_cost` twice rather than idling at `p_min` throughout.

> **Parameter names to know.** These differ from some other tools: it is
> `committable` (not `unit_commitment`), `p_min` (not
> `min_stable_generation`), and `ramp_up` / `ramp_down` (not `ramp_limit_up` /
> `ramp_limit_down`).

### Formulation notes

For committable links and non-clustered committable generators, the
`start_up` / `shut_down` indicators are declared **binary** and the state
transition compiles as two inequalities (`v[t] ≥ u[t] − u[t−1]`,
`w[t] ≥ u[t−1] − u[t]`) plus four upper-bound cuts (`v[t] ≤ u[t]`,
`v[t] ≤ 1 − u[t−1]`, `w[t] ≤ 1 − u[t]`, `w[t] ≤ u[t−1]`) — the PyPSA-style
LP-tight 3-bin form. The `v + w ≤ 1` mutex is implied and is not emitted as a
separate row. Clustered generators keep the equality transition form.

**Multi-state (hot / warm / cold) starts.** `start_up_segments` takes a list of
`(min_off_timesteps, startup_cost)` tuples — ascending in `min_off_timesteps`,
non-decreasing in cost, first entry `0`. When set it overrides the flat
`startup_cost` and uses the tight Morales-España (2013) start-type formulation.
Binary UC only; clustered UC falls back to the flat cost.

**Clustered UC.** `clustered=True` with `n_units=N` lumps N identical units
together with continuous `u, v, w ∈ [0, N]` — the GenX `UCommit: 2` analogue.
Far cheaper than N sets of binaries.

**Piecewise heat rates.** `heat_rate_segments` takes
`(p_MW_breakpoint, marginal_cost)` points defining a convex increasing
piecewise-linear fuel cost, overriding the flat `marginal_cost`.

---

## 9. Capacity expansion

Set `extendable=True` and give a `capital_cost`, and capacity becomes a
decision variable.

```python
import numpy as np, nexus_energy as ne

sys = ne.EnergySystem("expand")
elec = sys.add_bus("elec")
sys.set_timesteps(24)
sys.set_snapshot_weights(np.full(24, 365.0))     # one representative day × 365

cf = np.clip(np.sin(np.linspace(0, np.pi, 24)), 0, None)

sys.add_generator("solar", bus=elec, capacity=0.0, marginal_cost=0.0,
                  capital_cost=60_000.0,          # $/MW/YEAR
                  extendable=True, max_capacity=2000.0,
                  carrier_factor=cf)
sys.add_generator("gas", bus=elec, capacity=600.0, marginal_cost=90.0)
sys.add_load("demand", bus=elec, amount=np.full(24, 400.0))

r = sys.optimise()
print(r.status, r.total_cost)
print(r.capacity_additions)
```

```
optimal 110626907.4
{'solar': 769.8}
```

> ⚠ **The single most common beginner mistake: `capital_cost` is `$/MW/year`,
> but your model may only span a few hours.** If you model 6 hours and price
> capex annually, no plant will ever be worth building — the model sees a whole
> year of capital cost recovered by six hours of fuel savings, and builds
> nothing. `set_snapshot_weights()` is the fix: weight each snapshot by the
> number of hours of the year it represents, so the two sides of the trade-off
> are on the same clock. In the example above, 24 snapshots × 365 = 8760 hours.

Related parameters: `min_capacity` (a floor, PyPSA's `p_nom_min`), `fixed_om`
($/MW/year paid on built capacity whether or not it runs), `integer_investment`
with `unit_size` (build in discrete units), and `capex_segments` for piecewise
economies of scale.

Storage is extendable the same way, through `capital_cost_power` ($/MW/year)
and `capital_cost_energy` ($/MWh/year).

---

## 10. Networks and transmission

```python
import numpy as np, nexus_energy as ne

sys = ne.EnergySystem("network")
north = sys.add_bus("north")
south = sys.add_bus("south")
sys.set_timesteps(4)

sys.add_generator("wind", bus=north, capacity=400, marginal_cost=0)
sys.add_generator("gas",  bus=south, capacity=400, marginal_cost=70)
sys.add_load("city", bus=south, amount=np.full(4, 300.0))

sys.add_link("interconnector", bus_from=north, bus_to=south,
             capacity=200.0, efficiency=0.98)

r = sys.optimise()
print(r.status, r.total_cost)
print("flow:", r.link_flow["interconnector"])
```

```
optimal 29400.0
flow: [200. 200. 200. 200.]
```

The line runs at its 200 MW limit in every hour — free northern wind displaces
southern gas up to the point the wire allows, and the remaining 104 MW comes
from gas. A congested interconnector is exactly what you would expect the
shadow prices at the two buses to diverge across.

### Flow formulations

`Link.model_type` selects the physics: `"transport"` (default — a pipe with a
capacity), `"dc_opf"` (linearised DC power flow with voltage angles), `"ptdf"`
(power transfer distribution factors) or `"switched"`.

Internally each link resolves to one of two LP variable shapes:

- **`signed`** — PyPSA-style symmetric bidirectional transport: one
  `f[t] ∈ [−cap, cap]` variable per timestep, no forward/reverse mutex row.
  Halves the per-line variable count.
- **`fwd_rev`** — one non-negative flow variable, plus a reverse variable and a
  `fwd + rev ≤ cap` mutex when bidirectional.

Eligibility for `signed` is deliberately conservative: any feature applying a
coefficient to the *absolute* flow magnitude (losses, marginal cost, ramp cost,
ramp limits, multi-output efficiencies, CO₂ output, UC/switching, linepack)
keeps `fwd_rev` so the LP stays linear. Extendable bidirectional links also
stay `fwd_rev` — signed flow under a capacity *variable* needs a two-sided
coupling, and it measures slower.

The optimum is bit-identical either way. `optimise(link_formulation=…)` pins
the shape for A/B measurement; on a fixed-capacity transport LP, `signed`
measured −19 % variables and constraints and −17 % wall clock.

Other network capabilities: `set_n_minus_1()` for N-1 security,
`set_contingency_reserve()` for single-largest-unit reserve, HVDC via link
parameters, and the AC formulations in [section 20](#20-ac-power-flow).

---

## 11. Policy constraints

### Carbon

```python
sys.set_co2_price(80.0)            # $/tCO2 added to marginal costs
sys.set_emission_limit(600.0)      # tCO2 cap over the horizon
sys.set_co2_rate_cap(...)          # tCO2/MWh intensity cap
sys.set_co2_zone_cap(...)          # per-zone caps
```

Both levers on the same two-generator system:

```
with price 80/tCO2 : total 104400.0   coal 0.0     gas 1200.0
with a 600 t cap   : total  58800.0   coal 240.0   gas  960.0
```

A price of 80 makes coal (0.9 tCO₂/MWh) strictly worse than gas and pushes it
out entirely; a quantity cap lets coal run right up to the limit. Same physical
system, different instrument, different answer — which is the point of modelling
both.

**Pooled multi-zone cap groups** are one constraint over a *set* of buses (the
GenX `Cap_Zone` analogue), which is tighter than independent per-bus caps
because it forbids inter-zone emission averaging:

```python
sys.set_co2_cap_group(
    [bus_a, bus_b, bus_c], limit=0.05,   # tCO2/MWh
    is_rate=True,
    storage_losses_on_rhs=True,          # GenX CO2Cap=2 RHS term
    loss_accounting="net",               # "net" | "dissipation"
)
```

### Clean energy shares

```python
sys.set_rps(0.4, qualifying_techs=["wind"])       # 40% from tagged techs
sys.set_ces(0.6, scores={"wind": 1.0, "gas": 0.4})  # weighted clean standard
```

Both accept `slack_penalty=` to make the target a priced soft constraint rather
than a hard one. `qualifying_techs` matches against each generator's `tech`
tag, so remember to set `tech=` when you add the generator.

### Other instruments

`set_itc()` and `set_ptc()` (investment and production tax credits),
`set_hourly_matching()` (24/7 carbon-free matching), `set_capacity_bucket()`
(technology carve-outs), `set_reserve_margin()`, `set_spinning_reserve()`,
`set_regulation_reserve()`, `set_fuel_supply_limit()`, `set_outage()`,
`set_shared_capacity()`.

---

## 12. Coming from PyPSA

`from_pypsa(network)` converts a PyPSA `Network` into an `EnergySystem`.

```python
import pypsa
from nexus_energy.pypsa_compat import from_pypsa

n = pypsa.Network()
n.set_snapshots(range(4))
n.add("Bus", "elec")
n.add("Generator", "solar", bus="elec", p_nom=400, marginal_cost=0,
      p_max_pu=[0.0, 0.4, 0.9, 0.2])
n.add("Generator", "gas", bus="elec", p_nom=400, marginal_cost=60)
n.add("Load", "demand", bus="elec", p_set=[200, 250, 300, 280])

system = from_pypsa(n)                          # AC lines auto-route to DC-OPF
system = from_pypsa(n, line_model="transport")  # required by the diff layer

r = system.optimise()
print(r.status, r.total_cost)
print(r.generator_dispatch)
```

```
optimal 29400.0
{'solar': array([  0., 160., 300.,  80.]), 'gas': array([200.,  90.,   0., 200.])}
```

This is a **one-way adapter** that reads the network's dataframes. The library
does not depend on PyPSA and never calls it to solve. It exists so a benchmark
can put the identical network in front of both solvers, and so an existing
PyPSA workflow has an on-ramp.

### Two parity-critical conventions

- **Static (scalar) `p_max_pu` is honoured.** When a generator has no
  `p_max_pu` time series, a constant capacity-factor array is synthesised from
  the static column (for example PyPSA-Eur's nuclear at `p_max_pu = 0.781`), so
  `p[t] ≤ capacity · p_max_pu` is enforced exactly as PyPSA does. Dropping
  constant de-ratings used to over-dispatch baseload and under-price the
  system. Exports round-trip the factor back into `generators_t.p_max_pu`.
- **Cyclic storage level is free.** StorageUnits and Stores import with
  `cyclic_level="free"` — only continuity `soc(0) = soc(T)` is enforced and the
  optimiser picks the level, matching PyPSA's `cyclic_state_of_charge` /
  `e_cyclic`. A fixed-level pin over-constrains, and for extendable storage
  (whose energy capacity starts at 0) it forces start and end empty.

Both of these were found *because* the inputs were held identical across the
two solvers.

---

## 13. Multi-stage planning

`MultiStageSystem` chains per-year `EnergySystem` snapshots into one
investment problem with vintage tracking.

```python
import numpy as np, nexus_energy as ne

def stage(year, demand, hours=24):
    s = ne.EnergySystem(f"y{year}")
    b = s.add_bus("elec")
    s.set_timesteps(hours)
    cf = np.clip(np.sin(np.linspace(0, np.pi, hours)), 0, None)
    s.add_generator("solar", bus=b, capacity=0.0, marginal_cost=0.0,
                    capital_cost=500.0,        # see the note below
                    extendable=True, max_capacity=1000.0, carrier_factor=cf)
    s.add_generator("gas", bus=b, capacity=800.0, marginal_cost=95.0)
    s.add_load("demand", bus=b, amount=np.full(hours, demand))
    return s

ms = ne.MultiStageSystem("plan")
ms.add_stage(2030, stage(2030, 300.0))
ms.add_stage(2040, stage(2040, 420.0))

res = ms.optimise()
print(res.status, res.total_cost, res.years)
print("new_builds:     ", res.new_builds)
print("capacity_active:", res.capacity_active)
```

```
optimal 690637.1 [2030, 2040]
new_builds:      {'solar': [665.5, 0.0], 'gas': [0.0, 0.0]}
capacity_active: {'solar': [665.5, 665.5], 'gas': [800.0, 800.0]}
```

Solar is built once in 2030 and is still active in 2040 — that is the vintage
tracking. `MultiStageResult` also carries `stage_dispatch`, `stage_link_flow`,
`stage_storage_soc`, `storage_new_power` / `storage_new_energy` and the link
equivalents.

> **`MultiStageSystem` costs stage dispatch snapshot by snapshot and does not
> apply `set_snapshot_weights`.** Express `capital_cost` on the same basis as
> the modelled horizon, or scale the horizon to a year. The `capital_cost=500`
> above is per MW over the modelled 24 hours, not per year.

Vintage-related generator parameters: `build_year`, `lifetime_years`,
`retire_at_year`, `build_lead_years`, and `retrofit_of` for fuel switching
bounded by retiring host capacity.

Pass `myopic=True` to `optimise()` to solve each stage in sequence with no
foresight, which is the standard comparison against perfect-foresight planning.

---

## 14. Differentiable dispatch and inverse calibration

This is the part that does not exist elsewhere.

The dispatch problem is set up as a **differentiable layer**: analytic
`∂dispatch/∂parameter` Jacobians obtained by implicit differentiation of the
KKT system at the optimum. No sampling, no finite differences.

**Honest scope up front:** this is parameter learning on the inner **LP/QP**.
Every layer uses a mandatory strict-convexity ridge that shifts dispatch versus
the true LP by `O(ridge)` — disclose it in results. It is pure numpy; torch is
optional.

### Recovering a hidden CO₂ price

The headline use: you observe a system's dispatch and want to know what carbon
price explains it.

```python
from nexus_energy.pypsa_compat import from_pypsa
from nexus_energy.diff_bridge import fit_co2_price

system = from_pypsa(network, line_model="transport")
fit = fit_co2_price(system, observed_dispatch, price_bounds=(0.0, 1000.0))

print(fit.price, fit.n_solves, fit.converged)
```

`fit_co2_price` runs in two stages: a coarse **forward-only bracket** — dispatch
is piecewise-linear in price, so the loss has flat pieces that a pure gradient
method is born stuck on — followed by **safeguarded Gauss-Newton** with a
bisection fallback. It returns a `CO2FitResult(price, history, n_solves,
converged)`.

The price enters through `mc_eff = mc + price · emission`, so by the chain rule
`∂dispatch/∂price = d_dispatch_d_mc @ emission`, exposed directly as
`d_dispatch_d_co2_price`.

To inspect the bridge rather than fit through it:

```python
from nexus_energy.diff_bridge import multibus_problem_from_system

bridge = multibus_problem_from_system(system, co2_price=80.0, ridge=1e-2)
# bridge.problem is ready for solve_multibus_dispatch_with_sensitivities;
# bridge.emission and bridge.mc_base satisfy mc = mc_base + price * emission
```

**The bridge fails loudly rather than silently approximating.** It raises on:
extendable components (every capacity must be fixed), DC-OPF links (re-import
with `line_model="transport"`), storage, lossy or costed or multi-output links,
and any of UC / `p_min` / `must_run` / piecewise heat rates.

### Layer inventory

| Layer | Scope | Differentiable inputs |
|---|---|---|
| `solve_dispatch_with_sensitivities` | single-bus, single-period, closed form | mc, capacity, demand |
| `EconomicDispatchLayer` | stateful wrapper of the above | same |
| `MultiBusDispatchProblem` + `solve_multibus_dispatch_with_sensitivities` | multi-bus, multi-period transport QP | mc, capacity, demand, line_limit |
| `StorageDispatchProblem` + `solve_storage_dispatch_with_sensitivities` | single-bus multi-period + one storage | mc, capacity, demand, soc_init, η_c, η_d |
| `MultiBusStorageProblem` + `solve_multibus_storage_dispatch_with_sensitivities` | multi-bus + bus-attached storages, one stacked QP | mc, η_c, η_d, soc_init |
| `SmoothCommitmentLayer` / `fit_commitment_threshold` | sigmoid-smoothed commitment | threshold |
| `CapacityExpansionProblem` / `CapacityExpansionLayer` / `fit_component_params` | differentiable capacity expansion (design gradients) | capex, mc, demand, … |
| `TorchDispatchLayer` | cvxpylayers hook for problems beyond the numpy paths | — |

### Selective Jacobians

Jacobian blocks are expensive. Request only what you will use:

```python
sol = solve_multibus_dispatch_with_sensitivities(
    problem,
    jacobians=("mc",),      # () means forward-only
)
# sol.dispatch (G,T), sol.flows (L,T)
# sol.d_dispatch_d_mc (G·T, G), d_dispatch_d_capacity (G·T, G),
# d_dispatch_d_demand (G·T, B·T), d_dispatch_d_linelimit (G·T, L)
```

Skipped blocks come back as zeros. Pass `()` for the price a sampling baseline
pays per draw; pass `("mc",)` when only cost-side gradients are chained, as in
CO₂-price calibration.

`MultiBusDispatchProblem` also carries `availability` — `(G, T)` in `[0,1]`,
giving per-period derating `p[g,t] ≤ capacity[g] · availability[g,t]` for VRE
capacity factors and outages — and `line_min`, `(L,)` lower flow bounds, where
`None` keeps symmetric `−line_limit` and `0.0` entries make links
unidirectional.

### Storage efficiency Jacobians

`StorageDispatchSolution` exposes the constraint-matrix sensitivities
`d_dispatch_d_charge_eff` and `d_dispatch_d_discharge_eff`, which need the
equality duals recovered from the frozen-active-set KKT. These are what an
auto-calibration loop fits from battery telemetry.

**Dispatch-only telemetry identifies only the η_c · η_d product.** The SOC-trace
rows `d_soc_d_charge_eff` / `d_soc_d_discharge_eff` split charge from discharge
efficiency once observed SOC (BMS logs) is added to the residual. This is a
concrete example of the identifiability question the library is built to answer
honestly.

> `MultiBusStorageProblem` is solved as **one stacked QP over all periods**
> because SOC couples them. Keep the window small — representative days or
> weeks, T ≤ ~48. The dense stacked KKT is `O(((G+L+3S)·T)³)`.

---

## 15. MPC and auto-calibration

### Warm-started rolling horizons

`PersistentDispatchSession` does rolling-horizon LP re-solves **without
rebuilding the model**. `build()` solves once through the normal `optimise()`
path while capturing the assembled model, then loads it into a persistent HiGHS
instance; `advance()` pushes parameter changes straight into HiGHS columns and
rows and re-solves from the retained simplex basis.

```python
import numpy as np
from nexus_energy.mpc import PersistentDispatchSession

sess = PersistentDispatchSession(system)
base = sess.build()

res = sess.advance(
    demand={"elec": np.full(6, 350.0)},   # bus -> (T,) TOTAL demand
    cf={"wind": new_cf},                  # generator -> (T,) availability in [0,1]
    mc={"gas": 75.0},                     # generator -> scalar or (T,) marginal cost
    soc_init={"battery": 120.0},          # storage -> start SOC in MWh
)
print(sess.n_resolves, sess.n_rebuilds, sess.last_iterations)
```

A window re-solve costs a handful of simplex iterations instead of a rebuild
plus a cold solve.

**Honest scope** — `build()` raises otherwise: pure LP dispatch, so no
committable, integer-investment, switchable or piecewise-capex components;
fixed capacities (an extendable generator puts availability into a
`p ≤ cap_var · cf` matrix row this path cannot touch); and storage start-SOC
carry-over requires `soc_initial_free=True`, which makes the start SOC a
pinnable column. Anything out of scope at `advance()` time falls back to a full
rebuild — correct, just slower — and is counted in `n_rebuilds`.

### Keeping the model honest from telemetry

`autocal.fit_params` generalises the scalar CO₂-price recovery to an
m-dimensional parameter vector with a damped (scaled Levenberg-Marquardt)
Gauss-Newton driver. `AutoCalibrator` wraps it into a moving-horizon tracker so
a running MPC keeps correcting its own model.

```python
from nexus_energy.autocal import fit_params, AutoCalibrator

report = fit_params(
    make_solution,          # theta-dict -> SOLVED diff-layer solution
    observed,               # telemetry, flattened like the residual
    params={"eta_c": (0.95, 0.5, 1.0), "mc_gas": (50.0, 10.0, 200.0)},
    jacobian_fn=jac,        # (solution, theta) -> (len(r), m) analytic columns
    gate_threshold=1e-3,    # identifiability gate
    max_rel_step=0.25,      # per-run trust region
)
print(report.changed(), report.frozen)    # frozen[name] is True => data-silent

cal = AutoCalibrator(solution_factory, jac, observed_fn, params,
                     update_every=4, noise_std=0.5, smooth=0.3)
rep = cal.step(window)          # None on skipped cycles
cal.lock_param("eta_c")         # operator override
cal.set_param("mc_gas", 55.0, lock=True)
print(cal.believed)             # current point beliefs
```

Two design rules make this trustworthy in operations:

- **The identifiability gate** freezes any parameter whose scale-aware Jacobian
  column carries no signal in the window. It is flagged `data_silent` and never
  nudged by noise. This is the "it says so when the data is silent" property,
  made operational.
- **The per-cycle slew limit** (`max_rel_step`) bounds how far beliefs can move
  in one cycle, so a single bad window cannot wreck the model.

Noise-aware controls: `noise_std` enables EMA smoothing across windows
(`smooth` gain) and outright **rejection of outlier windows** whose post-fit
loss exceeds `outlier_zscore² · (½nσ²)` — a sensor fault or unmodelled event
leaves beliefs untouched. `residual_fn` switches the fitted observable between
dispatch shares, SOC traces and so on.

Any parameter chain-rulable from the exposed Jacobian blocks works: marginal
costs, fuel price and efficiency through `mc = fuel/η`, CO₂ price through
emissions, storage efficiencies through the efficiency blocks.

---

## 16. Solver controls

`EnergySystem.optimise()` exposes the full solver-control surface. Everything
here is accuracy-preserving.

| Kwarg | Default | Meaning |
|---|---|---|
| `lp_backend` | `None` | `auto` / `simplex` / `ipm` / `ipm_fast` / `pdlp` / `gpu`. `None` resolves: explicit argument > `.nexus_solver.json` sidecar > built-in default `"ipm_fast"`. Pure-LP only; MILP is never forced onto IPM. |
| `solver_method` | `None` | raw HiGHS `solver` option; always wins over `lp_backend`. `"ipm"` auto-forces `run_crossover="on"` so the result stays a bit-exact vertex. |
| `run_crossover` | `None` | `"on"` / `"off"` / `"choose"` — crossover recovers an exact vertex and duals |
| `parallel` | `None` | HiGHS parallel `"on"` / `"off"` / `"choose"`; auto-set `"on"` when `threads > 1` |
| `threads` | `None` | worker threads |
| `time_limit`, `gap` | `None` | stopping criteria |
| `scale_cleanup` | `True` | snap finite bounds/RHS with `0 < abs(v) < 1e-9` to exactly 0 |
| `simplex_scale_strategy` | `None` | HiGHS matrix-equilibration strategy, int 0–5 |
| `eliminate_redundant` | `True` | exact pre-solve pass dropping provably non-binding rows |
| `link_formulation` | `"auto"` | see [section 10](#10-networks-and-transmission) |
| `ramp_cost_formulation` | `"split"` | `"signed"` replaces each ramp-cost up/down aux pair with one aux variable `r ≥ ±Δ`; both price `abs(Δ)` identically at the optimum |
| `mip_strategy` | `"auto"` | see below |
| `uc_fix_schedule` | `None` | `{name: u-array}` pins `u[t]` for committable generators and links |
| `warm_start` | `None` | previous result forwarded to HiGHS `setSolution()` |
| `basis` | `None` | simplex basis hot-start |
| `myopic` | — | `MultiStageSystem.optimise()` only: no inter-stage foresight |

### `mip_strategy`

- **`"auto"`** (default) — for MIPs with ≤ 5000 estimated binaries, solve the LP
  relaxation first; if every binary lands within `1e-4` of {0,1} then the LP
  optimum *is* the MIP optimum and is returned directly (one LP, zero gap).
  Otherwise fall through to HiGHS MIP.
- **`"mip_only"`** — go straight to the MIP solver.
- **`"lp_first"`** — force the LP-first path at any size, on a vertex-producing
  backend (an interior point would spuriously look fractional). Exact by the
  LP-relaxation theorem; the LP cost is paid twice when fractional, hence
  opt-in.
- **`"fix_and_certify"`** — `lp_first` plus: on a fractional vertex, fix every
  `u[t]` that *is* integral there, solve the residual MIP, and return it when
  its cost is within `gap` of the relaxation bound — a valid optimality
  certificate. Falls back to the full MIP otherwise.

> **`ipm_fast` caveat.** It returns an interior (non-vertex) point, so
> degenerate duals and dispatch can differ from simplex, and basis warm-start
> is a no-op. Use `lp_backend="simplex"` when vertex duals or warm-start
> matter. A non-optimal `ipm_fast` result auto-falls back to simplex.

### Auto-tuning the LP method

Static shape heuristics proved unreliable — an early "≥ 50k columns ⇒ IPM" rule
regressed a real benchmark — so the tuner is **empirical**. It races `simplex`,
`ipm` and `ipm_fast` on reduced-horizon proxies of your system, checks the
winner is stable across two probe sizes (a clean simplex→IPM size-crossover
counts as stable), checks objective parity within each size, and writes a
`.nexus_solver.json` sidecar that `optimise()` then reads automatically.

```python
from nexus_energy.solver_tuner import tune_solver, recommend_lp_method

res = tune_solver(system)              # races, prints a report, writes the sidecar
print(res.recommended)                 # e.g. "ipm_fast"

# report-only / finer control:
res = recommend_lp_method(system, time_cap=120.0, threads=8,
                          need_duals=False, probe_hours=[365, 730],
                          verbose=True)
```

Library-agnostic — it works on any `from_pypsa` import. MILP systems
short-circuit to `simplex`, since IPM cannot solve branch-and-bound node LPs.
`need_duals=True` disqualifies `ipm_fast`. The sidecar is advisory: an explicit
`lp_backend=` always wins, and deleting the file reverts. Validated:
PyPSA-Eur → `ipm_fast`, CINDER → `simplex`.

---

## 17. Temporal aggregation

A full 8760-hour planning model is heavy. Cluster the year into representative
days instead.

```python
import nexus_energy as ne

rep = ne.aggregate_to_representative_days(
    timeseries={"demand": demand_8760, "solar_cf": solar_8760},
    n_days=12,
    hours_per_day=24,
    seed=42,
    extreme_periods=[("demand", "max"), ("solar_cf", "min")],
)

result = ne.apply_representative_days(
    system, rep,
    timeseries_map={"demand": "demand", "solar": "solar_cf"},
)
```

`extreme_periods` is important and easy to skip: pure k-medoids clustering
discards the peak-load and dark-doldrums days that determine system adequacy.
Naming them explicitly keeps them in the reduced set.

`aggregate_with_feature_embedding(timeseries, n_days=…, features=…)` embeds
load, capacity factors and ramp statistics in a feature space before running
k-medoids, which keeps the day count low while preserving the correlation
structure between wind and solar dropouts.

**Certified reduction bounds.** `certify_reduction` returns a `CertifiedBound`
— a provable two-sided bound on the cost error a temporal aggregation
introduces, turning "the representative days look close" into a certificate.
See `certified_reduction_demo` for a worked example.

**Variable resolution.** `ResolutionPlan`, `adaptive_resolution_plan`,
`multi_resolution_hierarchy` and `apply_adaptive_resolution` build non-uniform
timestep plans; `set_snapshot_durations()` applies per-snapshot hour lengths
directly, so a coarse merged block correctly moves `power × duration` of energy.

**Long-duration storage.** Set `long_duration=True` on a storage and, when the
system has representative periods plus a chronological mapping,
SOC = `soc_intra[t]` + `soc_inter[original_day]`, tracking carry-over across
the real calendar (Kotzur 2018 inter-period superposition).

---

## 18. Decomposition at scale

### Certified temporal decomposition

`optimise_temporal_certified` solves a long-horizon system as K temporal blocks
**with a global optimality certificate** — the same `(UB − LB)/UB ≤ gap`
contract a monolithic MIP solver gives, at an order of magnitude less wall time.
MILP wall clock scales roughly `T^2.5`, so cutting T is worth a great deal.

```python
from nexus_energy.temporal_certified import optimise_temporal_certified

res = optimise_temporal_certified(
    system_factory,            # callable(t0, t1) -> EnergySystem over [t0, t1)
    total_steps=8760,
    n_blocks=8,
    gap=1e-2,
    lb_blocks=4,               # fewer, longer LB blocks => tighter bound
    lb_rounds=3,               # subgradient ascent rounds on boundary prices
    lb_workers=4,              # process-parallel LB blocks (factory must pickle)
    ub_boundary="both",        # "prices" | "floors" | "both" | "none"
    reachability_envelopes=True,
)
print(res)   # TemporalCertifiedResult(status='certified', objective=…, gap=…)
```

- **Lower bound** — Lagrangian dual decomposition over interior SOC boundaries.
  Each block is a *relaxation* (free interior start SOC, telescoping ±λ boundary
  prices), so the sum of block dual bounds is ≤ the full optimum for arbitrary
  λ. `lb_rounds` refines λ by projected subgradient and the best round is
  reported.
- **Upper bound** — sequential block stitching, handing each block its
  predecessor's terminal SOC. `ub_boundary="prices"` prices terminal energy at
  −λ and subtracts the payment back out, so the UB is the stitched trajectory's
  *true* full-model cost.
- **Reachability envelopes** cap each LB block's free start and terminal SOC by
  what the full problem could physically have stored by that boundary — pure
  bound tightening, never validity-affecting.

Supported scope, guarded with `ValueError`: non-cyclic, non-LDS, non-extendable
storages; committable links with min up/down ≤ 1; no hard ramp limits; no
committable generators; uniform snapshot weights and durations.

Measured on the CINDER MILP: +0.91 % from parity at 58 s (6.1× faster than
monolithic), and beating the incumbent at 87 s.

### Benders

For networks too large for one machine, `BendersDecomposer` splits investment
(master) from dispatch (sub-problems) and solves the sub-problems in parallel.

```python
decomposer = ne.BendersDecomposer(
    system=sys,
    periods=[(0, 2190), (2190, 4380), (4380, 6570), (6570, 8760)],
    stabilisation="level",     # note the British spelling
    max_iter=30,
    tol=1e-3,
    n_jobs=4,
)
result = decomposer.solve()
```

Stabilisation prevents the sub-gradient oscillation that makes textbook Benders
crawl. Other drivers: `solve_with_temporal_benders`,
`solve_with_spatial_benders`, `solve_with_nested_benders`,
`solve_with_dantzig_wolfe`, `solve_with_column_generation`, and
`recommend_decomposition` to choose among them.

`rolling_horizon_solve(system_factory, total_timesteps, window_size, overlap)`
threads the simplex basis between windows for hot-started re-solves.

> Note: a `StageProblem` class exists in both `decomposition` and `stochastic`.
> The top-level aliases are `NestedStageProblem` and `SDDiPStageProblem`.

---

## 19. Uncertainty — stochastic and robust

### Two-stage stochastic programming

```python
scenarios = [
    ne.Scenario(name="low",  probability=0.25, demand_factor=0.8),
    ne.Scenario(name="base", probability=0.50, demand_factor=1.0),
    ne.Scenario(name="high", probability=0.25, demand_factor=1.3),
]

result = ne.solve_stochastic(sys, scenarios,
                             risk_measure="expected",   # or "cvar"
                             cvar_alpha=0.05,
                             method="benders")
```

`Scenario` fields are `name`, `probability`, `demand_factor`,
`carrier_factor_scale`, `fuel_cost_factor` and a free-form `overrides` dict.

### Robust optimisation

```python
uncertainty = ne.BudgetUncertaintySet(
    demand_up=0.15,      # demand can be 15% higher
    cf_down=0.30,        # capacity factors can be 30% lower
    fuel_cost_up=0.20,
    budget=2.0,          # Bertsimas-Sim uncertainty budget
)
result = ne.solve_robust(sys, uncertainty)
```

The `budget` is how many deviations may occur simultaneously — the classic
Bertsimas-Sim Γ. At `budget=0` you recover the deterministic problem; at the
full dimension you get the box-worst case.

### Chance constraints

```python
cc = ne.ChanceConstraint(name="reserve", alpha=0.05, threshold=0.0)
result = ne.solve_saa_chance_constrained(sys, scenarios, alpha=0.05,
                                         reserve_margin=0.15)
```

`alpha` is the violation probability, so `alpha=0.05` is a 95 % reliability
level.

### The extended toolkit

`solve_sddip` (multi-stage SDDiP), `solve_general_chance_constrained`,
`solve_wasserstein_dro` (distributionally robust over Wasserstein balls),
`solve_risk_averse_benders`, `generate_forced_outage_scenarios`,
`generate_demand_scenarios`, `generate_renewable_scenarios`,
`generate_moment_matching_scenarios`, `reduce_scenarios` and
`reduce_scenarios_wasserstein` (optimal-transport scenario reduction).

---

## 20. AC power flow

Linear transport models are insufficient for transmission studies that care
about voltage and reactive power.

### SOCP relaxation

The second-order cone (Jabr) relaxation is convex, so its optimum is a global
bound on the true AC optimum.

```python
res = ne.solve_socp_opf(
    system,
    snapshot=0,
    enable_obbt=True,        # optimisation-based bound tightening
    obbt_iters=3,
    obbt_tol=1e-4,
    cos_envelope_pieces=8,
    enforce_cycle_closure=False,
    enforce_tight_qc=False,
)
```

OBBT iteratively shrinks bounds on voltage magnitudes and angle differences,
tightening the relaxation on networks where plain Jabr is loose.

### Polar AC-OPF

```python
res = ne.solve_ac_opf_polar(system, snapshot=0, slack_bus="bus1")
```

The true non-convex formulation. It reflects real reactive power, at the cost
of being susceptible to local optima.

| Use | Choose |
|---|---|
| transmission **planning** | SOCP + OBBT — convex, gives a defensible envelope |
| operations **analysis** | polar AC-OPF — real non-linear physics |

Conic extensions: `solve_socp_opf_expansion` (capacity expansion on the Jabr
relaxation), `solve_socp_opf_multi` (multi-snapshot), `add_weymouth_pipe` (gas
network Weymouth relaxation) and `add_head_dependent_hydro`.

---

## 21. ML-guided solving

To accelerate rolling-horizon MPC, learned predictors warm-start the UC
binaries so the MILP does not start cold.

```python
sys_feat = ne.extract_system_features(system)
t_feat   = ne.extract_timestep_features(system)

prediction = ne.predict_unit_commitment(...)
schedule   = ne.warm_start_from_prediction(prediction,
                                           confidence_threshold=0.7,
                                           cold_start_fallback=True)
result = system.optimise(uc_fix_schedule=schedule)
```

Predictors: `MeritOrderPredictor` (cheap heuristic baseline),
`HistoricalNeighborPredictor` (k-NN over past solves), `GNNPredictor` (graph
neural network, torch optional), all behind `UCWarmstartPredictor`.

`confidence_threshold` is the safety dial — only predictions above it are
pinned, and `AdaptiveThresholdController` / `solve_with_adaptive_warmstart`
tune it automatically from observed hit rate. Also available: `LearnedVarFixer`
/ `apply_varfix`, `RLVarFixer` / `solve_with_rl_search`, and
`learned_representative_periods` / `feature_embedding_periods`.

---

## 22. The component library

223 components across 15 sectors, each available at several **fidelity** levels
— fidelity being how much physics the component carries. The same LFP cell is
available as:

| Level | Model | What it adds |
|---|---|---|
| `F0a` | round-trip efficiency curve | one lookup versus C-rate; cheapest |
| `F1a` | state of charge | energy in, energy out, SoC over time |
| `F1b` | + thermal | cell temperature moves the efficiency |
| `F1c` | + degradation | capacity fade over cycles |
| `F2a` | equivalent circuit (1RC) | real voltage dynamics |
| `F2d` | single-particle model | electrochemical detail |

**Pick the cheapest level that still answers the question — per component, not
per study.** A capacity screen can run everything at F0 while the one asset
under investigation runs at F2. That is the point of the axis.

Sector split: thermal 41 · power electronics 32 · batteries 26 · solar 18 ·
hydrogen 17 · conventional 13 · hydro and marine 12 · biomass 11 · carbon
capture 11 · gas systems 9 · thermoelectric 8 · desalination 7 · wind 6 ·
mechanical storage 6 · geothermal 6.

F0–F2 are built for all 223. **F3–F6 (distributed physics, AI surrogates,
PINNs) are scaffolded but not yet implemented.**

Access templates through the registry, and compose them with carrier checking:

```python
ne.registry.list_components()
template = ne.registry.get("lfp_cell_f1a")

from nexus_energy.components.composition import Subsystem, FIDELITY_LEVELS
```

`Subsystem` raises `CarrierMismatchError` when you wire a hydrogen output into
an electricity input.

---

## 23. Benchmarks

Full detail in [`COMPARISON_SCORECARD.md`](COMPARISON_SCORECARD.md); stored
results under [`benchmarks/results/`](benchmarks/results), so every number can
be re-run rather than taken on trust.

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
(a static scalar `p_max_pu` being ignored, and cyclic storage being
over-pinned).

### Where the speed comes from

1. **Rust constraint assembly.** Vectorised Python is the modelling speed limit
   in comparable libraries. Sparse CSC constraints are streamed to the solver
   from a PyO3 Rust kernel instead.
2. **Tighter-by-default UC formulations.** LP-tight 3-bin commitment reduces
   branch-and-bound work with no user-visible API change.
3. **Empirical LP-method selection.** The `solver_tuner` races simplex and IPM
   variants on reduced-horizon proxies of *your* problem and locks the winner
   in a sidecar; `ipm_fast` is the lean default for well-conditioned expansion
   LPs, while staircase MILPs keep warm-startable simplex.
4. **Feature-guided clustering.** Embedding load, capacity factors and ramp
   statistics before k-medoids keeps representative-day counts low while
   preserving the extreme periods that determine adequacy.
5. **In-place re-solves.** `PersistentDispatchSession` replaces rebuild plus
   cold solve with a parameter update and a hot-started simplex re-solve.

### An open problem, stated as a bug

Two GenX cases (`rate_co2` at −42.7 %, `mincapreq` at −3.90 %) look like large
wins and are **not**. Feeding the capacity nexus chose back into a real GenX
solve showed GenX's own cost for the nexus solution (5.823e9) essentially
matches GenX's optimum (5.808e9), while nexus *reports* 5.582e9 — a **~4.3 %
OPEX under-count bug here**, not a cheaper optimum. Lead suspect is
transmission-loss modelling (GenX piecewise-linear versus a linear `loss=%`).

The pattern: exact match when the build is determined, opex under-count when
multi-zone renewables are dispatched over transmission. The exact-parity rows
above are unaffected — none of them involve that case.

---

## 24. Honest scope and known limits

- **Not dynamic or EMT simulation.** No transients, no swing equation. Time is
  snapshots and rolling horizons coupled by algebraic constraints — a different
  category from Simulink and Modelica.
- **No integer (UC-MILP) differentiability.** Future work, not a claim.
- **Calibration is LP/QP class.** The small ridge term required to make the
  gradients well defined shifts economics by roughly 1–7 percentage points and
  is disclosed per result.
- **The speed headline is forward-only.** The calibration solve uses a denser
  path.
- **F3–F6 component fidelities are scaffolded, not implemented.**
- **`lp_backend="gpu"` requires cuOpt and a CUDA device**, and falls back to
  the CPU path with a printed warning when either is missing. It is not the
  source of the measured speed numbers above.
- **A first `optimise(threads=N)` call in a fresh process can return
  `status="unknown"`** — this comes from the HiGHS global-scheduler reset in
  the solver core. Check `status` before reading `total_cost`.

Third-party solvers and frameworks appear only as benchmark comparison rows.
They are never wrapped inside the library. `external_solvers` provides an
LP-export bridge towards Gurobi / CPLEX / SCIP / Mosek / Xpress for comparison
runs; `optimise()` itself rejects external solver names with a pointer to it.

---

## 25. Troubleshooting

**`status` is `"infeasible"`.** Demand cannot be met. Check that generation
capacity times availability covers peak load at every timestep, that links have
enough capacity to reach the load, and that a hard policy constraint
(`set_emission_limit`, `set_rps`) is not impossible to satisfy. Policy setters
accept `slack_penalty=` to convert a hard constraint into a priced soft one,
which turns an infeasibility into a diagnosable cost.

**Nothing gets built in a capacity-expansion model.** `capital_cost` is
`$/MW/year`. Weight your snapshots — see the warning in
[section 9](#9-capacity-expansion).

**`total_cost` is 0.0.** Everything was served by zero-marginal-cost
generation. That is usually correct, not a bug.

**Dispatch shows tiny negative numbers** like `-1e-11`. Solver tolerance.
Round before display.

**Shadow prices look wrong or degenerate.** `ipm_fast` (the default LP backend)
returns an interior point. Use `lp_backend="simplex"` when you need vertex
duals.

**A MILP is slow.** Try `mip_strategy="lp_first"`, cut the horizon with
representative periods ([section 17](#17-temporal-aggregation)), use
`clustered=True` for identical thermal units, or decompose
([section 18](#18-decomposition-at-scale)).

**The differentiable bridge raises.** By design — it never silently
approximates. Re-import with `from_pypsa(n, line_model="transport")`, fix all
capacities, and remove storage, UC and lossy links. See
[section 14](#14-differentiable-dispatch-and-inverse-calibration).

**A calibrated parameter never moves.** It was frozen by the identifiability
gate: the data in that window carries no signal about it. Check
`report.frozen`. This is the library working as intended, not a failure.

---

## 26. API index

### EnergySystem

**Building:** `add_bus` · `add_carrier` · `add_generator` · `add_load` ·
`add_storage` · `add_link`

**Time:** `set_timesteps` · `set_snapshot_weights` · `set_snapshot_durations` ·
`set_chronological_mapping`

**Policy:** `set_co2_price` · `set_emission_limit` · `set_co2_rate_cap` ·
`set_co2_zone_cap` · `set_co2_cap_group` · `set_rps` · `set_ces` · `set_itc` ·
`set_ptc` · `set_hourly_matching` · `set_capacity_bucket` ·
`set_fuel_supply_limit`

**Reliability:** `set_n_minus_1` · `set_reserve_margin` ·
`set_contingency_reserve` · `set_spinning_reserve` · `set_regulation_reserve` ·
`set_outage` · `set_shared_capacity`

**Solving and inspection:** `optimise` · `summary` · `n_buses` ·
`n_components` · `n_timesteps` *(the last three are properties)*

### Planning

`MultiStageSystem` (`add_stage`, `optimise`) · `MultiStageResult` · `annuity`

### Temporal

`aggregate_to_representative_days` · `aggregate_with_feature_embedding` ·
`apply_representative_days` · `RepresentativePeriods` ·
`representative_period_error` · `k_medoids` · `rolling_horizon_solve` ·
`ResolutionPlan` · `adaptive_resolution_plan` · `multi_resolution_hierarchy` ·
`apply_adaptive_resolution` · `certify_reduction` · `CertifiedBound`

### Decomposition

`BendersDecomposer` · `solve_with_temporal_benders` ·
`solve_with_spatial_benders` · `solve_with_nested_benders` ·
`solve_with_dantzig_wolfe` · `solve_with_column_generation` ·
`temporal_decomposition` · `recommend_decomposition` ·
`temporal_certified.optimise_temporal_certified`

### Uncertainty

`Scenario` · `solve_stochastic` · `solve_stochastic_ph` · `solve_robust` ·
`BudgetUncertaintySet` · `ChanceConstraint` · `solve_saa_chance_constrained` ·
`solve_general_chance_constrained` · `solve_sddip` · `solve_wasserstein_dro` ·
`solve_risk_averse_benders` · `evaluate_plan` · `reduce_scenarios` ·
`reduce_scenarios_wasserstein` · `generate_demand_scenarios` ·
`generate_renewable_scenarios` · `generate_forced_outage_scenarios` ·
`generate_moment_matching_scenarios` · `cvar_change_of_measure`

### Power flow

`solve_socp_opf` · `solve_socp_opf_multi` · `solve_socp_opf_expansion` ·
`solve_ac_opf_polar` · `obbt_tighten` · `add_weymouth_pipe` ·
`add_head_dependent_hydro`

### Differentiable

`solve_dispatch_with_sensitivities` · `EconomicDispatchLayer` ·
`MultiBusDispatchProblem` · `solve_multibus_dispatch_with_sensitivities` ·
`StorageDispatchProblem` · `solve_storage_dispatch_with_sensitivities` ·
`MultiBusStorageProblem` · `SmoothCommitmentLayer` ·
`fit_commitment_threshold` · `fit_demand_elasticity` ·
`CapacityExpansionProblem` · `CapacityExpansionLayer` ·
`solve_capacity_expansion_with_sensitivities` · `fit_component_params` ·
`TorchDispatchLayer`

### Modules

`nexus_energy.pypsa_compat` — `from_pypsa`
`nexus_energy.diff_bridge` — `multibus_problem_from_system` ·
`d_dispatch_d_co2_price` · `fit_co2_price`
`nexus_energy.autocal` — `fit_params` · `AutoCalibrator`
`nexus_energy.mpc` — `PersistentDispatchSession`
`nexus_energy.solver_tuner` — `tune_solver` · `recommend_lp_method`
`nexus_energy.temporal_certified` — `optimise_temporal_certified`
`nexus_energy.external_solvers` · `nexus_energy.io_tables`

### ML-guided

`extract_system_features` · `extract_timestep_features` ·
`UCWarmstartPredictor` · `MeritOrderPredictor` ·
`HistoricalNeighborPredictor` · `GNNPredictor` · `predict_unit_commitment` ·
`warm_start_from_prediction` · `LearnedVarFixer` · `apply_varfix` ·
`AdaptiveThresholdController` · `solve_with_adaptive_warmstart` · `RLVarFixer` ·
`solve_with_rl_search` · `learned_representative_periods` ·
`feature_embedding_periods`

### Sector coupling and components

`create_power_to_hydrogen` · `create_heat_system` · `create_power_to_gas` ·
`create_temperature_heat_network` · `create_multi_carrier_system` ·
`ComponentTemplate` · `ComponentRegistry` · `registry` · `add_component` ·
`Subsystem` · `CarrierMismatchError`

---

## Licence

MIT. See `LICENSE`.
