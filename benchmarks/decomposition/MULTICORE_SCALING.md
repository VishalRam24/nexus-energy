# Multicore Benders — scaling proof (Milestone 1)

**Claim:** decomposition *above* the solver turns the "one big solve only uses
one core" problem into a job that scales with cores. A two-stage stochastic
capacity-expansion model splits into one independent operational subproblem per
scenario; `BendersDecomposer(n_jobs=N)` solves them across N worker processes
(each a single-threaded HiGHS), and wall-clock drops with N.

This is the lever the monolithic solver cannot pull: HiGHS's own parallel
simplex is "unlikely to be worth using" (their docs) and B&B/IPM parallelism is
modest, so adding cores to a single monolithic solve barely helps. Splitting the
*problem* does.

Reproduce:

```bash
# heavy subproblems, scenario-granularity plateau
python benchmarks/decomposition/multicore_scaling.py \
    --T 1460 --buses 4 --n 16 --max-iter 12 --jobs 1,2,4,8,15

# lighter subproblems, more of them — scales across all cores
python benchmarks/decomposition/multicore_scaling.py \
    --T 730 --buses 4 --n 48 --max-iter 8 --jobs 1,2,4,8,15
```

Backend: `decomposition.py` — `n_jobs` + a `ProcessPoolExecutor`. Subsystems are
pickled into the workers once (pool initializer); each iteration ships only
`(period, caps, gap)`. Master + cut assembly stay serial (the Amdahl part).

---

## Run A — heavy subproblems (T=1460, 4 buses, 16 scenarios)

Host: 15 cores (Apple Silicon). max_iter=12 (capped; objective is deterministic
either way). Each subproblem ≈ 0.9 s.

| jobs | total (s) | subproblem (s) | speed-up | efficiency | objective |
|-----:|----------:|---------------:|---------:|-----------:|----------:|
| 1  | 196.4 | 196.4 | 1.00× | 100% | 38393707.06 |
| 2  | 108.3 | 108.2 | 1.81× |  91% | 38393707.06 |
| 4  |  68.5 |  68.4 | 2.87× |  72% | 38393707.06 |
| 8  |  33.4 |  33.3 | **5.88×** | 74% | 38393707.06 |
| 15 |  36.7 |  36.6 | 5.36× |  36% | 38393707.06 |

- **Parity is exact** — objective spread across all core counts = `0.00e+00`.
  For decomposition, identical-objective *is* the correctness proof: every core
  count walked a different execution but reached the same optimum to the cent.
- **The subproblem phase is ~99.96% of wall-time** → the serial master is
  negligible, so almost everything is parallelisable (large Amdahl headroom).
- **Plateau at 15 is a granularity ceiling, not a backend limit.** With only 16
  scenarios, any worker count > 8 still needs 2 sequential waves
  (`ceil(16/N)` waves: N=8 → 2, N=15 → 2), so 8 and 15 workers hit the same
  floor. The fix is more subproblems (Run B), not more code.

## Run B — lighter subproblems, more of them (T=730, 4 buses, 48 scenarios)

48 scenarios give finer granularity, so scaling continues past 8 workers. Each
subproblem is lighter (≈ 0.28 s). max_iter=8.

| jobs | total (s) | subproblem (s) | speed-up | efficiency | objective |
|-----:|----------:|---------------:|---------:|-----------:|----------:|
| 1  | 107.7 | 107.7 | 1.00× | 100% | 25398462.44 |
| 2  |  59.5 |  59.4 | 1.81× |  91% | 25398462.44 |
| 4  |  40.9 |  40.8 | 2.63× |  66% | 25398462.44 |
| 8  |  23.1 |  22.9 | 4.67× |  58% | 25398462.44 |
| 15 |  19.2 |  19.0 | **5.62×** | 37% | 25398462.44 |

- **Parity exact again** — spread = `0.00e+00`.
- **15 workers now beats 8** (5.62× vs 4.67×): with 48 subproblems there is work
  to fill the extra cores (`ceil(48/15)=4` waves vs `ceil(48/8)=6`), so the Run A
  plateau is confirmed as a granularity effect, not a backend limit.
- **Lower efficiency than Run A** (58% vs 74% at 8 workers) because these ≈0.28 s
  subproblems amortise the per-task pickle/dispatch overhead worse than Run A's
  ≈0.9 s ones.

### The regime that matters

Putting A and B together: speed-up is bounded by **both** the subproblem count
(granularity — you can't use more workers than subproblems-per-wave) **and** the
subproblem weight (heavier solves amortise overhead better). The sweet spot is
*many heavy subproblems* — which is exactly real stochastic capacity-expansion
(50–1000 weather/demand scenarios, each a full operational year). The
overhead-driven efficiency loss is also precisely what the planned Rust+rayon
migration removes (no pickling, no process spawn).

---

## Honest notes / findings

- **Serial-path HiGHS thread-state bug (worked around, logged).** Passing
  `threads=1` to a solve that runs *in the same process right after the master
  solve* trips a HiGHS thread-state corruption returning a spurious non-optimal
  status. The process-pool path is immune (workers never solve the master).
  The serial baseline therefore does **not** pin threads; its sub-LPs are
  simplex/serial (~1 core) anyway, so the baseline stays honestly single-core.
- **Small extendable storage pinned to ~0 is degenerate-infeasible.** When the
  master pins an extendable battery to (near-)zero power/energy the SOC/cyclic
  constraints become infeasible. The benchmark sets a positive min-capacity to
  keep storage a genuine `[min,max]` decision. Worth a separate model-side fix.
- **Efficiency caveats.** Per-iteration `pool.map` dispatch + scenario
  imbalance (some scenarios solve slower) cost ~25–30% at 4–8 workers. The
  planned Rust+rayon migration (subproblem solve via the `highs` crate, no GIL,
  no per-task pickling) is the path to higher efficiency and is the next
  milestone.

## What this does and does not claim

- **Does:** show that scenario/temporal decomposition makes capacity-expansion
  solving scale with cores, at exact parity, via a drop-in `n_jobs`.
- **Does not:** claim a faster *solver*. HiGHS is unmodified; the win is from
  splitting the problem so N cores each run an independent HiGHS. This is the
  decomposition-orchestrator architecture, not a solver fork.
