"""
N_En_Phase 17.3 — AC-OPF SOCP gap vs pandapower NLP on IEEE test cases.

Runs per case (default: case9 + case14):
  1) pandapower's `runopp` (PIPS-OPF interior-point NLP) with quadratic
     cost terms zeroed out (costs linearised) — the reference AC-OPF.
  2) nexus `solve_socp_opf` on the *same* case, built from pandapower's
     internal MATPOWER-format `_ppc` dict so both sides see identical
     per-unit admittances, taps, shunts, and bounds.

Reports the SOCP relaxation gap (lower bound ≤ NLP optimum — a tight
Jabr lift gives a small gap) + wall-clock + speedup.

pandapower runs in the isolated venv at
`test_projects/test_project_1/pandapower/.venv` (pandapower 2.x, numpy<2)
via subprocess. The nexus side runs in the nexus-energy venv.

Added 2026-04-20: case14 support via the new transformer-tap + bus-shunt
wiring in `solve_socp_opf` (N_En_Phase 10.3).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import nexus_energy as ne


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Reference pandapower lives in its own isolated venv outside the package.
# Override with NEXUS_PANDAPOWER_DIR if you keep the clone elsewhere.
PANDAPOWER_DIR = Path(
    os.environ.get(
        "NEXUS_PANDAPOWER_DIR",
        _REPO_ROOT / "test_projects/test_project_1/pandapower",
    )
)
PANDAPOWER_PY = PANDAPOWER_DIR / ".venv/bin/python"


# ---------------------------------------------------------------------------
# pandapower reference run (subprocess to isolated venv) — also returns
# the internal MATPOWER ppc dict so nexus can rebuild the exact per-unit
# admittance model.
# ---------------------------------------------------------------------------


_PANDAPOWER_SCRIPT = r"""
import json, sys, time
import numpy as np
import pandapower as pp
import pandapower.networks as nw

case_name = sys.argv[1]
net = getattr(nw, case_name)()

# N_En_Phase 10.4: nexus SOCP now supports both linear (cp1) and
# quadratic (cp2) polynomial cost terms via
# `generator.marginal_cost` / `generator.quadratic_cost`, so both
# sides can now run on the *full* MATPOWER cost function (not just
# the linearised version). Previous pre-10.4 builds zeroed cp2 here
# so both sides matched; that is no longer necessary.

# Warm-up, then timed solve.
pp.runopp(net, verbose=False, suppress_warnings=True)
t0 = time.perf_counter()
pp.runopp(net, verbose=False, suppress_warnings=True)
wall = time.perf_counter() - t0

ppc = net._ppc
# Branch arrays are complex-typed internally (Y-bus has complex admittances
# in later columns) — the leading per-unit columns we need are real-valued.
# Take .real and keep only the MATPOWER canonical columns.
bus = np.asarray(ppc["bus"], dtype=float)[:, :13].tolist()
branch = np.asarray(ppc["branch"].real, dtype=float)[:, :13].tolist()
gen = np.asarray(ppc["gen"], dtype=float)[:, :10].tolist()
gencost = np.asarray(ppc["gencost"], dtype=float).tolist()

out = dict(
    case=case_name,
    baseMVA=float(ppc["baseMVA"]),
    ppc=dict(bus=bus, branch=branch, gen=gen, gencost=gencost),
    objective=float(net.res_cost),
    solve_time_s=wall,
    bus_vm_pu={int(k): float(v) for k, v in net.res_bus["vm_pu"].items()},
    gen_p_mw={int(k): float(r.p_mw) for k, r in net.res_gen.iterrows()},
    gen_q_mvar={int(k): float(r.q_mvar) for k, r in net.res_gen.iterrows()},
    ext_grid_p_mw={int(k): float(r.p_mw) for k, r in net.res_ext_grid.iterrows()},
    ext_grid_q_mvar={int(k): float(r.q_mvar) for k, r in net.res_ext_grid.iterrows()},
    total_line_losses_mw=float(net.res_line["pl_mw"].sum()),
    total_trafo_losses_mw=(
        float(net.res_trafo["pl_mw"].sum()) if len(net.trafo) > 0 else 0.0
    ),
)
print("__PM_JSON__" + json.dumps(out))
"""


def pandapower_available() -> bool:
    """True only if the isolated pandapower venv interpreter exists."""
    return PANDAPOWER_PY.exists()


def run_pandapower(case: str) -> dict:
    proc = subprocess.run(
        [str(PANDAPOWER_PY), "-c", _PANDAPOWER_SCRIPT, case],
        capture_output=True, text=True, check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__PM_JSON__"):
            return json.loads(line[len("__PM_JSON__"):])
    raise RuntimeError(f"pandapower JSON not found:\n{proc.stdout}\n{proc.stderr}")


# ---------------------------------------------------------------------------
# Nexus-side rebuild from the MATPOWER ppc dict
# ---------------------------------------------------------------------------

# MATPOWER column indices (0-based; sliced to first N canonical cols above).
# bus:    BUS_I=0 BUS_TYPE=1 PD=2 QD=3 GS=4 BS=5 AREA=6 VM=7 VA=8 BASE_KV=9
#         ZONE=10 VMAX=11 VMIN=12
# branch: F_BUS=0 T_BUS=1 BR_R=2 BR_X=3 BR_B=4 RATE_A=5 RATE_B=6 RATE_C=7
#         TAP=8 SHIFT=9 STATUS=10 ANGMIN=11 ANGMAX=12
# gen:    GEN_BUS=0 PG=1 QG=2 QMAX=3 QMIN=4 VG=5 MBASE=6 STATUS=7 PMAX=8 PMIN=9
# gencost: MODEL=0 STARTUP=1 SHUTDOWN=2 NCOST=3 COST[4..4+NCOST]


def build_nexus_from_ppc(data: dict) -> tuple[ne.EnergySystem, float]:
    """
    Build a nexus `EnergySystem` that mirrors pandapower's internal ppc.

    Returns ``(system, cp0_sum)`` — `cp0_sum` is the sum of constant cost
    coefficients across generators (pandapower's objective includes them,
    the nexus linear objective does not; add after the solve to compare).
    """
    ppc = data["ppc"]
    base_mva = data["baseMVA"]
    sys = ne.EnergySystem(f"{data['case']}_socp")

    # ---- Buses + loads + bus shunts ----
    buses: dict[int, ne.Bus] = {}
    for row in ppc["bus"]:
        bus_i = int(row[0])
        pd = float(row[2])    # MW
        qd = float(row[3])    # Mvar
        gs = float(row[4])    # MW at V=1 pu ⇒ pu admittance = gs / baseMVA
        bs = float(row[5])    # Mvar at V=1 pu
        vmax = float(row[11])
        vmin = float(row[12])
        b = sys.add_bus(f"b{bus_i}", carrier="electricity")
        b.v_min = vmin
        b.v_max = vmax
        if gs != 0.0:
            b.g_shunt = gs / base_mva
        if bs != 0.0:
            b.b_shunt = bs / base_mva
        if pd != 0.0 or qd != 0.0:
            ld = sys.add_load(f"d_b{bus_i}", bus=b, amount=pd / base_mva)
            ld.q_amount = qd / base_mva
        buses[bus_i] = b

    # ---- Branches (lines + transformers; MATPOWER merges them) ----
    for i, row in enumerate(ppc["branch"]):
        status = int(row[10])
        if status == 0:
            continue
        f_bus = int(row[0])
        t_bus = int(row[1])
        br_r = float(row[2])         # pu
        br_x = float(row[3])         # pu
        br_b = float(row[4])         # total line-charging pu (split half/end)
        rate_a = float(row[5])       # MVA thermal limit; 0 ⇒ unlimited
        tap = float(row[8])          # 0 ⇒ no tap (treat as 1)
        shift_deg = float(row[9])
        if tap == 0.0:
            tap = 1.0
        s_max_pu = (rate_a / base_mva) if rate_a > 0 else 100.0
        l = sys.add_link(f"br{i}", bus_from=buses[f_bus], bus_to=buses[t_bus],
                         capacity=s_max_pu)
        l.resistance = br_r
        l.reactance = br_x
        l.b_fr = br_b / 2.0
        l.b_to = br_b / 2.0
        l.g_fr = 0.0
        l.g_to = 0.0
        if tap != 1.0:
            l.tap = tap
        if shift_deg != 0.0:
            l.shift = math.radians(shift_deg)
        # ANGMIN / ANGMAX (degrees); MATPOWER ships ±360° as the "effectively
        # unbounded" sentinel on the IEEE cases, so only honor them if they
        # fall in a realistic range. Branches without explicit bounds fall
        # through to the function-level `angle_diff_max` default.
        angmin_deg = float(row[11])
        angmax_deg = float(row[12])
        if -85.0 < angmin_deg < 0.0:
            l.angle_diff_min = math.radians(angmin_deg)
        if 0.0 < angmax_deg < 85.0:
            l.angle_diff_max = math.radians(angmax_deg)
        l.s_max = s_max_pu
        l.model_type = "socp_opf"

    # ---- Generators (ppc["gen"] includes ext_grid + PV gens) ----
    cp0_sum = 0.0
    for i, g_row in enumerate(ppc["gen"]):
        status = int(g_row[7])
        if status == 0:
            continue
        g_bus = int(g_row[0])
        qmax = float(g_row[3])
        qmin = float(g_row[4])
        pmax = float(g_row[8])
        pmin = float(g_row[9])

        cp2 = 0.0
        cp1 = 0.0
        cp0 = 0.0
        if i < len(ppc["gencost"]):
            cost_row = ppc["gencost"][i]
            model = int(cost_row[0])   # 2 = polynomial, 1 = PWL
            ncost = int(cost_row[3])
            coeffs = cost_row[4:4 + ncost]
            if model == 2:
                # Polynomial descending: cN, ..., c1, c0 where cost = Σ cK·p^K
                if ncost == 3:
                    cp2, cp1, cp0 = (float(coeffs[0]), float(coeffs[1]),
                                     float(coeffs[2]))
                elif ncost == 2:
                    cp1, cp0 = float(coeffs[0]), float(coeffs[1])
                elif ncost == 1:
                    cp0 = float(coeffs[0])
        cp0_sum += cp0

        g = sys.add_generator(f"gen{i}", bus=buses[g_bus],
                              capacity=pmax / base_mva,
                              marginal_cost=cp1 * base_mva)
        # p_min may be negative (ext_grid can export); SOCP builder supports it.
        g.p_min = pmin / base_mva
        g.q_min = qmin / base_mva
        g.q_max = qmax / base_mva
        # Quadratic cost (N_En_Phase 10.4). MATPOWER cp2 is in $/MW²; nexus
        # p is in pu, so cost = cp2 · (p_pu · base_mva)² = cp2·base_mva²·p_pu².
        if cp2 != 0.0:
            g.quadratic_cost = cp2 * base_mva * base_mva

    return sys, cp0_sum


def run_nexus(data: dict, angle_diff_max: float = math.pi / 6,
              formulation: str = "socp") -> dict:
    # angle_diff_max = 30° default (N_En_Phase 10.10 SOCP+AT tightening):
    # the Jabr per-branch rotated-cone alone admits dispatches with
    # physically impossible angle spread across mesh loops. ±30° is the
    # standard transmission stability window — well above any AC-OPF-
    # feasible solution on IEEE case9/14/30/118 (all ≤ 15° at optimum)
    # and tight enough to close the Kocuk-style relaxation gap.
    #
    # formulation:
    #   "socp"  - Jabr rotated-cone SOCP relaxation via Clarabel (default).
    #   "polar" - True polar AC-OPF NLP via IPOPT (N_En_Phase 17.3).
    system, cp0_sum = build_nexus_from_ppc(data)
    base_mva = data["baseMVA"]
    if formulation == "socp":
        _ = ne.solve_socp_opf(system, angle_diff_max=angle_diff_max)  # warm
        t0 = time.perf_counter()
        res = ne.solve_socp_opf(system, angle_diff_max=angle_diff_max)
        wall = time.perf_counter() - t0
        inner_time = float(res.solve_time)
    elif formulation == "polar":
        _ = ne.solve_ac_opf_polar(system, snapshot=0)  # warm
        t0 = time.perf_counter()
        res = ne.solve_ac_opf_polar(system, snapshot=0)
        wall = time.perf_counter() - t0
        inner_time = float(res.solve_time)
    else:
        raise ValueError(f"unknown formulation {formulation!r}; "
                         "use 'socp' or 'polar'")
    return dict(
        status=res.status,
        formulation=formulation,
        objective=float(res.total_cost),
        objective_plus_cp0=float(res.total_cost) + cp0_sum,
        cp0_sum=cp0_sum,
        solve_time_s=wall,
        clarabel_solve_time_s=inner_time,  # legacy key (SOCP); reused for polar
        voltage_mag={k: float(v) for k, v in res.voltage_mag.items()},
        gen_p_mw={k: float(v) * base_mva for k, v in res.gen_p.items()},
        gen_q_mvar={k: float(v) * base_mva for k, v in res.gen_q.items()},
        branch_p_mw={k: float(v) * base_mva for k, v in res.branch_p.items()},
        branch_loss_mw={k: float(v) * base_mva for k, v in res.branch_loss.items()},
    )


# ---------------------------------------------------------------------------
# Pretty-print + run
# ---------------------------------------------------------------------------


def tiny_ppc() -> dict:
    """A self-contained 3-bus MATPOWER ppc dict for the nexus-only smoke path.

    b1 (slack gen) — br0 — b2 — br1 — b3 (load 30 MW). r=0.01, x=0.10 pu,
    baseMVA=100. Lets the nexus AC-OPF/SOCP path run with no pandapower
    dependency, so the script is verifiably run-ready offline.

    Column layout matches the slices `run_pandapower` returns (see the
    MATPOWER index comments above).
    """
    return {
        "case": "tiny3",
        "baseMVA": 100.0,
        "ppc": {
            # BUS_I TYPE PD QD GS BS AREA VM VA BASE_KV ZONE VMAX VMIN
            "bus": [
                [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 100.0, 1.0, 1.1, 0.9],
                [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 100.0, 1.0, 1.1, 0.9],
                [3.0, 1.0, 30.0, 5.0, 0.0, 0.0, 1.0, 1.0, 0.0, 100.0, 1.0, 1.1, 0.9],
            ],
            # F_BUS T_BUS BR_R BR_X BR_B RATE_A RATE_B RATE_C TAP SHIFT STATUS ANGMIN ANGMAX
            "branch": [
                [1.0, 2.0, 0.01, 0.10, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
                [2.0, 3.0, 0.01, 0.10, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
            ],
            # GEN_BUS PG QG QMAX QMIN VG MBASE STATUS PMAX PMIN
            "gen": [
                [1.0, 0.0, 0.0, 100.0, -100.0, 1.0, 100.0, 1.0, 500.0, 0.0],
            ],
            # MODEL STARTUP SHUTDOWN NCOST  c2 c1 c0   (polynomial, $/MW)
            "gencost": [
                [2.0, 0.0, 0.0, 3.0, 0.0, 30.0, 0.0],
            ],
        },
        # Reference fields normally filled by pandapower; unused in nexus-only.
        "objective": float("nan"),
        "solve_time_s": float("nan"),
        "bus_vm_pu": {},
        "gen_p_mw": {},
        "gen_q_mvar": {},
        "ext_grid_p_mw": {},
        "ext_grid_q_mvar": {},
    }


def _int_keys(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def compare_and_report(case: str, formulation: str = "socp",
                       nexus_only: bool = False) -> dict:
    label = {"socp": "Jabr SOCP relaxation", "polar": "polar NLP (IPOPT)"}[formulation]
    print(f"\n{'=' * 76}")
    print(f"AC-OPF {formulation.upper()} vs pandapower NLP — IEEE {case}")
    print(f"{'=' * 76}")

    # --- Competitor side (guarded): pandapower in its isolated venv. ---
    if nexus_only or not pandapower_available():
        why = ("--nexus-only requested" if nexus_only
               else f"pandapower venv not found at {PANDAPOWER_PY}")
        print(f"pandapower SKIPPED ({why}) — solving nexus side on built-in "
              f"tiny 3-bus ppc.")
        nx = run_nexus(tiny_ppc(), formulation=formulation)
        assert nx["status"] in ("optimal", "OPTIMAL"), \
            f"nexus AC-OPF not optimal: {nx['status']}"
        print(f"  nexus {label} optimal: objective={nx['objective']:.6f}")
        return dict(case="tiny3", formulation=formulation, pandapower=None,
                    nexus=nx, competitor="skipped")

    print("Running pandapower runopp (NLP) …", flush=True)
    pm = run_pandapower(case)
    print(f"Running nexus {label} …", flush=True)
    nx = run_nexus(pm, formulation=formulation)

    pm["bus_vm_pu"] = _int_keys(pm["bus_vm_pu"])
    pm["gen_p_mw"] = _int_keys(pm["gen_p_mw"])
    pm["gen_q_mvar"] = _int_keys(pm["gen_q_mvar"])
    pm["ext_grid_p_mw"] = _int_keys(pm["ext_grid_p_mw"])
    pm["ext_grid_q_mvar"] = _int_keys(pm["ext_grid_q_mvar"])

    nx_bus_vm = {int(k[1:]): v for k, v in nx["voltage_mag"].items()}

    right_label = f"nexus ({formulation.upper()})"
    print(f"{'field':<28}{'pandapower (NLP)':>20}{right_label:>20}{'Δ':>10}")
    print("-" * 76)
    rows = [("total cost ($)", pm["objective"], nx["objective_plus_cp0"])]
    for i in sorted(pm["bus_vm_pu"]):
        rows.append((f"|V|[b{i}]", pm["bus_vm_pu"][i], nx_bus_vm.get(i, float('nan'))))

    for name, a, b in rows:
        d = a - b
        print(f"{name:<28}{a:>20.6f}{b:>20.6f}{d:>+10.3e}")

    gap = (pm["objective"] - nx["objective_plus_cp0"]) / abs(pm["objective"])
    print("-" * 76)
    gap_label = {"socp": "SOCP relaxation gap",
                 "polar": "polar NLP gap"}[formulation]
    print(f"{gap_label}:  {gap*100:+.4f} %   "
          f"(pandapower NLP {pm['objective']:.3f}, "
          f"nexus {formulation.upper()} {nx['objective_plus_cp0']:.3f})")
    pm_wall_ms = pm["solve_time_s"] * 1000
    nx_wall_ms = nx["clarabel_solve_time_s"] * 1000
    speedup = pm["solve_time_s"] / max(nx["clarabel_solve_time_s"], 1e-9)
    print(f"wall-clock: pandapower {pm_wall_ms:.1f} ms   "
          f"nexus {nx_wall_ms:.2f} ms   speedup {speedup:.2f}×")

    return dict(case=case, formulation=formulation, pandapower=pm, nexus=nx,
                gap_pct=gap * 100.0,
                speedup=speedup)


def _parse_args(argv: list[str]) -> tuple[str, list[str], bool]:
    """Tiny argv splitter to avoid pulling argparse (same shape as before)."""
    formulation = "socp"
    nexus_only = False
    cases: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--formulation":
            formulation = argv[i + 1]
            i += 2
        elif a.startswith("--formulation="):
            formulation = a.split("=", 1)[1]
            i += 1
        elif a == "--nexus-only":
            nexus_only = True
            i += 1
        else:
            cases.append(a)
            i += 1
    if formulation not in ("socp", "polar"):
        raise SystemExit(f"--formulation must be 'socp' or 'polar' (got {formulation!r})")
    if not cases:
        cases = ["case9", "case14"]
    return formulation, cases, nexus_only


def main():
    formulation, cases, nexus_only = _parse_args(sys.argv[1:])
    results = {}
    for case in cases:
        results[case] = compare_and_report(case, formulation=formulation,
                                           nexus_only=nexus_only)

    out_name = f"ac_opf_vs_pandapower_{formulation}.json"
    out_path = Path(__file__).resolve().parent / "results" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
