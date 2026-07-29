"""Phase 10.9 — external-solver bridge (Gurobi / CPLEX / SCIP / Mosek / Xpress).

nexus-opt solves natively with HiGHS (LP / MILP), OSQP / Clarabel (QP / conic)
and Ipopt-via-CasADi (NLP). This module adds an **LP/MILP bridge** to the major
third-party solvers by exporting any ``nexus_opt.Model`` to standard CPLEX-LP
format (``Model.to_lp()``) and handing it to the chosen solver's own reader.
Variable values are read back **by name** — the names carried in the LP file
are exactly the ones passed to ``model.variable(...)`` — so results map straight
back onto nexus variables.

Why a bridge rather than native routing: the third-party solvers are not part of
the nexus-opt core (and are typically licensed / not installed). The bridge keeps
them strictly optional — each is imported lazily and a clear, actionable error is
raised if the package is absent. Use it to cross-check the HiGHS optimum or to
solve a class HiGHS handles poorly:

    from nexus_energy.external_solvers import (
        available_solvers, solve_lp_external, solve_system_external)
    print(available_solvers())                 # which are importable here
    res = solve_system_external(system, "gurobi")
    print(res.status, res.objective, res.var_values["p_coal_0"])

Reference: every solver below ships a documented LP-format reader
(gurobipy ``read``, pyscipopt ``readProblem``, cplex ``Cplex.read``,
mosek ``Task.readdata``, xpress ``problem.read``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import tempfile


_SUPPORTED = ("gurobi", "cplex", "scip", "mosek", "xpress")

# package name to import-test for each solver
_IMPORT_NAME = {
    "gurobi": "gurobipy",
    "cplex": "cplex",
    "scip": "pyscipopt",
    "mosek": "mosek",
    "xpress": "xpress",
}


@dataclass
class ExternalSolveResult:
    """Outcome of an external-solver LP/MILP solve."""
    solver: str
    status: str                       # 'optimal' | 'infeasible' | 'unbounded' | 'unknown'
    objective: float | None
    var_values: dict = field(default_factory=dict)  # {var_name: value}


def available_solvers() -> list[str]:
    """Return the subset of supported external solvers importable in this env."""
    import importlib.util
    out = []
    for name, mod in _IMPORT_NAME.items():
        if importlib.util.find_spec(mod) is not None:
            out.append(name)
    return out


def _require(solver: str):
    s = solver.lower()
    if s not in _SUPPORTED:
        raise ValueError(
            f"Unknown external solver {solver!r}; supported: {_SUPPORTED}.")
    import importlib.util
    if importlib.util.find_spec(_IMPORT_NAME[s]) is None:
        raise ImportError(
            f"External solver {solver!r} requested but its Python package "
            f"'{_IMPORT_NAME[s]}' is not installed. Install it (and ensure a "
            f"valid licence where required) to use this path, or solve with "
            f"the native HiGHS/OSQP/Clarabel/Ipopt backends.")
    return s


def solve_lp_external(lp_text: str, solver: str,
                      *, time_limit: float | None = None) -> ExternalSolveResult:
    """Solve a CPLEX-LP-format model string with an external solver.

    Writes ``lp_text`` to a temporary ``.lp`` file, reads it with the chosen
    solver's native reader, solves, and returns status / objective / values.
    Raises ``ImportError`` (with an install hint) if the solver is unavailable.
    """
    s = _require(solver)
    fd, path = tempfile.mkstemp(suffix=".lp")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(lp_text)
        return _DISPATCH[s](path, time_limit)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def solve_system_external(system, solver: str, *,
                          time_limit: float | None = None,
                          objective: str = "min_cost") -> ExternalSolveResult:
    """Build ``system``'s LP/MILP and solve it with an external solver.

    Uses the same model assembly as :meth:`EnergySystem.optimise` — the fully
    built ``nexus_opt.Model`` is captured via the ``model_hook`` and exported to
    LP — then routes that LP to the third-party solver instead of returning the
    HiGHS result. Returns named variable values; map them back with the variable
    naming convention (``p_<gen>_<t>``, ``flow_<link>_<t>``, …).

    Note: capturing the model currently re-uses the native solve to assemble it;
    on the tiny models this bridge is meant for (cross-checks) that overhead is
    negligible.
    """
    captured = {}

    def _hook(model, sys, obj):
        captured["lp"] = model.to_lp()
        return None

    # Build (and natively solve) to obtain the assembled LP; we only keep the
    # exported LP text and discard the HiGHS result.
    system.optimise(objective=objective, model_hook=_hook)
    if "lp" not in captured:
        raise RuntimeError("solve_system_external: model capture failed.")
    return solve_lp_external(captured["lp"], solver, time_limit=time_limit)


# --------------------------------------------------------------------------
# per-solver readers (each lazily imports its package)
# --------------------------------------------------------------------------

def _solve_gurobi(path: str, time_limit):
    import gurobipy as gp
    m = gp.read(path)
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    m.Params.OutputFlag = 0
    m.optimize()
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.INFEASIBLE: "infeasible",
              gp.GRB.UNBOUNDED: "unbounded"}.get(m.Status, "unknown")
    vals, obj = {}, None
    if status == "optimal":
        vals = {v.VarName: v.X for v in m.getVars()}
        obj = m.ObjVal
    return ExternalSolveResult("gurobi", status, obj, vals)


def _solve_scip(path: str, time_limit):
    from pyscipopt import Model as SCIP
    m = SCIP()
    m.hideOutput(True)
    m.readProblem(path)
    if time_limit is not None:
        m.setParam("limits/time", float(time_limit))
    m.optimize()
    status = {"optimal": "optimal", "infeasible": "infeasible",
              "unbounded": "unbounded"}.get(m.getStatus(), "unknown")
    vals, obj = {}, None
    if status == "optimal":
        vals = {v.name: m.getVal(v) for v in m.getVars()}
        obj = m.getObjVal()
    return ExternalSolveResult("scip", status, obj, vals)


def _solve_cplex(path: str, time_limit):
    import cplex
    c = cplex.Cplex()
    c.set_results_stream(None)
    c.set_log_stream(None)
    c.read(path)
    if time_limit is not None:
        c.parameters.timelimit.set(float(time_limit))
    c.solve()
    st = c.solution.get_status_string()
    status = ("optimal" if "optimal" in st.lower()
              else "infeasible" if "infeasible" in st.lower() else "unknown")
    vals, obj = {}, None
    if status == "optimal":
        names = c.variables.get_names()
        xs = c.solution.get_values()
        vals = dict(zip(names, xs))
        obj = c.solution.get_objective_value()
    return ExternalSolveResult("cplex", status, obj, vals)


def _solve_mosek(path: str, time_limit):
    import mosek
    with mosek.Task() as task:
        task.readdata(path)
        if time_limit is not None:
            task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit))
        task.optimize()
        solsta = task.getsolsta(mosek.soltype.bas)
        status = ("optimal" if solsta == mosek.solsta.optimal
                  else "infeasible"
                  if solsta in (mosek.solsta.prim_infeas_cer,
                                mosek.solsta.dual_infeas_cer) else "unknown")
        vals, obj = {}, None
        if status == "optimal":
            n = task.getnumvar()
            xx = task.getxx(mosek.soltype.bas)
            names = [task.getvarname(i) for i in range(n)]
            vals = dict(zip(names, xx))
            obj = task.getprimalobj(mosek.soltype.bas)
    return ExternalSolveResult("mosek", status, obj, vals)


def _solve_xpress(path: str, time_limit):
    import xpress as xp
    p = xp.problem()
    p.read(path)
    if time_limit is not None:
        p.controls.maxtime = int(time_limit)
    p.solve()
    st = p.getProbStatusString()
    status = ("optimal" if "optimal" in st.lower()
              else "infeasible" if "infeasible" in st.lower() else "unknown")
    vals, obj = {}, None
    if status == "optimal":
        names = [v.name for v in p.getVariable()]
        xs = p.getSolution()
        vals = dict(zip(names, xs))
        obj = p.getObjVal()
    return ExternalSolveResult("xpress", status, obj, vals)


_DISPATCH = {
    "gurobi": _solve_gurobi,
    "scip": _solve_scip,
    "cplex": _solve_cplex,
    "mosek": _solve_mosek,
    "xpress": _solve_xpress,
}
