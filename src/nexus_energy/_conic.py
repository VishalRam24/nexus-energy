"""
Phase 10 — Clarabel-backed conic adapter.

Why this lives in nexus-energy and not in nexus-opt: nexus-opt's solver
dispatcher (``HiGHS``/``OSQP``/``scipy``) speaks LP, QP, MILP, and unconstrained
NLP. It has no API for second-order or PSD cones. Routing
``solver="clarabel"`` through nexus-opt would just shim Clarabel as another
LP/QP backend, which is strictly worse than HiGHS for that case. The unique
value Clarabel adds is cones, so we expose it here for the SOCP-OPF code path
(and any future conic features) and call its native interface directly.

Generic ``solver="clarabel"`` routing through nexus-opt's Rust crate is
deferred to **Phase 10.x** — it would require a cone API on
``nexus_opt.Model`` (e.g. ``model.add_cone(...)``) plus dispatcher
plumbing in ``lib.rs``. Nothing in the Phase 10 first-pass needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import clarabel
    _CLARABEL_OK = True
except ImportError:  # pragma: no cover
    clarabel = None  # type: ignore
    _CLARABEL_OK = False


def is_available() -> bool:
    """Whether the optional ``clarabel`` package is importable."""
    return _CLARABEL_OK


@dataclass
class ConicResult:
    """Minimal result analogue mirroring ``OptimisationResult`` core fields."""
    status: str
    objective: float
    x: np.ndarray
    z: np.ndarray  # dual on Ax + s = b
    iterations: int
    solve_time: float
    solver: str = "clarabel"


@dataclass
class ConicProblem:
    """
    Clarabel standard form:

        min   0.5 x' P x + q' x
        s.t.  A x + s = b
              s ∈ K   (Cartesian product of cones)

    Cone slot is a list of ``(kind, dim)`` pairs. Supported kinds:

        - ``"zero"``        — equality (``Ax = b``)
        - ``"nonneg"``      — ``Ax ≤ b``  (with ``A`` row-flipped at build time)
        - ``"soc"``         — second-order cone ``{(t, u): t ≥ ‖u‖₂}``

    Build the problem incrementally with the helpers below, then call
    :meth:`solve`.
    """
    n: int  # number of decision vars
    P_rows: list[int] = field(default_factory=list)
    P_cols: list[int] = field(default_factory=list)
    P_vals: list[float] = field(default_factory=list)
    q: list[float] = field(default_factory=list)
    # Constraint blocks (added in order, so cones line up with rows).
    cones: list[tuple[str, int]] = field(default_factory=list)
    A_rows: list[int] = field(default_factory=list)
    A_cols: list[int] = field(default_factory=list)
    A_vals: list[float] = field(default_factory=list)
    b: list[float] = field(default_factory=list)
    _next_row: int = 0

    def __post_init__(self) -> None:
        if len(self.q) == 0:
            self.q = [0.0] * self.n

    # ---- variable construction ----

    def add_var(self) -> int:
        """Allocate one fresh decision variable and return its column index.

        ``n`` is only consumed at :meth:`solve` time (it sizes the sparse
        ``P`` / ``A`` matrices), so growing it incrementally is safe and lets
        builders that need private auxiliaries (e.g. the rotated-SOC lift in
        :func:`nexus_energy.network_socp.add_weymouth_pipe`) allocate them
        without the caller pre-counting columns.
        """
        idx = self.n
        self.n += 1
        self.q.append(0.0)
        return idx

    def add_rotated_soc(self, x_var: int, y_var: int, u_vars: list[int]) -> int:
        """Rotated second-order cone ``2·x·y ≥ ‖u‖₂²``  (with ``x, y ≥ 0``).

        Clarabel exposes no rotated cone, so this posts the algebraically
        equivalent *standard* SOC ``‖(x − y, √2·u₁, …, √2·uₖ)‖₂ ≤ x + y``
        using auxiliary columns pinned by equalities to ``x + y``, ``x − y``
        and the scaled ``√2·uᵢ``. Expanding the SOC gives
        ``(x+y)² ≥ (x−y)² + 2‖u‖²`` ⇔ ``4xy ≥ (x−y)²... `` → ``2·x·y ≥ ‖u‖²``.
        Returns the first cone row index.

        The common use is a convex square lift ``w ≥ z²``: call
        ``add_rotated_soc(x_var=w, y_var=<col pinned to 1/2>, u_vars=[z])`` so
        ``2·w·(1/2) = w ≥ z²``.
        """
        import math as _math
        t_aux = self.add_var()   # = x + y  (cone t-axis, ≥ 0)
        d_aux = self.add_var()   # = x - y
        self.add_eq({t_aux: 1.0, x_var: -1.0, y_var: -1.0}, 0.0)
        self.add_eq({d_aux: 1.0, x_var: -1.0, y_var:  1.0}, 0.0)
        sqrt2 = _math.sqrt(2.0)
        scaled: list[int] = []
        for u in u_vars:
            s_aux = self.add_var()          # = √2 · u
            self.add_eq({s_aux: 1.0, u: -sqrt2}, 0.0)
            scaled.append(s_aux)
        return self.add_soc(t_var=t_aux, u_vars=[d_aux, *scaled])

    # ---- objective construction ----

    def add_linear_obj(self, coefs: dict[int, float]) -> None:
        for j, c in coefs.items():
            self.q[j] += float(c)

    def add_quadratic_obj(self, hessian_upper: list[tuple[int, int, float]]) -> None:
        """``hessian_upper`` is a list of (i, j, value) entries on the upper
        triangle of P (i ≤ j). Clarabel expects upper-triangular CSC."""
        for i, j, v in hessian_upper:
            assert i <= j, "Clarabel P must be upper triangular"
            self.P_rows.append(i)
            self.P_cols.append(j)
            self.P_vals.append(float(v))

    # ---- constraint construction ----

    def add_eq(self, row: dict[int, float], rhs: float) -> int:
        """``Σ_j a_j x_j = rhs``. Returns the assigned row index."""
        r = self._next_row
        for j, a in row.items():
            self.A_rows.append(r)
            self.A_cols.append(j)
            self.A_vals.append(float(a))
        self.b.append(float(rhs))
        self._next_row += 1
        self.cones.append(("zero", 1))
        return r

    def add_le(self, row: dict[int, float], rhs: float) -> int:
        """``Σ_j a_j x_j ≤ rhs``. Implemented as ``Ax + s = b, s ≥ 0``."""
        r = self._next_row
        for j, a in row.items():
            self.A_rows.append(r)
            self.A_cols.append(j)
            self.A_vals.append(float(a))
        self.b.append(float(rhs))
        self._next_row += 1
        self.cones.append(("nonneg", 1))
        return r

    def add_soc(self, t_var: int, u_vars: list[int]) -> int:
        """
        ``t ≥ ‖(u₁, …, uₖ)‖₂``.

        In Clarabel form this is the cone ``s ∈ SOC(k+1)`` with
        ``s = (t, u)`` and identity ``A`` rows so ``A x + s = 0``.
        Returns the first-row index of the block.
        """
        r0 = self._next_row
        block = [t_var, *u_vars]
        for k, j in enumerate(block):
            self.A_rows.append(r0 + k)
            self.A_cols.append(j)
            self.A_vals.append(-1.0)
            self.b.append(0.0)
        self._next_row += len(block)
        self.cones.append(("soc", len(block)))
        return r0

    # ---- solve ----

    def solve(self, *, verbose: bool = False, time_limit: float | None = None,
              eps: float = 1e-7) -> ConicResult:
        if not _CLARABEL_OK:
            raise RuntimeError(
                "clarabel is not installed. `pip install clarabel`. "
                "Required for SOCP/conic features.")
        import scipy.sparse as sp
        import time

        n_cons = self._next_row
        # Build CSC sparse matrices.
        if self.P_vals:
            P = sp.csc_matrix(
                (self.P_vals, (self.P_rows, self.P_cols)), shape=(self.n, self.n))
            P = P + P.T - sp.diags(P.diagonal())  # symmetrize, keep upper-tri only
            P = sp.triu(P, format="csc")
        else:
            P = sp.csc_matrix((self.n, self.n))
        A = sp.csc_matrix(
            (self.A_vals, (self.A_rows, self.A_cols)), shape=(n_cons, self.n))
        q = np.asarray(self.q, dtype=float)
        b = np.asarray(self.b, dtype=float)

        # Translate cone tags to Clarabel ConeT objects, merging consecutive
        # equality / nonneg rows into a single cone for efficiency.
        cones: list = []
        i = 0
        while i < len(self.cones):
            kind, dim = self.cones[i]
            if kind in ("zero", "nonneg"):
                # absorb run of same kind, all with dim 1 here
                run = dim
                j = i + 1
                while j < len(self.cones) and self.cones[j][0] == kind:
                    run += self.cones[j][1]
                    j += 1
                if kind == "zero":
                    cones.append(clarabel.ZeroConeT(run))
                else:
                    cones.append(clarabel.NonnegativeConeT(run))
                i = j
            elif kind == "soc":
                cones.append(clarabel.SecondOrderConeT(dim))
                i += 1
            else:
                raise ValueError(f"unknown cone kind {kind!r}")

        settings = clarabel.DefaultSettings()
        settings.verbose = verbose
        settings.tol_feas = eps
        settings.tol_gap_abs = eps
        settings.tol_gap_rel = eps
        if time_limit is not None:
            settings.time_limit = float(time_limit)

        t0 = time.perf_counter()
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        sol = solver.solve()
        elapsed = time.perf_counter() - t0

        status = str(sol.status)
        # Clarabel statuses: "Solved" | "PrimalInfeasible" | ...
        norm_status = (
            "optimal" if status.lower().startswith("solved")
            else "infeasible" if "infeasible" in status.lower()
            else "time_limit" if "time" in status.lower()
            else "unknown"
        )
        return ConicResult(
            status=norm_status,
            objective=float(sol.obj_val) if sol.x is not None else float("nan"),
            x=np.asarray(sol.x, dtype=float) if sol.x is not None else np.empty(self.n),
            z=np.asarray(sol.z, dtype=float) if sol.z is not None else np.empty(n_cons),
            iterations=int(getattr(sol, "iterations", 0) or 0),
            solve_time=elapsed,
        )
