"""
EC130 -- Small/Micro Hydropower -- F2a Physics-Lumped Penstock Transient

Physics-lumped (0D/1D) dynamic model of a micro-hydro plant. Couples three
first-order ODE states integrated with scipy.integrate.solve_ivp:

  State 1 -- Penstock flow velocity  v(t)     [m/s]   (rigid water-column momentum)
  State 2 -- Forebay / surge level   z(t)     [m]     (mass-oscillation buffer)
  State 3 -- Gate (nozzle) opening   g(t)     [-]     (first-order servo)

------------------------------------------------------------------------------
1.  Hydraulic power and electrical output (steady map at each instant)
------------------------------------------------------------------------------
    P_hyd = rho * g * Q * H_net                       (W)   hydraulic power
    P_el  = eta_turbine(q) * eta_gen * P_hyd          (W)   electrical output
    Q     = v * A_pipe                                       volumetric flow

------------------------------------------------------------------------------
2.  Net head with Darcy-Weisbach penstock head loss
------------------------------------------------------------------------------
    H_loss = ( f * L/D + K_minor ) * v^2 / (2 g)            (m)
    H_net  = H_static + z  -  H_loss
  where H_static = H_gross and z is the surge/forebay level deviation.
  (White 2011 eq. 6.10 Darcy-Weisbach; Chaudhry 2014 ch. 1.)

------------------------------------------------------------------------------
3.  Rigid water-column momentum ODE (simplified water-hammer / surge)
------------------------------------------------------------------------------
    (L/g) dv/dt = (H_static + z) - H_loss - H_turbine_demand
  The turbine, throttled by the gate g, imposes a back-head proportional to
  the available head scaled by the gate opening so that at equilibrium the
  flow matches the commanded gate. This is the lumped form of the elastic
  water-hammer PDE collapsed to a single water-column inertia term
  T_w = L v / (g H)  (water starting time). Chaudhry (2014) eq. 4.x.

------------------------------------------------------------------------------
4.  Forebay / surge-tank mass oscillation
------------------------------------------------------------------------------
    A_s dz/dt = Q_in - Q_pipe = Q_in - v A_pipe
  With Q_in held at the (slowly varying) inflow, the surge tank integrates the
  imbalance between supply and penstock draw, giving the classic U-tube
  mass-oscillation period. Jaeger (1977); Chaudhry (2014) ch. 10.

------------------------------------------------------------------------------
5.  Gate servo (nozzle / guide-vane actuator)
------------------------------------------------------------------------------
    tau dg/dt = g_cmd - g
  First-order lag on the wicket-gate / spear-valve position.

References:
    Harvey, A. et al. (1993) "Micro-Hydro Design Manual", IT Publications.
    Penche, C. (1998) "Layman's Guidebook on How to Develop a Small Hydro
        Site", European Commission DG XVII.
    Chaudhry, M. H. (2014) "Applied Hydraulic Transients", 3rd ed., Springer.
    White, F. M. (2011) "Fluid Mechanics", 7th ed., McGraw-Hill.
"""

import numpy as np
from scipy.integrate import solve_ivp


class MicroHydroF2a:
    """Micro-hydro plant -- lumped penstock + surge + gate transient model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_gross = u["H_gross"]["value"]            # m  static gross head
        self.Q_design = u["Q_design"]["value"]          # m3/s
        self.P_rated = u["P_rated"]["value"]            # kW
        self.eta_gen = u["eta_generator"]["value"]
        self.eta_t_peak = u["eta_turbine_peak"]["value"]
        self.k_eff = u["k_eff"]["value"]
        self.q_min = u["q_min"]["value"]
        self.q_max = u["q_max"]["value"]
        self.L = u["L_penstock"]["value"]               # m
        self.D = u["D_penstock"]["value"]               # m
        self.f = u["f_darcy"]["value"]
        self.K_minor = u["K_minor"]["value"]
        self.A_s = u["A_forebay"]["value"]              # m2 surge surface area
        self.tau = u["tau_nozzle"]["value"]             # s
        self.rho = u["rho"]["value"]                    # kg/m3
        self.g = u["g"]["value"]                        # m/s2

        self.A_pipe = np.pi * (self.D ** 2) / 4.0       # m2 penstock x-section
        # Design velocity from design flow
        self.v_design = self.Q_design / self.A_pipe

    # ------------------------------------------------------------------
    # Algebraic relations
    # ------------------------------------------------------------------
    def head_loss(self, v):
        """Darcy-Weisbach + minor losses [m]. White (2011) eq. 6.10."""
        v = np.asarray(v, dtype=float)
        return (self.f * self.L / self.D + self.K_minor) * v * v / (2.0 * self.g)

    def net_head(self, v, z):
        """Net head available to the turbine [m]."""
        return self.H_gross + z - self.head_loss(v)

    def turbine_efficiency(self, Q):
        """
        Part-load hydraulic efficiency (Francis parabolic curve).
        eta = eta_peak * (1 - k*(q-1)^2); zero outside [q_min, q_max].
        Harvey (1993) part-load curves.
        """
        Q = np.asarray(Q, dtype=float)
        q = Q / self.Q_design
        eta = self.eta_t_peak * (1.0 - self.k_eff * (q - 1.0) ** 2)
        eta = np.where((q < self.q_min) | (q > self.q_max), 0.0, eta)
        return np.clip(eta, 0.0, self.eta_t_peak)

    def hydraulic_power_kw(self, Q, H_net):
        """P_hyd = rho g Q H_net  [kW] (gross hydraulic power in the flow)."""
        Q = np.asarray(Q, dtype=float)
        H_net = np.asarray(H_net, dtype=float)
        return self.rho * self.g * Q * np.maximum(H_net, 0.0) / 1000.0

    def electrical_power_kw(self, Q, H_net):
        """P_el = eta_turbine * eta_gen * P_hyd  [kW]."""
        eta = self.turbine_efficiency(Q)
        P = eta * self.eta_gen * self.hydraulic_power_kw(Q, H_net)
        return np.clip(P, 0.0, self.P_rated)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, gate_cmd_fn, Q_in_fn):
        v, z, g = y
        g = np.clip(g, 0.0, 1.0)

        H_loss = self.head_loss(v)
        H_avail = self.H_gross + z - H_loss            # net head at gate

        # Turbine back-head: the gate throttles the column. The demanded head
        # drop across the turbine equals the available net head scaled so that
        # the steady velocity tracks the gate command (v_target = g * v_design).
        v_target = g * self.v_design
        # Quadratic valve characteristic: head consumed by the turbine/gate.
        # At equilibrium dv/dt = 0  ->  H_avail = H_turb, giving v = v_target.
        if g > 1e-6:
            H_turb = H_avail * (v / max(v_target, 1e-6)) ** 2
        else:
            # Gate (nearly) shut: large resistance brings flow to zero.
            H_turb = H_avail + 50.0 * v * v

        # 1) Rigid water-column momentum: (L/g) dv/dt = H_static+z - H_loss - H_turb
        dv_dt = (self.g / self.L) * (H_avail - H_turb)

        # 2) Surge/forebay continuity: A_s dz/dt = Q_in - Q_pipe
        Q_in = Q_in_fn(t)
        Q_pipe = v * self.A_pipe
        dz_dt = (Q_in - Q_pipe) / self.A_s

        # 3) Gate servo first-order lag
        dg_dt = (gate_cmd_fn(t) - g) / self.tau

        return [dv_dt, dz_dt, dg_dt]

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------
    def simulate(self, gate_cmd, Q_in=None, v0=None, z0=0.0, g0=None,
                 dt=0.1, duration_s=120.0):
        """
        Integrate the lumped transient with scipy.solve_ivp.

        Args:
            gate_cmd   : float in [0,1] OR callable t->[0,1]  (gate opening cmd)
            Q_in       : float (m3/s) OR callable t->m3/s  inflow to forebay.
                         Defaults to design flow scaled by the gate command.
            v0, z0, g0 : initial states (defaults: equilibrium at gate_cmd(0))
            dt         : output sample step [s]
            duration_s : total simulated time [s]

        Returns dict of numpy arrays (t, velocity, flow, head_net, head_loss,
            surge_level, gate, power_el, power_hyd, efficiency).
        """
        # Normalise commands to callables
        if callable(gate_cmd):
            gate_fn = gate_cmd
        else:
            gc = float(gate_cmd)
            gate_fn = lambda t: gc

        if Q_in is None:
            # supply tracks demand: design flow * gate command
            Q_in_fn = lambda t: gate_fn(t) * self.Q_design
        elif callable(Q_in):
            Q_in_fn = Q_in
        else:
            qin = float(Q_in)
            Q_in_fn = lambda t: qin

        g_init = gate_fn(0.0) if g0 is None else g0
        v_init = g_init * self.v_design if v0 is None else v0
        y0 = [v_init, z0, g_init]

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(gate_fn, Q_in_fn),
            method="RK45", rtol=1e-6, atol=1e-8, max_step=dt,
        )

        v = np.clip(sol.y[0], 0.0, None)
        z = sol.y[1]
        g = np.clip(sol.y[2], 0.0, 1.0)
        Q = v * self.A_pipe
        H_loss = self.head_loss(v)
        H_net = self.net_head(v, z)
        P_hyd = self.hydraulic_power_kw(Q, H_net)
        P_el = self.electrical_power_kw(Q, H_net)
        eta = self.turbine_efficiency(Q)

        return {
            "t": sol.t,
            "velocity": v,
            "flow": Q,
            "head_net": H_net,
            "head_loss": H_loss,
            "surge_level": z,
            "gate": g,
            "power_el": P_el,
            "power_hyd": P_hyd,
            "efficiency": eta,
        }

    # ------------------------------------------------------------------
    # Steady-state helper (algebraic fixed point for a given gate opening)
    # ------------------------------------------------------------------
    def steady_state(self, gate, tol=1e-9, itmax=200):
        """
        Solve the steady velocity for a fixed gate opening by fixed-point
        iteration on the head balance H_static = H_loss(v) + H_turb(v).
        At equilibrium v -> gate * v_design (target), but head losses reduce
        the achievable velocity slightly. Returns dict of steady outputs.
        """
        g = float(np.clip(gate, 0.0, 1.0))
        v_target = g * self.v_design
        # Steady velocity: balance momentum with H_turb = H_avail at v=v_target.
        # The quadratic valve law makes v=v_target the fixed point exactly,
        # so steady velocity is the target (losses already inside H_avail).
        v = v_target
        Q = v * self.A_pipe
        H_loss = self.head_loss(v)
        H_net = self.net_head(v, 0.0)
        P_el = self.electrical_power_kw(Q, H_net)
        return {
            "velocity": v, "flow": Q, "head_loss": H_loss,
            "head_net": H_net, "power_el": float(P_el),
            "efficiency": float(self.turbine_efficiency(Q)),
        }
