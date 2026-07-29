"""
EC073 -- Shell-and-Tube Heat Exchanger -- F2a Lumped-Capacitance Transient Model

Physics-lumped (1D) transient model. The exchanger is discretised into N
control volumes (CVs) along the flow path. Following the standard counter-flow
arrangement, the cold stream flows in the opposite direction to the hot stream
(TEMA E, 1 shell / 2 tube passes is well-approximated by a counter-flow cell
chain for the lumped energy balance). Each CV carries three lumped capacities:

    hot fluid node    (T_h[i])
    metal wall node   (T_w[i])
    cold fluid node   (T_c[i])

Energy balance ODEs (one set per CV i = 0..N-1):

    (rho_h*V_h/N)*cp_h * dT_h[i]/dt =
            m_dot_h*cp_h*(T_h_up - T_h[i]) - (UA/N)*(T_h[i] - T_w[i])

    (m_wall/N)*cp_w * dT_w[i]/dt =
            (UA/N)*(T_h[i] - T_w[i]) - (UA/N)*(T_w[i] - T_c[i])

    (rho_c*V_c/N)*cp_c * dT_c[i]/dt =
            m_dot_c*cp_c*(T_c_up - T_c[i]) + (UA/N)*(T_w[i] - T_c[i])

where "up" denotes the upstream neighbour for each stream. The hot stream is
indexed 0->N-1; the cold stream (counter-flow) enters at i = N-1 and exits at
i = 0, so its upstream neighbour is i+1. The interface conductance is split
symmetrically between the two fluid films so that the series resistance of the
two halves equals 1/UA per CV (each half = 2*UA/N), reproducing the lumped
overall coefficient UA at steady state.

This is solved with scipy.integrate.solve_ivp (LSODA, stiff-capable). At steady
state (dT/dt -> 0) the discretised chain converges to the classic counter-flow
effectiveness-NTU heat duty, which is verified against the closed-form
epsilon-NTU expression (the F1a model's steady-state limit).

References:
    Incropera & DeWitt (2006), Fundamentals of Heat and Mass Transfer, 6th ed.,
        ch.11 (effectiveness-NTU; eqs 11.29-11.32).
    Kakac, S. & Liu, H. (2002), Heat Exchangers: Selection, Rating and Thermal
        Design, 2nd ed., CRC Press.
    Roetzel, W. & Xuan, Y. (1999), Dynamic Behaviour of Heat Exchangers,
        WIT Press / Computational Mechanics Publications (lumped multi-cell
        transient HX models).
    Bohn, M. (2010), "Lumped-parameter dynamic models of heat exchangers",
        standard control-engineering formulation.

Fluid properties (water) are hardcoded with cited sources in parameters.json
(NIST/IAPWS-IF97 and Incropera Table A.6) -- CoolProp is NOT required.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ShellAndTubeHEX_F2a:
    """Lumped-capacitance transient shell-and-tube HX (counter-flow cell chain)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U = u["U"]["value"]            # W/m2K
        self.A = u["A"]["value"]            # m2
        self.cp_h = u["cp_hot"]["value"]    # J/kgK
        self.cp_c = u["cp_cold"]["value"]   # J/kgK
        self.cp_w = u["cp_wall"]["value"]   # J/kgK
        self.rho_h = u["rho_hot"]["value"]  # kg/m3
        self.rho_c = u["rho_cold"]["value"] # kg/m3
        self.V_h = u["V_hot"]["value"]      # m3
        self.V_c = u["V_cold"]["value"]     # m3
        self.m_wall = u["m_wall"]["value"]  # kg
        self.N = int(u["n_cv"]["value"])    # control volumes

        self.UA = self.U * self.A           # W/K

    # ------------------------------------------------------------------
    # Steady-state reference: closed-form counter-flow effectiveness-NTU
    # ------------------------------------------------------------------
    def epsilon_ntu_counterflow(self, m_dot_h, m_dot_c):
        """Counter-flow effectiveness (Incropera Eq. 11.29a)."""
        C_h = m_dot_h * self.cp_h
        C_c = m_dot_c * self.cp_c
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        C_r = C_min / C_max
        NTU = self.UA / C_min
        if abs(C_r - 1.0) < 1e-9:
            eps = NTU / (1.0 + NTU)
        else:
            ex = np.exp(-NTU * (1.0 - C_r))
            eps = (1.0 - ex) / (1.0 - C_r * ex)
        return eps, NTU, C_r, C_min

    def steady_state_duty(self, T_h_in, T_c_in, m_dot_h, m_dot_c):
        """Reference steady-state outputs via epsilon-NTU."""
        eps, NTU, C_r, C_min = self.epsilon_ntu_counterflow(m_dot_h, m_dot_c)
        Q = eps * C_min * (T_h_in - T_c_in)   # W
        C_h = m_dot_h * self.cp_h
        C_c = m_dot_c * self.cp_c
        T_h_out = T_h_in - Q / C_h
        T_c_out = T_c_in + Q / C_c
        return {
            "Q_kw": Q / 1000.0,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "effectiveness": eps,
            "ntu": NTU,
        }

    # ------------------------------------------------------------------
    # Transient: lumped-capacitance cell-chain ODEs
    # ------------------------------------------------------------------
    def _rhs(self, t, y, m_dot_h, m_dot_c, T_h_in_f, T_c_in_f):
        N = self.N
        T_h = y[0:N]
        T_w = y[N:2 * N]
        T_c = y[2 * N:3 * N]

        # Allow time-varying inlets (callables) or constants
        T_h_in = T_h_in_f(t) if callable(T_h_in_f) else T_h_in_f
        T_c_in = T_c_in_f(t) if callable(T_c_in_f) else T_c_in_f

        # Per-CV lumped capacities (J/K)
        C_cap_h = (self.rho_h * self.V_h / N) * self.cp_h
        C_cap_c = (self.rho_c * self.V_c / N) * self.cp_c
        C_cap_w = (self.m_wall / N) * self.cp_w

        # Advective heat-capacity rates (W/K)
        mc_h = m_dot_h * self.cp_h
        mc_c = m_dot_c * self.cp_c

        # Interface conductance per CV (W/K). Split symmetrically so the two
        # half-resistances in series equal 1/(UA/N).
        UA_cv = self.UA / N
        g_half = 2.0 * UA_cv   # conductance of each half (fluid<->wall)

        # Upstream temperatures.
        # Hot stream flows 0 -> N-1 : upstream of i is i-1 (inlet for i=0).
        T_h_up = np.empty(N)
        T_h_up[0] = T_h_in
        T_h_up[1:] = T_h[:-1]
        # Cold stream (counter-flow) flows N-1 -> 0 : upstream of i is i+1
        # (inlet for i=N-1).
        T_c_up = np.empty(N)
        T_c_up[-1] = T_c_in
        T_c_up[:-1] = T_c[1:]

        q_hw = g_half * (T_h - T_w)   # hot fluid -> wall
        q_wc = g_half * (T_w - T_c)   # wall -> cold fluid

        dT_h = (mc_h * (T_h_up - T_h) - q_hw) / C_cap_h
        dT_w = (q_hw - q_wc) / C_cap_w
        dT_c = (mc_c * (T_c_up - T_c) + q_wc) / C_cap_c

        return np.concatenate([dT_h, dT_w, dT_c])

    def simulate(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                 duration_s=300.0, dt=1.0, T_init=None):
        """Integrate the transient cell-chain ODEs.

        Parameters
        ----------
        T_h_in, T_c_in : float or callable(t)
            Hot / cold inlet temperatures [degC] (constant or time-varying).
        m_dot_hot, m_dot_cold : float
            Mass flow rates [kg/s].
        duration_s : float
            Simulation horizon [s].
        dt : float
            Output sample interval [s].
        T_init : float or None
            Uniform initial temperature [degC] for all nodes. Defaults to the
            cold inlet (a cold-start of the whole exchanger).

        Returns
        -------
        dict with time-series arrays and final steady outputs.
        """
        N = self.N
        T_h_in0 = T_h_in(0.0) if callable(T_h_in) else T_h_in
        T_c_in0 = T_c_in(0.0) if callable(T_c_in) else T_c_in
        if T_init is None:
            T_init = T_c_in0
        y0 = np.full(3 * N, float(T_init))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="LSODA",
            args=(m_dot_hot, m_dot_cold, T_h_in, T_c_in),
            rtol=1e-7, atol=1e-9,
        )
        T_h = sol.y[0:N, :]
        T_w = sol.y[N:2 * N, :]
        T_c = sol.y[2 * N:3 * N, :]

        # Outlet temperatures: hot exits at last CV (i=N-1),
        # cold exits at first CV (i=0).
        T_h_out = T_h[-1, :]
        T_c_out = T_c[0, :]

        # Heat duty from the hot-stream energy drop (W -> kW)
        Q_kw = (m_dot_hot * self.cp_h * (T_h_in0 - T_h_out)) / 1000.0

        return {
            "t": sol.t,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "Q_kw": Q_kw,
            "T_h_profile": T_h,    # (N, n_t) node temperatures
            "T_w_profile": T_w,
            "T_c_profile": T_c,
            "success": sol.success,
        }
