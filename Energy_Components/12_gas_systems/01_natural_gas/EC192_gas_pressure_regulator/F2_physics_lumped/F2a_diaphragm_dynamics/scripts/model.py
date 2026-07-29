"""
EC192 -- Gas Pressure Regulator -- F2a Physics-Lumped Diaphragm/Valve Dynamics

A self-operated, spring-loaded, downstream-sensing gas pressure regulator that
holds the outlet (downstream) pressure near a mechanical setpoint as the load
flow and upstream pressure vary. The model couples three first-principles ODEs
integrated with scipy.integrate.solve_ivp:

State vector y = [P_d, x, v]
    P_d : downstream control-volume pressure        [Pa]
    x   : valve/plug travel (diaphragm position)     [m]
    v   : valve velocity  dx/dt                       [m/s]

1) Downstream-volume pressure ODE (lumped, isothermal ideal-gas mass balance):
       dP_d/dt = (R_s * T_d / V_down) * (mdot_in - mdot_load)
   from m = P_d V/(R_s T_d) so dm/dt = (V/(R_s T_d)) dP_d/dt. This enforces
   MASS CONSERVATION of the control volume: pressure rises only when more gas
   enters through the valve than the load draws.

2) Valve / diaphragm position dynamics (spring-mass-damper force balance,
   Newton's 2nd law on the moving assembly):
       m_d * d2x/dt2 = F_spring - F_press - c * dx/dt
       F_press  = (P_d - P_set) * A_diaphragm        (downstream pressure feedback)
       F_spring = F_preload - k * x
   The diaphragm senses P_d: when P_d > P_set the net force pushes the plug
   CLOSED (x decreases); when P_d < P_set the spring pushes it OPEN. This
   negative-feedback loop regulates P_d toward P_set. The finite spring rate k
   produces proportional DROOP: holding the valve further open (higher flow)
   requires a larger spring deflection, i.e. a lower regulated P_d.

3) Valve flow (ISA-75.01.01-2012 gas sizing equation, choked + subsonic):
       Non-choked:  Q = N7 * Cv_eff * Y * sqrt(dP * P_up /(G T Z))
                    Y = 1 - dP /(3 Fk Xt P_up)        (>= 2/3)
       Choked:      dP >= Fk*Xt*P_up  ->  Q locked at critical dP, Y = 2/3
   Choked flow is independent of P_down (CHOKED-FLOW LIMIT enforced).

Joule-Thomson cooling on expansion (isenthalpic throttling):
       T_down = T_up - mu_JT * (P_up - P_down)
   mu_JT > 0 for NG below its inversion temperature -> the gas COOLS as it
   expands across the valve.

Regulator behaviour reproduced:
  * LOCKUP: at zero load the valve seats (x -> 0) and P_d settles slightly
    above P_set (lockup pressure) determined by spring preload.
  * DROOP : steady regulated P_d falls with increasing load flow.

References:
    ANSI/ISA-75.01.01-2012. Flow Equations for Sizing Control Valves.
    Emerson / Fisher (2019). Control Valve Handbook, 5th ed., Ch.2 & Ch.5
        (regulators, droop, lockup, boost).
    Driskell, L.R. (1983). Control Valve Selection and Sizing. ISA.
    Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press, Ch.4.
    Burnett (1999). Joule-Thomson coefficients for natural-gas mixtures.
"""

import numpy as np
from scipy.integrate import solve_ivp


class GasPressureRegulatorF2a:
    """Physics-lumped spring-loaded gas pressure regulator with valve dynamics."""

    def __init__(self, params: dict):
        u = params["unit"]
        vlv = u["valve"]
        act = u["actuator"]
        gas = u["gas"]
        dn = u["downstream"]

        # Valve / flow
        self.Cv_full = vlv["Cv_full"]["value"]
        self.Xt = vlv["Xt"]["value"]
        self.Fk = vlv["Fk"]["value"]
        self.x_max = vlv["x_max"]["value"]                 # m
        self.Cv_coeffs = vlv["Cv_PLR_coeffs"]["value"]
        self.N7 = vlv["N7"]["value"]
        self.Z = vlv["Z"]["value"]

        # Actuator / spring-mass-damper
        self.P_set = act["P_set"]["value"] * 1e5           # Pa
        self.A_dia = act["A_diaphragm"]["value"]            # m^2
        self.k = act["k_spring"]["value"]                  # N/m
        self.m_d = act["m_diaphragm"]["value"]             # kg
        self.c = act["c_damp"]["value"]                    # N.s/m
        self.F_preload = act["F_preload"]["value"]         # N

        # Gas
        self.G = gas["specific_gravity"]["value"]
        self.gamma = gas["gamma"]["value"]
        self.R_s = gas["R_specific"]["value"]              # J/(kg.K)
        self.cp = gas["cp"]["value"]
        self.rho_std = gas["rho_std"]["value"]             # kg/m^3
        self.mu_JT = gas["JT_coefficient"]["value"]        # K/Pa
        self.T_inv = gas["T_inversion"]["value"]           # K

        # Downstream lumped volume
        self.V_down = dn["V_down"]["value"]                # m^3
        self.T_down = dn["T_down"]["value"]                # K

    # ------------------------------------------------------------------
    # Valve travel -> effective Cv
    # ------------------------------------------------------------------
    def cv_effective(self, x):
        """Effective Cv from plug travel x [m]. Clamped to [0, Cv_full]."""
        s = np.clip(np.asarray(x, dtype=float) / self.x_max, 0.0, 1.0)
        a0, a1, a2 = self.Cv_coeffs
        f = (a0 + a1 * s + a2 * s ** 2) / (a0 + a1 + a2)
        f = np.clip(f, 0.0, 1.0)
        return self.Cv_full * f

    # ------------------------------------------------------------------
    # Choke detection and ISA expansion factor
    # ------------------------------------------------------------------
    def dp_choke(self, P_up_bar):
        """Critical pressure drop [bar]: dP_choke = Fk*Xt*P_up."""
        return self.Fk * self.Xt * np.asarray(P_up_bar, dtype=float)

    def is_choked(self, P_up_bar, P_down_bar):
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        return (P_up - P_down) >= self.dp_choke(P_up)

    def expansion_factor_Y(self, P_up_bar, P_down_bar):
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        dP = np.maximum(P_up - P_down, 0.0)
        Y = 1.0 - dP / (3.0 * self.Fk * self.Xt * P_up)
        return np.maximum(Y, 2.0 / 3.0)

    # ------------------------------------------------------------------
    # ISA gas flow through the valve (choked + subsonic)
    # ------------------------------------------------------------------
    def flow_std_m3_per_h(self, P_up_bar, P_down_bar, T_up_K, x):
        """
        Standard volumetric flow [m^3/h] (ANSI/ISA-75.01.01-2012 gas eqn):

            Q = N7 * Cv * P1 * Y * sqrt(x_r / (G * T1 * Z))

        with pressure-drop ratio x_r = dP/P1 (capped at the critical ratio
        x_choke = Fk*Xt at choke), expansion factor Y, and N7 = 417 for
        Q [m^3/h normal], P [bar abs]. At choke x_r locks at x_choke and
        Y = 2/3, so Q becomes independent of P_down (choked-flow limit).
        """
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        T_up = np.asarray(T_up_K, dtype=float)
        Cv_eff = self.cv_effective(x)

        x_ratio = np.maximum(P_up - P_down, 0.0) / P_up        # dP/P1
        x_choke = self.Fk * self.Xt                            # critical ratio
        x_eff = np.minimum(x_ratio, x_choke)                   # lock at choke
        Y = np.maximum(1.0 - x_eff / (3.0 * self.Fk * self.Xt), 2.0 / 3.0)
        Q = self.N7 * Cv_eff * P_up * Y * np.sqrt(
            x_eff / (self.G * T_up * self.Z))
        return Q

    def flow_kg_per_s(self, P_up_bar, P_down_bar, T_up_K, x):
        Q = self.flow_std_m3_per_h(P_up_bar, P_down_bar, T_up_K, x)
        return Q * self.rho_std / 3600.0

    # ------------------------------------------------------------------
    # Joule-Thomson cooling
    # ------------------------------------------------------------------
    def temperature_out(self, T_up_K, P_up_bar, P_down_bar):
        """Downstream temperature after isenthalpic JT expansion [K]."""
        T_up = np.asarray(T_up_K, dtype=float)
        dP_Pa = np.maximum(
            (np.asarray(P_up_bar, dtype=float) -
             np.asarray(P_down_bar, dtype=float)), 0.0) * 1e5
        mu = np.where(T_up < self.T_inv, self.mu_JT, -self.mu_JT)
        return T_up - mu * dP_Pa

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, P_up_fn, mdot_load_fn, T_up_fn):
        P_d, x, v = y
        P_d = max(P_d, 1.0)            # keep pressure positive
        P_up_bar = P_up_fn(t)
        P_d_bar = P_d / 1e5
        T_up = T_up_fn(t)

        # 1) Valve mass inflow (ISA), clamped: no backflow if P_d >= P_up
        if P_up_bar > P_d_bar:
            mdot_in = float(self.flow_kg_per_s(P_up_bar, P_d_bar, T_up, x))
        else:
            mdot_in = 0.0
        mdot_load = mdot_load_fn(t)

        # 1) Downstream pressure ODE (lumped isothermal mass balance)
        dPd_dt = (self.R_s * self.T_down / self.V_down) * (mdot_in - mdot_load)

        # 2) Valve / diaphragm spring-mass-damper (downstream feedback)
        F_spring = self.F_preload - self.k * x
        F_press = (P_d - self.P_set) * self.A_dia
        accel = (F_spring - F_press - self.c * v) / self.m_d

        # Travel limit stops (seat at x=0, full open at x_max)
        if x <= 0.0 and accel < 0.0:
            x = 0.0
            v = max(v, 0.0)
            accel = max(accel, 0.0)
        elif x >= self.x_max and accel > 0.0:
            v = min(v, 0.0)
            accel = min(accel, 0.0)

        return [dPd_dt, v, accel]

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, P_up, mdot_load, T_up=288.15, P_d0=None, x0=None,
                 duration_s=60.0, dt=0.05):
        """
        Integrate the coupled regulator ODEs.

        P_up       : upstream pressure [bar], scalar or callable f(t)->bar
        mdot_load  : downstream load draw [kg/s], scalar or callable f(t)->kg/s
        T_up       : upstream temperature [K], scalar or callable f(t)->K
        P_d0       : initial downstream pressure [Pa] (default = P_set)
        x0         : initial valve travel [m] (default = small open, x_max/2)
        Returns dict of time-series arrays.
        """
        P_up_fn = P_up if callable(P_up) else (lambda t, c=P_up: c)
        load_fn = mdot_load if callable(mdot_load) else (lambda t, c=mdot_load: c)
        T_up_fn = T_up if callable(T_up) else (lambda t, c=T_up: c)

        if P_d0 is None:
            P_d0 = self.P_set
        if x0 is None:
            x0 = 0.5 * self.x_max

        y0 = [P_d0, x0, 0.0]
        t_eval = np.arange(0.0, duration_s + dt / 2, dt)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0, t_eval=t_eval,
            args=(P_up_fn, load_fn, T_up_fn),
            method="BDF", rtol=1e-6, atol=[1e-2, 1e-9, 1e-9], max_step=dt)

        t = sol.t
        P_d = np.maximum(sol.y[0], 1.0)
        x = np.clip(sol.y[1], 0.0, self.x_max)
        v = sol.y[2]

        P_up_arr = np.array([P_up_fn(ti) for ti in t])
        T_up_arr = np.array([T_up_fn(ti) for ti in t])
        P_d_bar = P_d / 1e5

        Q_std = self.flow_std_m3_per_h(P_up_arr, P_d_bar, T_up_arr, x)
        mdot_in = Q_std * self.rho_std / 3600.0
        T_out = self.temperature_out(T_up_arr, P_up_arr, P_d_bar)
        choked = self.is_choked(P_up_arr, P_d_bar)
        load_arr = np.array([load_fn(ti) for ti in t])

        return {
            "t": t,
            "P_down_bar": P_d_bar,
            "P_down_Pa": P_d,
            "P_up_bar": P_up_arr,
            "valve_travel_m": x,
            "valve_travel_frac": x / self.x_max,
            "valve_velocity": v,
            "flow_std_m3_per_h": Q_std,
            "mdot_in_kg_s": mdot_in,
            "mdot_load_kg_s": load_arr,
            "T_downstream_K": T_out,
            "JT_cooling_K": T_up_arr - T_out,
            "is_choked": choked,
            "P_set_bar": np.full_like(t, self.P_set / 1e5),
        }

    # ------------------------------------------------------------------
    # Steady-state helper (droop / lockup curves)
    # ------------------------------------------------------------------
    def steady_state(self, P_up_bar, mdot_load, T_up=288.15,
                     duration_s=120.0, dt=0.05):
        """Run long enough to settle; return final regulated state."""
        r = self.simulate(P_up_bar, mdot_load, T_up,
                          duration_s=duration_s, dt=dt)
        return {
            "P_down_bar": float(r["P_down_bar"][-1]),
            "valve_travel_frac": float(r["valve_travel_frac"][-1]),
            "flow_std_m3_per_h": float(r["flow_std_m3_per_h"][-1]),
            "mdot_in_kg_s": float(r["mdot_in_kg_s"][-1]),
            "T_downstream_K": float(r["T_downstream_K"][-1]),
            "is_choked": bool(r["is_choked"][-1]),
        }
