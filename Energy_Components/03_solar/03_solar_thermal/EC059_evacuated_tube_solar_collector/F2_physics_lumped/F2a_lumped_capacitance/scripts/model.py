"""
EC059 — Evacuated Tube Solar Collector — F2a Lumped-Capacitance Dynamic Model

Physics-lumped (0D) dynamic energy balance for the absorber/fluid node.
This is the first-principles ODE upgrade of the F1 algebraic Hottel-Whillier-Bliss
models (F1a constant-U_L, F1b second-order U_L(T)+IAM): instead of assuming the
collector is always at thermal steady state, F2a integrates the transient
lumped-capacitance heat balance with scipy.integrate.solve_ivp.

Lumped-capacitance energy balance (single absorber/fluid node at temperature T):

    C * dT/dt = Q_abs(t) - Q_loss(T, Ta) - Q_useful(T, T_in)

where
  Q_abs   = optical_eff * IAM(theta) * G(t) * A          absorbed solar [W]
  Q_loss  = Q_rad + Q_cond                                heat loss to ambient [W]
  Q_useful= m_dot * cp * (T_out - T_in)                   heat carried off by fluid [W]

Loss model — physics of the evacuated tube
-------------------------------------------
The vacuum annulus between the selective absorber and the outer glass tube
suppresses gas conduction and convection, so the dominant loss path is
RADIATION from the (low-emissivity) absorber to the surroundings, plus a small
residual non-radiative conduction term (end caps, headers, residual gas):

    Q_rad  = emissivity * sigma * A_abs * (T^4 - Ta^4)        [W]   (Stefan-Boltzmann)
    Q_cond = U_residual * A * (T - Ta)                        [W]   (residual conduction)

Linearising the radiation term about Tm recovers the familiar overall loss
coefficient U_L of the Hottel-Whillier framework:

    U_rad(Tm) = emissivity * sigma * (A_abs/A) * (T^2 + Ta^2)(T + Ta) ~ 4*eps*sigma*Tm^3*(A_abs/A)
    U_L(Tm)   = U_rad(Tm) + U_residual

which is strongly nonlinear in temperature (rises as Tm^3) — exactly the
behaviour seen in EN 12975 / ISO 9806 second-order test curves where the a2
term dominates at high mean temperature. This reproduces the F1b a1, a2 form
from first principles rather than fitting.

Useful-heat / outlet-temperature closure
-----------------------------------------
The node temperature T is taken as the collector mean fluid temperature Tm.
With single-pass plug flow the outlet relates to the node by an effectiveness
(Duffie & Beckman Ch.6, flow-factor F''):

    Q_useful = m_dot * cp * (T_out - T_in)
    T_out    = 2*T - T_in            (mean = (T_in+T_out)/2 closure)

Steady-state efficiency curve (recovered, not assumed)
------------------------------------------------------
At dT/dt = 0 the model collapses to the EN 12975 / Hottel-Whillier-Bliss form:

    eta(x) = eta_0 * IAM - a1 * x - a2 * G * x^2 ,   x = (Tm - Ta)/G

so efficiency is monotonically decreasing in the reduced temperature
x = (Tm - Ta)/G, eta -> eta_0*IAM as x -> 0, and Q -> 0 at night (G = 0).

Incidence Angle Modifier (IAM)
------------------------------
    IAM(theta) = 1 - b0 * (1/cos(theta) - 1)            (Duffie & Beckman Eq. 6.17.6)

References:
    Duffie & Beckman (2013). 'Solar Engineering of Thermal Processes', 4th ed.,
        Wiley, Ch.6 (collector performance, U_L, IAM) & Ch.10 (transient capacitance).
    EN 12975 / ISO 9806:2017. Solar thermal collectors — quasi-dynamic test method,
        effective thermal capacity and second-order efficiency equation.
    Incropera & DeWitt, Fundamentals of Heat and Mass Transfer (radiation exchange).
"""

import numpy as np
from scipy.integrate import solve_ivp

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/m2K4


class EvacuatedTubeF2a:
    """Evacuated tube collector — lumped-capacitance dynamic energy balance."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.area = u["area"]["value"]
        self.optical_eff = u["optical_eff"]["value"]
        self.emissivity = u["emissivity"]["value"]
        self.area_ratio = u["absorber_area_ratio"]["value"]
        self.U_residual = u["U_residual"]["value"]
        self.b0 = u["b0"]["value"]
        self.C = u["C_thermal"]["value"]
        self.m_dot = u["m_dot"]["value"]
        self.cp = u["cp_fluid"]["value"]
        self.T_in_default = u.get("T_in_default", {"value": 40.0})["value"]
        self.A_abs = self.area * self.area_ratio

    # ------------------------------------------------------------------
    # Incidence angle modifier
    # ------------------------------------------------------------------
    def iam(self, theta_deg):
        """IAM(theta) = 1 - b0*(1/cos theta - 1), Duffie & Beckman Eq. 6.17.6."""
        theta = np.asarray(theta_deg, dtype=float)
        cos_t = np.cos(np.radians(theta))
        safe_cos = np.where(theta < 80.0, np.maximum(cos_t, 0.02), 0.02)
        val = np.where(theta < 80.0, 1.0 - self.b0 * (1.0 / safe_cos - 1.0), 0.0)
        return float(np.clip(val, 0.0, 1.0)) if np.ndim(val) == 0 else np.clip(val, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Heat flows (all in Watts), T in degC
    # ------------------------------------------------------------------
    def q_absorbed(self, G, theta_deg=0.0):
        """Absorbed solar power [W] = eta_opt * IAM * G * A."""
        return self.optical_eff * self.iam(theta_deg) * np.asarray(G, float) * self.area

    def q_loss(self, T_c, Ta_c):
        """
        Heat loss [W] = radiation (vacuum-annulus dominant) + residual conduction.
        Stefan-Boltzmann radiation in absolute temperature.
        """
        T_k = np.asarray(T_c, float) + 273.15
        Ta_k = np.asarray(Ta_c, float) + 273.15
        q_rad = self.emissivity * SIGMA * self.A_abs * (T_k**4 - Ta_k**4)
        q_cond = self.U_residual * self.area * (np.asarray(T_c, float) - np.asarray(Ta_c, float))
        return q_rad + q_cond

    def U_L(self, Tm_c, Ta_c):
        """Effective overall loss coefficient [W/m2K], linearised radiation + residual."""
        Tm_k = np.asarray(Tm_c, float) + 273.15
        Ta_k = np.asarray(Ta_c, float) + 273.15
        U_rad = self.emissivity * SIGMA * self.area_ratio * (Tm_k**2 + Ta_k**2) * (Tm_k + Ta_k)
        return U_rad + self.U_residual

    def q_useful(self, T_c, T_in_c):
        """
        Useful heat removed by fluid [W]. Node T is collector mean temperature,
        so T_out = 2*T - T_in (mean closure). Q clamped >= 0 (no reverse heating
        of the absorber by the fluid below ambient is reported as negative gain).
        """
        T_out = 2.0 * np.asarray(T_c, float) - np.asarray(T_in_c, float)
        return self.m_dot * self.cp * (T_out - np.asarray(T_in_c, float))

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, G_func, Ta_func, Tin_func, theta_func):
        T = y[0]
        G = G_func(t)
        Ta = Ta_func(t)
        T_in = Tin_func(t)
        theta = theta_func(t)
        q_abs = self.q_absorbed(G, theta)
        q_loss = self.q_loss(T, Ta)
        q_use = self.q_useful(T, T_in)
        dTdt = (q_abs - q_loss - q_use) / self.C
        return [dTdt]

    @staticmethod
    def _as_func(x):
        """Coerce scalar or callable into a time function."""
        if callable(x):
            return x
        val = float(x)
        return lambda t: val

    def simulate(self, irradiance, T_ambient_c=20.0, T_inlet_c=None,
                 theta_deg=0.0, T0_c=None, dt=10.0, duration_s=3600.0):
        """
        Integrate the lumped-capacitance ODE with scipy.solve_ivp.

        Each of irradiance / T_ambient_c / T_inlet_c / theta_deg may be a scalar
        or a callable f(t)->value (time-varying boundary condition).

        Returns dict of time-series arrays.
        """
        if T_inlet_c is None:
            T_inlet_c = self.T_in_default
        G_func = self._as_func(irradiance)
        Ta_func = self._as_func(T_ambient_c)
        Tin_func = self._as_func(T_inlet_c)
        theta_func = self._as_func(theta_deg)

        if T0_c is None:
            T0_c = Ta_func(0.0)  # start at ambient (cold start)

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [float(T0_c)],
            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-6,
            args=(G_func, Ta_func, Tin_func, theta_func),
            max_step=dt,
        )

        t = sol.t
        T = sol.y[0]
        G = np.array([G_func(ti) for ti in t])
        Ta = np.array([Ta_func(ti) for ti in t])
        Tin = np.array([Tin_func(ti) for ti in t])
        theta = np.array([theta_func(ti) for ti in t])

        q_abs = self.q_absorbed(G, theta)
        q_loss = self.q_loss(T, Ta)
        q_use_raw = self.q_useful(T, Tin)
        q_use = np.maximum(0.0, q_use_raw)
        T_out = 2.0 * T - Tin

        denom = self.area * np.where(G > 1.0, G, 1.0)
        eta = np.where(G > 1.0, q_use / denom, 0.0)
        eta = np.clip(eta, 0.0, self.optical_eff)

        Tm = T
        x_reduced = np.where(G > 1.0, (Tm - Ta) / np.where(G > 1.0, G, 1.0), 0.0)

        return {
            "t": t,
            "T_absorber_c": T,
            "T_outlet_c": T_out,
            "T_mean_c": Tm,
            "q_absorbed_w": q_abs,
            "q_loss_w": q_loss,
            "useful_heat_w": q_use,
            "efficiency": eta,
            "U_L_w_m2k": self.U_L(Tm, Ta),
            "reduced_temp": x_reduced,
            "irradiance": G,
            "T_ambient_c": Ta,
        }

    # ------------------------------------------------------------------
    # Steady-state convenience (root of dT/dt=0) — recovers HWB curve
    # ------------------------------------------------------------------
    def steady_state(self, irradiance, T_ambient_c=20.0, T_inlet_c=None,
                     theta_deg=0.0):
        """
        Long-time solution: integrate to steady state and report the
        Hottel-Whillier efficiency point. Useful for the efficiency curve.
        """
        r = self.simulate(irradiance, T_ambient_c, T_inlet_c, theta_deg,
                          dt=30.0, duration_s=7200.0)
        return {k: (v[-1] if isinstance(v, np.ndarray) else v) for k, v in r.items()}

    def efficiency_steady(self, irradiance, T_inlet_c, T_ambient_c, theta_deg=0.0):
        """Steady-state instantaneous efficiency at a single operating point."""
        ss = self.steady_state(irradiance, T_ambient_c, T_inlet_c, theta_deg)
        return float(ss["efficiency"])
