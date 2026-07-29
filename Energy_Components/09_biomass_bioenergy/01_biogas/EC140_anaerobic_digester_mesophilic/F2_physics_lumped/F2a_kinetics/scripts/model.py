"""
EC140 -- Anaerobic Digester (Mesophilic) -- F2a Monod Kinetics Model

Simplified ADM1 with three state variables and Monod kinetics.

State equations:
    dS/dt = (S_in - S)/HRT - (mu_max * S / (K_s + S)) * X / Y_xs
    dX/dt = (mu_max * S / (K_s + S)) * X - k_d * X - X/HRT
    (optionally): dV_ch4_cum/dt = Y_ch4 * (mu_max * S / (K_s + S)) * X * V_reactor

Monod growth:
    mu = mu_max * S / (K_s + S) * f_T * f_pH

Temperature correction (Arrhenius):
    f_T = exp(E_a / R * (1/T_ref - 1/T))

pH inhibition (empirical bell curve):
    f_pH = exp(-3 * ((pH - pH_opt) / (pH_high - pH_low))^2)  for pH outside optimum

Methane production rate:
    V_ch4 = Y_ch4 * mu / Y_xs * X * V_reactor   [L/d at STP]

Reference:
    Batstone, D.J. et al. (2002). IWA Anaerobic Digestion Model No. 1 (ADM1).
    Rittmann, B.E. & McCarty, P.L. (2001). Environmental Biotechnology. McGraw-Hill.
"""

import numpy as np
from scipy.integrate import solve_ivp


class AnaerobicDigesterF2a:
    """Anaerobic digester -- simplified ADM1 with Monod kinetics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_r = u["V_reactor"]["value"]
        self.HRT = u["HRT"]["value"]
        self.S_in = u["S_in"]["value"]
        self.X_in = u.get("X_in", {}).get("value", 0.0)

        self.mu_max = u["mu_max"]["value"]
        self.K_s = u["K_s"]["value"]
        self.Y_xs = u["Y_xs"]["value"]
        self.k_d = u["k_d"]["value"]
        self.Y_ch4 = u["Y_ch4"]["value"]

        self.T_ref = u["T_ref"]["value"]
        self.E_a = u["E_a"]["value"]
        self.R_gas = u["R_gas"]["value"]

        self.pH_opt = u["pH_opt"]["value"]
        self.pH_low = u["pH_low"]["value"]
        self.pH_high = u["pH_high"]["value"]

        self.T_op = u["T_operating"]["value"]
        self.S0 = u["S0"]["value"]
        self.X0 = u["X0"]["value"]

    # ---- Kinetic rate functions ----

    def monod(self, S):
        """Monod specific growth rate (without corrections)."""
        return self.mu_max * S / (self.K_s + S)

    def temperature_factor(self, T=None):
        """
        Arrhenius temperature correction factor.

        f_T = exp(E_a/R * (1/T_ref - 1/T))
        f_T = 1.0 at T = T_ref
        """
        if T is None:
            T = self.T_op
        return np.exp(self.E_a / self.R_gas * (1.0 / self.T_ref - 1.0 / T))

    def ph_inhibition(self, pH=None):
        """
        pH inhibition factor (empirical bell curve).

        f_pH = exp(-3 * ((pH - pH_opt) / (pH_high - pH_low))^2)
        Returns 1.0 at optimal pH.
        """
        if pH is None:
            return 1.0  # assume optimal
        delta = (pH - self.pH_opt) / (self.pH_high - self.pH_low)
        return np.exp(-3.0 * delta ** 2)

    def effective_mu(self, S, T=None, pH=None):
        """Effective specific growth rate with all corrections."""
        return self.monod(S) * self.temperature_factor(T) * self.ph_inhibition(pH)

    # ---- Steady-state analysis ----

    def steady_state(self, S_in=None, HRT=None, T=None, pH=None):
        """
        Analytical steady-state solution (set all derivatives to zero).

        From dX/dt = 0:  mu = k_d + 1/HRT
        From Monod:       S_ss = K_s * mu / (mu_max_eff - mu)
        From dS/dt = 0:   X_ss = Y_xs * (S_in - S_ss) / (1 + k_d * HRT)
                          (accounting for washout via HRT)
        """
        if S_in is None:
            S_in = self.S_in
        if HRT is None:
            HRT = self.HRT

        f_T = self.temperature_factor(T)
        f_pH = self.ph_inhibition(pH)
        mu_max_eff = self.mu_max * f_T * f_pH

        # Minimum HRT to avoid washout
        mu_required = self.k_d + 1.0 / HRT
        if mu_required >= mu_max_eff:
            # Washout: biomass cannot sustain
            return {
                "S_ss": S_in,
                "X_ss": 0.0,
                "V_ch4_d": 0.0,
                "COD_removal_pct": 0.0,
                "washout": True,
                "HRT_min_d": 1.0 / (mu_max_eff - self.k_d) if mu_max_eff > self.k_d else float("inf"),
            }

        S_ss = self.K_s * mu_required / (mu_max_eff - mu_required)
        S_ss = max(S_ss, 0.0)

        # Biomass steady state from dS/dt = 0:
        # (S_in - S_ss)/HRT = mu * X / Y_xs
        # X_ss = Y_xs * (S_in - S_ss) / (mu * HRT)
        X_ss = self.Y_xs * (S_in - S_ss) / (mu_required * HRT)
        X_ss = max(X_ss, 0.0)

        # Methane production
        COD_removed_rate = (S_in - S_ss) * self.V_r / HRT  # gCOD/d (V_r in m^3 = 1000 L)
        # Convert to gCOD/d properly: concentration is gCOD/L, flow = V_r/HRT [m^3/d]
        # But since S is in gCOD/L and V_r in m^3: Q = V_r/HRT [m^3/d], COD_in = S_in * 1000 [g/m^3]
        Q_m3_d = self.V_r / HRT
        COD_removed_g_d = (S_in - S_ss) * 1000.0 * Q_m3_d  # gCOD/d
        V_ch4_d = self.Y_ch4 * COD_removed_g_d  # L_CH4/d at STP

        COD_removal = (S_in - S_ss) / S_in * 100.0 if S_in > 0 else 0.0

        # Energy content: CH4 has ~35.8 MJ/m^3 at STP
        E_ch4_kW = V_ch4_d / 1000.0 * 35800.0 / 86400.0  # kW

        HRT_min = 1.0 / (mu_max_eff - self.k_d)

        return {
            "S_ss": S_ss,
            "X_ss": X_ss,
            "V_ch4_d": V_ch4_d,
            "V_ch4_m3_d": V_ch4_d / 1000.0,
            "COD_removal_pct": COD_removal,
            "COD_removed_g_d": COD_removed_g_d,
            "E_ch4_kW": E_ch4_kW,
            "washout": False,
            "HRT_min_d": HRT_min,
            "mu_eff": mu_required,
        }

    # ---- Dynamic ODE simulation ----

    def derivatives_ode(self, t, state, S_in_func, T_func, pH_func):
        """
        ODE derivatives for anaerobic digester.

        States: [S, X]
            S: substrate concentration [gCOD/L]
            X: biomass concentration [gVSS/L]
        """
        S, X = state
        S = max(S, 0.0)
        X = max(X, 0.0)

        S_in = S_in_func(t) if callable(S_in_func) else S_in_func
        T = T_func(t) if callable(T_func) else T_func
        pH = pH_func(t) if callable(pH_func) else pH_func

        mu = self.effective_mu(S, T, pH)

        # Substrate balance
        dS_dt = (S_in - S) / self.HRT - mu * X / self.Y_xs

        # Biomass balance
        dX_dt = mu * X - self.k_d * X - X / self.HRT

        return [dS_dt, dX_dt]

    def simulate(self, S_in=None, T=None, pH=None, dt=0.1, duration_d=60.0,
                 x0=None, HRT=None):
        """
        Simulate anaerobic digester dynamics.

        Args:
            S_in:       influent substrate [gCOD/L] (scalar or callable(t))
            T:          temperature [K] (scalar or callable(t)), default: T_operating
            pH:         pH (scalar or callable(t)), default: None (optimal)
            dt:         output time step [days]
            duration_d: simulation duration [days]
            x0:         initial state [S0, X0]
            HRT:        hydraulic retention time [d] (default: self.HRT)

        Returns:
            dict with time-series: t, S, X, V_ch4_rate, V_ch4_cumulative, mu_eff, etc.
        """
        if S_in is None:
            S_in = self.S_in
        if T is None:
            T = self.T_op
        if HRT is not None:
            old_HRT = self.HRT
            self.HRT = HRT

        if x0 is None:
            x0 = [self.S0, self.X0]

        _S_in = S_in if callable(S_in) else lambda t: S_in
        _T = T if callable(T) else lambda t: T
        _pH = pH if callable(pH) else lambda t: pH

        t_span = (0.0, duration_d)
        t_eval = np.arange(0.0, duration_d + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_d]

        def rhs(t, state):
            return self.derivatives_ode(t, state, _S_in, _T, _pH)

        sol = solve_ivp(
            rhs, t_span, x0, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt * 5,
        )

        if HRT is not None:
            self.HRT = old_HRT  # restore

        t = sol.t
        S = np.maximum(sol.y[0], 0.0)
        X = np.maximum(sol.y[1], 0.0)

        # Compute methane production at each time step
        mu_arr = np.array([
            self.effective_mu(S[i], _T(t[i]), _pH(t[i]))
            for i in range(len(t))
        ])

        # COD removal rate [gCOD/L/d] and methane production [L/d]
        COD_removal_rate = mu_arr * X / self.Y_xs
        # V_ch4 = Y_ch4 * COD_removal_rate * V_reactor * 1000 (L per m^3)
        V_ch4_rate = self.Y_ch4 * COD_removal_rate * self.V_r * 1000.0  # L_CH4/d

        # Cumulative methane
        V_ch4_cum = np.zeros_like(t)
        if len(t) > 1:
            dt_arr = np.diff(t)
            V_ch4_cum[1:] = np.cumsum(V_ch4_rate[1:] * dt_arr)

        S_in_arr = np.array([_S_in(ti) for ti in t])
        COD_removal_pct = np.where(S_in_arr > 0, (S_in_arr - S) / S_in_arr * 100, 0.0)

        return {
            "t": t,
            "S": S,
            "X": X,
            "mu_eff": mu_arr,
            "V_ch4_rate_L_d": V_ch4_rate,
            "V_ch4_cumulative_L": V_ch4_cum,
            "COD_removal_pct": COD_removal_pct,
            "S_in": S_in_arr,
        }
