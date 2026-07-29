"""
EC058 -- Flat Plate Solar Collector -- F2a Dynamic Lumped-Capacitance Thermal Model

Dynamic energy balance on a single thermal node (lumped capacitance):

    (C_eff * A_c) * dT_m/dt = Q_solar - Q_loss - Q_useful

where:
    T_m   = mean collector temperature = (T_in + T_out) / 2
    Q_solar  = A_c * F_R * tau_alpha * K(theta) * G_T
    Q_loss   = A_c * [a1*(T_m - T_amb) + a2*(T_m - T_amb)^2]
    Q_useful = m_dot * cp * (T_out - T_in)

The collector effectiveness couples T_out to T_m:
    Q_useful = epsilon * (m_dot * cp) * (T_m - T_in)
    where epsilon = A_c * F_R * U_L / (m_dot * cp) ... capped at 1

This allows the ODE to be expressed purely in terms of T_m, which is
integrated forward in time using scipy.integrate.solve_ivp.

Improvements over F1a/F1b (steady-state):
    - Tracks thermal inertia (startup, cloud transients, cooldown)
    - Temperature-dependent loss coefficient (a1 + a2*dT)
    - IAM correction for off-normal incidence
    - Time-resolved outlet temperature and useful heat

References:
    Duffie & Beckman (2013), 'Solar Engineering of Thermal Processes',
    4th ed., John Wiley & Sons, Ch. 6.
    Fischer et al. (2004), 'Dynamic testing of solar thermal collectors',
    Solar Energy 76(1-3), 117-123.
"""

import numpy as np
from scipy.integrate import solve_ivp


class FlatPlateCollectorF2a:
    """Flat-plate solar collector -- dynamic lumped-capacitance thermal model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_c = u["A_c"]["value"]               # m2
        self.eta_0 = u["eta_0"]["value"]            # optical efficiency
        self.a1 = u["a1"]["value"]                  # W/(m2.K)
        self.a2 = u["a2"]["value"]                  # W/(m2.K2)
        self.tau_alpha_n = u["tau_alpha_n"]["value"] # tau*alpha at normal incidence
        self.b0 = u["b0"]["value"]                  # IAM coefficient
        self.C_eff = u["C_eff"]["value"]            # J/(m2.K) effective capacitance
        self.m_dot_spec = u["m_dot_spec"]["value"]  # kg/(s.m2)
        self.cp = u["cp_fluid"]["value"]            # J/(kg.K)
        self.F_R = u["F_R"]["value"]                # heat removal factor

        # Derived
        self.m_dot = self.m_dot_spec * self.A_c     # kg/s total flow
        self.C_total = self.C_eff * self.A_c        # J/K total capacitance

    # ----- IAM -----
    def iam(self, theta_deg):
        """
        Incidence Angle Modifier: K(theta) = 1 - b0*(1/cos(theta) - 1).
        Returns 0 for theta >= 80 deg.
        """
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        cos_theta = np.cos(theta_rad)
        safe_cos = np.where(theta < 80.0, np.maximum(cos_theta, 0.01), 0.01)
        k = 1.0 - self.b0 * (1.0 / safe_cos - 1.0)
        k = np.where(theta < 80.0, k, 0.0)
        return np.clip(k, 0.0, 1.0)

    # ----- Loss coefficient (temperature-dependent) -----
    def U_L(self, dT):
        """Overall loss coefficient U_L = a1 + a2*dT [W/(m2.K)]."""
        return self.a1 + self.a2 * np.abs(dT)

    # ----- Collector effectiveness -----
    def effectiveness(self, m_dot, T_m, T_amb):
        """
        Heat-exchanger effectiveness relating T_out to T_m.
        epsilon = A_c * F_R * U_L / (m_dot * cp), capped at 1.
        """
        dT = T_m - T_amb
        u_l = self.U_L(dT)
        if m_dot < 1e-8:
            return 0.0
        eps = self.A_c * self.F_R * u_l / (m_dot * self.cp)
        return min(eps, 1.0)

    # ----- Steady-state solution (for validation) -----
    def steady_state(self, G_T, T_in, T_amb, theta_deg=0.0):
        """
        Steady-state solution matching Hottel-Whillier with IAM.

        Returns dict with Q_useful, T_out, efficiency, T_mean.
        """
        K = float(self.iam(theta_deg))
        tau_alpha_eff = self.tau_alpha_n * K

        # HWB: Q_u = A_c * F_R * [tau_alpha * G - U_L * (T_in - T_amb)]
        # With temperature-dependent U_L, iterate
        T_m_guess = T_in
        for _ in range(20):
            dT_loss = T_m_guess - T_amb
            u_l = self.U_L(dT_loss)
            Q_u = self.A_c * self.F_R * (tau_alpha_eff * G_T - u_l * (T_in - T_amb))
            Q_u = max(0.0, Q_u)
            T_out = T_in + Q_u / (self.m_dot * self.cp) if self.m_dot > 1e-8 else T_in
            T_m_new = (T_in + T_out) / 2.0
            if abs(T_m_new - T_m_guess) < 1e-6:
                break
            T_m_guess = T_m_new

        eta = Q_u / (self.A_c * G_T) if G_T > 1.0 else 0.0
        return {
            "Q_useful_W": Q_u,
            "T_outlet_C": T_out,
            "T_mean_C": T_m_guess,
            "efficiency": eta,
        }

    # ----- Dynamic ODE RHS -----
    def _rhs(self, t, T_m, G_func, T_in_func, T_amb_func, theta_func, m_dot):
        """
        Right-hand side of the energy balance ODE.

        dT_m/dt = [Q_solar - Q_loss - Q_useful] / C_total
        """
        G_T = G_func(t)
        T_in = T_in_func(t)
        T_amb = T_amb_func(t)
        theta = theta_func(t)

        K = float(self.iam(theta))
        tau_alpha_eff = self.tau_alpha_n * K

        dT = T_m - T_amb
        u_l = self.a1 + self.a2 * abs(dT)

        # Solar gain
        Q_solar = self.A_c * self.F_R * tau_alpha_eff * G_T

        # Thermal losses
        Q_loss = self.A_c * (self.a1 * dT + self.a2 * dT ** 2)

        # Useful heat extraction via effectiveness
        eps = self.A_c * self.F_R * u_l / (m_dot * self.cp) if m_dot > 1e-8 else 0.0
        eps = min(eps, 1.0)
        Q_useful = eps * m_dot * self.cp * (T_m - T_in) if m_dot > 1e-8 else 0.0
        Q_useful = max(0.0, Q_useful)

        dTm_dt = (Q_solar - Q_loss - Q_useful) / self.C_total
        return dTm_dt

    # ----- Dynamic simulation -----
    def simulate(self, t_span, t_eval, T_m0, G_func, T_in_func, T_amb_func,
                 theta_func=None, m_dot=None, method="RK45", max_step=60.0):
        """
        Run dynamic simulation over a time span.

        Parameters
        ----------
        t_span    : (t_start, t_end) in seconds
        t_eval    : array of times at which to report solution [s]
        T_m0      : initial mean collector temperature [degC]
        G_func    : callable(t) -> irradiance [W/m2]
        T_in_func : callable(t) -> inlet temperature [degC]
        T_amb_func: callable(t) -> ambient temperature [degC]
        theta_func: callable(t) -> incidence angle [deg], default 0
        m_dot     : mass flow rate [kg/s], default self.m_dot
        method    : ODE solver method
        max_step  : maximum time step [s]

        Returns
        -------
        dict with keys: t, T_mean, T_outlet, Q_useful, Q_loss, Q_solar, efficiency
        """
        if theta_func is None:
            theta_func = lambda t: 0.0
        if m_dot is None:
            m_dot = self.m_dot

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y[0], G_func, T_in_func, T_amb_func,
                                        theta_func, m_dot),
            t_span=t_span,
            y0=[T_m0],
            t_eval=t_eval,
            method=method,
            max_step=max_step,
            rtol=1e-8,
            atol=1e-10,
        )

        T_m = sol.y[0]
        n = len(T_m)

        # Reconstruct outputs at each time step
        T_out = np.zeros(n)
        Q_useful = np.zeros(n)
        Q_loss = np.zeros(n)
        Q_solar = np.zeros(n)
        eta = np.zeros(n)

        for i in range(n):
            ti = sol.t[i]
            G_T = G_func(ti)
            T_in = T_in_func(ti)
            T_amb = T_amb_func(ti)
            theta = theta_func(ti)

            K = float(self.iam(theta))
            tau_alpha_eff = self.tau_alpha_n * K

            dT = T_m[i] - T_amb
            u_l = self.a1 + self.a2 * abs(dT)

            Q_solar[i] = self.A_c * self.F_R * tau_alpha_eff * G_T
            Q_loss[i] = self.A_c * (self.a1 * dT + self.a2 * dT ** 2)

            eps = self.A_c * self.F_R * u_l / (m_dot * self.cp) if m_dot > 1e-8 else 0.0
            eps = min(eps, 1.0)
            Q_useful[i] = eps * m_dot * self.cp * (T_m[i] - T_in) if m_dot > 1e-8 else 0.0
            Q_useful[i] = max(0.0, Q_useful[i])

            if m_dot > 1e-8:
                T_out[i] = T_in + Q_useful[i] / (m_dot * self.cp)
            else:
                T_out[i] = T_m[i]

            eta[i] = Q_useful[i] / (self.A_c * G_T) if G_T > 1.0 else 0.0

        return {
            "t": sol.t,
            "T_mean_C": T_m,
            "T_outlet_C": T_out,
            "Q_useful_W": Q_useful,
            "Q_loss_W": Q_loss,
            "Q_solar_W": Q_solar,
            "efficiency": eta,
            "solver_success": sol.success,
            "solver_message": sol.message,
        }
