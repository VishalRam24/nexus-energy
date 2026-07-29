"""
EC061 — Unglazed Solar Collector (Pool Heating) — F2a Physics-Lumped

Dynamic lumped (0D) energy balance for an UNGLAZED solar collector used for
pool heating. Because there is no glass cover, the absorber is exposed directly
to the wind, so the thermal-loss coefficient is strongly wind-dependent and the
zero-loss (optical) efficiency is slightly reduced at high wind. Convective,
long-wave radiative and (optionally) evaporative losses are all lumped into an
effective overall loss coefficient that rises sharply with wind speed.

Governing ODE (lumped capacitance, per unit area):

    C_th * dT_p/dt = eta0_eff(u) * G                       (absorbed solar)
                     - U_L(u) * (T_p - Ta)                  (convective + still-air)
                     - eps * sigma * (T_p^4 - T_sky^4)      (long-wave radiation)
                     - h_evap * (T_p - Ta)                  (evaporative, if wetted)
                     - (mdot * cp / A) * (T_p - Tf_in)      (useful heat removed by flow)

with the wind-dependent Hottel–Whillier loss / gain terms (ISO 9806 unglazed):

    U_L(u)    = b0 + b1 * u            (still-air + wind convection slope)
    eta0_eff  = eta0 * (1 - bu * u)    (wind degrades effective optical gain)

The useful (collected) heat delivered to the pool loop is

    Q_use = mdot * cp * (T_p - Tf_in)        [W per unit area, clamped >= 0]

and the instantaneous efficiency

    eta = Q_use / G                          (0 at night, G -> 0)

Steady-state of this ODE reproduces the F1 Hottel–Whillier curve
    eta ≈ eta0_eff - U_L * (Tm - Ta)/G
so efficiency drops sharply with both (Tm - Ta)/G and wind speed u.

Time integration uses scipy.integrate.solve_ivp (LSODA / RK45).

References:
    Duffie, J.A. & Beckman, W.A. (2013). Solar Engineering of Thermal
        Processes, 4th ed., Wiley, ch.6 (flat-plate collector energy balance,
        Hottel–Whillier–Bliss equation, useful gain).
    ISO 9806:2017. Solar energy — Solar thermal collectors — Test methods
        (unglazed collector efficiency with wind-speed dependent coefficients).
    Soltau, H. (1992). Testing the thermal performance of uncovered solar
        collectors. Solar Energy 49(4), 263–272 (wind-dependent loss model).
"""

import numpy as np
from scipy.integrate import solve_ivp


class UnglazedCollectorF2a:
    """Unglazed solar collector — physics-lumped dynamic energy balance."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta0 = u["eta0"]["value"]          # -
        self.b0 = u["b0"]["value"]              # W/m2K  (still-air loss)
        self.b1 = u["b1"]["value"]              # W*s/m3*K  (wind slope)
        self.bu = u["bu"]["value"]              # s/m  (optical wind degradation)
        self.A = u["A"]["value"]                # m2
        self.C_th = u["C_th"]["value"]          # J/m2K
        self.mdot = u["mdot"]["value"]          # kg/s per m2
        self.cp = u["cp_water"]["value"]        # J/kgK
        self.eps = u["epsilon"]["value"]        # -
        self.sigma = u["sigma"]["value"]        # W/m2K4
        self.h_evap = u["h_evap"]["value"]      # W/m2K

    # ------------------------------------------------------------------
    # Wind-dependent coefficients (ISO 9806 unglazed)
    # ------------------------------------------------------------------
    def loss_coefficient(self, u_wind):
        """Overall convective loss coefficient U_L(u) [W/m2K]."""
        return self.b0 + self.b1 * float(u_wind)

    def optical_efficiency(self, u_wind):
        """Effective zero-loss optical efficiency, degraded by wind [-]."""
        eta = self.eta0 * (1.0 - self.bu * float(u_wind))
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Net flux on the absorber [W/m2] at plate temperature Tp_C (degC)
    # ------------------------------------------------------------------
    def _net_flux(self, Tp_C, G, Ta_C, Tsky_C, u_wind, Tf_in_C):
        """Net heat flux into the lumped thermal mass [W/m2]."""
        eta0_eff = self.optical_efficiency(u_wind)
        U_L = self.loss_coefficient(u_wind)

        absorbed = eta0_eff * G
        conv_loss = U_L * (Tp_C - Ta_C)

        Tp_K = Tp_C + 273.15
        Tsky_K = Tsky_C + 273.15
        rad_loss = self.eps * self.sigma * (Tp_K ** 4 - Tsky_K ** 4)

        evap_loss = self.h_evap * (Tp_C - Ta_C)

        # useful heat extracted by the circulating pool water
        q_use = (self.mdot * self.cp) * (Tp_C - Tf_in_C)
        q_use = max(q_use, 0.0)

        return absorbed - conv_loss - rad_loss - evap_loss - q_use

    def useful_heat(self, Tp_C, Tf_in_C):
        """Useful collected heat delivered to pool loop [W/m2], clamped >=0."""
        q = (self.mdot * self.cp) * (Tp_C - Tf_in_C)
        return max(q, 0.0)

    # ------------------------------------------------------------------
    # Steady-state plate temperature (root of net flux = 0)
    # ------------------------------------------------------------------
    def steady_state(self, G, Ta_C=20.0, Tsky_C=None, u_wind=1.0, Tf_in_C=None):
        """Bisection for steady-state plate temperature [degC]."""
        if Tsky_C is None:
            Tsky_C = Ta_C - 6.0     # clear-sky approx (Duffie & Beckman)
        if Tf_in_C is None:
            Tf_in_C = Ta_C

        lo, hi = -30.0, 150.0
        f_lo = self._net_flux(lo, G, Ta_C, Tsky_C, u_wind, Tf_in_C)
        f_hi = self._net_flux(hi, G, Ta_C, Tsky_C, u_wind, Tf_in_C)
        if f_lo * f_hi > 0:
            # monotone net flux — return the bound closer to zero
            return lo if abs(f_lo) < abs(f_hi) else hi
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = self._net_flux(mid, G, Ta_C, Tsky_C, u_wind, Tf_in_C)
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------
    # Dynamic simulation via solve_ivp
    # ------------------------------------------------------------------
    def simulate(self, G, Ta=20.0, Tsky=None, u_wind=1.0, Tf_in=None,
                 Tp0=None, dt=60.0, duration_s=3600.0):
        """
        Integrate the lumped energy-balance ODE.

        Any of G, Ta, Tsky, u_wind, Tf_in may be a scalar or a callable f(t).
        Returns dict of time-series arrays.
        """
        def as_fn(x, default=None):
            if x is None:
                x = default
            if callable(x):
                return x
            val = float(x)
            return lambda t: val

        G_f = as_fn(G, 0.0)
        Ta_f = as_fn(Ta, 20.0)
        u_f = as_fn(u_wind, 1.0)
        # sky default tied to Ta if not given
        if Tsky is None:
            Tsky_f = lambda t: Ta_f(t) - 6.0
        else:
            Tsky_f = as_fn(Tsky)
        if Tf_in is None:
            Tf_in_f = lambda t: Ta_f(t)
        else:
            Tf_in_f = as_fn(Tf_in)

        if Tp0 is None:
            Tp0 = Ta_f(0.0)

        def rhs(t, y):
            Tp = y[0]
            flux = self._net_flux(Tp, G_f(t), Ta_f(t), Tsky_f(t),
                                  u_f(t), Tf_in_f(t))
            return [flux / self.C_th]

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        sol = solve_ivp(rhs, (0.0, t_eval[-1]), [Tp0], t_eval=t_eval,
                        method="LSODA", rtol=1e-6, atol=1e-6, max_step=dt)

        t = sol.t
        Tp = sol.y[0]

        Gv = np.array([G_f(ti) for ti in t])
        Tav = np.array([Ta_f(ti) for ti in t])
        uv = np.array([u_f(ti) for ti in t])
        Tfv = np.array([Tf_in_f(ti) for ti in t])

        q_use = np.maximum((self.mdot * self.cp) * (Tp - Tfv), 0.0)   # W/m2
        Q_use_W = q_use * self.A                                       # W (panel)

        G_safe = np.where(Gv < 1e-6, 1.0, Gv)
        eta = np.where(Gv < 1e-6, 0.0, q_use / G_safe)
        eta = np.clip(eta, 0.0, 1.0)

        U_L = self.b0 + self.b1 * uv
        eta0_eff = np.clip(self.eta0 * (1.0 - self.bu * uv), 0.0, None)

        return {
            "t": t,
            "T_plate": Tp,           # degC
            "q_use": q_use,          # W/m2
            "Q_use_W": Q_use_W,      # W (full panel)
            "eta": eta,              # -
            "U_L": U_L,              # W/m2K
            "eta0_eff": eta0_eff,    # -
            "G": Gv,
            "Ta": Tav,
            "u_wind": uv,
        }
