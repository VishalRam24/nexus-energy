"""
EC215 -- Solar Still / Humidification-Dehumidification (HDH)
F2a -- Single-Basin Solar Still, Dunkle Lumped Energy Balance

Physics-lumped (0D) transient model of a passive single-basin solar still.
Two coupled lumped capacitances -- the saline basin water and the glass
cover -- exchange heat by the three classic interior modes of Dunkle (1961):

    1. Evaporative   q_ew  (latent transport of vapour water -> cover)
    2. Convective    q_cw  (free convection across the humid air gap)
    3. Radiative     q_rw  (long-wave exchange, parallel-plate)

plus solar gain and external losses. The instantaneous distillate
production rate follows directly from the evaporative flux divided by the
latent heat of vaporisation (mass from latent heat), guaranteeing that the
yield tracks solar input and vanishes at night (G = 0 => q_ew -> 0 once the
water cools toward the cover).

State (ODE integrated with scipy.integrate.solve_ivp):
    T_w  : basin water temperature [K]
    T_g  : glass cover temperature [K]
    m_d  : cumulative distillate mass [kg]   (monotone accumulator)

Energy balances (per Dunkle 1961; Tiwari & Tiwari 2007, Ch.3):
    m_w cp_w dT_w/dt = A * [ alpha_w * tau_g * G                       (solar in)
                              - (q_ew + q_cw + q_rw)                    (to cover)
                              - U_b (T_w - T_amb) ]                     (basin loss)
    m_g cp_g dT_g/dt = A * [ alpha_g * G                               (solar in)
                              + (q_ew + q_cw + q_rw)                    (from water)
                              - h_ga (T_g - T_amb) ]                    (top loss)
    dm_d/dt          = A * q_ew / h_fg                                  (latent->mass)

Dunkle interior-correlation closure (evaporative & convective):
    q_cw = h_cw (T_w - T_g)
    h_cw = 0.884 * [ (T_w - T_g)
                     + (P_w - P_g)(T_w+273.15') / (268.9e3 - P_w) ]^(1/3)   [W/m2K]
    q_ew = 16.273e-3 * h_cw * (P_w - P_g)                                    [W/m2]
where P = saturation vapour pressure [Pa] at the local surface temperature
(Antoine/Tiwari fit), and the radiative term is the parallel-plate grey body:
    q_rw = eps_eff * sigma (T_w^4 - T_g^4)

References:
    Dunkle, R.V. (1961). Solar water distillation: the roof-type still and a
        multiple effect diffusion still. Int. Developments in Heat Transfer,
        ASME Proc., Part V, 895-902.
    Cooper, P.I. (1973). The maximum efficiency of single-effect solar stills.
        Solar Energy 12(3), 313-331.
    Tiwari, G.N. & Tiwari, A.K. (2007). Solar Distillation Practice for Water
        Desalination Systems. Anamaya/Springer.
"""

import numpy as np
from scipy.integrate import solve_ivp

SIGMA = 5.670374419e-8  # Stefan-Boltzmann [W/(m2.K4)]


class SolarStillF2a:
    """Single-basin solar still -- Dunkle lumped two-node energy balance."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_basin"]["value"]              # m2 (water surface)
        self.A_g = u["A_glass"]["value"]            # m2 (cover area)
        self.m_w = u["m_water"]["value"]            # kg
        self.cp_w = u["cp_water"]["value"]          # J/(kg.K)
        self.m_g = u["m_glass"]["value"]            # kg
        self.cp_g = u["cp_glass"]["value"]          # J/(kg.K)
        self.alpha_w = u["alpha_water"]["value"]
        self.alpha_g = u["alpha_glass"]["value"]
        self.tau_g = u["tau_glass"]["value"]
        self.eps_w = u["eps_water"]["value"]
        self.eps_g = u["eps_glass"]["value"]
        self.h_ga = u["h_glass_amb"]["value"]       # W/(m2.K)
        self.U_b = u["U_basin_amb"]["value"]        # W/(m2.K)
        self.T_amb = u["T_amb_K"]["value"]          # K
        self.G_peak = u["G_peak"]["value"]          # W/m2

        # effective grey-body emissivity for parallel surfaces
        self.eps_eff = 1.0 / (1.0 / self.eps_w + 1.0 / self.eps_g - 1.0)

    # ------------------------------------------------------------------
    # Water properties (hardcoded, cited)
    # ------------------------------------------------------------------
    @staticmethod
    def latent_heat(T_K):
        """Latent heat of vaporisation of water [J/kg].

        Tiwari & Tiwari (2007) cubic fit used in still analyses, valid
        0-100 C (T in deg C); h_fg falls from ~2.50 MJ/kg at 0 C to
        ~2.26 MJ/kg at 100 C:
            h_fg = 2.5036e6 - 2369.2 T + 0.2657 T^2 - 1.5266e-3 T^3
        """
        T = T_K - 273.15
        return (2.5036e6 - 2369.2 * T + 0.2657 * T**2 - 1.5266e-3 * T**3)

    @staticmethod
    def p_sat(T_K):
        """Saturation vapour pressure of water [Pa].

        Tiwari & Tiwari (2007) exponential fit used in still analyses:
            P = exp(25.317 - 5144 / T_K)                    [Pa], T in K
        Equivalent to the Antoine form over the 10-90 C still range.
        """
        return np.exp(25.317 - 5144.0 / T_K)

    # ------------------------------------------------------------------
    # Dunkle interior heat-transfer correlations
    # ------------------------------------------------------------------
    def h_conv(self, T_w, T_g):
        """Internal free-convection coefficient water->cover [W/(m2.K)].

        Dunkle (1961) correlation (buoyancy of humid air in the still cavity).
        """
        Pw = self.p_sat(T_w)
        Pg = self.p_sat(T_g)
        dT = T_w - T_g
        # buoyancy-augmented driving term; guard against negative/zero
        term = dT + (Pw - Pg) * T_w / (268.9e3 - Pw)
        if term <= 0.0:
            return 0.0
        return 0.884 * term ** (1.0 / 3.0)

    def q_conv(self, T_w, T_g):
        """Convective flux water->cover [W/m2]."""
        return self.h_conv(T_w, T_g) * (T_w - T_g)

    def q_evap(self, T_w, T_g):
        """Evaporative flux water->cover [W/m2] (Dunkle 1961).

        q_ew = 16.273e-3 * h_cw * (P_w - P_g);  only positive (no
        re-condensation back onto basin) -> distillate is one-way.
        """
        hcw = self.h_conv(T_w, T_g)
        Pw = self.p_sat(T_w)
        Pg = self.p_sat(T_g)
        q = 16.273e-3 * hcw * (Pw - Pg)
        return max(q, 0.0)

    def q_rad(self, T_w, T_g):
        """Radiative flux water->cover [W/m2] (grey parallel plates)."""
        return self.eps_eff * SIGMA * (T_w**4 - T_g**4)

    # ------------------------------------------------------------------
    # Solar input profile (diurnal half-sine, zero at night)
    # ------------------------------------------------------------------
    def irradiance(self, t, G_peak=None, day_s=86400.0,
                   sunrise_frac=0.25, sunset_frac=0.75):
        """Solar irradiance G(t) [W/m2].

        Half-sine clear-sky profile between sunrise and sunset, exactly 0
        outside daylight -> enforces Q=0 (no distillate) at night.
        """
        Gp = self.G_peak if G_peak is None else G_peak
        tod = np.mod(t, day_s) / day_s
        if tod < sunrise_frac or tod > sunset_frac:
            return 0.0
        phase = (tod - sunrise_frac) / (sunset_frac - sunrise_frac)
        return Gp * np.sin(np.pi * phase)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, G_func):
        T_w, T_g, _ = y
        G = G_func(t)

        q_ew = self.q_evap(T_w, T_g)
        q_cw = self.q_conv(T_w, T_g)
        q_rw = self.q_rad(T_w, T_g)
        q_int = q_ew + q_cw + q_rw           # total water->cover [W/m2]

        # Water node
        solar_w = self.alpha_w * self.tau_g * G
        loss_b = self.U_b * (T_w - self.T_amb)
        dTw = (self.A * (solar_w - q_int - loss_b)) / (self.m_w * self.cp_w)

        # Glass node
        solar_g = self.alpha_g * G
        loss_t = self.h_ga * (T_g - self.T_amb)
        dTg = (self.A * solar_g + self.A * q_int - self.A_g * loss_t) \
            / (self.m_g * self.cp_g)

        # Distillate accumulation (mass from latent heat)
        h_fg = self.latent_heat(T_w)
        dmd = self.A * q_ew / h_fg

        return [dTw, dTg, dmd]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, G_peak=None, T_w0=None, T_g0=None, T_amb=None,
                 duration_s=86400.0, dt=600.0):
        """Integrate the still over `duration_s`.

        Parameters
        ----------
        G_peak : float or callable(t)->W/m2
            Peak irradiance (half-sine diurnal) OR a custom G(t) profile.
        T_w0, T_g0 : float
            Initial water / glass temperatures [K] (default = ambient).
        T_amb : float
            Ambient temperature [K] (override of parameter file).
        duration_s : float
            Simulation horizon [s] (default one day).
        dt : float
            Output sampling interval [s].

        Returns
        -------
        dict of time-series arrays + scalar daily_yield_L_m2.
        """
        if T_amb is not None:
            self.T_amb = T_amb
        if T_w0 is None:
            T_w0 = self.T_amb
        if T_g0 is None:
            T_g0 = self.T_amb

        if callable(G_peak):
            G_func = G_peak
        else:
            Gp = G_peak
            G_func = lambda t: self.irradiance(t, G_peak=Gp)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T_w0, T_g0, 0.0],
            t_eval=t_eval, args=(G_func,),
            method="LSODA", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        T_w = sol.y[0]
        T_g = sol.y[1]
        m_d = sol.y[2]            # cumulative kg
        N = len(t)

        G = np.array([G_func(ti) for ti in t])
        q_ew = np.zeros(N)
        q_cw = np.zeros(N)
        q_rw = np.zeros(N)
        rate = np.zeros(N)        # instantaneous distillate [kg/s]
        for i in range(N):
            q_ew[i] = self.q_evap(T_w[i], T_g[i])
            q_cw[i] = self.q_conv(T_w[i], T_g[i])
            q_rw[i] = self.q_rad(T_w[i], T_g[i])
            rate[i] = self.A * q_ew[i] / self.latent_heat(T_w[i])

        daily_yield_kg = float(m_d[-1])
        # productivity per m2 of basin (water density ~1 kg/L)
        daily_yield_L_m2 = daily_yield_kg / self.A

        return {
            "t": t,
            "T_water": T_w,
            "T_glass": T_g,
            "G": G,
            "q_evap": q_ew,
            "q_conv": q_cw,
            "q_rad": q_rw,
            "distillate_rate_kg_s": rate,
            "distillate_rate_L_h": rate * 3600.0,
            "cumulative_distillate_kg": m_d,
            "daily_yield_kg": daily_yield_kg,
            "daily_yield_L_m2": daily_yield_L_m2,
        }
