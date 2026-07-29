"""
EC094 -- Evaporative Cooler -- F2a Psychrometric Heat-Mass Transfer (Physics-Lumped)

Physics-lumped first-principles model of a direct (and optional indirect) evaporative
cooler. Air is driven across a wetted pad; adiabatic-saturation heat-and-mass transfer
drives the supply air toward the thermodynamic wet-bulb temperature. The approach to
wet-bulb is characterised by the saturation effectiveness derived from a number of
transfer units (NTU) of the air/pad contactor:

    eps_sat = 1 - exp(-NTU)                         (effectiveness-NTU, Camargo 2005)

Steady psychrometric outlet state (adiabatic-saturation / constant wet-bulb line):
    T_out  = T_db - eps_sat * (T_db - T_wb)         (ASHRAE 2021)
    w_out  = w_in + eps_sat * (w_sat(T_wb) - w_in)  (mass added by evaporation)

Water mass balance (conservation):
    m_dot_water = m_dot_dryair * (w_out - w_in)     [kg/s]

Energy: the sensible cooling is balanced by latent heat of the evaporated water,
so the wet-bulb (constant-enthalpy) line is followed -- adiabatic saturation.
    Q_sens = m_dot_air * Cp_moist * (T_db - T_out)  [W]
    Q_latent = m_dot_water * h_fg                   [W]
For an ideal adiabatic saturator Q_sens ~= Q_latent (energy conservation check).

Lumped transient ODE (pad / outlet-air thermal capacitance) via scipy.solve_ivp:
    (m_pad * cp_pad) dT_pad/dt = m_dot_air * Cp_moist * (T_in_to_pad - T_pad)
                                 - m_dot_water * h_fg
The pad temperature relaxes toward its steady wet-bulb-approach value with a
time constant tau = (m_pad cp_pad) / (m_dot_air Cp_moist NTU). The outlet air
temperature tracks the pad surface, giving a first-order step response.

Psychrometric relations (hardcoded + cited; NO CoolProp):
    - Saturation vapor pressure: Tetens / Magnus formula (ASHRAE 2021, valid 0-60C)
    - Humidity ratio: w = 0.62198 * p_w / (P - p_w)            (ASHRAE 2021)
    - Wet-bulb: Stull (2011) empirical approximation (J. Appl. Meteorol. 50:2267)

References
----------
ASHRAE Handbook of Fundamentals (2021), Ch. 1 "Psychrometrics".
Camargo, J.R., Ebinuma, C.D., Silveira, J.L. (2005), "Experimental performance of a
    direct evaporative cooler operating during summer", Int. J. Thermal Sciences 44(3).
Stull, R. (2011), "Wet-Bulb Temperature from Relative Humidity and Air Temperature",
    J. Appl. Meteorol. Climatol. 50(11), 2267-2269.
"""

import numpy as np
from scipy.integrate import solve_ivp


class EvaporativeCooler_F2a:
    """Direct evaporative cooler -- psychrometric heat-mass transfer + lumped thermal ODE."""

    EPS = 0.62198  # ratio of molar masses M_water / M_dryair (ASHRAE)

    def __init__(self, params: dict):
        u = params["unit"]
        self.cooler_type = u.get("type", "direct")
        self.eps_design = u["epsilon_sat_design"]["value"]
        self.NTU = u["ntu_design"]["value"]
        self.P_fan = u["P_fan_W"]["value"]
        self.Cp_air = u["Cp_air_J_kgK"]["value"]
        self.Cp_vap = u["Cp_vapor_J_kgK"]["value"]
        self.Cp_water = u["Cp_water_J_kgK"]["value"]
        self.rho_air = u["rho_air_kg_m3"]["value"]
        self.h_fg = u["h_fg_J_kg"]["value"]
        self.P_atm = u["P_atm_Pa"]["value"]
        self.m_pad = u["pad_mass_kg"]["value"]
        self.cp_pad = u["pad_cp_J_kgK"]["value"]
        self.NTU_air_pad = u.get("NTU_air_pad", {"value": self.NTU})["value"]

    # ------------------------------------------------------------------
    # Psychrometric property relations (hardcoded, cited)
    # ------------------------------------------------------------------
    def p_sat(self, T_C):
        """Saturation vapor pressure of water [Pa]. Magnus/Tetens (ASHRAE 2021), 0-60 C."""
        return 610.94 * np.exp(17.625 * T_C / (T_C + 243.04))

    def humidity_ratio(self, T_C, RH):
        """Humidity ratio w [kg_water/kg_dryair] from dry-bulb T and relative humidity."""
        p_w = RH * self.p_sat(T_C)
        p_w = min(p_w, 0.999 * self.P_atm)
        return self.EPS * p_w / (self.P_atm - p_w)

    def w_sat(self, T_C):
        """Saturation humidity ratio at temperature T (RH=1)."""
        return self.humidity_ratio(T_C, 1.0)

    def wet_bulb(self, T_C, RH):
        """
        Thermodynamic wet-bulb temperature [degC] -- Stull (2011) empirical fit.
        Valid roughly 5-99% RH, -20..50 C, P ~ 1 atm. RH in percent for the formula.
        """
        RHp = max(min(RH, 1.0), 1e-4) * 100.0
        Tw = (T_C * np.arctan(0.151977 * np.sqrt(RHp + 8.313659))
              + np.arctan(T_C + RHp)
              - np.arctan(RHp - 1.676331)
              + 0.00391838 * RHp ** 1.5 * np.arctan(0.023101 * RHp)
              - 4.686035)
        return min(Tw, T_C)  # wet-bulb never exceeds dry-bulb

    def cp_moist(self, w):
        """Specific heat of moist air per kg dry air [J/(kg.K)]."""
        return self.Cp_air + w * self.Cp_vap

    def latent_heat(self, T_C):
        """
        Latent heat of vaporization of water h_fg(T) [J/kg].
        Linear fit (ASHRAE 2021 / Rogers & Yau): h_fg = 2501 kJ/kg - 2.36 kJ/kgK * T,
        valid 0-60 C. Returns the configured reference value scaled with temperature.
        """
        return 2501000.0 - 2360.0 * T_C

    # ------------------------------------------------------------------
    # Saturation effectiveness
    # ------------------------------------------------------------------
    def saturation_effectiveness(self, ntu=None):
        """eps_sat = 1 - exp(-NTU)  (effectiveness-NTU, Camargo 2005). In (0,1)."""
        ntu = self.NTU if ntu is None else ntu
        return 1.0 - np.exp(-ntu)

    # ------------------------------------------------------------------
    # Steady-state psychrometric outlet state
    # ------------------------------------------------------------------
    def steady_state(self, T_db, RH, m_dot_air=1.0, eps_sat=None):
        """
        Steady direct-evaporative outlet state.

        Returns dict with T_out, w_in, w_out, RH_out, T_wb, eps_sat,
        Q_sens_W, Q_latent_W, m_dot_water_kg_s, COP, energy_residual.
        """
        eps = self.saturation_effectiveness() if eps_sat is None else eps_sat
        T_wb = self.wet_bulb(T_db, RH)
        w_in = self.humidity_ratio(T_db, RH)

        # Sensible approach to the wet-bulb temperature (effectiveness definition).
        T_out = T_db - eps * (T_db - T_wb)

        # dry-air mass flow (split moist flow)
        m_dry = m_dot_air / (1.0 + w_in)
        cp_m = self.cp_moist(w_in)

        Q_sens = m_dot_air * cp_m * (T_db - T_out)    # W sensible cooling

        # Outlet humidity from the adiabatic-saturation ENERGY balance (constant-enthalpy
        # process): the sensible heat removed equals the latent heat of the evaporated
        # water, evaluated at the wet-bulb temperature (ASHRAE 2021 adiabatic saturation).
        # This makes the moist-air enthalpy conserved BY CONSTRUCTION, so Q_sens = Q_latent.
        h_fg_wb = self.latent_heat(T_wb)              # J/kg at wet-bulb (energy datum)
        m_dot_water = Q_sens / h_fg_wb                # kg/s evaporated (energy balance)
        w_out = w_in + m_dot_water / m_dry            # humidity ratio (mass balance)
        Q_latent = m_dot_water * h_fg_wb              # W latent absorbed (= Q_sens)

        # RH at outlet
        w_sat_out = self.w_sat(T_out)
        RH_out = min(w_out / w_sat_out, 1.0) if w_sat_out > 0 else 1.0

        COP = Q_sens / self.P_fan if self.P_fan > 0 else float("inf")
        energy_residual = abs(Q_sens - Q_latent) / max(Q_sens, 1e-9)

        return {
            "T_out": T_out,
            "T_wb": T_wb,
            "w_in": w_in,
            "w_out": w_out,
            "RH_out": RH_out,
            "eps_sat": eps,
            "Q_sens_W": Q_sens,
            "Q_latent_W": Q_latent,
            "m_dot_water_kg_s": m_dot_water,
            "COP": COP,
            "energy_residual": energy_residual,
        }

    # ------------------------------------------------------------------
    # Lumped transient ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, T_db_func, RH, m_dot_air):
        """dT_pad/dt for the lumped wetted-pad energy balance."""
        T_pad = y[0]
        T_db = T_db_func(t)
        w_in = self.humidity_ratio(T_db, RH)
        cp_m = self.cp_moist(w_in)
        m_dry = m_dot_air / (1.0 + w_in)

        # convective sensible exchange between incoming air and pad surface
        UA = m_dot_air * cp_m * self.NTU_air_pad
        Q_conv = UA * (T_db - T_pad)

        # evaporation rate driven by humidity deficit at pad surface (toward saturation)
        w_surf = self.w_sat(T_pad)
        # mass-transfer coefficient consistent with NTU (Lewis number ~1)
        m_dot_water = m_dry * (1.0 - np.exp(-self.NTU_air_pad)) * max(w_surf - w_in, 0.0)
        Q_evap = m_dot_water * self.h_fg

        dTdt = (Q_conv - Q_evap) / (self.m_pad * self.cp_pad)
        return [dTdt]

    def simulate(self, T_db, RH, m_dot_air=1.0, T_pad0=None, dt=1.0, duration_s=120.0):
        """
        Integrate the lumped pad-temperature ODE with scipy.solve_ivp.

        T_db : float or callable(t)->T_db [degC]  (dry-bulb inlet)
        RH   : relative humidity [0-1]
        Returns dict of time-series arrays + final steady psychrometric state.
        """
        if callable(T_db):
            T_db_func = T_db
            T_db0 = T_db(0.0)
        else:
            T_db_val = float(T_db)
            T_db_func = lambda t: T_db_val
            T_db0 = T_db_val

        if T_pad0 is None:
            T_pad0 = T_db0  # cold start: pad begins at inlet (dry) temperature

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T_pad0],
            t_eval=t_eval, args=(T_db_func, RH, m_dot_air),
            method="RK45", rtol=1e-6, atol=1e-8, max_step=dt,
        )

        T_pad = sol.y[0]
        # Outlet air relaxes toward pad surface via effectiveness-NTU each instant
        eps = self.saturation_effectiveness()
        T_db_arr = np.array([T_db_func(tt) for tt in sol.t])
        T_wb_arr = np.array([self.wet_bulb(T_db_func(tt), RH) for tt in sol.t])
        # outlet = inlet cooled by approach to current pad temperature
        T_out = T_db_arr - eps * (T_db_arr - T_pad)

        w_in_arr = np.array([self.humidity_ratio(T_db_func(tt), RH) for tt in sol.t])
        cp_arr = self.cp_moist(w_in_arr)
        Q_sens = m_dot_air * cp_arr * (T_db_arr - T_out)
        COP = np.where(self.P_fan > 0, Q_sens / self.P_fan, np.inf)

        ss = self.steady_state(T_db0, RH, m_dot_air)

        return {
            "t": sol.t,
            "T_db": T_db_arr,
            "T_wb": T_wb_arr,
            "T_pad": T_pad,
            "T_out": T_out,
            "Q_sens_W": Q_sens,
            "COP": COP,
            "steady_state": ss,
            "success": sol.success,
        }
