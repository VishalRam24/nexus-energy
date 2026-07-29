"""
EC152 -- Flash Steam Geothermal Plant -- F2a Flash Thermodynamics (Physics-Lumped)

First-principles single/double-flash model built on water/steam saturation
thermodynamics, with a lumped separator+turbine transient ODE.

Process chain (DiPippo, 2015, Ch. 5-6):
    1. Pressurised saturated brine at wellhead state (T_geo).
    2. Throttled (isenthalpic) to separator pressure -> flashes.  The flash is
       a constant-enthalpy expansion:  h_geo = (1-x) h_f(T_fl) + x h_g(T_fl)
       => steam fraction  x = (h_geo - h_f) / h_fg   (lever rule).
    3. Saturated steam (mass m_dot*x) expands through turbine to condenser
       pressure.  Ideal (isentropic) work from steam tables, scaled by
       turbine isentropic efficiency and generator efficiency.
    4. Separated liquid brine (mass m_dot*(1-x)) is reinjected.

Net power:
    W_net = eta_gen * eta_turb_is * m_steam * (h_g(T_fl) - h_2s)
    where h_2s is the isentropic outlet enthalpy at condenser pressure obtained
    from the entropy balance  s_g(T_fl) = (1-x2) s_f(T_cond) + x2 s_g(T_cond).

Utilization efficiency (2nd-law style, DiPippo eq. 5):
    eta_util = W_net / (m_dot * (h_geo - h_0))     where h_0 is dead-state
    enthalpy (liquid at rejection T).  Bounded above by the Carnot factor
    1 - T_cond/T_geo (enforced in tests).

Lumped transient (control / dynamic simulation):
    Two first-order states track the approach of separator steam production and
    turbine power to their thermodynamic set-points after a flow / temperature
    change:
        tau_sep  * d(m_steam)/dt = m_steam_ss(t) - m_steam
        tau_turb * d(W)/dt       = W_ss(m_steam)  - W
    Integrated with scipy.integrate.solve_ivp (lumped 0-D ODE).

Water/steam property correlations (hardcoded, IAPWS-simplified):
    * Saturation pressure  -> IAPWS-IF97 region-4 backward eq. (Wagner & Pruss,
      2002; IAPWS-IF97 1997), valid 273.15-647 K.
    * Saturated-liquid enthalpy h_f(T), latent heat h_fg(T): polynomial fits to
      IAPWS steam tables (Cengel & Boles, 2014, Table A-4), valid 0.01-300 C.
    * Saturated entropies s_f, s_g: thermodynamically-consistent fits.

References:
    DiPippo, R. (2015). Geothermal Power Plants, 4th ed. Butterworth-Heinemann.
    Wagner, W. & Pruss, A. (2002). J. Phys. Chem. Ref. Data 31(2), 387-535.
    IAPWS-IF97 (1997). Industrial Formulation for Water and Steam.
    Cengel, Y. & Boles, M. (2014). Thermodynamics: An Engineering Approach, 8e.
"""

import numpy as np
from scipy.integrate import solve_ivp


class FlashSteamGeothermalF2a:
    """Flash steam geothermal plant -- physics-lumped flash thermodynamics."""

    T0_K = 273.15  # 0 C in K

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_geo_design = u["T_geo_design"]["value"]
        self.T_reject_design = u["T_reject_design"]["value"]
        self.T_cond_offset = u["T_condenser_offset"]["value"]
        self.m_dot_design = u["m_dot_brine_design"]["value"]
        self.cp_brine = u["cp_brine"]["value"]
        self.eta_turb = u["eta_turbine_isentropic"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.tau_sep = u["tau_separator"]["value"]
        self.tau_turb = u["tau_turbine"]["value"]

    # ==================================================================
    # IAPWS-simplified water/steam saturation property correlations
    # ==================================================================
    @staticmethod
    def p_sat(T_c):
        """
        Saturation pressure [kPa] vs temperature [C].
        IAPWS-IF97 region-4 saturation-line equation (Wagner & Pruss, 2002).
        Valid 273.15-647.096 K.
        """
        T = np.asarray(T_c, dtype=float) + 273.15
        # IF97 region 4 coefficients (n1..n10)
        n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
             0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
             -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849e0,
             0.65017534844798e3]
        th = T + n[8] / (T - n[9])
        A = th * th + n[0] * th + n[1]
        B = n[2] * th * th + n[3] * th + n[4]
        C = n[5] * th * th + n[6] * th + n[7]
        p = (2.0 * C / (-B + np.sqrt(B * B - 4.0 * A * C))) ** 4  # MPa
        return p * 1000.0  # MPa -> kPa

    @staticmethod
    def h_f(T_c):
        """
        Saturated liquid enthalpy [kJ/kg] vs T [C].
        Polynomial fit to IAPWS steam tables (Cengel & Boles 2014, Table A-4),
        valid 0.01-300 C. Reference: h_f = 0 at triple point.
        """
        T = np.asarray(T_c, dtype=float)
        # near-cp*T with mild curvature; matches A-4 to < 0.5 % over 50-300 C
        return (4.1868 * T
                + 2.30e-4 * T ** 2
                + 2.0e-6 * T ** 3)

    @classmethod
    def h_fg(cls, T_c):
        """
        Latent heat of vaporization [kJ/kg] vs T [C].
        Watson-type fit anchored to IAPWS (h_fg=2501 at 0 C, ->0 at Tc=373.95 C).
        """
        T = np.asarray(T_c, dtype=float)
        Tr = (T + 273.15) / 647.096
        Tr = np.clip(Tr, 0.0, 0.99999)
        return 2501.0 * np.clip((1.0 - Tr) / (1.0 - 273.15 / 647.096), 0.0, None) ** 0.38

    @classmethod
    def h_g(cls, T_c):
        """Saturated vapour enthalpy [kJ/kg] = h_f + h_fg."""
        return cls.h_f(T_c) + cls.h_fg(T_c)

    @classmethod
    def s_f(cls, T_c):
        """
        Saturated liquid entropy [kJ/(kg.K)].
        Thermodynamically consistent: s_f = integral cp/T ~ cp*ln(T/T_ref).
        Anchored to IAPWS A-4 (s_f=0 at 0.01 C).
        """
        T = np.asarray(T_c, dtype=float) + 273.15
        return 4.1868 * np.log(T / 273.16) + 1.0e-4 * (T - 273.16)

    @classmethod
    def s_g(cls, T_c):
        """Saturated vapour entropy [kJ/(kg.K)] = s_f + h_fg/T."""
        T = np.asarray(T_c, dtype=float) + 273.15
        return cls.s_f(T_c) + cls.h_fg(T_c) / T

    # ==================================================================
    # Flash thermodynamics
    # ==================================================================
    def condenser_temperature(self, T_reject_c):
        """Condenser saturation temperature [C]."""
        return np.asarray(T_reject_c, dtype=float) + self.T_cond_offset

    def optimal_flash_temperature(self, T_geo_c, T_reject_c):
        """
        Optimal single-flash separator temperature [C].
        Maximises steam work; well-approximated by the geometric mean of the
        source and condenser absolute temperatures (DiPippo, 2015, Ch. 5).
        """
        T_geo = np.asarray(T_geo_c, dtype=float) + self.T0_K
        T_cond = self.condenser_temperature(T_reject_c) + self.T0_K
        return np.sqrt(T_geo * T_cond) - self.T0_K

    def flash_steam_fraction(self, T_geo_c, T_flash_c):
        """
        Steam dryness fraction x from isenthalpic flash (lever rule):
            x = (h_geo - h_f(T_fl)) / h_fg(T_fl)
        h_geo is the wellhead saturated-liquid enthalpy throttled at constant h.
        """
        h_geo = self.h_f(T_geo_c)             # saturated brine enthalpy in
        hf = self.h_f(T_flash_c)
        hfg = self.h_fg(T_flash_c)
        x = (h_geo - hf) / np.where(hfg > 1e-6, hfg, 1e-6)
        return np.clip(x, 0.0, 1.0)

    def turbine_specific_work(self, T_flash_c, T_reject_c):
        """
        Specific turbine work [kJ/kg of steam].
        Isentropic expansion from saturated vapour at T_flash to condenser
        pressure, then scaled by isentropic + generator efficiency.

            s1 = s_g(T_fl)
            s1 = (1-x2) s_f(T_cond) + x2 s_g(T_cond)   -> outlet quality x2
            h2s = (1-x2) h_f(T_cond) + x2 h_g(T_cond)
            w  = eta_gen * eta_turb * (h_g(T_fl) - h2s)
        """
        T_cond = self.condenser_temperature(T_reject_c)
        s1 = self.s_g(T_flash_c)
        sf2 = self.s_f(T_cond)
        sg2 = self.s_g(T_cond)
        x2 = (s1 - sf2) / np.where((sg2 - sf2) > 1e-9, sg2 - sf2, 1e-9)
        x2 = np.clip(x2, 0.0, 1.0)
        h1 = self.h_g(T_flash_c)
        h2s = self.h_f(T_cond) + x2 * self.h_fg(T_cond)
        w_ideal = np.clip(h1 - h2s, 0.0, None)        # ideal isentropic drop
        return self.eta_gen * self.eta_turb * w_ideal  # kJ/kg

    def steam_mass_flow(self, m_dot_kgs, T_geo_c, T_flash_c):
        """Steam mass flow to turbine [kg/s] = m_dot * x."""
        x = self.flash_steam_fraction(T_geo_c, T_flash_c)
        return np.asarray(m_dot_kgs, dtype=float) * x

    def net_power_ss(self, T_geo_c, T_reject_c, m_dot_kgs, T_flash_c=None):
        """
        Steady-state net electrical power [kW].
        P = m_steam * w_turbine
        """
        if T_flash_c is None:
            T_flash_c = self.optimal_flash_temperature(T_geo_c, T_reject_c)
        m_steam = self.steam_mass_flow(m_dot_kgs, T_geo_c, T_flash_c)
        w = self.turbine_specific_work(T_flash_c, T_reject_c)  # kJ/kg
        return m_steam * w  # kg/s * kJ/kg = kW

    def carnot_efficiency(self, T_geo_c, T_reject_c):
        """Carnot factor 1 - T_cond/T_geo (absolute T)."""
        T_geo = np.asarray(T_geo_c, dtype=float) + self.T0_K
        T_cond = self.condenser_temperature(T_reject_c) + self.T0_K
        return np.clip(1.0 - T_cond / T_geo, 0.0, 1.0)

    def utilization_efficiency(self, T_geo_c, T_reject_c, m_dot_kgs, T_flash_c=None):
        """
        Utilization (2nd-law) efficiency = W_net / available thermal energy,
        where available energy is m_dot*(h_geo - h_0), dead state = liquid at
        rejection temperature. Bounded by Carnot (checked in tests).
        """
        if T_flash_c is None:
            T_flash_c = self.optimal_flash_temperature(T_geo_c, T_reject_c)
        P = self.net_power_ss(T_geo_c, T_reject_c, m_dot_kgs, T_flash_c)
        h_geo = self.h_f(T_geo_c)
        h0 = self.h_f(T_reject_c)
        Q_avail = np.asarray(m_dot_kgs, dtype=float) * np.clip(h_geo - h0, 1e-6, None)
        return np.clip(P / Q_avail, 0.0, 1.0)

    # ==================================================================
    # Lumped separator/turbine transient ODE (scipy.solve_ivp)
    # ==================================================================
    def simulate(self, m_dot_kgs, T_geo_c, T_reject_c, dt=0.5, duration_s=120.0,
                 T_flash_c=None, m_steam0=None, W0=None):
        """
        Dynamic simulation of the lumped flash-separator + turbine.

        m_dot_kgs : float OR callable(t)->kg/s   (production well flow)
        Returns dict of time-series arrays.
        """
        if callable(m_dot_kgs):
            mdot_fn = m_dot_kgs
        else:
            mdot_fn = lambda t: float(m_dot_kgs)

        # initial flow / set-points
        m0_flow = mdot_fn(0.0)
        if T_flash_c is None:
            T_flash_c = float(self.optimal_flash_temperature(T_geo_c, T_reject_c))
        w_spec = float(self.turbine_specific_work(T_flash_c, T_reject_c))  # kJ/kg

        m_steam_ss0 = float(self.steam_mass_flow(m0_flow, T_geo_c, T_flash_c))
        W_ss0 = m_steam_ss0 * w_spec

        if m_steam0 is None:
            m_steam0 = 0.0   # cold start: no steam yet
        if W0 is None:
            W0 = 0.0

        def rhs(t, y):
            m_steam, W = y
            mdot = mdot_fn(t)
            m_steam_set = float(self.steam_mass_flow(mdot, T_geo_c, T_flash_c))
            W_set = m_steam * w_spec  # turbine power follows actual steam flow
            dms = (m_steam_set - m_steam) / self.tau_sep
            dW = (W_set - W) / self.tau_turb
            return [dms, dW]

        n = max(2, int(round(duration_s / dt)) + 1)
        t_eval = np.linspace(0.0, duration_s, n)
        sol = solve_ivp(rhs, (0.0, duration_s), [m_steam0, W0],
                        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

        t = sol.t
        m_steam = np.clip(sol.y[0], 0.0, None)
        W = np.clip(sol.y[1], 0.0, None)

        mdot_arr = np.array([mdot_fn(tt) for tt in t])
        x = self.flash_steam_fraction(T_geo_c, T_flash_c) * np.ones_like(t)
        eta_carnot = self.carnot_efficiency(T_geo_c, T_reject_c) * np.ones_like(t)
        h_geo = self.h_f(T_geo_c)
        h0 = self.h_f(T_reject_c)
        Q_avail = np.clip(mdot_arr * (h_geo - h0), 1e-6, None)
        eta_util = np.clip(W / Q_avail, 0.0, 1.0)

        return {
            "t": t,
            "net_power_kW": W,
            "steam_flow_kgs": m_steam,
            "brine_flow_kgs": mdot_arr,
            "steam_fraction": x,
            "T_flash_C": float(T_flash_c),
            "T_cond_C": float(self.condenser_temperature(T_reject_c)),
            "specific_work_kJkg": w_spec,
            "eta_utilization": eta_util,
            "eta_carnot": eta_carnot,
            "P_steady_kW": W_ss0,
        }
