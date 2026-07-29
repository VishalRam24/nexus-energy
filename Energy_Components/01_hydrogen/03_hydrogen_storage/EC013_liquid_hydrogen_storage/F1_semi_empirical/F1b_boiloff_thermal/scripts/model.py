"""
EC013 -- Liquid Hydrogen Storage -- F1b Boil-Off Thermal Model

Extends F1a by adding:
1. T_ambient variation with MLI conductivity T-dependence.
2. Tank pressurization dynamics when vent is closed (dormancy model).
3. Pressure-corrected saturation temperature (Clausius-Clapeyron).

Physical model
--------------
MLI effective U(T_amb):
    The apparent conductivity of multi-layer insulation grows with mean wall
    temperature.  A widely-used empirical fit (Johnson 2010, NREL/TP-560-47503):
        k_eff(T_mean) = k0 * (T_mean / T0_ref)^n_mli
    where k0=0.0015 W/(m.K) at T0_ref=160 K, n_mli=1.5 (representative for
    30-layer perforated MLI in residual vacuum).
    T_mean = (T_amb + T_sat) / 2.

Effective overall U:
    U_eff = k_eff / d_mli    [W/(m2.K)]

Heat leak:
    Q_in = U_eff(T_amb) * A_surf * (T_amb - T_sat(P))   [W]

Boil-off mass rate (when vent is open or at NBP):
    m_dot = Q_in / h_vap                                 [kg/s]

Dormancy / pressurization (closed vent):
    All boil-off gas stays in ullage.  Pressure rise modeled using ideal gas
    in ullage volume with vapor mass accumulation:
        V_ull = V_tank * (1 - fill_fraction)
        dm_vap/dt = m_dot_gen                 [kg/s]
        dP/dt = (R_H2 / V_ull) * dm_vap/dt * T_sat   [Pa/s]
    When P >= P_vent -> gas is vented, pressure held at P_vent.

Saturation temperature correction:
    T_sat(P) = T_sat_1atm + dTsat_dP * (P - 1.01325)     [K]  (Clausius-Clapeyron linear)

References:
    Sherif et al. (1997) Int. J. Hydrogen Energy, 22(7), 683.
    Johnson (2010) NREL/TP-560-47503: MLI k(T) characterisation.
    Petitpas (2018) NREL: LH2 boil-off pathway analysis.
    Barron (1999) Cryogenic Systems, Oxford: Clausius-Clapeyron slope.
"""

import numpy as np
from scipy.integrate import solve_ivp


class LH2ThermalModel:
    """Liquid hydrogen storage with T_amb variation, MLI T-dependence, pressurization."""

    def __init__(self, params: dict):
        t = params["tank"]
        h = params["hydrogen"]
        a = params["ambient"]

        self.V_tank   = t["volume"]["value"]
        self.A_surf   = t["surface_area"]["value"]
        self.m_tank   = t["mass_empty"]["value"]
        self.U_ref    = t["U_ref"]["value"]
        self.n_mli    = 1.5         # MLI exponent; Johnson (2010)
        self.T0_ref   = 160.0       # K reference for MLI fit
        self.d_mli    = t["MLI_thickness"]["value"]
        self.fill_max = t["fill_fraction_max"]["value"]
        self.P_vent   = t["vent_pressure"]["value"]  # bar
        self.P_max    = t["max_pressure"]["value"]   # bar

        self.T_sat0   = h["T_sat_1atm"]["value"]
        self.rho_L    = h["rho_liquid"]["value"]
        self.rho_V    = h["rho_vapor"]["value"]
        self.h_vap    = h["h_vap"]["value"] * 1000.0   # J/kg
        self.cp_L     = h["cp_liquid"]["value"] * 1000.0
        self.LHV      = h["LHV"]["value"]
        self.R_H2     = h["R_H2"]["value"]
        self.dTsat_dP = h["dTsat_dP"]["value"]         # K/bar

        self.T_amb_default = a["T_ambient_default"]["value"]

    # ------------------------------------------------------------------
    # MLI thermal conductance
    # ------------------------------------------------------------------

    def k_mli(self, T_amb_K, P_bar=1.01325):
        """
        Effective MLI conductivity [W/(m.K)].
        Johnson (2010) power-law fit: k = k0*(T_mean/T0_ref)^n_mli
        where T_mean = (T_amb + T_sat) / 2.
        """
        T_sat = self.t_sat(P_bar)
        T_mean = (np.asarray(T_amb_K, dtype=float) + T_sat) / 2.0
        k0 = self.U_ref * self.d_mli  # back-calculate k0 at reference conditions
        # k0 at T0_ref=160 K, so k0 corresponds to T_mean at reference
        T_mean_ref = (self.T_amb_default + T_sat) / 2.0
        k_scale = (T_mean / T_mean_ref) ** self.n_mli
        return k0 * k_scale

    def u_eff(self, T_amb_K, P_bar=1.01325):
        """Effective overall heat transfer coefficient [W/(m2.K)]."""
        return self.k_mli(T_amb_K, P_bar) / self.d_mli

    # ------------------------------------------------------------------
    # Saturation temperature (pressure-corrected)
    # ------------------------------------------------------------------

    def t_sat(self, P_bar):
        """Saturation temperature at pressure P [K]."""
        return self.T_sat0 + self.dTsat_dP * (np.asarray(P_bar, dtype=float) - 1.01325)

    # ------------------------------------------------------------------
    # Heat leak and boil-off
    # ------------------------------------------------------------------

    def heat_leak(self, T_amb_K=None, P_bar=1.01325):
        """Heat leak through MLI [W]."""
        T_amb = self.T_amb_default if T_amb_K is None else np.asarray(T_amb_K, dtype=float)
        P_bar = np.asarray(P_bar, dtype=float)
        T_sat = self.t_sat(P_bar)
        U = self.u_eff(T_amb, P_bar)
        dT = T_amb - T_sat
        return U * self.A_surf * dT

    def boiloff_mass_rate(self, T_amb_K=None, P_bar=1.01325):
        """Boil-off mass rate [kg/s]."""
        Q = self.heat_leak(T_amb_K, P_bar)
        return np.maximum(Q, 0.0) / self.h_vap

    def boiloff_rate_percent_day(self, fill_fraction, T_amb_K=None, P_bar=1.01325):
        """Daily boil-off as % of stored mass."""
        m_dot = self.boiloff_mass_rate(T_amb_K, P_bar)
        m = self.stored_mass(fill_fraction)
        safe = np.where(m > 0, m, 1.0)
        return np.where(m > 0, m_dot * 86400.0 / safe * 100.0, 0.0)

    # ------------------------------------------------------------------
    # Stored mass and energy
    # ------------------------------------------------------------------

    def stored_mass(self, fill_fraction):
        f = np.clip(np.asarray(fill_fraction, dtype=float), 0.0, self.fill_max)
        return self.rho_L * self.V_tank * f

    def energy_stored(self, fill_fraction):
        return self.stored_mass(fill_fraction) * self.LHV   # MJ

    # ------------------------------------------------------------------
    # Dormancy / pressurization dynamics
    # ------------------------------------------------------------------

    def pressurization_transient(self, fill_fraction, T_amb_K, P0_bar, t_span, n_steps=500):
        """
        Closed-vent dormancy simulation.

        State: [P_bar, m_liq_kg]
        When P >= P_vent: venting opens, P held at P_vent and boil-off exits.

        Parameters
        ----------
        fill_fraction : float -- initial fill fraction (0-0.95)
        T_amb_K       : float -- ambient temperature (constant during dormancy)
        P0_bar        : float -- initial pressure [bar]
        t_span        : (t0, tf) in seconds
        n_steps       : int

        Returns
        -------
        t, P_arr, m_liq_arr, bor_arr
        """
        m_liq0 = float(self.stored_mass(fill_fraction))
        P0 = float(P0_bar)
        T_amb = float(T_amb_K)

        def ode(t, y):
            P = y[0]
            m_liq = y[1]

            # Current fill fraction
            f = np.clip(m_liq / (self.rho_L * self.V_tank), 0.0, self.fill_max)
            V_ull = self.V_tank * (1.0 - f)  # ullage volume m3

            m_dot = float(self.boiloff_mass_rate(T_amb, P))

            if P >= self.P_vent:
                # Venting: pressure stays at P_vent, liquid depletes
                dP_dt = 0.0
                dm_liq_dt = -m_dot
            else:
                # Closed: vapor accumulates in ullage, pressure rises
                # Ideal gas: PV = m_vap * R_H2 * T_sat => dP/dt = R_H2*T_sat/V_ull * dm_vap/dt
                T_sat = float(self.t_sat(P))
                if V_ull > 1e-6:
                    dP_dt = (self.R_H2 * T_sat / (V_ull * 1e5)) * m_dot  # Pa/s -> bar/s /1e5
                else:
                    dP_dt = 0.0
                dm_liq_dt = -m_dot

            return [dP_dt, dm_liq_dt]

        t_eval = np.linspace(t_span[0], t_span[1], n_steps)
        sol = solve_ivp(ode, t_span, [P0, m_liq0], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-8,
                        events=None)

        P_arr   = np.maximum(sol.y[0], P0)  # pressure can only rise
        m_liq_arr = np.maximum(sol.y[1], 0.0)
        f_arr = m_liq_arr / (self.rho_L * self.V_tank)
        bor_arr = np.array([
            float(self.boiloff_rate_percent_day(float(f), T_amb, float(p)))
            for f, p in zip(f_arr, P_arr)
        ])
        return sol.t, P_arr, m_liq_arr, bor_arr

    def evaluate(self, fill_fraction, T_amb_K, P_bar=1.01325):
        """
        Full steady-state evaluation.

        Returns
        -------
        dict
        """
        f = np.asarray(fill_fraction, dtype=float)
        T_amb = np.asarray(T_amb_K, dtype=float)
        P = np.asarray(P_bar, dtype=float)

        m = self.stored_mass(f)
        E = self.energy_stored(f)
        Q = self.heat_leak(T_amb, P)
        m_dot = self.boiloff_mass_rate(T_amb, P)
        bor = self.boiloff_rate_percent_day(f, T_amb, P)
        U = self.u_eff(T_amb, P)
        T_sat = self.t_sat(P)

        return {
            "stored_mass_kg":       m,
            "energy_stored_MJ":     E,
            "heat_leak_W":          Q,
            "boiloff_rate_kg_s":    m_dot,
            "BOR_pct_day":          bor,
            "U_eff_W_m2_K":         U,
            "T_sat_K":              T_sat,
        }
