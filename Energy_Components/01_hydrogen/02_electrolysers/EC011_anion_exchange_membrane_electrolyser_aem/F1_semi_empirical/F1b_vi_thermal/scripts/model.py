"""
EC011 -- Anion Exchange Membrane Electrolyser (AEM) -- F1b V-I Thermal Model

Extends F1a by adding an explicit cell thermal balance so that stack temperature
is solved from the heat budget rather than fixed externally.

Electrochemical model (same as F1a, already has Arrhenius j0):
    E_rev(T) = 1.229 - 0.0009*(T - 298.15)                          [V]
    j0_a(T)  = j0_a_ref * exp(-Ea_a/R * (1/T - 1/T_ref))           [A/cm2]
    j0_c(T)  = j0_c_ref * exp(-Ea_c/R * (1/T - 1/T_ref))           [A/cm2]
    V_act_a  = (RT / alpha_a / F) * ln(j / j0_a(T))                 [V]
    V_act_c  = (RT / alpha_c / F) * ln(j / j0_c(T))                 [V]
    ASR(T)   = r_ref * (1 + r_T*(T - T_ref))                        [Ohm.cm2]
    V_ohm    = ASR(T) * j                                           [V]
    V_cell   = E_rev + V_act_a + V_act_c + V_ohm                    [V]

Thermal balance (lumped, steady-state solve or transient ODE):
    Q_gen  = N * I * (V_cell - E_tn)     [W]   -- heat of reaction loss
             (= N * I * V_cell - N * I * E_tn; positive when V_cell > E_tn)
    Q_cool = UA_cool * (T_stack - T_coolant)  [W]  -- convective removal
    Steady-state: Q_gen = Q_cool => T_stack = T_coolant + Q_gen / UA_cool
    Transient:    Cp * dT/dt = Q_gen - Q_cool

Physical insight: higher j -> higher Q_gen -> higher T_stack -> lower ASR,
higher j0 -> slightly lower V_cell (net effect at electrolyser: V_cell
DECREASES with T because E_rev drops faster than losses recover at high j;
at low j, activation losses decrease more, so V_cell can decrease or stay flat).

References:
    Vincent & Bessarabov (2018) RSE Rev., 81, 1690.
    Henkensmeier et al. (2021) J. Electrochem. Energy Conv. Storage, 18, 024001.
    Schalenbach et al. (2016) J. Electrochem. Soc., 163(11), F3197.
"""

import numpy as np
from scipy.integrate import solve_ivp


class AEMThermalModel:
    """
    AEM electrolyser with temperature-coupled polarization and thermal balance.

    Operating range: 303-363 K (30-90 C).
    """

    # Physical constants
    R_gas = 8.314      # J/(mol K)
    F = 96485.0        # C/mol
    E_tn = 1.481       # thermoneutral voltage [V] (HHV basis, 25 C ref)
    H2_LHV = 241800.0  # J/mol LHV

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["constants"]

        self.N_cells = u["N_cells"]["value"]
        self.A_m2 = u["electrode_area"]["value"]
        self.A_cm2 = self.A_m2 * 1.0e4

        self.T_ref = u["T_ref"]["value"]
        self.T_cool_default = u["T_coolant"]["value"]

        self.j0_a_ref = u["j0_anode"]["value"]
        self.j0_c_ref = u["j0_cathode"]["value"]
        self.Ea_a = u["Ea_anode"]["value"]
        self.Ea_c = u["Ea_cathode"]["value"]
        self.alpha_a = u["alpha_anode"]["value"]
        self.alpha_c = u["alpha_cathode"]["value"]

        self.r_ref = u["r_membrane_ref"]["value"]
        self.r_T = u["r_temp_coeff"]["value"]

        self.eta_F = u["eta_F"]["value"]
        self.Cp_stack = u["thermal_mass"]["value"]
        self.UA_cool = u["UA_cool"]["value"]

        self.E_rev_ref = c["E_rev_ref"]["value"]
        self.E_rev_T_coeff = c["E_rev_T_coeff"]["value"]

    # ------------------------------------------------------------------
    # Electrochemical sub-models (T-dependent)
    # ------------------------------------------------------------------

    def e_rev(self, T_K):
        """Reversible cell voltage [V]."""
        T_K = np.asarray(T_K, dtype=float)
        return self.E_rev_ref - self.E_rev_T_coeff * (T_K - 298.15)

    def asr(self, T_K):
        """
        Area-specific resistance [Ohm.cm2].
        Linear T-dependence; r_T < 0 so ASR decreases with temperature.
        Ref: Henkensmeier et al. (2021).
        """
        T_K = np.asarray(T_K, dtype=float)
        return self.r_ref * (1.0 + self.r_T * (T_K - self.T_ref))

    def exchange_current_density(self, T_K):
        """Arrhenius exchange current densities [A/cm2]."""
        T_K = np.asarray(T_K, dtype=float)
        j0_a = self.j0_a_ref * np.exp(-self.Ea_a / self.R_gas * (1.0 / T_K - 1.0 / self.T_ref))
        j0_c = self.j0_c_ref * np.exp(-self.Ea_c / self.R_gas * (1.0 / T_K - 1.0 / self.T_ref))
        return j0_a, j0_c

    def cell_voltage(self, j_A_m2, T_K):
        """
        Single-cell voltage [V].

        Parameters
        ----------
        j_A_m2 : float or array -- current density [A/m2]
        T_K    : float or array -- temperature [K]
        """
        j = np.asarray(j_A_m2, dtype=float)
        T_K = np.asarray(T_K, dtype=float)

        j_cm2 = j / 1.0e4
        j_safe = np.where(j_cm2 > 1e-12, j_cm2, 1e-12)

        E_rev = self.e_rev(T_K)
        RT_F = self.R_gas * T_K / self.F

        j0_a, j0_c = self.exchange_current_density(T_K)

        V_act_a = np.where(j_cm2 > 1e-12,
                           (RT_F / self.alpha_a) * np.log(j_safe / j0_a), 0.0)
        V_act_c = np.where(j_cm2 > 1e-12,
                           (RT_F / self.alpha_c) * np.log(j_safe / j0_c), 0.0)
        V_act_a = np.maximum(V_act_a, 0.0)
        V_act_c = np.maximum(V_act_c, 0.0)

        ASR = self.asr(T_K)
        V_ohm = ASR * j_cm2

        return E_rev + V_act_a + V_act_c + V_ohm

    def stack_voltage(self, j_A_m2, T_K):
        """Total stack voltage [V]."""
        return self.N_cells * self.cell_voltage(j_A_m2, T_K)

    # ------------------------------------------------------------------
    # Thermal sub-model
    # ------------------------------------------------------------------

    def heat_generation(self, j_A_m2, T_K):
        """
        Total stack heat generation [W].

        Q = N * I * (V_cell - E_tn)
        Positive (exothermic) when V_cell > E_tn (always true in electrolysis).
        Ref: Schalenbach et al. (2016), eq. 2.
        """
        j = np.asarray(j_A_m2, dtype=float)
        I = j * self.A_m2
        V_cell = self.cell_voltage(j, T_K)
        return self.N_cells * I * (V_cell - self.E_tn)

    def steady_state_temperature(self, j_A_m2, T_coolant_K=None):
        """
        Steady-state stack temperature [K] from thermal balance:
            Q_gen = UA_cool * (T_stack - T_coolant)
        """
        T_cool = self.T_cool_default if T_coolant_K is None else float(T_coolant_K)
        j = np.asarray(j_A_m2, dtype=float)

        # Self-consistent solve: T_stack appears in V_cell which feeds Q_gen
        from scipy.optimize import brentq

        def residual(T_stk, j_val):
            Q_gen = self.heat_generation(float(j_val), T_stk)
            Q_cool = self.UA_cool * (T_stk - T_cool)
            return Q_gen - Q_cool

        if j.ndim == 0:
            T_lo, T_hi = T_cool, T_cool + 200.0
            try:
                T_ss = brentq(residual, T_lo, T_hi, args=(float(j),), xtol=1e-4)
            except ValueError:
                T_ss = T_cool
            return float(T_ss)
        else:
            out = np.empty_like(j)
            for k, jv in enumerate(j.flat):
                T_lo, T_hi = T_cool, T_cool + 200.0
                try:
                    out.flat[k] = brentq(residual, T_lo, T_hi, args=(float(jv),), xtol=1e-4)
                except ValueError:
                    out.flat[k] = T_cool
            return out

    def transient_temperature(self, j_A_m2, T_cool_K, T0_K, t_span, n_steps=200):
        """
        Transient stack temperature profile.

        Parameters
        ----------
        j_A_m2  : float -- constant current density
        T_cool_K: float -- coolant temperature
        T0_K    : float -- initial stack temperature
        t_span  : (t0, tf) in seconds
        n_steps : int -- number of output time points

        Returns
        -------
        t_arr, T_arr : arrays
        """
        def ode(t, y):
            T_stk = y[0]
            Q_gen = float(self.heat_generation(j_A_m2, T_stk))
            Q_cool = self.UA_cool * (T_stk - T_cool_K)
            dTdt = (Q_gen - Q_cool) / self.Cp_stack
            return [dTdt]

        t_eval = np.linspace(t_span[0], t_span[1], n_steps)
        sol = solve_ivp(ode, t_span, [T0_K], t_eval=t_eval, method="RK45",
                        rtol=1e-6, atol=1e-8)
        return sol.t, sol.y[0]

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def hydrogen_rate(self, j_A_m2):
        """Hydrogen production rate [mol/s]."""
        j = np.asarray(j_A_m2, dtype=float)
        I = j * self.A_m2
        return self.eta_F * self.N_cells * I / (2.0 * self.F)

    def efficiency_lhv(self, j_A_m2, T_K):
        """LHV efficiency = (H2 LHV power) / (electrical power)."""
        j = np.asarray(j_A_m2, dtype=float)
        I = j * self.A_m2
        V_stack = self.stack_voltage(j, T_K)
        P_el = V_stack * I
        n_H2 = self.hydrogen_rate(j)
        safe = np.where(P_el > 0, P_el, 1.0)
        return np.where(P_el > 0, np.clip(n_H2 * self.H2_LHV / safe, 0.0, 1.0), 0.0)

    def evaluate(self, j_A_m2, T_K):
        """
        Full operating-point evaluation at specified T.

        Parameters
        ----------
        j_A_m2 : float or array -- current density [A/m2]
        T_K    : float or array -- stack temperature [K]

        Returns
        -------
        dict
        """
        j = np.asarray(j_A_m2, dtype=float)
        T_K = np.asarray(T_K, dtype=float)

        V_cell = self.cell_voltage(j, T_K)
        V_stack = self.N_cells * V_cell
        I = j * self.A_m2
        P_el = V_stack * I / 1000.0       # kW
        Q_gen = self.heat_generation(j, T_K)
        ASR_val = self.asr(T_K)
        n_H2 = self.hydrogen_rate(j)
        eta = self.efficiency_lhv(j, T_K)
        j0_a, j0_c = self.exchange_current_density(T_K)
        E_rev = self.e_rev(T_K)

        return {
            "cell_voltage_V": V_cell,
            "stack_voltage_V": V_stack,
            "power_kW": P_el,
            "heat_generation_W": Q_gen,
            "ASR_ohm_cm2": ASR_val,
            "hydrogen_rate_mol_s": n_H2,
            "efficiency_lhv": eta,
            "E_rev_V": E_rev,
            "j0_anode": j0_a,
            "j0_cathode": j0_c,
        }
