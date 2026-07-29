"""
EC009 -- Alkaline Electrolyser (AEL) -- F2a Electrochemical Model

Physics-lumped dynamic model with detailed electrochemistry:
    V_cell = E_rev(T) + eta_act_anode + eta_act_cathode + eta_ohm + eta_bubble

    E_rev(T) = 1.229 - 0.0009*(T - 298)

    Activation (Butler-Volmer):
        eta_act = (R*T)/(alpha*n*F) * arcsinh(j / (2*j0))

    Ohmic:
        KOH conductivity: sigma(T, w) from empirical correlation
        Diaphragm resistance: R_dia = t_dia / (sigma_dia * A)
        eta_ohm = j * (1/sigma_KOH * d_gap + R_dia) / (1 - theta)

    Bubble coverage:
        theta = k_bubble * j^0.3
        Effective area = A * (1 - theta)

    H2 production: n_H2 = eta_F * N_cells * I / (2*F)
    Faraday efficiency: eta_F = f1*I^2 / (f2 + I^2)

Dynamic: dT/dt = (P_heat - Q_cool) / (m*cp)   [optional thermal coupling]

Reference:
    Ulleberg (2003), Int. J. Hydrogen Energy, 28(1), 21-33.
    Haug et al. (2017), Int. J. Hydrogen Energy, 42(25), 15689-15707.
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314
F_CONST = 96485.0


class AELF2a:
    """Alkaline electrolyser -- physics-lumped electrochemical model with bubble coverage."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = int(u["N_cells"]["value"])
        self.A_m2 = u["electrode_area_m2"]["value"]
        self.n = int(u["n"]["value"])
        self.alpha = u["alpha"]["value"]
        self.j0_anode = u["j0_anode"]["value"]       # A/m2
        self.j0_cathode = u["j0_cathode"]["value"]    # A/m2
        self.d_gap_m = u["d_gap_m"]["value"]           # electrode gap
        self.t_dia_m = u["t_dia_m"]["value"]           # diaphragm thickness
        self.sigma_dia = u["sigma_dia"]["value"]       # S/m diaphragm conductivity
        self.k_bubble = u["k_bubble"]["value"]         # bubble coverage coefficient
        self.f1 = u["f1"]["value"]                     # Faraday eff param
        self.f2 = u["f2"]["value"]                     # Faraday eff param
        self.E_rev_ref = u["E_rev_ref"]["value"]       # V at 298K
        self.E_rev_T_coeff = u["E_rev_T_coeff"]["value"]  # V/K

    def koh_conductivity(self, T_K, w_pct):
        """
        KOH electrolyte conductivity [S/m].
        Empirical correlation (See & White, 1997).
        w_pct: KOH weight percent (typically 25-35%).
        """
        T = np.asarray(T_K, dtype=float)
        w = np.asarray(w_pct, dtype=float)
        # Simplified correlation: sigma = a*w - b*w^2 + c*T - d
        sigma = -2.041 * w - 0.0028 * w**2 + 0.005332 * w * T + 207.2 * w / T - 0.1043 * w**3 / T
        return np.maximum(sigma, 0.1)

    def e_rev(self, T_K):
        """Reversible voltage [V]."""
        T = np.asarray(T_K, dtype=float)
        return self.E_rev_ref - self.E_rev_T_coeff * (T - 298.15)

    def bubble_coverage(self, j_A_m2):
        """Bubble coverage fraction theta [-]."""
        j = np.abs(np.asarray(j_A_m2, dtype=float))
        theta = self.k_bubble * j**0.3
        return np.clip(theta, 0.0, 0.95)

    def eta_activation(self, j_A_m2, T_K):
        """Total activation overpotential (anode + cathode) [V]."""
        j = np.abs(np.asarray(j_A_m2, dtype=float))
        T = np.asarray(T_K, dtype=float)
        Vt = R_GAS * T / (self.alpha * self.n * F_CONST)
        eta_a = Vt * np.arcsinh(j / (2.0 * self.j0_anode))
        eta_c = Vt * np.arcsinh(j / (2.0 * self.j0_cathode))
        return eta_a + eta_c

    def eta_ohmic(self, j_A_m2, T_K, w_pct):
        """Ohmic overpotential [V] including electrolyte and diaphragm."""
        j = np.abs(np.asarray(j_A_m2, dtype=float))
        T = np.asarray(T_K, dtype=float)
        theta = self.bubble_coverage(j)

        sigma_koh = self.koh_conductivity(T, w_pct)
        # Bruggeman correction for bubble coverage
        sigma_eff = sigma_koh * (1.0 - theta)**1.5

        R_elec = self.d_gap_m / np.maximum(sigma_eff, 0.01)
        R_dia = self.t_dia_m / self.sigma_dia
        return j * (R_elec + R_dia)

    def cell_voltage(self, j_A_m2, T_K, w_pct):
        """Cell voltage [V]."""
        j = np.asarray(j_A_m2, dtype=float)
        E = self.e_rev(T_K)
        eta_act = self.eta_activation(j, T_K)
        eta_ohm = self.eta_ohmic(j, T_K, w_pct)
        return E + eta_act + eta_ohm

    def stack_voltage(self, j_A_m2, T_K, w_pct):
        """Stack voltage [V]."""
        return self.N_cells * self.cell_voltage(j_A_m2, T_K, w_pct)

    def faraday_efficiency(self, j_A_m2):
        """Faraday efficiency [-]."""
        j = np.abs(np.asarray(j_A_m2, dtype=float))
        I = j * self.A_m2
        eta_F = self.f1 * I**2 / (self.f2 + I**2)
        return np.clip(eta_F, 0.0, 1.0)

    def h2_production_rate(self, j_A_m2):
        """H2 production rate [mol/s]."""
        j = np.abs(np.asarray(j_A_m2, dtype=float))
        I = j * self.A_m2
        eta_F = self.faraday_efficiency(j)
        return eta_F * self.N_cells * I / (2.0 * F_CONST)

    def efficiency(self, j_A_m2, T_K, w_pct):
        """Stack efficiency: H2 LHV power / electrical input."""
        j = np.abs(np.asarray(j_A_m2, dtype=float))
        H2_LHV = 241800.0  # J/mol
        n_H2 = self.h2_production_rate(j)
        I = j * self.A_m2
        V_stack = self.stack_voltage(j, T_K, w_pct)
        P_el = V_stack * I
        P_chem = n_H2 * H2_LHV
        return np.where(P_el > 0, np.clip(P_chem / P_el, 0.0, 1.0), 0.0)

    def simulate(self, current_density, T_K, koh_wt_pct, dt, duration_s):
        """
        Simulate AEL operation over time.

        Args:
            current_density: A/m2 (scalar or callable(t))
            T_K:            Temperature K (scalar or callable(t))
            koh_wt_pct:     KOH weight % (scalar)
            dt:             Time step [s]
            duration_s:     Duration [s]

        Returns:
            dict with arrays: t, voltage, h2_production, efficiency, bubble_coverage
        """
        _j = current_density if callable(current_density) else lambda t: current_density
        _T = T_K if callable(T_K) else lambda t: T_K

        t_arr = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_arr = t_arr[t_arr <= duration_s]
        N = len(t_arr)

        voltage = np.zeros(N)
        h2_prod = np.zeros(N)
        eff = np.zeros(N)
        bubble = np.zeros(N)

        for i in range(N):
            j_t = _j(t_arr[i])
            T_t = _T(t_arr[i])
            voltage[i] = float(self.stack_voltage(j_t, T_t, koh_wt_pct))
            h2_prod[i] = float(self.h2_production_rate(j_t))
            eff[i] = float(self.efficiency(j_t, T_t, koh_wt_pct))
            bubble[i] = float(self.bubble_coverage(j_t))

        return {
            "t": t_arr,
            "voltage": voltage,
            "h2_production": h2_prod,
            "efficiency": eff,
            "bubble_coverage": bubble,
        }
