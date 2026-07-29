"""
EC008 -- PEM Electrolyser (PEMEL) -- F2a Electrochemical Model

Physics-lumped model: reversible voltage + Butler-Volmer activation
(both electrodes) + Springer membrane ohmic + Faradaic efficiency.

Voltage (electrolyser: V > E_rev):
    V = E_rev(T, P) + eta_act_anode(j,T) + eta_act_cathode(j,T) + eta_ohm(j,T)

H2 production:
    n_H2 = eta_F * I / (n * F)   [mol/s]

Thermal ODE:
    m*cp * dT/dt = Q_gen - Q_cool
    Q_gen = N_cells * A * j * (V - E_th)

Reference:
    Garcia-Valverde et al. (2012), Int. J. Hydrogen Energy, 37(2), 1927-1938
    Springer et al. (1991), J. Electrochem. Soc., 138(8), 2334-2342
"""

import numpy as np
from scipy.integrate import solve_ivp


class PEMEL_F2a:
    """PEM Electrolyser -- full electrochemical + thermal ODE model."""

    R = 8.314
    F = 96485.0
    n = 2

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]
        self.t_mem = u["t_mem"]["value"]
        self.lambda_mem = u["lambda_mem"]["value"]
        self.j0_a_ref = u["j0_anode_ref"]["value"]
        self.j0_c_ref = u["j0_cathode_ref"]["value"]
        self.alpha_a = u["alpha_anode"]["value"]
        self.alpha_c = u["alpha_cathode"]["value"]
        self.E_act_a = u["E_act_anode"]["value"]
        self.E_act_c = u["E_act_cathode"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.j_max = u["j_max"]["value"]
        self.eta_F_a = u["eta_F_a"]["value"] / 1000.0  # convert mA/cm2 -> A/cm2
        self.eta_F_b = u["eta_F_b"]["value"]
        self.m_stack = u["m_stack"]["value"]
        self.cp_stack = u["cp_stack"]["value"]
        self.hA_cool = u["hA_cool"]["value"]
        self.T_coolant = u["T_coolant"]["value"]

    # ------------------------------------------------------------------
    # Reversible (Nernst) voltage for water electrolysis
    # ------------------------------------------------------------------
    def reversible_voltage(self, T, P_bar=1.0):
        """Reversible voltage [V] with T and P correction."""
        E0 = 1.229 - 8.5e-4 * (T - 298.15)
        # Pressure correction (assumes P_h2 = P_o2 ~ P cathode)
        if P_bar > 1.0:
            E0 += (self.R * T) / (self.n * self.F) * np.log(P_bar / 1.0)
        return E0

    def thermoneutral_voltage(self, T):
        return 1.481 - 2.26e-4 * (T - 298.15)

    # ------------------------------------------------------------------
    # Butler-Volmer activation (Tafel approximation)
    # ------------------------------------------------------------------
    def activation_anode(self, j, T):
        if j <= 0:
            return 0.0
        j0 = self.j0_a_ref * np.exp(
            (-self.E_act_a / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        return (self.R * T) / (self.alpha_a * self.n * self.F) * np.log(max(j, 1e-12) / max(j0, 1e-15))

    def activation_cathode(self, j, T):
        if j <= 0:
            return 0.0
        j0 = self.j0_c_ref * np.exp(
            (-self.E_act_c / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        return max(0.0, (self.R * T) / (self.alpha_c * self.n * self.F) * np.log(max(j, 1e-12) / max(j0, 1e-15)))

    # ------------------------------------------------------------------
    # Springer membrane ohmic
    # ------------------------------------------------------------------
    def membrane_conductivity(self, T):
        lam = self.lambda_mem
        return (0.005139 * lam - 0.00326) * np.exp(1268.0 * (1.0 / 303.15 - 1.0 / T))

    def ohmic_overpotential(self, j, T):
        sigma = max(self.membrane_conductivity(T), 1e-6)
        return j * self.t_mem / sigma

    # ------------------------------------------------------------------
    # Faradaic efficiency
    # ------------------------------------------------------------------
    def faradaic_efficiency(self, j):
        """Faradaic efficiency [-] -- empirical fit."""
        if j <= 0:
            return 0.0
        return min(1.0, self.eta_F_b * (j ** 2) / (self.eta_F_a ** 2 + j ** 2))

    # ------------------------------------------------------------------
    # Cell voltage (electrolyser: V > E_rev)
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, P_bar=1.0):
        E_rev = self.reversible_voltage(T, P_bar)
        eta_a = self.activation_anode(j, T)
        eta_c = self.activation_cathode(j, T)
        eta_ohm = self.ohmic_overpotential(j, T)
        return E_rev + eta_a + eta_c + eta_ohm

    # ------------------------------------------------------------------
    # H2 production rate
    # ------------------------------------------------------------------
    def h2_production_rate(self, j, T):
        """H2 production [mol/s per cell]."""
        eta_F = self.faradaic_efficiency(j)
        I = j * self.A_cell  # A
        return eta_F * I / (self.n * self.F)

    # ------------------------------------------------------------------
    # Thermal ODE
    # ------------------------------------------------------------------
    def dTdt(self, T, j, P_bar=1.0):
        V = self.cell_voltage(j, T, P_bar)
        E_th = self.thermoneutral_voltage(T)
        # For electrolyser: heat = I*(V - E_th) when V > E_th
        Q_gen = self.N_cells * self.A_cell * j * max(V - E_th, 0.0)
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, current_density, T_K, P_bar=1.0, dt=0.1, duration_s=60.0):
        _j = current_density if callable(current_density) else lambda t: current_density

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], _j(t), P_bar)]

        sol = solve_ivp(rhs, (0.0, duration_s), [T_K],
                        t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
                        max_step=dt)

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        voltage = np.zeros(N)
        h2_prod = np.zeros(N)
        efficiency = np.zeros(N)
        heat_gen = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            V = self.cell_voltage(j, T, P_bar)
            voltage[i] = V
            h2_mol_s = self.h2_production_rate(j, T) * self.N_cells
            h2_prod[i] = h2_mol_s * 2.016e-3  # kg/s
            E_rev = self.reversible_voltage(T, P_bar)
            P_elec = self.N_cells * self.A_cell * j * V
            P_h2 = h2_mol_s * 241800.0  # LHV J/mol * mol/s
            efficiency[i] = P_h2 / max(P_elec, 1e-9) if P_elec > 0 else 0.0
            E_th = self.thermoneutral_voltage(T)
            heat_gen[i] = self.N_cells * self.A_cell * j * max(V - E_th, 0.0)

        return {
            "t": t_out,
            "voltage": voltage,
            "h2_production_kg_s": h2_prod,
            "efficiency": efficiency,
            "heat_generation_W": heat_gen,
            "temperature": T_out,
        }
