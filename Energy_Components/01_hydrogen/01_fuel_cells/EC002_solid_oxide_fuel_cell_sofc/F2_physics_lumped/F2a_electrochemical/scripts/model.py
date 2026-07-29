"""
EC002 -- Solid Oxide Fuel Cell (SOFC) -- F2a Electrochemical Model

Physics-lumped electrochemical model with coupled thermal ODE.

Voltage:
    V = E_nernst(T, pH2, pO2, pH2O)
        - eta_act_anode(j, T)       Butler-Volmer
        - eta_act_cathode(j, T)     Butler-Volmer
        - eta_ohm(j, T)             YSZ ionic conductivity
        - eta_conc(j, T)            anode + cathode diffusion

Thermal ODE:
    m*cp * dT/dt = Q_gen - Q_cool

Reference:
    Chan et al. (2001), J. Power Sources, 93, 130-140
    Campanari & Iora (2004), J. Power Sources, 132, 113-126
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class SOFC_F2a:
    """Solid Oxide Fuel Cell -- full electrochemical + thermal ODE model."""

    R = 8.314
    F = 96485.0
    n = 2  # electrons per H2

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]
        self.t_el = u["t_electrolyte"]["value"]
        self.sigma_0 = u["sigma_0_ysz"]["value"]
        self.E_act_ysz = u["E_act_ysz"]["value"]
        self.j0_a_ref = u["j0_anode_ref"]["value"]
        self.j0_c_ref = u["j0_cathode_ref"]["value"]
        self.E_act_a = u["E_act_anode"]["value"]
        self.E_act_c = u["E_act_cathode"]["value"]
        self.alpha_a = u["alpha_anode"]["value"]
        self.alpha_c = u["alpha_cathode"]["value"]
        self.j_L_a = u["j_L_anode"]["value"]
        self.j_L_c = u["j_L_cathode"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.pH2 = u["pH2"]["value"]
        self.pO2 = u["pO2"]["value"]
        self.pH2O = u["pH2O"]["value"]
        self.m_stack = u["m_stack"]["value"]
        self.cp_stack = u["cp_stack"]["value"]
        self.hA_cool = u["hA_cool"]["value"]
        self.T_air_in = u["T_air_in"]["value"]
        self.uf_nom = u["uf_nominal"]["value"]

    # ------------------------------------------------------------------
    # Nernst voltage
    # ------------------------------------------------------------------
    def nernst_voltage(self, T, pH2=None, pO2=None, pH2O=None):
        pH2 = pH2 if pH2 is not None else self.pH2
        pO2 = pO2 if pO2 is not None else self.pO2
        pH2O = pH2O if pH2O is not None else self.pH2O
        E0 = 1.253 - 2.4516e-4 * T
        E = E0 + (self.R * T) / (2.0 * self.F) * np.log(
            pH2 * np.sqrt(pO2) / max(pH2O, 1e-6)
        )
        return E

    def thermoneutral_voltage(self, T):
        return 1.285 - 2.0e-4 * (T - 298.15)

    # ------------------------------------------------------------------
    # Butler-Volmer activation (implicit solve for eta given j)
    # ------------------------------------------------------------------
    def _butler_volmer_eta(self, j, j0, alpha, T):
        """Solve Butler-Volmer for activation overpotential."""
        if j <= 0 or j0 <= 0:
            return 0.0
        a = alpha * self.n * self.F / (self.R * T)
        # Use analytic approximation (Tafel) for j >> j0
        if j > 5.0 * j0:
            return (self.R * T) / (alpha * self.n * self.F) * np.log(j / j0)
        # Otherwise solve BV: j = j0 * [exp(a*eta) - exp(-(1-alpha)*nF*eta/RT)]
        def bv_residual(eta):
            return j0 * (np.exp(a * eta) - np.exp(-(1 - alpha) * self.n * self.F * eta / (self.R * T))) - j
        try:
            eta = brentq(bv_residual, 0.0, 2.0, xtol=1e-8)
        except ValueError:
            eta = (self.R * T) / (alpha * self.n * self.F) * np.log(j / j0 + 1.0)
        return max(eta, 0.0)

    def activation_anode(self, j, T):
        j0 = self.j0_a_ref * np.exp(
            (-self.E_act_a / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        return self._butler_volmer_eta(j, j0, self.alpha_a, T)

    def activation_cathode(self, j, T):
        j0 = self.j0_c_ref * np.exp(
            (-self.E_act_c / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        return self._butler_volmer_eta(j, j0, self.alpha_c, T)

    # ------------------------------------------------------------------
    # Ohmic (YSZ electrolyte)
    # ------------------------------------------------------------------
    def ysz_conductivity(self, T):
        return (self.sigma_0 / T) * np.exp(-self.E_act_ysz / T)

    def ohmic_overpotential(self, j, T):
        sigma = self.ysz_conductivity(T)
        return j * self.t_el / max(sigma, 1e-8)

    # ------------------------------------------------------------------
    # Concentration
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, T):
        eta_a = 0.0
        eta_c = 0.0
        if j > 0:
            r_a = j / self.j_L_a
            r_c = j / self.j_L_c
            if r_a < 1.0:
                eta_a = -(self.R * T) / (self.n * self.F) * np.log(1.0 - r_a)
            else:
                eta_a = 5.0
            if r_c < 1.0:
                eta_c = -(self.R * T) / (4.0 * self.F) * np.log(1.0 - r_c)
            else:
                eta_c = 5.0
        return eta_a + eta_c

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, fuel_comp=None):
        pH2 = self.pH2
        pH2O = self.pH2O
        pO2 = self.pO2
        if fuel_comp:
            pH2 = fuel_comp.get("pH2", pH2)
            pH2O = fuel_comp.get("pH2O", pH2O)
            pO2 = fuel_comp.get("pO2", pO2)

        E = self.nernst_voltage(T, pH2, pO2, pH2O)
        eta_act_a = self.activation_anode(j, T)
        eta_act_c = self.activation_cathode(j, T)
        eta_ohm = self.ohmic_overpotential(j, T)
        eta_conc = self.concentration_overpotential(j, T)
        V = E - eta_act_a - eta_act_c - eta_ohm - eta_conc
        return max(V, 0.0)

    # ------------------------------------------------------------------
    # Fuel utilization
    # ------------------------------------------------------------------
    def fuel_utilization(self, j, T):
        # uf = I / (n*F*n_H2_in)  -- simplified: at nominal, uf = uf_nom at j=0.5
        j_nom = 0.5
        uf = self.uf_nom * (j / j_nom) if j_nom > 0 else 0.0
        return min(uf, 0.99)

    # ------------------------------------------------------------------
    # Thermal ODE
    # ------------------------------------------------------------------
    def dTdt(self, T, j, fuel_comp=None):
        V = self.cell_voltage(j, T, fuel_comp)
        E_th = self.thermoneutral_voltage(T)
        Q_gen = self.N_cells * self.A_cell * j * (E_th - V)
        Q_cool = self.hA_cool * (T - self.T_air_in)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, current_density, T_cell_K, fuel_composition=None,
                 dt=1.0, duration_s=600.0):
        _j = current_density if callable(current_density) else lambda t: current_density

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], _j(t), fuel_composition)]

        sol = solve_ivp(rhs, (0.0, duration_s), [T_cell_K],
                        t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
                        max_step=dt)

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        voltage = np.zeros(N)
        power_density = np.zeros(N)
        efficiency = np.zeros(N)
        fuel_util = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            voltage[i] = self.cell_voltage(j, T, fuel_composition)
            power_density[i] = j * voltage[i]
            E_th = self.thermoneutral_voltage(T)
            efficiency[i] = voltage[i] / E_th if E_th > 0 else 0.0
            fuel_util[i] = self.fuel_utilization(j, T)

        return {
            "t": t_out,
            "voltage": voltage,
            "power_density": power_density,
            "efficiency": efficiency,
            "temperature": T_out,
            "fuel_utilization": fuel_util,
        }
