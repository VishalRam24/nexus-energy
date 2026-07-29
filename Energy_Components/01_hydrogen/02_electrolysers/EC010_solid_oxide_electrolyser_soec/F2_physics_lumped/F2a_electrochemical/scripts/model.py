"""
EC010 -- Solid Oxide Electrolyser Cell (SOEC) -- F2a Electrochemical Model

Physics-lumped model with detailed electrochemistry for high-temperature steam electrolysis.

Cell voltage:
    V_cell = E_nernst(T, p_H2O, p_H2) + eta_act_anode + eta_act_cathode + eta_ohm

    E_nernst = E0(T) + (R*T)/(2*F) * ln(p_H2 * p_O2^0.5 / p_H2O)

    Steam utilization effect on Nernst:
        p_H2O = p_total * (1 - U * x_in_H2O)  where U = steam utilization
        p_H2  = p_total * U * x_in_H2O

    Activation (Butler-Volmer):
        eta_act = (R*T)/(alpha*n*F) * arcsinh(j/(2*j0))
        j0 follows Arrhenius: j0 = j0_ref * exp(-E_act/(R) * (1/T - 1/T_ref))

    Ohmic (YSZ electrolyte):
        sigma_YSZ = (A_YSZ/T) * exp(-E_YSZ/(R*T))
        eta_ohm = j * t_elec / sigma_YSZ

Thermal mode:
    V_cell < V_tn => endothermic (uses external/waste heat)
    V_cell > V_tn => exothermic

H2 production: n_H2 = N_cells * j * A / (2*F) (100% Faraday eff at high T)

Reference:
    Ni et al. (2007), Chem. Eng. Tech., 29(6), 636-642.
    Udagawa et al. (2007), J. Power Sources, 166(1), 127-136.
"""

import numpy as np

R_GAS = 8.314
F_CONST = 96485.0


class SOECF2a:
    """SOEC -- physics-lumped electrochemical model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2 = u["electrode_area_cm2"]["value"]
        self.A_m2 = self.A_cm2 * 1e-4
        self.t_elec_m = u["t_electrolyte_m"]["value"]       # YSZ thickness [m]
        self.A_YSZ = u["A_YSZ"]["value"]                     # pre-exponential [S*K/m]
        self.E_YSZ = u["E_YSZ"]["value"]                     # activation energy [J/mol]
        self.alpha = u["alpha"]["value"]
        self.n = int(u["n"]["value"])
        self.j0_cathode_ref = u["j0_cathode_ref"]["value"]    # A/m2
        self.j0_anode_ref = u["j0_anode_ref"]["value"]        # A/m2
        self.E_act_cathode = u["E_act_cathode"]["value"]      # J/mol
        self.E_act_anode = u["E_act_anode"]["value"]          # J/mol
        self.T_ref = u["T_ref"]["value"]                      # K
        self.V_tn = u["V_thermoneutral"]["value"]             # V
        self.E0_ref = u["E0_ref"]["value"]                    # V at T_ref
        self.dE0_dT = u["dE0_dT"]["value"]                   # V/K
        self.p_total = u.get("p_total_bar", {}).get("value", 1.0)

    def e_nernst(self, T_K, steam_utilization):
        """Nernst potential [V] accounting for steam utilization."""
        T = np.asarray(T_K, dtype=float)
        U = np.clip(np.asarray(steam_utilization, dtype=float), 0.01, 0.95)

        # Standard potential at temperature
        E0_T = self.E0_ref + self.dE0_dT * (T - self.T_ref)

        # Partial pressures (assuming inlet is mostly steam)
        x_H2O = 0.9  # inlet mole fraction steam
        p_H2O = self.p_total * x_H2O * (1.0 - U)
        p_H2 = self.p_total * x_H2O * U
        p_O2 = 0.21 * self.p_total  # air side

        p_H2O = np.maximum(p_H2O, 1e-6)
        p_H2 = np.maximum(p_H2, 1e-6)

        # Nernst: for electrolysis, V increases with products concentration
        Vt = R_GAS * T / (2.0 * F_CONST)
        E_nernst = E0_T + Vt * np.log(p_H2 * p_O2**0.5 / p_H2O)
        return E_nernst

    def sigma_ysz(self, T_K):
        """YSZ ionic conductivity [S/m]."""
        T = np.asarray(T_K, dtype=float)
        return (self.A_YSZ / T) * np.exp(-self.E_YSZ / (R_GAS * T))

    def eta_ohmic(self, j_A_cm2, T_K):
        """Ohmic overpotential [V]."""
        j_m2 = np.abs(np.asarray(j_A_cm2, dtype=float)) * 1e4  # A/cm2 -> A/m2
        sigma = self.sigma_ysz(T_K)
        return j_m2 * self.t_elec_m / sigma

    def j0_electrode(self, T_K, j0_ref, E_act):
        """Exchange current density with Arrhenius T-dependence [A/m2]."""
        T = np.asarray(T_K, dtype=float)
        return j0_ref * np.exp(-E_act / R_GAS * (1.0 / T - 1.0 / self.T_ref))

    def eta_activation(self, j_A_cm2, T_K):
        """Total activation overpotential (cathode + anode) [V]."""
        j_m2 = np.abs(np.asarray(j_A_cm2, dtype=float)) * 1e4
        T = np.asarray(T_K, dtype=float)
        Vt = R_GAS * T / (self.alpha * self.n * F_CONST)

        j0_c = self.j0_electrode(T, self.j0_cathode_ref, self.E_act_cathode)
        j0_a = self.j0_electrode(T, self.j0_anode_ref, self.E_act_anode)

        eta_c = Vt * np.arcsinh(j_m2 / (2.0 * j0_c))
        eta_a = Vt * np.arcsinh(j_m2 / (2.0 * j0_a))
        return eta_c + eta_a

    def cell_voltage(self, j_A_cm2, T_K, steam_utilization):
        """Cell voltage [V]. j in A/cm2."""
        E_n = self.e_nernst(T_K, steam_utilization)
        eta_act = self.eta_activation(j_A_cm2, T_K)
        eta_ohm = self.eta_ohmic(j_A_cm2, T_K)
        return E_n + eta_act + eta_ohm

    def stack_voltage(self, j_A_cm2, T_K, steam_utilization):
        """Stack voltage [V]."""
        return self.N_cells * self.cell_voltage(j_A_cm2, T_K, steam_utilization)

    def thermal_mode(self, j_A_cm2, T_K, steam_utilization):
        """
        Thermal mode: 'endothermic' if V < V_tn, 'exothermic' if V > V_tn.
        Returns string array.
        """
        V = self.cell_voltage(j_A_cm2, T_K, steam_utilization)
        V = np.atleast_1d(V)
        modes = np.where(V < self.V_tn, 1, -1)  # 1=endothermic, -1=exothermic
        return modes

    def h2_production_rate(self, j_A_cm2):
        """H2 production rate [mol/s]. 100% Faraday eff at high T."""
        j = np.abs(np.asarray(j_A_cm2, dtype=float))
        I = j * self.A_cm2
        return self.N_cells * I / (2.0 * F_CONST)

    def efficiency(self, j_A_cm2, T_K, steam_utilization):
        """Electrical efficiency = H2_LHV / P_electrical."""
        j = np.abs(np.asarray(j_A_cm2, dtype=float))
        H2_LHV = 241800.0
        n_H2 = self.h2_production_rate(j)
        I = j * self.A_cm2
        V_stack = self.stack_voltage(j, T_K, steam_utilization)
        P_el = V_stack * I
        P_chem = n_H2 * H2_LHV
        # SOEC can have > 100% electrical efficiency in endothermic mode
        return np.where(P_el > 0, P_chem / P_el, 0.0)

    def simulate(self, current_density, T_K, steam_utilization, dt, duration_s):
        """
        Simulate SOEC operation.

        Args:
            current_density: A/cm2 (scalar or callable(t))
            T_K:            Temperature [K] (scalar or callable(t))
            steam_utilization: fraction [0-1] (scalar or callable(t))
            dt:             Time step [s]
            duration_s:     Duration [s]

        Returns:
            dict with arrays: t, voltage, h2_production, efficiency, thermal_mode
        """
        _j = current_density if callable(current_density) else lambda t: current_density
        _T = T_K if callable(T_K) else lambda t: T_K
        _U = steam_utilization if callable(steam_utilization) else lambda t: steam_utilization

        t_arr = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_arr = t_arr[t_arr <= duration_s]
        N = len(t_arr)

        voltage = np.zeros(N)
        h2_prod = np.zeros(N)
        eff = np.zeros(N)
        th_mode = np.zeros(N)

        for i in range(N):
            j_t = _j(t_arr[i])
            T_t = _T(t_arr[i])
            U_t = _U(t_arr[i])
            voltage[i] = float(self.stack_voltage(j_t, T_t, U_t))
            h2_prod[i] = float(self.h2_production_rate(j_t))
            eff[i] = float(self.efficiency(j_t, T_t, U_t))
            th_mode[i] = float(np.atleast_1d(self.thermal_mode(j_t, T_t, U_t)).flat[0])

        return {
            "t": t_arr,
            "voltage": voltage,
            "h2_production": h2_prod,
            "efficiency": eff,
            "thermal_mode": th_mode,
        }
