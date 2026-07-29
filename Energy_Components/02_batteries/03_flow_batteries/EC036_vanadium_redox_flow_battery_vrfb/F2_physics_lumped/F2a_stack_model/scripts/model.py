"""
EC036 -- Vanadium Redox Flow Battery (VRFB) -- F2a Stack Model

Physics-lumped dynamic model of a VRFB stack coupled with tank SOC dynamics.

Cell voltage:
    E_cell = E_nernst(SOC) - eta_act - eta_ohm - eta_conc

    E_nernst = E0 + 2*(R*T)/(n*F) * ln(SOC/(1-SOC))
    eta_ohm  = j * R_ohm_cm2 / 10000   (per cell, j in A/m2)
    eta_act  = (R*T)/(alpha*n*F) * arcsinh(j / (2*j0))
    eta_conc = -(R*T)/(n*F) * ln(1 - |j|/j_lim)

    j_lim depends on flow rate: j_lim = n*F*c_total*Q_flow / A_cell

Tank SOC dynamics (Euler integration):
    dSOC/dt = -I / (n * F * c_total * V_tank)
    (positive I = discharge -> SOC decreases)

Hydraulic loss:
    P_pump = delta_P * Q_flow / eta_pump
    delta_P = k_hydraulic * Q_flow  (linear approximation)

Stack power:
    P_stack = N_cells * E_cell * I
    P_net   = P_stack - P_pump   (discharge)
            = P_stack + P_pump   (charge: pump still consumes)

Reference:
    Blanc, C., Rufer, A. (2010). Multiphysics and Energetic Modeling of a VRFB.
    Shah, A.A. et al. (2011). Dynamic modelling of hydrogen evolution effects in VRFB.
    Electrochimica Acta, 56(3), 1570-1578.
"""

import numpy as np


# Physical constants
R_GAS = 8.314      # J/(mol*K)
F_CONST = 96485.0  # C/mol


class VRFBF2a:
    """VRFB stack + tank dynamic model with electrochemical detail."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2 = u["electrode_area_cm2"]["value"]          # cm2
        self.A_m2 = self.A_cm2 * 1e-4                         # m2
        self.E0 = u["E0"]["value"]                             # V
        self.R_ohm_cm2 = u["R_ohm_cm2"]["value"]              # Ohm*cm2
        self.T = u["T_K"]["value"]                             # K
        self.n = int(u["n"]["value"])
        self.c_total = u["c_total_mol_L"]["value"]             # mol/L
        self.V_tank = u["V_tank_L"]["value"]                   # L
        self.alpha = u["alpha"]["value"]                       # charge transfer coeff
        self.j0 = u["j0_A_m2"]["value"]                       # exchange current density A/m2
        self.k_hydraulic = u["k_hydraulic_Pa_s_m3"]["value"]   # Pa/(m3/s)
        self.eta_pump = u["eta_pump"]["value"]                 # pump efficiency
        self.Q_flow_nom = u["Q_flow_nom_L_min"]["value"]       # L/min nominal

        # Derived
        self.c_total_mol_m3 = self.c_total * 1000.0            # mol/m3
        self.V_tank_m3 = self.V_tank / 1000.0                  # m3
        self.total_charge_C = self.n * F_CONST * self.c_total_mol_m3 * self.V_tank_m3

    def _thermal_voltage(self):
        """R*T/(n*F) in Volts."""
        return R_GAS * self.T / (self.n * F_CONST)

    SOC_MIN = 0.01
    SOC_MAX = 0.99

    def e_nernst(self, soc):
        """Nernst cell potential [V]."""
        soc = np.clip(np.asarray(soc, dtype=float), self.SOC_MIN, self.SOC_MAX)
        Vt = self._thermal_voltage()
        return self.E0 + 2.0 * Vt * np.log(soc / (1.0 - soc))

    def eta_ohmic(self, current_A):
        """Ohmic overpotential per cell [V]. current_A is stack current."""
        j = np.abs(np.asarray(current_A, dtype=float)) / self.A_m2  # A/m2
        return j * self.R_ohm_cm2 / 1e4  # Ohm*cm2 -> Ohm*m2 -> V

    def eta_activation(self, current_A):
        """Activation overpotential per cell [V] (Butler-Volmer symmetric)."""
        j = np.abs(np.asarray(current_A, dtype=float)) / self.A_m2
        Vt = self._thermal_voltage()
        return Vt / self.alpha * np.arcsinh(j / (2.0 * self.j0))

    def j_limiting(self, flow_rate_L_min):
        """Limiting current density [A/m2] based on flow rate."""
        Q_m3_s = np.asarray(flow_rate_L_min, dtype=float) / 60.0 / 1000.0
        # j_lim = n*F*c*Q / A  (for one cell in a stack)
        j_lim = self.n * F_CONST * self.c_total_mol_m3 * Q_m3_s / self.A_m2
        return np.maximum(j_lim, 1.0)  # minimum 1 A/m2 to avoid div/0

    def eta_concentration(self, current_A, flow_rate_L_min, soc):
        """Concentration overpotential per cell [V]."""
        j = np.abs(np.asarray(current_A, dtype=float)) / self.A_m2
        j_lim = self.j_limiting(flow_rate_L_min)
        soc = np.clip(np.asarray(soc, dtype=float), self.SOC_MIN, self.SOC_MAX)
        # Effective limiting current scales with SOC (less reactant at low SOC during discharge)
        j_lim_eff = j_lim * soc
        j_lim_eff = np.maximum(j_lim_eff, 1.0)
        ratio = np.clip(j / j_lim_eff, 0.0, 0.99)
        Vt = self._thermal_voltage()
        return -Vt * np.log(1.0 - ratio)

    def cell_voltage(self, soc, current_A, flow_rate_L_min):
        """
        Cell terminal voltage [V].
        current_A > 0 = discharge, < 0 = charge.
        """
        soc = np.asarray(soc, dtype=float)
        current_A = np.asarray(current_A, dtype=float)

        E_n = self.e_nernst(soc)
        eta_o = self.eta_ohmic(current_A)
        eta_a = self.eta_activation(current_A)
        eta_c = self.eta_concentration(current_A, flow_rate_L_min, soc)

        sign = np.sign(current_A)
        # Discharge: V = E_nernst - losses; Charge: V = E_nernst + losses
        V = E_n - sign * (eta_o + eta_a + eta_c)
        return V

    def stack_voltage(self, soc, current_A, flow_rate_L_min):
        """Stack terminal voltage [V]."""
        return self.N_cells * self.cell_voltage(soc, current_A, flow_rate_L_min)

    def pump_power_w(self, flow_rate_L_min):
        """Pump power [W] for both half-cells (2x)."""
        Q_m3_s = np.asarray(flow_rate_L_min, dtype=float) / 60.0 / 1000.0
        delta_P = self.k_hydraulic * Q_m3_s  # Pa
        P_pump = 2.0 * delta_P * Q_m3_s / self.eta_pump
        return np.maximum(P_pump, 0.0)

    def dsoc_dt(self, soc, current_A):
        """Rate of SOC change [1/s]. Positive current = discharge = SOC decreases."""
        I = np.asarray(current_A, dtype=float)
        return -I / self.total_charge_C

    def simulate(self, current_A, flow_rate_L_min, dt, duration_s, soc_init=0.5):
        """
        Simulate VRFB stack + tank dynamics.

        Args:
            current_A:        Stack current [A] (scalar or callable(t))
            flow_rate_L_min:  Electrolyte flow rate [L/min] (scalar or callable(t))
            dt:               Time step [s]
            duration_s:       Simulation duration [s]
            soc_init:         Initial SOC [0-1]

        Returns:
            dict with arrays: t, voltage, soc, power_stack, power_pump, net_power, efficiency
        """
        _I = current_A if callable(current_A) else lambda t: current_A
        _Q = flow_rate_L_min if callable(flow_rate_L_min) else lambda t: flow_rate_L_min

        t_arr = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_arr = t_arr[t_arr <= duration_s]
        N = len(t_arr)

        voltage = np.zeros(N)
        soc = np.zeros(N)
        power_stack = np.zeros(N)
        power_pump = np.zeros(N)
        net_power = np.zeros(N)
        efficiency = np.zeros(N)

        soc[0] = np.clip(soc_init, self.SOC_MIN, self.SOC_MAX)

        for i in range(N):
            t = t_arr[i]
            I_t = _I(t)
            Q_t = _Q(t)

            V_stack = float(self.stack_voltage(soc[i], I_t, Q_t))
            P_stack = V_stack * I_t  # W
            P_pump = float(self.pump_power_w(Q_t))

            voltage[i] = V_stack
            power_stack[i] = P_stack
            power_pump[i] = P_pump

            if I_t > 0:  # discharge
                net_power[i] = P_stack - P_pump
            else:  # charge
                net_power[i] = P_stack - P_pump  # both negative (charge) minus pump

            # Efficiency: |net_power / stack_power| or stack/(stack+pump)
            if abs(P_stack) > 1e-3:
                efficiency[i] = abs(net_power[i]) / (abs(P_stack) + P_pump)
            else:
                efficiency[i] = 0.0

            # Euler integration for SOC
            if i < N - 1:
                dsoc = float(self.dsoc_dt(soc[i], I_t))
                soc[i + 1] = np.clip(soc[i] + dsoc * dt, self.SOC_MIN, self.SOC_MAX)

        # Fix power_pump array name collision
        pp = np.zeros(N)
        for i in range(N):
            pp[i] = float(self.pump_power_w(_Q(t_arr[i])))

        return {
            "t": t_arr,
            "voltage": voltage,
            "soc": soc,
            "power_stack": power_stack,
            "power_pump": pp,
            "net_power": net_power,
            "efficiency": efficiency,
        }
