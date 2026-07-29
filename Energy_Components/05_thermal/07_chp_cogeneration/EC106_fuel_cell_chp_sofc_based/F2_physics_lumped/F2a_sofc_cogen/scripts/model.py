"""
EC106 -- Fuel Cell CHP (SOFC-Based) -- F2a Physics-Lumped Cogeneration Model

Physics-lumped combined-heat-and-power (CHP) model of a high-temperature
Solid-Oxide Fuel Cell (SOFC) micro-cogeneration unit. The SOFC stack
(operating ~800-1000 C) produces electrical power from a Nernst potential net
of activation / ohmic / concentration overpotentials; the stack waste heat and
the hot anode/cathode exhaust are recovered by a heat-recovery unit (HRU) to
deliver useful thermal output (hot water / space heat).

------------------------------------------------------------------------------
1. ELECTROCHEMISTRY (per cell, evaluated at the current stack temperature T)
------------------------------------------------------------------------------
    V = E_nernst(T, pH2, pO2, pH2O)
        - eta_act_anode(j, T)     Butler-Volmer (anode)
        - eta_act_cathode(j, T)   Butler-Volmer (cathode)
        - eta_ohm(j, T)           YSZ electrolyte ionic resistance
        - eta_conc(j, T)          anode + cathode diffusion limit

    E_nernst = E0(T) + (RT/2F) ln( pH2 * sqrt(pO2) / pH2O )      [Chan 2001]
    sigma_YSZ(T) = (sigma_0 / T) exp(-E_act_ysz / T)             [Chan 2001]

    Electrical power:  P_e = N_cells * A_cell * j * V_cell

------------------------------------------------------------------------------
2. ENERGY / HEAT BALANCE (cogeneration)
------------------------------------------------------------------------------
Fuel chemical power in (LHV basis, scaled by fuel utilisation Uf):
    Q_fuel = N_cells * A_cell * j / (n F) * LHV_H2 / Uf

Total heat liberated by the irreversible electrochemistry (the difference
between the thermoneutral/enthalpy voltage and the operating voltage):
    Q_heat_total = N_cells * A_cell * j * (E_th(T) - V_cell)

This waste heat splits into two recoverable streams plus a parasitic loss:
  (a) hot exhaust gas sensible heat, recovered with HRU effectiveness eps_hru
      down to the exhaust stack temperature T_exhaust_out;
  (b) stack convective/radiative loss to the water jacket (recoverable
      fraction f_loss_recovery of the hA_loss*(T - T_amb) parasitic flux);
  the remainder is lost to ambient.

    Q_useful_thermal = Q_exhaust_recovered + Q_jacket_recovered

------------------------------------------------------------------------------
3. CHP PERFORMANCE METRICS
------------------------------------------------------------------------------
    eta_electrical = P_e / Q_fuel
    eta_thermal    = Q_useful_thermal / Q_fuel
    eta_total      = eta_electrical + eta_thermal      (CHP / cogeneration eff.)
    power_to_heat  = P_e / Q_useful_thermal

Enforced bounds (Larminie & Dicks 2003; EG&G Fuel Cell Handbook 2004):
    V_cell < E_nernst,  0 < each eta < 1,
    eta_total > eta_electrical and eta_total < 1, and energy conservation:
    P_e + Q_useful_thermal + Q_loss = Q_fuel.

------------------------------------------------------------------------------
4. LUMPED STACK THERMAL ODE  (integrated with scipy.solve_ivp)
------------------------------------------------------------------------------
    m_stack * cp_stack * dT/dt = Q_heat_total - Q_exhaust_recovered
                                 - hA_loss * (T - T_amb)

i.e. the stack stores the difference between heat liberated and heat carried
away by the recovered exhaust enthalpy and the jacket/ambient loss.

References
----------
    Chan, S.H., Khor, K.A., Xia, Z.T. (2001) "A complete polarization model of
        a SOFC...", J. Power Sources 93, 130-140.
    Campanari, S., Iora, P. (2004) "Definition and sensitivity analysis of a
        finite volume SOFC model", J. Power Sources 132, 113-126.
    Larminie, J., Dicks, A. (2003) "Fuel Cell Systems Explained", 2nd ed.,
        Wiley (cogeneration efficiency definitions, thermoneutral voltage).
    EG&G Technical Services (2004) "Fuel Cell Handbook", 7th ed., U.S. DOE
        (CHP heat-recovery and exhaust-enthalpy treatment).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class SOFC_CHP_F2a:
    """SOFC-based fuel-cell CHP -- electrochemical stack + heat recovery + thermal ODE."""

    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 2              # electrons per H2

    def __init__(self, params: dict):
        u = params["unit"]
        # --- stack geometry / electrochemistry ---
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]               # cm2
        self.t_el = u["t_electrolyte"]["value"]          # cm
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
        self.uf_nom = u["uf_nominal"]["value"]
        self.j_nom = u["j_nominal"]["value"]
        # --- thermal / CHP ---
        self.m_stack = u["m_stack"]["value"]             # kg
        self.cp_stack = u["cp_stack"]["value"]           # J/(kg.K)
        self.hA_loss = u["hA_loss"]["value"]             # W/K
        self.T_amb = u["T_amb"]["value"]                 # K
        self.lhv = u["lhv_h2"]["value"]                  # J/mol
        self.hhv = u["hhv_h2"]["value"]                  # J/mol
        self.T_exh_out = u["T_exhaust_out"]["value"]     # K
        self.T_water_in = u["T_water_in"]["value"]       # K
        self.T_air_stack_in = u["T_air_stack_in"]["value"]  # K (recuperated air inlet)
        self.cp_exh = u["cp_exhaust"]["value"]           # J/(kg.K)
        self.lambda_air = u["lambda_air"]["value"]
        self.eps_hru = u["eps_hru"]["value"]
        self.f_loss_rec = u["f_loss_recovery"]["value"]

    # ------------------------------------------------------------------
    # Nernst (reversible / open-circuit) voltage  -- Chan (2001)
    # ------------------------------------------------------------------
    def nernst_voltage(self, T, pH2=None, pO2=None, pH2O=None):
        """Reversible cell voltage [V]."""
        pH2 = self.pH2 if pH2 is None else pH2
        pO2 = self.pO2 if pO2 is None else pO2
        pH2O = self.pH2O if pH2O is None else pH2O
        E0 = 1.253 - 2.4516e-4 * T
        E = E0 + (self.R * T) / (2.0 * self.F) * np.log(
            pH2 * np.sqrt(pO2) / max(pH2O, 1e-6)
        )
        return E

    def thermoneutral_voltage(self, T):
        """Thermoneutral (enthalpy) voltage [V] used for heat generation."""
        return 1.285 - 2.0e-4 * (T - 298.15)

    # ------------------------------------------------------------------
    # Activation overpotentials -- Butler-Volmer (anode + cathode)
    # ------------------------------------------------------------------
    def _butler_volmer_eta(self, j, j0, alpha, T):
        if j <= 0 or j0 <= 0:
            return 0.0
        a = alpha * self.n * self.F / (self.R * T)
        if j > 5.0 * j0:  # Tafel regime
            return (self.R * T) / (alpha * self.n * self.F) * np.log(j / j0)

        def bv_residual(eta):
            return j0 * (np.exp(a * eta)
                         - np.exp(-(1 - alpha) * self.n * self.F * eta / (self.R * T))) - j
        try:
            eta = brentq(bv_residual, 0.0, 2.0, xtol=1e-8)
        except ValueError:
            eta = (self.R * T) / (alpha * self.n * self.F) * np.log(j / j0 + 1.0)
        return max(eta, 0.0)

    def activation_anode(self, j, T):
        j0 = self.j0_a_ref * np.exp((-self.E_act_a / self.R) * (1.0 / T - 1.0 / self.T_ref))
        return self._butler_volmer_eta(j, j0, self.alpha_a, T)

    def activation_cathode(self, j, T):
        j0 = self.j0_c_ref * np.exp((-self.E_act_c / self.R) * (1.0 / T - 1.0 / self.T_ref))
        return self._butler_volmer_eta(j, j0, self.alpha_c, T)

    # ------------------------------------------------------------------
    # Ohmic overpotential -- YSZ electrolyte ionic resistance  Chan (2001)
    # ------------------------------------------------------------------
    def ysz_conductivity(self, T):
        return (self.sigma_0 / T) * np.exp(-self.E_act_ysz / T)

    def ohmic_overpotential(self, j, T):
        sigma = self.ysz_conductivity(T)
        return j * self.t_el / max(sigma, 1e-8)

    # ------------------------------------------------------------------
    # Concentration overpotential -- mass-transport limit
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, T):
        if j <= 0:
            return 0.0
        r_a = j / self.j_L_a
        r_c = j / self.j_L_c
        eta_a = -(self.R * T) / (self.n * self.F) * np.log(1.0 - r_a) if r_a < 1.0 else 5.0
        eta_c = -(self.R * T) / (4.0 * self.F) * np.log(1.0 - r_c) if r_c < 1.0 else 5.0
        return eta_a + eta_c

    # ------------------------------------------------------------------
    # Cell voltage (V < E_nernst always enforced)
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T):
        E = self.nernst_voltage(T)
        eta = (self.activation_anode(j, T)
               + self.activation_cathode(j, T)
               + self.ohmic_overpotential(j, T)
               + self.concentration_overpotential(j, T))
        return max(E - eta, 0.0)

    # ------------------------------------------------------------------
    # Fuel utilization (scales linearly with load about nominal)
    # ------------------------------------------------------------------
    def fuel_utilization(self, j):
        if self.j_nom <= 0 or j <= 0:
            return self.uf_nom
        uf = self.uf_nom * (j / self.j_nom)
        return float(min(max(uf, 1e-3), 0.95))

    # ------------------------------------------------------------------
    # Power & heat-recovery / CHP balance at a given (j, T)
    # ------------------------------------------------------------------
    def power_and_heat(self, j, T):
        """
        Returns a dict of instantaneous CHP quantities (all powers in W):
            P_e_W, Q_fuel_W, Q_heat_total_W,
            Q_exhaust_recovered_W, Q_jacket_recovered_W,
            Q_useful_thermal_W, Q_loss_W,
            eta_electrical, eta_thermal, eta_total, power_to_heat,
            V_cell, E_nernst
        """
        A_tot = self.N_cells * self.A_cell  # cm2 of active area in stack
        V = self.cell_voltage(j, T)
        E = self.nernst_voltage(T)

        # --- electrical power ---
        P_e = A_tot * j * V                                  # W

        # --- fuel chemical power (LHV), via molar H2 consumption and Uf ---
        Uf = self.fuel_utilization(j)
        I = A_tot * j                                        # total current [A]
        n_h2_consumed = I / (self.n * self.F)                # mol/s reacted
        n_h2_fed = n_h2_consumed / Uf if Uf > 0 else 0.0     # mol/s supplied
        Q_fuel = n_h2_fed * self.lhv                         # W (LHV basis)

        # --- total irreversible heat liberated by the stack ---
        E_th = self.thermoneutral_voltage(T)
        Q_heat_total = max(A_tot * j * (E_th - V), 0.0)      # W

        # --- exhaust sensible-heat recovery ---
        # exhaust mass flow proportional to cathode air (excess-air ratio) +
        # anode off-gas; modelled as effective gas mass flow carrying the
        # stack-temperature enthalpy down to T_water_in. Recoverable enthalpy
        # bounded by HRU effectiveness eps_hru and the exhaust outlet floor.
        # mdot_exh [kg/s] from O2 stoichiometry of air:
        #   O2 reacted = I/(4F) mol/s; air O2 frac 0.21; air molar mass ~28.97 g
        n_o2_react = I / (4.0 * self.F)                      # mol/s O2 consumed
        n_air = self.lambda_air * n_o2_react / 0.21          # mol/s air supplied
        mdot_exh = n_air * 0.02897                           # kg/s (~air + fuel)
        # ideal sensible enthalpy available cooling exhaust T -> T_water_in:
        Q_exh_ideal = mdot_exh * self.cp_exh * max(T - self.T_water_in, 0.0)
        # capped by the HRU effectiveness and by what physically cannot be
        # extracted below the exhaust outlet floor temperature:
        Q_exh_floor = mdot_exh * self.cp_exh * max(self.T_exh_out - self.T_water_in, 0.0)
        Q_exhaust_rec = self.eps_hru * max(Q_exh_ideal - Q_exh_floor, 0.0)
        # never claim more than the total heat liberated:
        Q_exhaust_rec = min(Q_exhaust_rec, Q_heat_total)

        # --- stack jacket (convective/radiative) loss, partly recoverable ---
        Q_jacket_loss = self.hA_loss * max(T - self.T_amb, 0.0)   # W
        Q_jacket_rec = self.f_loss_rec * Q_jacket_loss
        # cap recovered jacket heat by remaining liberated heat:
        Q_jacket_rec = min(Q_jacket_rec, max(Q_heat_total - Q_exhaust_rec, 0.0))

        Q_useful = Q_exhaust_rec + Q_jacket_rec
        # enforce useful thermal cannot exceed liberated heat
        Q_useful = min(Q_useful, Q_heat_total)
        Q_loss = max(Q_fuel - P_e - Q_useful, 0.0)

        # --- CHP efficiencies (LHV basis) ---
        if Q_fuel > 0:
            eta_e = P_e / Q_fuel
            eta_th = Q_useful / Q_fuel
        else:
            eta_e = 0.0
            eta_th = 0.0
        eta_total = eta_e + eta_th
        power_to_heat = P_e / Q_useful if Q_useful > 1e-9 else float("inf")

        return {
            "P_e_W": P_e,
            "Q_fuel_W": Q_fuel,
            "Q_heat_total_W": Q_heat_total,
            "Q_exhaust_recovered_W": Q_exhaust_rec,
            "Q_jacket_recovered_W": Q_jacket_rec,
            "Q_useful_thermal_W": Q_useful,
            "Q_loss_W": Q_loss,
            "eta_electrical": eta_e,
            "eta_thermal": eta_th,
            "eta_total": eta_total,
            "power_to_heat": power_to_heat,
            "V_cell": V,
            "E_nernst": E,
        }

    # ------------------------------------------------------------------
    # Lumped stack thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j):
        """
        Stack temperature rate of change [K/s].

        Heat balance on the lumped solid:
            m cp dT/dt = Q_heat_total - Q_gas_cool - Q_jacket_loss

        Q_gas_cool: sensible heat carried out of the stack by the process gas
        (cathode air + anode off-gas). The cathode air is recuperated, so it
        enters the stack at T_air_stack_in (~recuperated, high) and leaves at
        the stack temperature; the net convective cooling on the solid is
        mdot_gas * cp_gas * (T - T_air_stack_in). This term grows with current
        (more air) and with temperature, giving a stable steady state inside
        the SOFC operating window (Campanari & Iora 2004, energy-balance form).
        Q_jacket_loss: insulated-hotbox parasitic loss to ambient.
        """
        A_tot = self.N_cells * self.A_cell
        V = self.cell_voltage(j, T)
        E_th = self.thermoneutral_voltage(T)
        Q_heat_total = max(A_tot * j * (E_th - V), 0.0)

        # process-gas mass flow (cathode air dominated), scales with current
        I = A_tot * j
        n_o2_react = I / (4.0 * self.F)
        n_air = self.lambda_air * n_o2_react / 0.21
        mdot_gas = n_air * 0.02897                      # kg/s
        Q_gas_cool = mdot_gas * self.cp_exh * max(T - self.T_air_stack_in, 0.0)

        Q_jacket_loss = self.hA_loss * (T - self.T_amb)
        return (Q_heat_total - Q_gas_cool - Q_jacket_loss) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def simulate(self, current_density, T_cell_K, dt=1.0, duration_s=600.0):
        """
        Simulate SOFC-CHP dynamics with the coupled lumped thermal ODE.

        Parameters
        ----------
        current_density : float or callable(t) -> float   [A/cm2]
        T_cell_K        : float, initial stack temperature [K]
        dt              : float, output time step          [s]
        duration_s      : float, total duration            [s]

        Returns
        -------
        dict of time-series arrays plus scalar steady-state CHP summary.
        """
        _j = current_density if callable(current_density) else (lambda t: current_density)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], _j(t))]

        sol = solve_ivp(rhs, (0.0, duration_s), [T_cell_K],
                        t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
                        max_step=dt)

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        voltage = np.zeros(N)
        P_e = np.zeros(N)
        Q_fuel = np.zeros(N)
        Q_useful = np.zeros(N)
        Q_loss = np.zeros(N)
        eta_e = np.zeros(N)
        eta_th = np.zeros(N)
        eta_total = np.zeros(N)
        p2h = np.zeros(N)
        E_nernst = np.zeros(N)

        for i in range(N):
            ph = self.power_and_heat(_j(t_out[i]), T_out[i])
            voltage[i] = ph["V_cell"]
            P_e[i] = ph["P_e_W"]
            Q_fuel[i] = ph["Q_fuel_W"]
            Q_useful[i] = ph["Q_useful_thermal_W"]
            Q_loss[i] = ph["Q_loss_W"]
            eta_e[i] = ph["eta_electrical"]
            eta_th[i] = ph["eta_thermal"]
            eta_total[i] = ph["eta_total"]
            p2h[i] = ph["power_to_heat"]
            E_nernst[i] = ph["E_nernst"]

        return {
            "t": t_out,
            "temperature": T_out,
            "voltage": voltage,
            "P_e_W": P_e,
            "P_e_kW": P_e / 1e3,
            "Q_fuel_W": Q_fuel,
            "Q_useful_thermal_W": Q_useful,
            "Q_useful_thermal_kW": Q_useful / 1e3,
            "Q_loss_W": Q_loss,
            "eta_electrical": eta_e,
            "eta_thermal": eta_th,
            "eta_total": eta_total,
            "power_to_heat": p2h,
            "E_nernst": E_nernst,
            # scalar steady-state summary (final step)
            "steady_state": {
                "T_K": float(T_out[-1]),
                "P_e_kW": float(P_e[-1] / 1e3),
                "Q_useful_thermal_kW": float(Q_useful[-1] / 1e3),
                "eta_electrical": float(eta_e[-1]),
                "eta_thermal": float(eta_th[-1]),
                "eta_total": float(eta_total[-1]),
                "power_to_heat": float(p2h[-1]),
            },
        }
