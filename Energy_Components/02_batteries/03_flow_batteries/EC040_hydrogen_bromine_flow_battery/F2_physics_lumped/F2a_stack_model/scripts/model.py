"""
EC040 -- Hydrogen-Bromine Flow Battery (HBrFB) -- F2a Physics-Lumped Stack Model

A 0D physics-lumped model of an H2/Br2 flow-battery stack with TWO coupled
state ODEs (SOC and temperature) integrated with scipy.integrate.solve_ivp.

Cell chemistry
--------------
    Positive (catholyte) : Br2 + 2 e-  <-> 2 Br-     E0 = +1.065 V vs SHE
    Negative (anolyte)   : 2 H+ + 2 e- <-> H2(g)     E0 =  0.000 V vs SHE
    Cell                 : Br2 + H2 <-> 2 HBr         E0_cell ~ 1.09 V (activity-corrected)

Terminal voltage (sign convention: I>0 discharge, I<0 charge)
-------------------------------------------------------------
    V_cell(I, SOC, T) = E_Nernst(SOC, T)
                        - sign(I) * [ eta_act_Br(|I|,T)   (Butler-Volmer, Br2 slow)
                                    + eta_act_H2(|I|,T)    (Butler-Volmer, HOR/HER fast)
                                    + eta_conc(|I|) ]      (Br2 mass transport)
                        - I * R_ohm(T)                     (low ASR -> high power density)

So on discharge (I>0) all losses subtract -> V_disch < E_Nernst,
and on charge (I<0) all losses add        -> V_charge > E_Nernst,
guaranteeing V_charge > V_disch at equal |I| (round-trip eff < 1).

Nernst potential (Br2/Br- vs H2/H+ with SOC activity term)
----------------------------------------------------------
    E_Nernst = E0(T) + (R T)/(n F) * ln( SOC / (1 - SOC) )
    E0(T)    = E0_ref + dOCV_dT * (T - T_ref)         (entropic temperature correction)
    SOC == fractional state of charge = (Br2 available) / (Br2 at full charge)

State ODEs
----------
    Coulomb / SOC balance (Faraday's law + Br2 crossover self-discharge):
        d(SOC)/dt = -I / (n_cell_basis * Q_nom_C)  -  k_cross * SOC
        (I>0 discharge lowers SOC; crossover always lowers SOC -> Coulombic eff < 1)

    Lumped thermal balance:
        m cp dT/dt = Q_gen - Q_cool
        Q_gen  = N_cells*|I|*(|eta_act+eta_conc+eta_ohm|)   (irreversible, always >=0)
                 + N_cells*I*T*dOCV_dT                       (reversible/entropic)
        Q_cool = hA*(T - T_coolant)

Kinetics
--------
Butler-Volmer inverted to Tafel-like form for each couple. The H2/H+ couple has a
large exchange current density (Pt/C) -> negligible activation loss; the Br2/Br-
couple is the rate-limiting electrode with a modest j0 and Arrhenius T-dependence.

References
----------
    Livshits, V. et al. (2006). J. Power Sources 160, 1298-1301.
    Cho, K. T. et al. (2012). J. Electrochem. Soc. 159, A1806-A1815.
    Tucker, M. C. et al. (2015). J. Electrochem. Soc. 162, A2159-A2165.
    Kreutzer, H. et al. (2012). J. Electrochem. Soc. 159, F331-F337.
    Bard, A. J. & Faulkner, L. R. (2001). Electrochemical Methods, 2nd ed., Wiley.
"""

import numpy as np
from scipy.integrate import solve_ivp


class HydrogenBromineFlowF2a:
    """Physics-lumped H2/Br2 flow-battery stack with SOC + thermal ODEs."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol

    SOC_MIN = 0.02
    SOC_MAX = 0.98

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cell = u["A_cell"]["value"]               # cm2
        self.E0_ref = u["E0"]["value"]                   # V
        self.n = int(u["n"]["value"])

        # Kinetics
        self.j0_Br_ref = u["j0_Br_ref"]["value"]         # A/cm2
        self.j0_H2_ref = u["j0_H2_ref"]["value"]         # A/cm2
        self.alpha_Br = u["alpha_Br"]["value"]
        self.alpha_H2 = u["alpha_H2"]["value"]
        self.E_act_Br = u["E_act_Br"]["value"]           # J/mol

        # Ohmic
        self.R_ohm_cm2_ref = u["R_ohm_cm2_ref"]["value"] # Ohm.cm2
        self.E_act_R = u["E_act_R"]["value"]             # J/mol

        # Mass transport
        self.j_L = u["j_L"]["value"]                     # A/cm2

        # Capacity / crossover
        self.Q_nom_Ah = u["Q_nom_Ah"]["value"]           # Ah (stack)
        self.Q_nom_C = self.Q_nom_Ah * 3600.0            # Coulombs
        self.k_cross = u["k_cross"]["value"]             # 1/s

        # Parasitics / thermal
        self.k_pump = u["pump_loss_coefficient"]["value"]  # W/A^2
        self.m_stack = u["m_stack"]["value"]               # kg
        self.cp_stack = u["cp_stack"]["value"]             # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]               # W/K
        self.T_coolant = u["T_coolant"]["value"]           # K
        self.dOCV_dT = u["dOCV_dT"]["value"]               # V/K
        self.T_ref = u["T_ref"]["value"]                   # K

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------
    def e0_thermal(self, T):
        """Temperature-corrected standard cell potential [V]."""
        return self.E0_ref + self.dOCV_dT * (T - self.T_ref)

    def nernst_voltage(self, soc, T):
        """Open-circuit (Nernst) cell voltage [V] at given SOC and T."""
        s = float(np.clip(soc, self.SOC_MIN, self.SOC_MAX))
        thermal = self.R * T / (self.n * self.F)
        return self.e0_thermal(T) + thermal * np.log(s / (1.0 - s))

    # ------------------------------------------------------------------
    # Kinetics -- Butler-Volmer (Tafel-inverted) per couple
    # ------------------------------------------------------------------
    def _j0_Br(self, T):
        return self.j0_Br_ref * np.exp(
            (-self.E_act_Br / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )

    def activation_overpotential_Br(self, j_abs, T):
        """Br2/Br- activation overpotential [V] (rate-limiting electrode)."""
        if j_abs <= 0:
            return 0.0
        j0 = max(self._j0_Br(T), 1e-12)
        # asinh form of Butler-Volmer (valid for all j, reduces to Tafel at high j)
        return (self.R * T) / (self.alpha_Br * self.n * self.F) * \
            np.arcsinh(j_abs / (2.0 * j0))

    def activation_overpotential_H2(self, j_abs, T):
        """H2/H+ activation overpotential [V] (fast Pt/C kinetics -> small)."""
        if j_abs <= 0:
            return 0.0
        j0 = max(self.j0_H2_ref, 1e-12)
        return (self.R * T) / (self.alpha_H2 * self.n * self.F) * \
            np.arcsinh(j_abs / (2.0 * j0))

    # ------------------------------------------------------------------
    # Ohmic
    # ------------------------------------------------------------------
    def r_ohm_cell(self, T):
        """Per-cell ohmic resistance [Ohm] (Arrhenius membrane conductivity)."""
        asr = self.R_ohm_cm2_ref * np.exp(
            self.E_act_R / self.R * (1.0 / T - 1.0 / self.T_ref)
        )
        return asr / self.A_cell

    def ohmic_overpotential(self, j_abs, T):
        """Per-cell ohmic loss magnitude [V]."""
        asr = self.R_ohm_cm2_ref * np.exp(
            self.E_act_R / self.R * (1.0 / T - 1.0 / self.T_ref)
        )
        return j_abs * asr

    # ------------------------------------------------------------------
    # Mass transport (Br2 concentration overpotential)
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j_abs):
        """Br2 mass-transport overpotential magnitude [V]."""
        if j_abs <= 0:
            return 0.0
        ratio = j_abs / self.j_L
        if ratio >= 1.0:
            return 10.0  # flooded / depleted
        return -(self.R * self.T_ref) / (self.n * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell / stack voltage (signed current: I>0 discharge, I<0 charge)
    # ------------------------------------------------------------------
    def cell_voltage(self, current_A, soc, T):
        """Single-cell terminal voltage [V]. current_A>0 discharge, <0 charge."""
        j = current_A / self.A_cell        # A/cm2 (signed)
        j_abs = abs(j)
        E = self.nernst_voltage(soc, T)
        eta_act = (self.activation_overpotential_Br(j_abs, T) +
                   self.activation_overpotential_H2(j_abs, T))
        eta_conc = self.concentration_overpotential(j_abs)
        eta_ohm = self.ohmic_overpotential(j_abs, T)
        s = np.sign(current_A) if current_A != 0 else 0.0
        # losses always oppose the driven direction
        return E - s * (eta_act + eta_conc + eta_ohm)

    def stack_voltage(self, current_A, soc, T):
        return self.N_cells * self.cell_voltage(current_A, soc, T)

    def overpotential_sum(self, current_A, T):
        """Total per-cell irreversible loss magnitude [V] (>=0)."""
        j_abs = abs(current_A) / self.A_cell
        return (self.activation_overpotential_Br(j_abs, T) +
                self.activation_overpotential_H2(j_abs, T) +
                self.concentration_overpotential(j_abs) +
                self.ohmic_overpotential(j_abs, T))

    def pump_loss(self, current_A):
        """Parasitic pump power [W]."""
        return self.k_pump * current_A ** 2

    # ------------------------------------------------------------------
    # State derivatives
    # ------------------------------------------------------------------
    def dSOCdt(self, soc, current_A):
        """SOC rate [1/s]. Faraday + Br2 crossover self-discharge."""
        # I>0 discharge -> SOC decreases
        faradaic = -current_A / self.Q_nom_C
        crossover = -self.k_cross * np.clip(soc, 0.0, 1.0)  # always self-discharges
        ds = faradaic + crossover
        # clamp at bounds (no overshoot)
        if soc <= self.SOC_MIN and ds < 0:
            return 0.0
        if soc >= self.SOC_MAX and ds > 0:
            return 0.0
        return ds

    def dTdt(self, soc, current_A, T):
        """Temperature rate [K/s]."""
        q_irrev = self.N_cells * abs(current_A) * self.overpotential_sum(current_A, T)
        # reversible heat: -N*I*T*dE0/dT (with our sign, discharge of an exothermic
        # entropy-negative cell releases heat); magnitude is small
        q_rev = self.N_cells * current_A * T * (-self.dOCV_dT)
        q_pump = self.pump_loss(current_A)  # pump work dissipated as heat
        Q_gen = q_irrev + q_rev + q_pump
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation (coupled SOC + thermal ODE)
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.5, T0=298.15, dt=10.0, duration_s=3600.0):
        """
        Integrate coupled SOC + temperature ODEs.

        Parameters
        ----------
        current_A : float or callable(t)
            Stack current [A]. >0 discharge, <0 charge.
        soc0 : float        initial state of charge [-]
        T0   : float        initial temperature [K]
        dt   : float        output time step [s]
        duration_s : float  total simulation time [s]

        Returns
        -------
        dict of time-series arrays.
        """
        _I = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            soc, T = y
            I = _I(t)
            return [self.dSOCdt(soc, I), self.dTdt(soc, I, T)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [soc0, T0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        soc_out = np.clip(sol.y[0], 0.0, 1.0)
        T_out = sol.y[1]
        N = len(t_out)

        v_stack = np.zeros(N)
        v_cell = np.zeros(N)
        E_nernst = np.zeros(N)
        power_W = np.zeros(N)
        eta_act_Br = np.zeros(N)
        eta_act_H2 = np.zeros(N)
        eta_ohm = np.zeros(N)
        eta_conc = np.zeros(N)

        for i in range(N):
            I = _I(t_out[i])
            T = T_out[i]
            s = soc_out[i]
            jabs = abs(I) / self.A_cell
            E_nernst[i] = self.nernst_voltage(s, T)
            v_cell[i] = self.cell_voltage(I, s, T)
            v_stack[i] = self.N_cells * v_cell[i]
            # net electrical power (terminal) minus pump parasitics
            power_W[i] = v_stack[i] * I - self.pump_loss(I)
            eta_act_Br[i] = self.activation_overpotential_Br(jabs, T)
            eta_act_H2[i] = self.activation_overpotential_H2(jabs, T)
            eta_ohm[i] = self.ohmic_overpotential(jabs, T)
            eta_conc[i] = self.concentration_overpotential(jabs)

        return {
            "t": t_out,
            "soc": soc_out,
            "temperature": T_out,
            "cell_voltage": v_cell,
            "stack_voltage": v_stack,
            "power_W": power_W,
            "E_nernst": E_nernst,
            "overpotentials": {
                "activation_Br": eta_act_Br,
                "activation_H2": eta_act_H2,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }

    # ------------------------------------------------------------------
    # Round-trip efficiency (charge then discharge at fixed |I|)
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, current_A, soc=0.5, T=298.15):
        """Voltaic*Coulombic round-trip efficiency estimate at a fixed SOC/T."""
        I = abs(current_A)
        V_dis = self.cell_voltage(+I, soc, T)
        V_chg = self.cell_voltage(-I, soc, T)
        voltaic = V_dis / V_chg if V_chg > 0 else 0.0
        # coulombic eff: faradaic vs faradaic+crossover loss over a nominal hour
        far_rate = I / self.Q_nom_C
        cross_rate = self.k_cross * soc
        coulombic = far_rate / (far_rate + cross_rate) if (far_rate + cross_rate) > 0 else 0.0
        return voltaic * coulombic
