"""
EC037 -- Zinc-Bromine Flow Battery (ZBFB) -- F2a Physics-Lumped Stack Model

A 0D first-principles dynamic stack model for a hybrid Zn/Br2 flow battery.
The negative electrode plates metallic zinc (a *deposition* reaction, so the
storable charge is plating-limited, NOT tank-limited like an all-soluble
flow battery); the positive electrolyte stores soluble Br2/Br3- complexed
with a quaternary-ammonium agent. Both electrolytes are pumped from tanks.

State vector  y = [SOC_neg, c_Br2, T]   integrated with scipy.solve_ivp.

(1) Nernst stack voltage from SOC + bromine concentration
    Negative: Zn2+ + 2e-  <-> Zn(s)        E0 = -0.763 V (vs SHE)
    Positive: Br2 + 2e-    <-> 2 Br-        E0 = +1.087 V (vs SHE)
    E0_cell ~= 1.85 V
    Activity of plated Zn(s) = 1; the SOC-dependence enters through the
    Zn2+ activity (negative) and the Br2/Br- ratio (positive):
        E = E0(T) + (RT)/(2F) * ln( a_Zn2+ * a_Br2 / a_Br-^2 )
    Parameterised in SOC and the bromine concentration c_Br2 so that
    charging (SOC up, c_Br2 up) raises the OCV (V_charge > V_discharge).

(2) Overpotentials at the working current density j = I / (N_cells*A):
    - activation (Butler-Volmer / Tafel, both electrodes lumped)
    - ohmic      (area-specific resistance, Arrhenius in T)
    - concentration (mass-transport limit j_L, flow-rate dependent)

(3) Electrolyte SOC ODE from current and flow:
    dSOC/dt = -eta_C * I / Q_plating          (I>0 = discharge)
    Q_plating = plating-limited areal capacity * N_cells * A   [Coulombs]
    Coulombic efficiency eta_C < 1 on CHARGE because part of the charge
    current is lost to Br2 self-discharge / crossover (shuttle).

(4) Br2 self-discharge / crossover (coulombic loss):
    Soluble Br2 diffuses across the separator to the Zn electrode and is
    reduced directly (chemical shuttle), self-discharging the cell:
        dc_Br2/dt|loss = -k_sd * c_Br2          (1st-order)
    This shuttle current i_shuttle = n F V_pos k_sd c_Br2 is the dominant
    coulombic loss; eta_C = I/(I + i_shuttle) on charge.

(5) Lumped thermal ODE:
    m cp dT/dt = Q_gen - Q_loss
    Q_gen = I*(E_th - V_terminal)/... irreversibility + I*T*dOCV/dT (reversible)
    Q_loss = hA (T - T_amb)
    For Zn/Br2  dOCV/dT ~ -1.5e-4 V/K per cell (from Zn(s)+Br2 entropy).

Plating-limited capacity: zinc areal loading is capped (~ <=120 mAh/cm2 in
practice; high loading -> dendrites/shorting), so usable Coulombs scale with
electrode area, not tank volume.

References:
    Lim, H. S., Lackner, A. M., Knechtli, R. C. (1977). "Zinc-Bromine
        Secondary Battery." J. Electrochem. Soc. 124(8), 1154-1157.
    Skyllas-Kazacos, M., et al. (2011). "Progress in Flow Battery Research
        and Development." J. Electrochem. Soc. 158(8), R55-R79.
    Wu, M. C., Zhao, T. S., et al. (2017). "A zinc-bromine flow battery
        with improved design of cell structure and electrodes." Energy
        Conversion and Management 138, 271-277.
    Suresh, S., et al. (2018). "Zinc-bromine hybrid flow battery: effect of
        zinc utilization and performance characteristics." RSC Advances 8,
        24, 13374-13385.  (zinc plating utilisation / areal capacity limit)
    Wang, C., Lai, Q., et al. (2018). "Bromine complexing agents and the
        self-discharge of Zn-Br flow batteries." J. Power Sources.
    Newman, J. & Thomas-Alyea (2004). Electrochemical Systems, 3rd ed.
        (Butler-Volmer, mass-transport limiting current).
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314      # J/(mol.K)
F_CONST = 96485.0  # C/mol


class ZnBrFlowF2a:
    """Physics-lumped Zn/Br2 flow-battery stack with coupled SOC, Br2 and thermal ODEs."""

    SOC_MIN = 0.01
    SOC_MAX = 0.99
    n = 2  # electrons per reaction

    def __init__(self, params: dict):
        u = params["unit"]
        k = params["kinetics"]
        th = params["thermal"]
        sd = params["self_discharge"]

        # Geometry / stack
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2 = float(u["electrode_area_cm2"]["value"])
        self.E0 = float(u["E0"]["value"])
        self.c_Br2_max = float(u["c_Br2_max_M"]["value"])      # mol/L at SOC=1
        self.areal_cap_Ah_cm2 = float(u["areal_capacity_Ah_cm2"]["value"])

        # Plating-limited charge capacity [Coulombs] = Ah/cm2 * 3600 * A * N
        self.Q_plating = (
            self.areal_cap_Ah_cm2 * 3600.0 * self.A_cm2 * self.N_cells
        )

        # Kinetics
        self.j0 = float(k["j0_A_cm2"]["value"])                 # exchange current density
        self.alpha = float(k["alpha"]["value"])                # charge transfer coeff
        self.R_ohm_cm2_ref = float(k["R_cell_ohm_cm2_ref"]["value"])
        self.E_a_R = float(k["E_a_R"]["value"])                # J/mol, ohmic Arrhenius
        self.jL_ref = float(k["jL_ref_A_cm2"]["value"])        # limiting j at ref flow
        self.Q_flow_ref = float(k["flow_ref_Lpm"]["value"])    # L/min reference flow

        # Thermal
        self.T_ref = float(th["T_ref"]["value"])
        self.dOCV_dT = float(th["dOCV_dT"]["value"])           # V/K per cell
        self.m_cp = float(th["m_stack"]["value"]) * float(th["cp_stack"]["value"])  # J/K
        self.hA = float(th["hA"]["value"])                     # W/K
        self.T_amb = float(th["T_amb"]["value"])               # K

        # Self-discharge / crossover (shuttle)
        self.k_sd = float(sd["k_sd_per_s"]["value"])           # 1/s, 1st-order Br2 loss

        self.A_stack = self.A_cm2 * self.N_cells               # total cm2

    # ------------------------------------------------------------------
    # Electrochemistry
    # ------------------------------------------------------------------
    def _clip_soc(self, soc):
        return np.clip(soc, self.SOC_MIN, self.SOC_MAX)

    def r_cell(self, T):
        """Per-cell ohmic resistance [Ohm] via Arrhenius on area-specific R."""
        R_asr = self.R_ohm_cm2_ref * np.exp(
            self.E_a_R / R_GAS * (1.0 / T - 1.0 / self.T_ref)
        )
        return R_asr / self.A_cm2

    def e_nernst(self, soc, c_Br2, T):
        """
        Per-cell Nernst OCV [V].

        E = E0(T) + (RT)/(2F) * ln[ (SOC) * (c_Br2/c_max) / (1-SOC)^2 ]

        SOC scales Zn2+ depletion / Br- consumption; c_Br2 the oxidant
        activity. Monotonic increasing in SOC and c_Br2, so OCV rises on
        charge -> guarantees V_charge > V_discharge at equal |I|.
        """
        soc = self._clip_soc(np.asarray(soc, dtype=float))
        c_Br2 = np.maximum(np.asarray(c_Br2, dtype=float), 1e-6)
        T = np.asarray(T, dtype=float)
        E0_T = self.E0 + self.dOCV_dT * (T - self.T_ref)
        x_Br2 = c_Br2 / self.c_Br2_max
        arg = soc * x_Br2 / (1.0 - soc) ** 2
        return E0_T + (R_GAS * T) / (self.n * F_CONST) * np.log(arg)

    def _j_lim(self, flow_Lpm):
        """Mass-transport limiting current density [A/cm2], ~ proportional to flow."""
        flow_Lpm = max(float(flow_Lpm), 1e-6)
        return self.jL_ref * (flow_Lpm / self.Q_flow_ref) ** 0.5

    def activation_overpotential(self, I, T):
        """Lumped (both electrodes) activation overpotential magnitude [V/cell].

        Tafel form from Butler-Volmer (Newman & Thomas-Alyea 2004):
            eta_act = (RT)/(alpha n F) * ln(|j|/j0),  |j|>j0.
        """
        j = abs(float(I)) / self.A_stack
        if j <= self.j0:
            return 0.0
        return (R_GAS * T) / (self.alpha * self.n * F_CONST) * np.log(j / self.j0)

    def concentration_overpotential(self, I, T, flow_Lpm):
        """Mass-transport overpotential magnitude [V/cell] (Nernst diffusion)."""
        j = abs(float(I)) / self.A_stack
        jL = self._j_lim(flow_Lpm)
        ratio = j / jL
        if ratio >= 1.0:
            return 5.0  # flooded / starved electrode
        return -(R_GAS * T) / (self.n * F_CONST) * np.log(1.0 - ratio)

    def cell_voltage(self, soc, c_Br2, I, T, flow_Lpm):
        """
        Terminal per-cell voltage [V]. Sign convention: I>0 = discharge.
        Discharge: V = E - eta_act - eta_ohm - eta_conc   (V < E)
        Charge   : V = E + eta_act + eta_ohm + eta_conc   (V > E)
        """
        E = float(self.e_nernst(soc, c_Br2, T))
        eta_act = self.activation_overpotential(I, T)
        eta_ohm = abs(I) * self.r_cell(T)
        eta_conc = self.concentration_overpotential(I, T, flow_Lpm)
        losses = eta_act + eta_ohm + eta_conc
        if I >= 0.0:           # discharge
            return E - losses
        return E + losses      # charge

    def stack_voltage(self, soc, c_Br2, I, T, flow_Lpm):
        """Stack terminal voltage [V] = N_cells * cell voltage."""
        return self.N_cells * self.cell_voltage(soc, c_Br2, I, T, flow_Lpm)

    # ------------------------------------------------------------------
    # Self-discharge / crossover and coulombic efficiency
    # ------------------------------------------------------------------
    def shuttle_current(self, c_Br2):
        """
        Equivalent crossover/shuttle current [A] from 1st-order Br2 loss.
        Br2 reduced at the Zn side self-discharges the cell:
            i_shuttle = n F * V_pos * (k_sd * c_Br2)
        With positive electrolyte volume folded into k_sd's calibration we
        express it directly as a current proportional to the stored Br2
        fraction times the plating capacity scale.
        """
        c_Br2 = max(float(c_Br2), 0.0)
        frac = c_Br2 / self.c_Br2_max
        # i_shuttle scaled so it is a few % of a typical charge current
        return self.k_sd * frac * self.Q_plating

    def coulombic_efficiency(self, I, c_Br2):
        """
        Coulombic efficiency in (0,1).
        Charge (I<0): part of charge current lost to the Br2 shuttle ->
            eta_C = |I| / (|I| + i_shuttle).
        Discharge (I>0): shuttle adds to self-discharge ->
            eta_C = |I| / (|I| + i_shuttle) as well (delivered < drawn).
        """
        i_sh = self.shuttle_current(c_Br2)
        denom = abs(float(I)) + i_sh
        if denom <= 0.0:
            return 1.0
        eta = abs(float(I)) / denom
        return float(np.clip(eta, 1e-6, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # Heat generation
    # ------------------------------------------------------------------
    def heat_generation(self, soc, c_Br2, I, T, flow_Lpm):
        """
        Stack heat rate [W]:
            Q = |I_irrev * eta_total|*N + reversible I*N*T*dOCV/dT
        Irreversible part = |I| * (overpotential sum) * N_cells  (always >=0).
        Reversible (entropic) part = I * N_cells * T * dOCV_dT.
        """
        eta_act = self.activation_overpotential(I, T)
        eta_ohm = abs(I) * self.r_cell(T)
        eta_conc = self.concentration_overpotential(I, T, flow_Lpm)
        q_irrev = abs(I) * (eta_act + eta_ohm + eta_conc) * self.N_cells
        q_rev = I * self.N_cells * T * self.dOCV_dT
        return q_irrev + q_rev

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, I_func, flow_Lpm):
        soc, c_Br2, T = y
        soc = float(np.clip(soc, 0.0, 1.0))
        c_Br2 = max(float(c_Br2), 0.0)
        I = float(I_func(t))

        eta_C = self.coulombic_efficiency(I, c_Br2)
        # SOC ODE: I>0 discharge lowers SOC. On charge only eta_C of current
        # is stored (rest shuttles); on discharge eta_C fraction reaches load.
        if I >= 0.0:          # discharge
            dsoc = -I / (eta_C * self.Q_plating)
        else:                 # charge
            dsoc = -(eta_C * I) / self.Q_plating
        # hard stops at the rails
        if soc >= 1.0 and dsoc > 0:
            dsoc = 0.0
        if soc <= 0.0 and dsoc < 0:
            dsoc = 0.0

        # Br2 concentration tracks SOC of the positive side plus self-discharge
        c_Br2_eq = soc * self.c_Br2_max
        # relax toward equilibrium with charge/discharge, decay by shuttle
        dc = (c_Br2_eq - c_Br2) * 0.05 - self.k_sd * c_Br2

        # Thermal ODE
        Q_gen = self.heat_generation(soc, c_Br2, I, T, flow_Lpm)
        Q_loss = self.hA * (T - self.T_amb)
        dT = (Q_gen - Q_loss) / self.m_cp

        return [dsoc, dc, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.5, T0=298.15, flow_Lpm=2.0,
                 c_Br2_0=None, dt=1.0, duration_s=600.0):
        """
        Integrate the coupled SOC / Br2 / thermal ODEs.

        Parameters
        ----------
        current_A : float or callable(t)  -- stack current [A], +discharge.
        soc0      : float  -- initial state of charge.
        T0        : float  -- initial temperature [K].
        flow_Lpm  : float  -- electrolyte volumetric flow [L/min].
        c_Br2_0   : float  -- initial Br2 concentration [mol/L] (default soc0*c_max).
        dt        : float  -- output time step [s].
        duration_s: float  -- total simulation time [s].

        Returns
        -------
        dict of time-series arrays.
        """
        I_func = current_A if callable(current_A) else (lambda t: current_A)
        if c_Br2_0 is None:
            c_Br2_0 = soc0 * self.c_Br2_max

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [soc0, c_Br2_0, T0],
            t_eval=t_eval, args=(I_func, flow_Lpm),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        soc = np.clip(sol.y[0], 0.0, 1.0)
        c_Br2 = np.maximum(sol.y[1], 0.0)
        T = sol.y[2]
        N = len(t)

        V_stack = np.zeros(N)
        E_ocv = np.zeros(N)
        I_arr = np.zeros(N)
        eta_C = np.zeros(N)
        i_shuttle = np.zeros(N)
        power = np.zeros(N)
        for i in range(N):
            Ii = float(I_func(t[i]))
            I_arr[i] = Ii
            V_stack[i] = self.stack_voltage(soc[i], c_Br2[i], Ii, T[i], flow_Lpm)
            E_ocv[i] = self.N_cells * float(self.e_nernst(soc[i], c_Br2[i], T[i]))
            eta_C[i] = self.coulombic_efficiency(Ii, c_Br2[i])
            i_shuttle[i] = self.shuttle_current(c_Br2[i])
            power[i] = V_stack[i] * Ii

        return {
            "t": t,
            "soc": soc,
            "c_Br2": c_Br2,
            "temperature": T,
            "voltage": V_stack,
            "ocv": E_ocv,
            "current": I_arr,
            "coulombic_efficiency": eta_C,
            "shuttle_current": i_shuttle,
            "power": power,
        }
