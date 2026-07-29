"""
EC202 -- Direct Air Capture (DAC), Liquid Solvent -- F2a Lumped Contactor + Calciner

Physics-lumped (0D) first-principles model of a Carbon Engineering-style KOH/Ca
caustic-loop DAC plant. Two coupled lumped units integrated with scipy.solve_ivp:

  1) AIR CONTACTOR (mass-transfer limited, very dilute CO2).
     Ambient air (~420 ppm CO2) is blown through a packed slab where CO2 is
     absorbed into a strongly alkaline hydroxide solution:
         CO2(g) + 2 OH-  ->  CO3^2- + H2O          (fast, OH- in large excess)
     The capture is gas-film / mass-transfer controlled, NOT reaction-rate
     controlled, because the liquid OH- is in vast excess relative to the trace
     CO2.  Using a lumped overall volumetric mass-transfer coefficient Kg*a and a
     plug-flow air pass through packing depth L, the single-pass capture fraction
     is the classic exponential of a transfer unit (NTU):

         eta_single = 1 - exp( -Kg*a * L / u_air )          (Holmes & Keith 2012)

     The molar CO2 capture rate from the dilute air stream is then
         R_abs = eta_single * (y_CO2 * P / (R*T_air)) * (u_air * A_contactor)   [mol/s]
     i.e. (inlet CO2 molar concentration) * (volumetric air flow) * capture frac.

  2) CAUSTIC / CALCINER LOOP (high-temp regeneration, large thermal demand).
     Captured carbonate is pelletized to CaCO3 (causticization + slaking, lumped
     into a loop efficiency) and fed to a calciner held near 900 C, where the
     endothermic decomposition releases the CO2:
         CaCO3(s)  ->  CaO(s) + CO2(g)     dH = +178 kJ/mol     (calcination)
     The lumped calcination rate follows Arrhenius kinetics on the accumulated
     CaCO3 inventory, and a thermal ODE tracks the calciner temperature driven by
     burner heat against the endothermic reaction heat sink:

         dn_CaCO3/dt = R_feed - R_calc
         R_calc      = k(T) * n_CaCO3,   k(T) = A_pre * exp(-Ea/(R*T))
         m*cp dT/dt  = UA*(T_flame - T) - dH * R_calc

ODE state vector  x = [n_CO2_abs, n_CaCO3, T_calc]:
   n_CO2_abs : cumulative CO2 absorbed in contactor          [mol]
   n_CaCO3   : CaCO3 inventory awaiting calcination          [mol]
   T_calc    : calciner solids temperature                   [K]

Conservation / sanity:
   * Carbon: every mol entering the contactor either stays in the loop
     (n_CaCO3) or is released as product CO2; absorbed = released + in-process.
   * CO2 captured from genuinely dilute air (420 ppm) -> low single-pass eta and
     a huge air throughput, exactly as in real DAC.
   * Regeneration energy is large and mostly THERMAL (>~5 GJ/tCO2), dominated by
     CaCO3 calcination enthalpy plus solids sensible heat -- matching Keith 2018.

References:
   Keith, D.W. et al. (2018). A Process for Capturing CO2 from the Atmosphere.
     Joule 2(8), 1573-1594.
   Sabatino, F. et al. (2021). A comparative energy and costs assessment ...
     of DAC technologies. Joule 5(8), 2047-2076.
   Holmes, G. & Keith, D.W. (2012). An air-liquid contactor for large-scale
     capture of CO2 from air. Phil. Trans. R. Soc. A 370, 4380-4403.
"""

import numpy as np
from scipy.integrate import solve_ivp


class DAC_F2a:
    """Lumped contactor + calciner DAC model (liquid KOH/Ca caustic loop)."""

    R = 8.314          # J/(mol.K) universal gas constant

    def __init__(self, params: dict):
        u = params["unit"]
        # --- air / contactor ---
        self.CO2_ppm = u["CO2_ppm_air"]["value"]
        self.T_air = u["T_air_K"]["value"]
        self.P_atm = u["P_atm"]["value"]
        self.M_CO2 = u["CO2_molar_mass"]["value"]          # kg/mol
        self.A_contactor = u["area_contactor_m2"]["value"]
        self.u_air = u["air_velocity_m_s"]["value"]
        self.KgA = u["KgA_contactor"]["value"]             # 1/s
        self.L = u["contactor_depth_m"]["value"]
        self.V_sol = u["V_solution_m3"]["value"]
        self.OH_conc = u["OH_conc_mol_m3"]["value"]
        # --- calciner ---
        self.m_calc = u["m_calciner_kg"]["value"]
        self.cp_solids = u["cp_solids"]["value"]
        self.T_calc_set = u["T_calciner_setpoint_K"]["value"]
        self.dH_calc = u["dH_calcination_J_mol"]["value"]
        self.A_pre = u["A_calc_pre"]["value"]
        self.Ea = u["Ea_calc_J_mol"]["value"]
        self.UA = u["UA_calciner_W_K"]["value"]
        self.T_flame = u["T_flame_K"]["value"]
        self.loop_eff = u["loop_efficiency"]["value"]
        self.heat_loss_frac = u["heat_loss_frac"]["value"]
        self.T_feed_solids = u["T_feed_solids_K"]["value"]

    # ------------------------------------------------------------------
    # Contactor: dilute-air CO2 absorption (mass-transfer limited)
    # ------------------------------------------------------------------
    def single_pass_capture(self, u_air=None, KgA=None):
        """Single-pass CO2 capture fraction = 1 - exp(-Kg*a * L / u_air).

        Number-of-transfer-units form for a gas-film controlled absorber with
        OH- in large excess (Holmes & Keith 2012).
        """
        u_air = self.u_air if u_air is None else u_air
        KgA = self.KgA if KgA is None else KgA
        ntu = KgA * self.L / max(u_air, 1e-9)
        return 1.0 - np.exp(-ntu)

    def inlet_co2_conc(self, ppm=None, T_air=None):
        """Inlet CO2 molar concentration in air [mol/m3] via ideal gas."""
        ppm = self.CO2_ppm if ppm is None else ppm
        T_air = self.T_air if T_air is None else T_air
        y = ppm * 1e-6
        return y * self.P_atm / (self.R * T_air)   # mol/m3

    def absorption_rate(self, u_air=None, ppm=None, T_air=None, KgA=None):
        """CO2 molar absorption rate from dilute air [mol/s].

        R_abs = eta_single * c_in_CO2 * (u_air * A_contactor)
              = (capture fraction) * (CO2 conc) * (volumetric air flow)
        """
        u_air = self.u_air if u_air is None else u_air
        eta = self.single_pass_capture(u_air, KgA)
        c_in = self.inlet_co2_conc(ppm, T_air)
        Q_air = u_air * self.A_contactor            # m3/s volumetric air flow
        return eta * c_in * Q_air                   # mol/s

    # ------------------------------------------------------------------
    # Calciner kinetics + thermal
    # ------------------------------------------------------------------
    def calcination_rate(self, n_CaCO3, T_calc):
        """Lumped Arrhenius calcination rate [mol/s] of CaCO3 -> CaO + CO2."""
        n_CaCO3 = max(n_CaCO3, 0.0)
        k = self.A_pre * np.exp(-self.Ea / (self.R * T_calc))
        return k * n_CaCO3

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, x, ppm_fn, u_air_fn):
        n_abs, n_CaCO3, T_calc = x
        n_CaCO3 = max(n_CaCO3, 0.0)

        ppm = ppm_fn(t)
        u_air = u_air_fn(t)

        # 1) contactor absorption from dilute air
        R_abs = self.absorption_rate(u_air=u_air, ppm=ppm)

        # 2) carbonate feed to calciner (after caustic-loop losses)
        R_feed = self.loop_eff * R_abs

        # 3) calcination releasing product CO2
        R_calc = self.calcination_rate(n_CaCO3, T_calc)

        # thermal ODE for calciner solids -- thermostatic burner holding setpoint.
        Q_burner = self._burner_duty(T_calc, R_calc, R_feed)
        Q_rxn = self.dH_calc * R_calc                          # endothermic sink [W]
        Q_sens = (R_feed * self.M_solids_per_mol_CO2 *         # heat cold feed solids
                  self.cp_solids * (T_calc - self.T_feed_solids))
        Q_loss = self.heat_loss_frac * Q_burner               # wall/flue losses [W]
        dT = (Q_burner - Q_rxn - Q_sens - Q_loss) / (self.m_calc * self.cp_solids)

        dn_abs = R_abs
        dn_CaCO3 = R_feed - R_calc
        return [dn_abs, dn_CaCO3, dT]

    # Molar mass of CaCO3 solids carried per mol of CO2 (1:1 stoichiometry).
    M_solids_per_mol_CO2 = 0.10009   # kg/mol CaCO3

    def _burner_duty(self, T_calc, R_calc, R_feed):
        """Thermostatic burner duty [W].

        Proportional controller toward the calciner setpoint, supplying reaction
        enthalpy + feed sensible heat + losses, capped at the burner's UA-limited
        maximum UA*(T_flame - T_calc).  This keeps the calciner near ~900 C instead
        of drifting to the flame temperature.
        """
        Q_demand = (self.dH_calc * R_calc
                    + R_feed * self.M_solids_per_mol_CO2 * self.cp_solids
                    * (self.T_calc_set - self.T_feed_solids))
        Q_demand /= max(1.0 - self.heat_loss_frac, 1e-3)
        # proportional trim to hold setpoint
        Q_trim = self.UA * (self.T_calc_set - T_calc)
        Q = Q_demand + Q_trim
        Q_max = self.UA * (self.T_flame - T_calc)
        return float(np.clip(Q, 0.0, max(Q_max, 0.0)))

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, ppm=None, u_air=None, T_calc0=None,
                 dt=60.0, duration_s=3600.0, n_CaCO3_0=0.0):
        """Integrate the coupled contactor+calciner ODE.

        Parameters
        ----------
        ppm    : float or callable(t)->ppm   ambient CO2 (default param value)
        u_air  : float or callable(t)->m/s   air face velocity
        T_calc0: initial calciner temperature [K] (default setpoint)
        dt     : output sampling interval [s]
        duration_s : total simulated time [s]
        n_CaCO3_0  : initial CaCO3 inventory [mol]

        Returns dict of time-series arrays + scalar energy metrics.
        """
        ppm0 = self.CO2_ppm if ppm is None else (ppm if callable(ppm) else float(ppm))
        u0 = self.u_air if u_air is None else (u_air if callable(u_air) else float(u_air))
        ppm_fn = ppm if callable(ppm) else (lambda t: ppm0)
        u_air_fn = u_air if callable(u_air) else (lambda t: u0)
        T0 = self.T_calc_set if T_calc0 is None else T_calc0

        x0 = [0.0, n_CaCO3_0, T0]
        t_eval = np.arange(0.0, duration_s + 1e-9, dt)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), x0,
            t_eval=t_eval, args=(ppm_fn, u_air_fn),
            method="LSODA", rtol=1e-7, atol=1e-3, max_step=dt,
        )

        t = sol.t
        n_abs = sol.y[0]
        n_CaCO3 = np.maximum(sol.y[1], 0.0)
        T_calc = sol.y[2]

        # instantaneous rates (re-evaluate for reporting)
        R_abs = np.array([self.absorption_rate(u_air=u_air_fn(tt), ppm=ppm_fn(tt))
                          for tt in t])
        R_calc = np.array([self.calcination_rate(n, T)
                           for n, T in zip(n_CaCO3, T_calc)])

        # cumulative product CO2 released (integrate R_calc)
        n_released = np.concatenate([[0.0], np.cumsum(0.5 * (R_calc[1:] + R_calc[:-1])
                                                      * np.diff(t))]) if len(t) > 1 else np.zeros_like(t)

        # captured (from air) and product (released) mass [kg, then tonnes]
        co2_captured_kg = n_abs * self.M_CO2
        co2_product_kg = n_released * self.M_CO2

        # thermal energy delivered by the thermostatic burner [W] -- this is the
        # plant thermal demand that sets the specific thermal energy consumption.
        R_feed = self.loop_eff * R_abs
        Q_in = np.array([self._burner_duty(T, rc, rf)
                         for T, rc, rf in zip(T_calc, R_calc, R_feed)])
        E_thermal_J = (np.concatenate([[0.0], np.cumsum(0.5 * (Q_in[1:] + Q_in[:-1])
                       * np.diff(t))]) if len(t) > 1 else np.zeros_like(t))

        # specific thermal energy [GJ / tCO2 of product released]
        prod_t = np.maximum(co2_product_kg / 1000.0, 1e-12)
        sec_thermal = E_thermal_J / 1e9 / prod_t              # GJ/tCO2

        return {
            "t": t,
            "n_CO2_absorbed_mol": n_abs,
            "n_CaCO3_mol": n_CaCO3,
            "T_calciner_K": T_calc,
            "R_absorption_mol_s": R_abs,
            "R_calcination_mol_s": R_calc,
            "co2_captured_kg": co2_captured_kg,
            "co2_product_kg": co2_product_kg,
            "Q_thermal_W": Q_in,
            "E_thermal_J": E_thermal_J,
            "sec_thermal_GJ_tCO2": sec_thermal,
            "single_pass_capture": self.single_pass_capture(u0),
        }
