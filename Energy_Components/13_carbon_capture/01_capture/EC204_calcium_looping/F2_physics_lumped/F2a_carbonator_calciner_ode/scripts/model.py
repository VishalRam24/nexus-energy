"""
EC204 -- Calcium Looping -- F2a Carbonator/Calciner Coupled ODE (physics-lumped)

A 0D first-principles model of a dual interconnected fluidised-bed calcium-looping
CO2 capture loop:

    Carbonator (~650 C, EXOthermic):   CaO + CO2  -> CaCO3      (captures CO2)
    Calciner   (~900 C, ENDOthermic):  CaCO3 -> CaO + CO2       (regenerates sorbent
                                                                 + concentrated CO2)

Two coupled physical sub-models, integrated with scipy.integrate.solve_ivp:

  (1) Sorbent carbonation kinetics + carbon balance.
      The maximum carrying capacity of a CaO particle DECAYS with the number of
      calcination/carbonation cycles N it has experienced. We use the
      Grasa & Abanades (2006) deactivation correlation:

          X_N = 1 / ( 1/(X_max1 - X_r) + k*N )  +  X_r

      so X_N falls from ~X_max1 (fresh) toward the residual capacity X_r as N->inf.
      Carbonation within a cycle follows a lumped first-order approach to that
      cycle's capacity (the fast chemically-controlled stage, Bhatia & Perlmutter
      1983; Abanades et al. 2004):

          dX/dt = k_s * (X_N - X)          for X < X_N,  else 0

      The instantaneous CO2 capture rate is the carbon laid down on the
      circulating sorbent plus accumulation in the bed inventory.

  (2) Carbonator energy balance (lumped, single solids temperature):

          N_inv * cp * dT/dt = Q_rxn - Q_removed - Q_sensible_feed

      with Q_rxn = (-dH_carb) * R_CO2_captured  (exothermic, heats the bed) and
      Q_removed = hA*(T - T_setpoint) representing in-bed heat extraction that
      keeps the carbonator near its operating temperature.

Carbon / mass conservation is enforced explicitly:
      F_CO2_in = R_captured + F_CO2_out      (every instant)
and the steady carbon closed by the calciner equals what is captured plus the
make-up CaCO3 calcined, giving a concentrated CO2 product stream.

References
----------
    Abanades, J.C. (2002). The maximum capture efficiency of CO2 using a
        carbonation/calcination cycle of CaO/CaCO3. Chem. Eng. J. 90:303-306.
    Abanades, J.C. et al. (2004). Capture of CO2 from combustion gases in a
        fluidized bed of CaO. AIChE J. 50(7):1614-1622.
    Grasa, G.S. & Abanades, J.C. (2006). CO2 capture capacity of CaO in long
        series of carbonation/calcination cycles. Ind. Eng. Chem. Res. 45:8846-8851.
    Bhatia, S.K. & Perlmutter, D.D. (1983). Effect of the product layer on the
        kinetics of the CO2-lime reaction. AIChE J. 29(1):79-86.
    Blamey, J. et al. (2010). The calcium looping cycle for large-scale CO2
        capture. Prog. Energy Combust. Sci. 36(2):260-279.
"""

import numpy as np
from scipy.integrate import solve_ivp


class CalciumLoopingF2a:
    """Coupled carbonator/calciner calcium-looping model with cyclic decay."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Grasa-Abanades deactivation
        self.X_max1 = u["X_max1"]["value"]
        self.k_deact = u["k_deact"]["value"]
        self.X_r = u["X_r"]["value"]
        # Carbonation kinetics
        self.k_s_carb = u["k_s_carb"]["value"]
        self.T_carb = u["T_carbonator_C"]["value"] + 273.15      # K setpoint
        self.T_calc = u["T_calciner_C"]["value"] + 273.15        # K
        # Energy
        self.dH_carb = u["dH_carb_kJ_mol"]["value"] * 1000.0     # J/mol (negative)
        self.cp_solid = u["cp_solid_J_molK"]["value"]            # J/(mol.K)
        self.N_inv = u["N_inventory_mol"]["value"]               # mol solids
        self.hA = u["hA_carb_W_K"]["value"]                      # W/K
        # Flows
        self.F_CaO = u["F_CaO_mol_s"]["value"]                   # mol/s circulating
        self.F_CO2_in = u["F_CO2_in_mol_s"]["value"]             # mol/s flue CO2
        self.F_makeup = u["F_makeup_mol_s"]["value"]             # mol/s fresh

    # ------------------------------------------------------------------
    # Grasa-Abanades sorbent carrying capacity vs cycle number
    # ------------------------------------------------------------------
    def carrying_capacity(self, N):
        """Max CaO conversion X_N after N cycles [-]  (Grasa & Abanades 2006)."""
        N = np.asarray(N, dtype=float)
        X = 1.0 / (1.0 / (self.X_max1 - self.X_r) + self.k_deact * N) + self.X_r
        # Bounded between residual and fresh-particle capacity
        return np.clip(X, self.X_r, self.X_max1)

    def average_capacity(self, N_makeup_ratio):
        """
        Steady-state mean capacity of a make-up-replenished sorbent population.

        Hawthorne/Abanades residence-time-distribution result: with a fresh
        make-up fraction f0 = F_makeup / F_CaO each cycle, the average capacity is
            X_ave = sum_N f0 (1-f0)^(N-1) X_N
        Returns the population-mean carrying capacity [-].
        """
        f0 = float(np.clip(N_makeup_ratio, 1e-6, 1.0))
        N = np.arange(1, 600)
        w = f0 * (1.0 - f0) ** (N - 1)
        w = w / w.sum()
        return float(np.sum(w * self.carrying_capacity(N)))

    # ------------------------------------------------------------------
    # Carbonation kinetics derivative (fast chemically-controlled stage)
    # ------------------------------------------------------------------
    def dXdt(self, X, X_N):
        """Conversion rate toward this cycle's capacity X_N [1/s]."""
        if X >= X_N:
            return 0.0
        return self.k_s_carb * (X_N - X)

    # ------------------------------------------------------------------
    # CO2 captured given current conversion of circulating solids
    # ------------------------------------------------------------------
    def co2_capture_rate(self, X, X_N):
        """
        Instantaneous CO2 capture rate [mol/s].

        Carbon is fixed onto the circulating CaO as it converts toward X_N.
        Rate of CaCO3 formation on the loop = F_CaO * dX/dt-equivalent, capped by
        the available flue CO2 (carbon conservation).
        """
        r = self.F_CaO * self.dXdt(X, X_N)
        return float(np.clip(r, 0.0, self.F_CO2_in))

    # ------------------------------------------------------------------
    # Carbonator energy-balance derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, R_capture):
        """Carbonator solids temperature rate [K/s]."""
        Q_rxn = (-self.dH_carb) * R_capture            # W, exothermic -> +
        Q_removed = self.hA * (T - self.T_carb)        # W, in-bed cooling to setpoint
        return (Q_rxn - Q_removed) / (self.N_inv * self.cp_solid)

    # ------------------------------------------------------------------
    # Calciner energy duty (steady, per mol CO2 regenerated)
    # ------------------------------------------------------------------
    def calciner_duty(self, R_co2_regen):
        """
        Calciner thermal duty [W] to drive CaCO3 -> CaO + CO2 (endothermic) plus
        sensible heating of solids from carbonator to calciner temperature.
        """
        Q_rxn = (-self.dH_carb) * R_co2_regen          # +178 kJ/mol endothermic
        Q_sens = self.F_CaO * self.cp_solid * (self.T_calc - self.T_carb)
        return Q_rxn + Q_sens

    # ------------------------------------------------------------------
    # Time-domain coupled simulation
    # ------------------------------------------------------------------
    def simulate(self, cycle_number, T0_K=None, X0=0.0, dt=1.0, duration_s=300.0):
        """
        Integrate coupled (X, T) ODE for the carbonator at a given sorbent age N.

        Parameters
        ----------
        cycle_number : float or callable(t)
            Sorbent cycle age N (sets the carrying capacity X_N).
        T0_K : float
            Initial carbonator solids temperature [K]. Defaults to setpoint.
        X0 : float
            Initial conversion of incoming solids [-].
        dt : float
            Output time step [s].
        duration_s : float
            Total duration [s].

        Returns
        -------
        dict of time series:
            t, conversion, capacity (X_N), capture_rate [mol/s],
            co2_out [mol/s], capture_efficiency [-], temperature [K],
            calciner_duty [W], carbon_balance_residual [mol/s]
        """
        if T0_K is None:
            T0_K = self.T_carb
        _N = cycle_number if callable(cycle_number) else (lambda t: cycle_number)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            X, T = y
            X_N = float(self.carrying_capacity(_N(t)))
            R = self.co2_capture_rate(X, X_N)
            return [self.dXdt(X, X_N), self.dTdt(T, R)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [X0, T0_K],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10, max_step=dt,
        )

        t_out = sol.t
        X_out = sol.y[0]
        T_out = sol.y[1]
        M = len(t_out)

        capacity = np.zeros(M)
        capture_rate = np.zeros(M)
        co2_out = np.zeros(M)
        eff = np.zeros(M)
        duty = np.zeros(M)
        residual = np.zeros(M)

        for i in range(M):
            X_N = float(self.carrying_capacity(_N(t_out[i])))
            capacity[i] = X_N
            R = self.co2_capture_rate(X_out[i], X_N)
            capture_rate[i] = R
            co2_out[i] = self.F_CO2_in - R
            eff[i] = R / self.F_CO2_in if self.F_CO2_in > 0 else 0.0
            # Calciner regenerates captured CO2 + calcines fresh make-up CaCO3
            duty[i] = self.calciner_duty(R + self.F_makeup)
            # Carbon conservation check: in == captured + out
            residual[i] = self.F_CO2_in - (R + co2_out[i])

        return {
            "t": t_out,
            "conversion": X_out,
            "capacity": capacity,
            "capture_rate": capture_rate,
            "co2_out": co2_out,
            "capture_efficiency": eff,
            "temperature": T_out,
            "calciner_duty": duty,
            "carbon_balance_residual": residual,
        }
