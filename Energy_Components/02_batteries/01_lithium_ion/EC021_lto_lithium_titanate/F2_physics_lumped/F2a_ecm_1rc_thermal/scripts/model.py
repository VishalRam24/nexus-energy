"""
EC021 -- LTO Battery (Lithium Titanate Oxide) -- F2a Thevenin 1-RC ECM + Thermal

Physics-lumped equivalent-circuit model with coupled state ODEs solved by
scipy.integrate.solve_ivp. This is the first-principles (ODE) upgrade of the
F1b algebraic SOC-thermal model: instead of an instantaneous V = OCV(SOC) - I*R,
the polarization is now a *dynamic* state with a finite time constant, and SOC
and cell temperature are integrated simultaneously.

State vector  y = [SOC, V_RC, T]:

    Coulomb counting (capacity is temperature-corrected):
        dSOC/dt = -I / (3600 * Q_eff(T))                     [1/s]
        Q_eff(T) = Q_ref * (1 + alpha_c * (T - T_ref))       [Ah]

    Single RC (Thevenin) polarization branch:
        dV_RC/dt = -V_RC / (R1*C1) + I / C1                  [V/s]
        tau1 = R1 * C1  (~10 s)  -- charge-transfer/diffusion relaxation

    Terminal voltage (sign convention: I>0 discharge):
        V_t = OCV(SOC) - I*R0(T) - V_RC

    Lumped thermal balance (Bernardi 1985 heat source):
        m*cp * dT/dt = Q_gen - Q_cool
        Q_gen  = I*R0*I + V_RC^2/R1            (irreversible, Joule)
               + I * T * dOCV/dT               (reversible, entropic)
        Q_cool = hA * (T - T_amb)              (Newton cooling)

    Resistances follow an Arrhenius temperature dependence:
        R(T) = R_ref * exp( E_a/R_gas * (1/T - 1/T_ref) )

LTO-specific physics:
    - Very flat OCV curve with a ~1.5 V (cell-format) / ~2.4 V (full-cell) plateau
      from the two-phase Li4Ti5O12 <-> Li7Ti5O12 spinel reaction (zero-strain).
    - Very high rate capability and low internal resistance (R0 ~ 12 mOhm),
      tolerating >10C currents -- captured by small R0/R1 and low activation energy.
    - Excellent low-temperature operation (low E_a) and long cycle life.

References:
    Takami, N. et al. (2011). J. Power Sources 196, 6989-6995.   (LTO cell, OCV plateau)
    Hu, X. et al. (2012). J. Power Sources 198, 359-367.         (Thevenin RC ECM comparison)
    He, H. et al. (2013). J. Power Sources 239, 269-276.         (Arrhenius resistance, LTO)
    Bernardi, D. et al. (1985). J. Electrochem. Soc. 132, 5-12.  (battery heat-generation balance)
    Kobayashi, Y. et al. (2013). J. Power Sources 244, 727-734.  (entropic coefficient)
"""

import numpy as np
from scipy.integrate import solve_ivp


class LTO_ECM_F2a:
    """LTO cell -- Thevenin 1-RC equivalent circuit with lumped thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_ref = u["capacity_ref"]["value"]          # Ah at T_ref
        self.v_max = u["voltage_max"]["value"]            # V
        self.v_min = u["voltage_min"]["value"]            # V
        self.R0_ref = u["R0_ref"]["value"]                # Ohm
        self.R1_ref = u["R1_ref"]["value"]                # Ohm
        self.C1 = u["C1"]["value"]                        # F
        self.E_a = u["E_a_R"]["value"]                    # J/mol
        self.T_ref = u["T_ref"]["value"]                  # K
        self.R_gas = u["R_gas"]["value"]                  # J/(mol.K)
        self.alpha_c = u["alpha_c"]["value"]              # 1/K
        self.dOCV_dT = u["dOCV_dT"]["value"]              # V/K
        self.m_cell = u["m_cell"]["value"]                # kg
        self.cp_cell = u["cp_cell"]["value"]              # J/(kg.K)
        self.hA = u["hA_cool"]["value"]                   # W/K
        self.T_amb = u["T_amb"]["value"]                  # K

        self.ocv_coeff = np.array(
            [u["ocv_coefficients"][f"a{i}"] for i in range(6)]
        )

    # ------------------------------------------------------------------
    # Open-circuit voltage OCV(SOC) -- flat LTO plateau (5th-order poly)
    # ------------------------------------------------------------------
    def ocv(self, soc):
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    # ------------------------------------------------------------------
    # Arrhenius temperature-dependent resistances
    # ------------------------------------------------------------------
    def R0(self, T):
        """Series ohmic resistance [Ohm]."""
        T = np.asarray(T, dtype=float)
        return self.R0_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def R1(self, T):
        """RC-branch polarization resistance [Ohm]."""
        T = np.asarray(T, dtype=float)
        return self.R1_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def tau1(self, T):
        """RC time constant [s]."""
        return self.R1(T) * self.C1

    # ------------------------------------------------------------------
    # Temperature-corrected capacity
    # ------------------------------------------------------------------
    def effective_capacity(self, T):
        """Effective capacity [Ah]."""
        T = np.asarray(T, dtype=float)
        return self.Q_ref * (1.0 + self.alpha_c * (T - self.T_ref))

    # ------------------------------------------------------------------
    # Terminal voltage from state
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, v_rc, current, T):
        """V_t = OCV(SOC) - I*R0(T) - V_RC  [V]."""
        return self.ocv(soc) - current * self.R0(T) - v_rc

    # ------------------------------------------------------------------
    # Heat generation (Bernardi 1985)
    # ------------------------------------------------------------------
    def heat_generation(self, v_rc, current, T):
        """
        Q_gen = I^2*R0 + V_RC^2/R1  (irreversible Joule, >= 0)
              + I*T*dOCV/dT          (reversible entropic, signed)
        """
        R0 = self.R0(T)
        R1 = self.R1(T)
        q_irr = current**2 * R0 + v_rc**2 / R1
        q_rev = current * T * self.dOCV_dT
        return q_irr + q_rev

    # ------------------------------------------------------------------
    # Coupled state derivatives  y = [SOC, V_RC, T]
    # ------------------------------------------------------------------
    def derivatives(self, soc, v_rc, T, current):
        Q_eff = self.effective_capacity(T)
        dsoc = -current / (3600.0 * Q_eff)
        dvrc = -v_rc / self.tau1(T) + current / self.C1
        Q_gen = self.heat_generation(v_rc, current, T)
        Q_cool = self.hA * (T - self.T_amb)
        dT = (Q_gen - Q_cool) / (self.m_cell * self.cp_cell)
        return dsoc, dvrc, dT

    # ------------------------------------------------------------------
    # Time-domain simulation via solve_ivp
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.9, T0=298.15, v_rc0=0.0,
                 dt=1.0, duration_s=600.0):
        """
        Simulate LTO cell dynamics with coupled SOC + RC + thermal ODEs.

        Parameters
        ----------
        current_A : float or callable(t)
            Current [A], positive = discharge, negative = charge.
        soc0 : float
            Initial state of charge (0-1).
        T0 : float
            Initial cell temperature [K].
        v_rc0 : float
            Initial RC-branch polarization voltage [V].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation duration [s].

        Returns
        -------
        dict with arrays: t, soc, voltage, v_rc, current, temperature,
             power, heat_gen.
        """
        _I = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            soc, v_rc, T = y
            I = _I(t)
            dsoc, dvrc, dT = self.derivatives(soc, v_rc, T, I)
            return [dsoc, dvrc, dT]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [soc0, v_rc0, T0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        soc_out = np.clip(sol.y[0], 0.0, 1.0)
        vrc_out = sol.y[1]
        T_out = sol.y[2]
        N = len(t_out)

        I_out = np.array([_I(t) for t in t_out])
        voltage = self.terminal_voltage(soc_out, vrc_out, I_out, T_out)
        voltage = np.clip(voltage, self.v_min, self.v_max)
        power = voltage * I_out
        heat = np.array(
            [self.heat_generation(vrc_out[i], I_out[i], T_out[i]) for i in range(N)]
        )

        return {
            "t": t_out,
            "soc": soc_out,
            "voltage": voltage,
            "v_rc": vrc_out,
            "current": I_out,
            "temperature": T_out,
            "power": power,
            "heat_gen": heat,
        }
