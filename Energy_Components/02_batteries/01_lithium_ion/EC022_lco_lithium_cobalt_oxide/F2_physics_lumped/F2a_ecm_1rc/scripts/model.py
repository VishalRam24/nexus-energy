"""
EC022 -- LCO Battery (Lithium Cobalt Oxide) -- F2a Thevenin 1-RC ECM

Physics-lumped equivalent-circuit model coupling a Thevenin 1-RC electrical
network, Coulomb-counted state of charge, and a lumped Bernardi thermal ODE,
integrated together with scipy.integrate.solve_ivp.

Sign convention:  I > 0 = discharge,  I < 0 = charge.

State vector y = [SOC, V_RC, T]:

  (1) Coulomb counting (charge conservation):
        dSOC/dt = -eta_c * I / (3600 * Q)            [Plett (2015), BMS Vol.1]
      where eta_c = coulombic efficiency on charge (1 on discharge).

  (2) Thevenin 1-RC polarization branch:
        dV_RC/dt = I/C1 - V_RC/(R1*C1)               [Hu, Li & Peng (2012)]
        V_term   = OCV(SOC) - I*R0 - V_RC

  (3) OCV(SOC): 5th-order polynomial fit to a representative LiCoO2/graphite
      discharge curve -- the characteristic sloping 3.0-4.2 V profile of the
      ordered layered LiCoO2 cathode [Reimers & Dahn (1992)].

  (4) Arrhenius temperature scaling of the resistances:
        R0(T) = R0_ref * exp( E_a_R0/Rg * (1/T - 1/T_ref) )
        R1(T) = R1_ref * exp( E_a_R1/Rg * (1/T - 1/T_ref) )

  (5) Lumped Bernardi thermal ODE (Joule + entropic heat vs Newton cooling):
        m*cp * dT/dt = Q_gen - Q_cool
        Q_gen  = I^2*R0 + V_RC^2/R1            (irreversible Joule heat)
                 + I * T * dOCV/dT             (reversible entropic heat)
        Q_cool = hA * (T - T_amb)
      [Bernardi, Pawlikowski & Newman (1985); Forgez et al. (2010)]

References:
    Reimers, J.N. & Dahn, J.R. (1992). J. Electrochem. Soc. 139(8), 2091-2097.
    Bernardi, D., Pawlikowski, E. & Newman, J. (1985). J. Electrochem. Soc.
        132(1), 5-12.   (general energy balance for battery systems)
    Hu, X., Li, S. & Peng, H. (2012). J. Power Sources 198, 359-367.
        (comparison of equivalent-circuit models for Li-ion; 1-RC / 2-RC)
    Forgez, C. et al. (2010). J. Power Sources 195, 2961-2968.
        (thermal lumped model of a cylindrical LiFePO4/LCO-type cell)
    Thomas, K.E. & Newman, J. (2003). J. Electrochem. Soc. 150(2), A176-A192.
        (entropic / reversible heat coefficients)
    Plett, G. (2015). Battery Management Systems, Vol. 1, Artech House.
        (Coulomb counting / SOC bookkeeping)
"""

import numpy as np
from scipy.integrate import solve_ivp


class LCO_ECM_1RC:
    """LCO cell -- Thevenin 1-RC equivalent circuit + lumped thermal ODE."""

    R_GAS = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q = u["capacity_ref"]["value"]            # Ah
        self.v_max = u["voltage_max"]["value"]
        self.v_min = u["voltage_min"]["value"]
        self.R0_ref = u["R0_ref"]["value"]             # Ohm
        self.R1_ref = u["R1_ref"]["value"]             # Ohm
        self.C1_ref = u["C1_ref"]["value"]             # F
        self.T_ref = u["T_ref"]["value"]               # K
        self.E_a_R0 = u["E_a_R0"]["value"]             # J/mol
        self.E_a_R1 = u["E_a_R1"]["value"]             # J/mol
        self.dOCV_dT = u["dOCV_dT"]["value"]           # V/K
        self.eta_c = u["coulombic_eff"]["value"]       # -
        self.m_cell = u["m_cell"]["value"]             # kg
        self.cp_cell = u["cp_cell"]["value"]           # J/(kg.K)
        self.hA = u["hA_cell"]["value"]                # W/K
        self.T_amb = u["T_amb"]["value"]               # K

        self.ocv_coeff = np.array(
            [params["ocv_coefficients"][f"a{i}"] for i in range(6)]
        )

    # ------------------------------------------------------------------
    # Open-circuit voltage (sloping LiCoO2 profile)
    # ------------------------------------------------------------------
    def ocv(self, soc):
        """OCV(SOC) [V] -- monotone-increasing 5th-order polynomial."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def docv_dsoc(self, soc):
        """Analytic dOCV/dSOC [V] (>0 everywhere -> monotone OCV)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack(
            [i * soc ** (i - 1) if i >= 1 else np.zeros_like(soc)
             for i in range(6)], axis=-1
        )
        return np.dot(powers, self.ocv_coeff)

    # ------------------------------------------------------------------
    # Arrhenius-scaled resistances
    # ------------------------------------------------------------------
    def R0(self, T):
        return self.R0_ref * np.exp(
            self.E_a_R0 / self.R_GAS * (1.0 / T - 1.0 / self.T_ref)
        )

    def R1(self, T):
        return self.R1_ref * np.exp(
            self.E_a_R1 / self.R_GAS * (1.0 / T - 1.0 / self.T_ref)
        )

    def C1(self, T):
        # Keep tau roughly constant: C1 scales inversely with R1.
        return self.C1_ref * self.R1_ref / max(self.R1(T), 1e-12)

    # ------------------------------------------------------------------
    # Terminal voltage
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, v_rc, current, T):
        """V_term = OCV(SOC) - I*R0(T) - V_RC  [V]."""
        v = self.ocv(soc) - current * self.R0(T) - v_rc
        return v

    # ------------------------------------------------------------------
    # Heat generation -- Bernardi energy balance
    # ------------------------------------------------------------------
    def heat_generation(self, v_rc, current, T):
        """Q_gen [W] = Joule (R0 + R1) + reversible entropic heat."""
        R1 = self.R1(T)
        q_joule = current**2 * self.R0(T) + v_rc**2 / max(R1, 1e-12)
        q_rev = current * T * self.dOCV_dT
        return q_joule + q_rev

    # ------------------------------------------------------------------
    # ODE right-hand side  y = [SOC, V_RC, T]
    # ------------------------------------------------------------------
    def _rhs(self, t, y, I_of_t):
        soc, v_rc, T = y
        I = I_of_t(t)

        # Coulombic efficiency applies only on charge (I < 0).
        eta = self.eta_c if I < 0 else 1.0
        dsoc = -eta * I / (3600.0 * self.Q)

        R1 = self.R1(T)
        C1 = self.C1(T)
        dv_rc = I / C1 - v_rc / (R1 * C1)

        Q_gen = self.heat_generation(v_rc, I, T)
        Q_cool = self.hA * (T - self.T_amb)
        dT = (Q_gen - Q_cool) / (self.m_cell * self.cp_cell)

        return [dsoc, dv_rc, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.9, T0=None, v_rc0=0.0,
                 dt=1.0, duration_s=600.0):
        """
        Integrate the coupled ECM + thermal ODE.

        Parameters
        ----------
        current_A : float or callable(t)   I>0 discharge, I<0 charge [A]
        soc0      : float                   initial SOC (0-1)
        T0        : float                   initial cell temperature [K]
        v_rc0     : float                   initial RC-branch voltage [V]
        dt        : float                   output time step [s]
        duration_s: float                   total duration [s]

        Returns
        -------
        dict of time-series arrays: t, soc, voltage, current, power,
             v_rc, temperature, heat_gen, R0, R1, efficiency.
        """
        if T0 is None:
            T0 = self.T_amb
        I_of_t = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        # Stop integration if SOC leaves [0,1].
        def soc_low(t, y, *a):
            return y[0]
        def soc_high(t, y, *a):
            return y[0] - 1.0
        soc_low.terminal = True
        soc_low.direction = -1
        soc_high.terminal = True
        soc_high.direction = 1

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [soc0, v_rc0, T0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt, args=(I_of_t,), events=(soc_low, soc_high),
        )

        t_out = sol.t
        soc_out = sol.y[0]
        vrc_out = sol.y[1]
        T_out = sol.y[2]
        N = len(t_out)

        voltage = np.zeros(N)
        current = np.zeros(N)
        power = np.zeros(N)
        heat = np.zeros(N)
        R0_arr = np.zeros(N)
        R1_arr = np.zeros(N)
        eff = np.zeros(N)

        for i in range(N):
            I = I_of_t(t_out[i])
            current[i] = I
            voltage[i] = self.terminal_voltage(soc_out[i], vrc_out[i], I, T_out[i])
            power[i] = voltage[i] * I
            heat[i] = self.heat_generation(vrc_out[i], I, T_out[i])
            R0_arr[i] = self.R0(T_out[i])
            R1_arr[i] = self.R1(T_out[i])
            ocv_i = self.ocv(soc_out[i])
            # Round-trip-style voltaic efficiency: useful work fraction of OCV.
            if I > 0 and ocv_i > 0:        # discharge: V_term/OCV
                eff[i] = voltage[i] / ocv_i
            elif I < 0 and voltage[i] > 0:  # charge: OCV/V_term
                eff[i] = ocv_i / voltage[i]
            else:
                eff[i] = 1.0
            eff[i] = min(max(eff[i], 0.0), 1.0)

        return {
            "t": t_out,
            "soc": soc_out,
            "voltage": voltage,
            "current": current,
            "power": power,
            "v_rc": vrc_out,
            "temperature": T_out,
            "heat_gen": heat,
            "R0": R0_arr,
            "R1": R1_arr,
            "efficiency": eff,
        }
