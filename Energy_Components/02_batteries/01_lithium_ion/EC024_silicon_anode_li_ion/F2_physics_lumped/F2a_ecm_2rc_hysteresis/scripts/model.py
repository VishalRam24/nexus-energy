"""
EC024 -- Silicon-Anode Li-ion Battery (Si/NMC) -- F2a Thevenin ECM (1/2-RC)

Physics-lumped equivalent-circuit model (ECM) with:
  * Coulomb-counted state of charge (SOC) with coulombic efficiency,
  * Si-anode-specific OCV(SOC) with a LARGE charge/discharge hysteresis envelope
    (the lithiation and delithiation curves split by ~50-120 mV -- a hallmark of
    silicon electrodes arising from the path-dependent amorphous Li-Si phase
    transformation and the mechanical work of ~280-300% volume swelling),
  * Arrhenius temperature dependence of all resistances R(T),
  * a 1-RC or 2-RC Thevenin network for the dynamic (transient) overpotential,
  * a one-state hysteresis model (Plett) blended between the OCV branches, and
  * a lumped thermal ODE (Bernardi energy balance) integrated with
    scipy.integrate.solve_ivp.

State vector y = [SOC, V_rc1, V_rc2, h, T]
    SOC    -- state of charge          [-]
    V_rc1  -- fast RC overpotential    [V]
    V_rc2  -- slow RC overpotential    [V]  (0 if n_rc == 1)
    h      -- hysteresis state in [-1, +1]; -1 -> discharge branch, +1 -> charge
    T      -- cell temperature         [K]

Terminal voltage (sign convention: I > 0 discharge, I < 0 charge):
    V_t = OCV_hyst(SOC, h) - I*R0(T) - V_rc1 - V_rc2

Coulomb counting:
    dSOC/dt = -eta(I) * I / (Q * 3600)         [Q in Ah]
    eta(I)  = eta_c on charge (I<0), 1 on discharge -> Coulomb conservation
              of *delivered* charge with efficiency loss on charge.

RC dynamics (Thevenin):
    dV_rc_k/dt = -V_rc_k / (R_k(T)*C_k) + I / C_k

Hysteresis (Plett one-state):
    dh/dt = -gamma * |eta*I| / (Q*3600) * (sign(I) + h)
    OCV_hyst(SOC,h) = OCV_dis(SOC) + (h+1)/2 * (OCV_chg(SOC) - OCV_dis(SOC))

Thermal ODE (lumped Bernardi 1985 energy balance):
    m*cp dT/dt = Q_gen - hA*(T - T_amb)
    Q_gen = I*(OCV_hyst - V_t)          irreversible (I^2*R-equivalent)
          + I*T*dOCV/dT                 reversible (entropic)

Si swelling: electrode thickness strain ~ swelling_coeff * SOC is reported as a
diagnostic and is the mechanical origin of the OCV hysteresis amplitude M_hyst.

References:
    Plett, G. (2015). Battery Management Systems, Vol. 2: Equivalent-Circuit
        Methods. Artech House. (ESC model, one-state hysteresis, R0/R1/C1)
    Hu, X., Li, S., Peng, H. (2012). A comparative study of equivalent circuit
        models for Li-ion batteries. J. Power Sources 198, 359-367. (n-RC)
    Obrovac, M.N., Christensen, L. (2004). Structural changes in silicon anodes
        during lithium insertion/extraction. Electrochem. Solid-State Lett.
        7(5), A93-A96. (lithiation/delithiation OCV hysteresis)
    McDowell, M.T. et al. (2013). 25th anniversary article: Understanding the
        lithiation of silicon and other alloying anodes. Adv. Mater. 25, 4966.
        (volume swelling ~280-300%, high capacity 3579 mAh/g)
    Bernardi, D., Pawlikowski, E., Newman, J. (1985). A general energy balance
        for battery systems. J. Electrochem. Soc. 132(1), 5-12. (thermal ODE)
    Geng, Z. et al. (2020). J. Electrochem. Soc. 167, 090504. (entropy dOCV/dT)
"""

import numpy as np
from scipy.integrate import solve_ivp


class SiAnodeECM_F2a:
    """Silicon-anode Li-ion Thevenin ECM with hysteresis + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q = u["capacity_ref"]["value"]          # Ah
        self.v_max = u["voltage_max"]["value"]
        self.v_min = u["voltage_min"]["value"]
        self.R0_ref = u["R0_ref"]["value"]
        self.R1_ref = u["R1_ref"]["value"]
        self.C1 = u["C1"]["value"]
        self.R2_ref = u["R2_ref"]["value"]
        self.C2 = u["C2"]["value"]
        self.M_hyst = u["M_hyst"]["value"]
        self.gamma = u["gamma_hyst"]["value"]
        self.n_rc = int(u["n_rc"]["value"])
        self.eta_c = u["eta_coulombic"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.E_a = u["E_a"]["value"]
        self.dOCV_dT = u["dOCV_dT"]["value"]
        self.m_cell = u["m_cell"]["value"]
        self.cp_cell = u["cp_cell"]["value"]
        self.hA = u["hA_cell"]["value"]
        self.T_amb = u["T_ambient"]["value"]
        self.swelling = u["swelling_coeff"]["value"]
        self.R_gas = u["R_gas"]["value"]

        self.a_chg = np.array([params["ocv_charge_coefficients"][f"a{i}"] for i in range(6)])
        self.a_dis = np.array([params["ocv_discharge_coefficients"][f"a{i}"] for i in range(6)])

    # ------------------------------------------------------------------
    # OCV branches (charge = lithiation envelope top, discharge = bottom)
    # ------------------------------------------------------------------
    def _poly(self, coeff, soc):
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc ** i for i in range(6)], axis=-1)
        return np.dot(powers, coeff)

    def ocv_charge(self, soc):
        """Lithiation (charge) OCV branch [V] -- upper envelope."""
        return self._poly(self.a_chg, soc)

    def ocv_discharge(self, soc):
        """Delithiation (discharge) OCV branch [V] -- lower envelope."""
        return self._poly(self.a_dis, soc)

    def ocv_hyst(self, soc, h):
        """OCV blended between branches by hysteresis state h in [-1, +1].

        h = -1 -> discharge branch, h = +1 -> charge branch.
        """
        lo = self.ocv_discharge(soc)
        hi = self.ocv_charge(soc)
        frac = (np.clip(h, -1.0, 1.0) + 1.0) / 2.0
        return lo + frac * (hi - lo)

    def hysteresis_gap(self, soc):
        """Charge-minus-discharge OCV gap [V] at given SOC (>=0)."""
        return self.ocv_charge(soc) - self.ocv_discharge(soc)

    # ------------------------------------------------------------------
    # Arrhenius resistances
    # ------------------------------------------------------------------
    def _arrhenius(self, R_ref, T):
        T = np.asarray(T, dtype=float)
        return R_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def R0(self, T):
        return self._arrhenius(self.R0_ref, T)

    def R1(self, T):
        return self._arrhenius(self.R1_ref, T)

    def R2(self, T):
        return self._arrhenius(self.R2_ref, T)

    # ------------------------------------------------------------------
    # Coulombic efficiency: <1 on charge, 1 on discharge (Coulomb balance)
    # ------------------------------------------------------------------
    def coulombic_eff(self, current):
        """eta in (0,1] applied to the charge that actually changes SOC."""
        return self.eta_c if current < 0 else 1.0

    # ------------------------------------------------------------------
    # Swelling diagnostic
    # ------------------------------------------------------------------
    def swelling_strain(self, soc):
        """Fractional electrode thickness change [-] at given SOC."""
        return self.swelling * np.clip(soc, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Terminal voltage from full state
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, v_rc1, v_rc2, h, T, current):
        R0 = self.R0(T)
        v = self.ocv_hyst(soc, h) - current * R0 - v_rc1 - v_rc2
        return float(np.clip(v, self.v_min, self.v_max))

    # ------------------------------------------------------------------
    # State derivative for solve_ivp
    # ------------------------------------------------------------------
    def _deriv(self, t, y, current_fn):
        soc, v_rc1, v_rc2, h, T = y
        I = current_fn(t)
        eta = self.coulombic_eff(I)

        # Coulomb counting (Q in Ah -> seconds)
        dsoc = -eta * I / (self.Q * 3600.0)

        # RC transients
        tau1 = self.R1(T) * self.C1
        dv1 = -v_rc1 / tau1 + I / self.C1
        if self.n_rc >= 2:
            tau2 = self.R2(T) * self.C2
            dv2 = -v_rc2 / tau2 + I / self.C2
        else:
            dv2 = -v_rc2 / 1.0  # force unused state to decay to 0

        # One-state hysteresis (Plett). Throughput drives h toward the branch
        # set by the sign of the current: discharge (I>0) -> h=-1 (discharge
        # branch), charge (I<0) -> h=+1 (charge branch).
        throughput = abs(eta * I) / (self.Q * 3600.0)
        dh = -self.gamma * throughput * (np.sign(I) + h) if I != 0 else 0.0

        # Thermal ODE (Bernardi energy balance)
        Vt = self.terminal_voltage(soc, v_rc1, v_rc2, h, T, I)
        ocv = self.ocv_hyst(soc, h)
        Q_irr = I * (ocv - Vt)              # irreversible (>=0 for I>0 discharge)
        Q_rev = I * T * self.dOCV_dT        # reversible entropic
        Q_gen = Q_irr + Q_rev
        dT = (Q_gen - self.hA * (T - self.T_amb)) / (self.m_cell * self.cp_cell)

        return [dsoc, dv1, dv2, dh, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0, T0, dt, duration_s, h0=None):
        """
        Simulate the Thevenin ECM dynamics.

        Parameters
        ----------
        current_A : float or callable(t)
            Load current [A], I>0 discharge, I<0 charge.
        soc0 : float        initial SOC [-]
        T0   : float        initial temperature [K]
        dt   : float        output time step [s]
        duration_s : float  total duration [s]
        h0   : float or None  initial hysteresis state; default = -sign(I0)

        Returns
        -------
        dict of time-series arrays: t, soc, voltage, ocv, current, power,
            v_rc1, v_rc2, hysteresis, temperature, swelling_strain,
            efficiency, components (dict).
        """
        I_fn = current_A if callable(current_A) else (lambda t: current_A)

        I0 = I_fn(0.0)
        if h0 is None:
            h0 = -float(np.sign(I0)) if I0 != 0 else 0.0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [float(np.clip(soc0, 0.0, 1.0)), 0.0, 0.0, float(h0), float(T0)]

        sol = solve_ivp(
            lambda t, y: self._deriv(t, y, I_fn),
            (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=max(dt, 1.0),
        )

        t = sol.t
        soc = np.clip(sol.y[0], 0.0, 1.0)
        v_rc1 = sol.y[1]
        v_rc2 = sol.y[2]
        h = np.clip(sol.y[3], -1.0, 1.0)
        T = sol.y[4]
        N = len(t)

        voltage = np.zeros(N)
        ocv = np.zeros(N)
        current = np.zeros(N)
        power = np.zeros(N)
        efficiency = np.zeros(N)
        strain = np.zeros(N)
        for i in range(N):
            Ii = I_fn(t[i])
            current[i] = Ii
            ocv[i] = self.ocv_hyst(soc[i], h[i])
            voltage[i] = self.terminal_voltage(soc[i], v_rc1[i], v_rc2[i], h[i], T[i], Ii)
            power[i] = voltage[i] * Ii
            strain[i] = self.swelling_strain(soc[i])
            # round-trip voltage efficiency proxy: |Vt|/OCV on discharge,
            # OCV/|Vt| on charge -> always in (0,1)
            if ocv[i] > 1e-6:
                if Ii >= 0:
                    efficiency[i] = voltage[i] / ocv[i]
                else:
                    efficiency[i] = ocv[i] / voltage[i] if voltage[i] > 1e-6 else 0.0
            efficiency[i] = float(np.clip(efficiency[i], 0.0, 1.0))

        return {
            "t": t,
            "soc": soc,
            "voltage": voltage,
            "ocv": ocv,
            "current": current,
            "power": power,
            "v_rc1": v_rc1,
            "v_rc2": v_rc2,
            "hysteresis": h,
            "temperature": T,
            "swelling_strain": strain,
            "efficiency": efficiency,
            "components": {
                "R0_T": self.R0(T),
                "R1_T": self.R1(T),
                "ocv_charge_branch": self.ocv_charge(soc),
                "ocv_discharge_branch": self.ocv_discharge(soc),
            },
        }
