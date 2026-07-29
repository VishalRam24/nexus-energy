"""
EC030 -- Nickel-Cadmium Battery (NiCd) -- F2a Thevenin Equivalent-Circuit Model

Physics-lumped dynamic model. The cell is represented as a Thevenin
equivalent circuit with a series ohmic resistance R0 and one or two
parallel RC pairs (charge-transfer + diffusion polarisation), driven by a
Coulomb-counted state of charge and coupled to a lumped thermal ODE.

State vector  y = [SOC, V_rc1, (V_rc2), T]   integrated with scipy.solve_ivp.

Terminal voltage (sign convention: I > 0 = discharge):
    V_t = OCV(SOC) - I*R0(T) - V_rc1 - V_rc2

Coulomb counting (charge conservation):
    dSOC/dt = -eta_c * I / (3600 * C_eff(T))     on charge  (I < 0)
    dSOC/dt =        - I / (3600 * C_eff(T))     on discharge (I >= 0)
    with coulombic efficiency 0 < eta_c <= 1 applied only to charge.

RC branch dynamics (each pair):
    dV_rc/dt = -V_rc / (R_k * C_k) + I / C_k          (first-order relaxation)
    tau_k = R_k(T) * C_k

Arrhenius temperature dependence of all resistances:
    R(T) = R_ref * exp( E_a / R_gas * (1/T - 1/T_ref) )

Lumped thermal ODE (energy balance):
    m*cp * dT/dt = Q_gen - Q_loss
    Q_gen  = I^2*R0 + V_rc1^2/R1 + V_rc2^2/R2   (irreversible Joule, all branches)
             + I * T * dOCV/dT                  (reversible entropic heat)
    Q_loss = hA * (T - T_amb)

NiCd-specific physics:
    * Flat OCV plateau ~1.20-1.25 V (NiOOH/Ni(OH)2 couple).
    * Very low R0 (~10 mOhm/10Ah) from sintered plates + KOH electrolyte.
    * Strongly negative entropic coefficient dOCV/dT ~ -0.60 mV/K, so the
      reversible heat term is positive on discharge -> NiCd warms markedly.
    * Coulombic efficiency on charge ~85-92% (oxygen-recombination / gassing).

References:
    Berndt, D. (2003). Maintenance-Free Batteries. Wiley.
    Shepherd, C.M. (1965). J. Electrochem. Soc. 112, 657-664.
    Thomas, K.E. & Newman, J. (2003). J. Electrochem. Soc. 150, A176.
    Plett, G.L. (2015). Battery Management Systems, Vol. 2. Artech House.
    Hu, X., Li, S. & Peng, H. (2012). J. Power Sources 198, 359-367
        (comparative study of equivalent circuit models).
"""

import numpy as np
from scipy.integrate import solve_ivp


class NiCdTheveninF2a:
    """NiCd cell -- Thevenin ECM (1-RC or 2-RC) with thermal ODE."""

    def __init__(self, params: dict, n_rc: int = 2):
        if n_rc not in (1, 2):
            raise ValueError("n_rc must be 1 or 2")
        self.n_rc = n_rc
        u = params["unit"]

        self.capacity_ref = u["capacity_ref"]["value"]      # Ah
        self.v_max = u["voltage_max"]["value"]
        self.v_min = u["voltage_min"]["value"]

        self.R0_ref = u["R0_ref"]["value"]
        self.R1_ref = u["R1_ref"]["value"]
        self.C1 = u["C1_ref"]["value"]
        self.R2_ref = u["R2_ref"]["value"]
        self.C2 = u["C2_ref"]["value"]

        self.ocv_coeff = np.array([u[f"ocv_a{i}"]["value"] for i in range(6)])

        self.T_ref = u["T_ref"]["value"]
        self.E_a = u["E_a"]["value"]
        self.alpha_c = u["alpha_c"]["value"]
        self.dOCV_dT = u["dOCV_dT"]["value"]
        self.R_gas = u["R_gas"]["value"]
        self.eta_c = u["coulombic_eff"]["value"]

        self.m_cell = u["m_cell"]["value"]
        self.cp_cell = u["cp_cell"]["value"]
        self.hA_cell = u["hA_cell"]["value"]
        self.T_amb = u["T_amb"]["value"]

    # ------------------------------------------------------------------
    # Open-circuit voltage (flat NiCd plateau)
    # ------------------------------------------------------------------
    def ocv(self, soc):
        """Open-circuit voltage [V] vs SOC (0..1). 5th-order polynomial."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc ** i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    # ------------------------------------------------------------------
    # Arrhenius temperature scaling of resistances
    # ------------------------------------------------------------------
    def _arrhenius(self, R_ref, T):
        return R_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def R0(self, T):
        return self._arrhenius(self.R0_ref, T)

    def R1(self, T):
        return self._arrhenius(self.R1_ref, T)

    def R2(self, T):
        return self._arrhenius(self.R2_ref, T)

    def tau1(self, T):
        return self.R1(T) * self.C1

    def tau2(self, T):
        return self.R2(T) * self.C2

    def effective_capacity(self, T):
        """Temperature-corrected capacity [Ah]."""
        return self.capacity_ref * (1.0 + self.alpha_c * (T - self.T_ref))

    # ------------------------------------------------------------------
    # Terminal voltage from full state
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, current, T, V_rc1, V_rc2=0.0):
        """Terminal voltage [V]. I>0 discharge. Clipped to [v_min, v_max]."""
        v = self.ocv(soc) - current * self.R0(T) - V_rc1 - V_rc2
        return float(np.clip(v, self.v_min, self.v_max))

    # ------------------------------------------------------------------
    # SOC derivative -- Coulomb counting with coulombic efficiency
    # ------------------------------------------------------------------
    def dSOC_dt(self, current, T):
        """dSOC/dt [1/s]. Coulombic efficiency applied to charge only."""
        C_eff = self.effective_capacity(T)
        eta = self.eta_c if current < 0 else 1.0   # charge less than 100% effective
        return -eta * current / (C_eff * 3600.0)

    # ------------------------------------------------------------------
    # Heat generation -- irreversible (all branches) + reversible entropic
    # ------------------------------------------------------------------
    def heat_generation(self, current, T, V_rc1, V_rc2=0.0):
        """Total heat rate [W]."""
        q_ohmic = current ** 2 * self.R0(T)
        q_rc1 = V_rc1 ** 2 / max(self.R1(T), 1e-12)
        q_rc2 = (V_rc2 ** 2 / max(self.R2(T), 1e-12)) if self.n_rc == 2 else 0.0
        q_rev = current * T * self.dOCV_dT
        return q_ohmic + q_rc1 + q_rc2 + q_rev

    # ------------------------------------------------------------------
    # Full RHS for solve_ivp
    # ------------------------------------------------------------------
    def _rhs(self, t, y, current_fn):
        I = current_fn(t)
        if self.n_rc == 2:
            soc, vrc1, vrc2, T = y
        else:
            soc, vrc1, T = y
            vrc2 = 0.0

        dsoc = self.dSOC_dt(I, T)
        dvrc1 = -vrc1 / max(self.tau1(T), 1e-9) + I / self.C1
        if self.n_rc == 2:
            dvrc2 = -vrc2 / max(self.tau2(T), 1e-9) + I / self.C2

        Q_gen = self.heat_generation(I, T, vrc1, vrc2)
        Q_loss = self.hA_cell * (T - self.T_amb)
        dT = (Q_gen - Q_loss) / (self.m_cell * self.cp_cell)

        if self.n_rc == 2:
            return [dsoc, dvrc1, dvrc2, dT]
        return [dsoc, dvrc1, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=1.0, T0=None, dt=1.0, duration_s=600.0):
        """
        Integrate the coupled ECM + thermal ODE.

        Parameters
        ----------
        current_A : float or callable(t)
            Load current [A]. I>0 discharge, I<0 charge.
        soc0 : float
            Initial state of charge (0..1).
        T0 : float
            Initial cell temperature [K]. Defaults to T_amb.
        dt : float
            Output sampling step [s].
        duration_s : float
            Total simulated time [s].

        Returns
        -------
        dict of time-series arrays:
            t, soc, voltage, current, power, efficiency, temperature,
            ocv, V_rc1, V_rc2, heat_W
        """
        if T0 is None:
            T0 = self.T_amb
        current_fn = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [soc0, 0.0, 0.0, T0] if self.n_rc == 2 else [soc0, 0.0, T0]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(current_fn,),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t_out = sol.t
        N = len(t_out)
        soc = np.clip(sol.y[0], 0.0, 1.0)
        vrc1 = sol.y[1]
        if self.n_rc == 2:
            vrc2 = sol.y[2]
            T = sol.y[3]
        else:
            vrc2 = np.zeros(N)
            T = sol.y[2]

        voltage = np.zeros(N)
        ocv_arr = np.zeros(N)
        current = np.zeros(N)
        heat = np.zeros(N)
        for i in range(N):
            I = current_fn(t_out[i])
            current[i] = I
            ocv_arr[i] = self.ocv(soc[i])
            voltage[i] = self.terminal_voltage(soc[i], I, T[i], vrc1[i], vrc2[i])
            heat[i] = self.heat_generation(I, T[i], vrc1[i], vrc2[i])

        power = voltage * current   # W, +ve = delivered on discharge
        # Round-trip-style voltage efficiency: V_t / OCV on discharge,
        # OCV / V_t on charge -> always in (0,1) and < 1 for nonzero loss.
        eff = np.ones(N)
        for i in range(N):
            if current[i] > 1e-9:        # discharge: terminal below OCV
                eff[i] = voltage[i] / ocv_arr[i]
            elif current[i] < -1e-9:     # charge: terminal above OCV
                eff[i] = ocv_arr[i] / voltage[i]
            else:
                eff[i] = 1.0
        eff = np.clip(eff, 0.0, 1.0)

        return {
            "t": t_out,
            "soc": soc,
            "voltage": voltage,
            "current": current,
            "power": power,
            "efficiency": eff,
            "temperature": T,
            "ocv": ocv_arr,
            "V_rc1": vrc1,
            "V_rc2": vrc2,
            "heat_W": heat,
        }
