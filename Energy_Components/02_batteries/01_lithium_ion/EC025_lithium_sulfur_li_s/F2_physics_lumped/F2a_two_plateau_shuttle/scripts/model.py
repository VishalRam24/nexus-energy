"""
EC025 -- Lithium-Sulfur Battery (Li-S) -- F2a Two-Plateau + Polysulfide-Shuttle + Thermal

Physics-lumped 0D electrochemical / Thevenin hybrid model that captures the three
phenomena that distinguish Li-S from intercalation Li-ion:

  1. TWO-PLATEAU discharge OCV.
       upper plateau ~2.35 V : S8 + 2e- -> S8^2-, then -> Li2S4   (long-chain
                               polysulfides, ~25% of capacity)
       lower plateau ~2.10 V : Li2S4 -> Li2S2 / Li2S              (short-chain,
                               ~75% of capacity)
     Modelled as a smooth (logistic) blend of two sloped plateaus so the OCV(SOC)
     curve has the characteristic Li-S "staircase" shape.

  2. POLYSULFIDE SHUTTLE parasitic current (Mikhaylik & Akridge, 2004).
       Dissolved high-order polysulfides diffuse to the Li anode, are reduced, and
       diffuse back to the cathode -- an internal redox shuttle. Lumped here as a
       parasitic current
           I_sh = k_s(T) * f_H * S_high(SOC)
       that ALWAYS drains charge from the cathode. It (a) causes self-discharge at
       rest, (b) limits charge acceptance ("infinite" overcharge plateau), and
       (c) makes coulombic efficiency eta_C < 1 on every discharge.

  3. THERMAL ODE with a POSITIVE entropic coefficient dOCV/dT = +0.35 mV/K (unique
     to Li-S): discharge is reversibly endothermic, partially self-cooling the cell.

State vector integrated by scipy.integrate.solve_ivp:
       y = [SOC, V_rc, T]
       dSOC/dt = -(I_app + I_sh) / (3600 * Q)        (shuttle adds to drain)
       dV_rc/dt = I_app/C1 - V_rc/(R1 C1)            (1-RC Thevenin relaxation)
       m cp dT/dt = Q_ohm + Q_rev + Q_shuttle - hA (T - T_amb)

Terminal voltage:  V = OCV(SOC) - I_app*R0(T) - V_rc    (sign: I>0 discharge)

Conservation: Q_discharged + Q_shuttle_lost = Q_removed_from_cathode, and the
coulombic efficiency eta_C = Q_out / Q_removed = I_app / (I_app + I_sh) < 1.

References
----------
Mikhaylik, Y.V. & Akridge, J.R. (2004). "Polysulfide Shuttle Study in the Li/S
    Battery System." J. Electrochem. Soc. 151(11), A1969-A1976.
Wild, M. et al. (2015). Energy Environ. Sci. 8, 3477.
Kumaresan, K., Mikhaylik, Y. & White, R.E. (2008). J. Electrochem. Soc. 155(6), A576.
Cuisinier, M. et al. (2014). J. Phys. Chem. Lett. 5, 3227.
"""

import numpy as np
from scipy.integrate import solve_ivp


class LiS_F2a:
    """Lithium-sulfur 0D two-plateau + shuttle + thermal model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q = u["capacity_ref"]["value"]            # Ah
        self.v_max = u["voltage_max"]["value"]
        self.v_min = u["voltage_min"]["value"]

        # OCV two-plateau shape
        self.ocv_high = u["ocv_high_plateau"]["value"]
        self.ocv_low = u["ocv_low_plateau"]["value"]
        self.ocv_high_frac = u["ocv_high_frac"]["value"]
        self.ocv_top = u["ocv_top"]["value"]
        self.ocv_bottom = u["ocv_bottom"]["value"]
        self.ocv_k = u["ocv_sharpness"]["value"]
        self.ocv_slope = u["ocv_slope"]["value"]

        # Thevenin
        self.R0_ref = u["R0"]["value"]
        self.R1_ref = u["R1"]["value"]
        self.C1 = u["C1"]["value"]

        # thermal / Arrhenius
        self.T_ref = u["T_ref"]["value"]
        self.E_a = u["E_a"]["value"]
        self.dOCV_dT = u["dOCV_dT"]["value"]
        self.R_gas = u["R_gas"]["value"]
        self.F = u["F_const"]["value"]

        # shuttle
        self.k_shuttle = u["k_shuttle"]["value"]
        self.f_high = u["f_high"]["value"]
        self.shuttle_E = u["shuttle_act_E"]["value"]

        # thermal lump
        self.m_cell = u["m_cell"]["value"]
        self.cp_cell = u["cp_cell"]["value"]
        self.hA = u["hA_cool"]["value"]
        self.T_amb = u["T_amb"]["value"]

    # ------------------------------------------------------------------
    # Two-plateau open-circuit voltage
    # ------------------------------------------------------------------
    def ocv(self, soc):
        """
        Two-plateau Li-S OCV [V] as a function of SOC in [0, 1].

        A logistic switch at SOC = ocv_high_frac blends an upper plateau
        (~2.35 V, S8 -> polysulfides) into a lower plateau (~2.1 V -> Li2S),
        each carrying a gentle internal slope so the curve is monotone.
        """
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        # logistic weight: ~1 in the high-SOC plateau, ~0 in the low-SOC plateau
        w = 1.0 / (1.0 + np.exp(-self.ocv_k * (soc - self.ocv_high_frac)))
        # within-plateau gentle slopes (higher SOC -> higher V)
        v_high = self.ocv_high + self.ocv_slope * (soc - 1.0)        # toward ocv_top at SOC=1
        v_low = self.ocv_low + self.ocv_slope * (soc - 0.0) * 1.5    # toward ocv_bottom at SOC=0
        v = w * v_high + (1.0 - w) * v_low
        # anchor the extremes
        return v

    def ocv_with_T(self, soc, T):
        """OCV including the positive entropic temperature correction."""
        return self.ocv(soc) + self.dOCV_dT * (T - self.T_ref)

    # ------------------------------------------------------------------
    # Temperature-dependent resistances (Arrhenius)
    # ------------------------------------------------------------------
    def R0(self, T):
        return self.R0_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def R1(self, T):
        return self.R1_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    # ------------------------------------------------------------------
    # Polysulfide shuttle parasitic current (Mikhaylik & Akridge 2004)
    # ------------------------------------------------------------------
    def shuttle_constant(self, T):
        """Temperature-dependent shuttle constant k_s(T) [1/s]."""
        return self.k_shuttle * np.exp(
            self.shuttle_E / self.R_gas * (1.0 / self.T_ref - 1.0 / T)
        )

    def high_polysulfide_fraction(self, soc):
        """
        Normalised concentration of HIGH-order polysulfides [S_high in 0..1].

        High-order polysulfides only exist while the upper plateau is active
        (high SOC). They vanish once the cell is deep into the lower plateau.
        Logistic in SOC, peaking near the upper plateau.
        """
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-self.ocv_k * (soc - self.ocv_high_frac)))

    def shuttle_current(self, soc, T):
        """
        Lumped polysulfide-shuttle parasitic current [A] (always discharging).

        Mikhaylik-Akridge form: I_sh = k_s * f_H * S_high, scaled to amperes
        through the cell capacity. Always >= 0 -> always drains the cathode.
        """
        ks = self.shuttle_constant(T)
        s_high = self.high_polysulfide_fraction(soc)
        # k_s [1/s] * fraction * capacity [A.s] -> A ; /3600 converts Ah->A·s consistently
        return ks * self.f_high * s_high * self.Q

    def coulombic_efficiency(self, I_app, soc, T):
        """Instantaneous coulombic efficiency eta_C = I_app/(I_app + I_sh), in (0,1)."""
        I_app = float(I_app)
        if I_app <= 0:
            return 1.0  # defined only for discharge
        I_sh = float(self.shuttle_current(soc, T))
        return I_app / (I_app + I_sh)

    # ------------------------------------------------------------------
    # Terminal voltage
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, I_app, V_rc, T):
        """Terminal voltage [V]; I>0 discharge lowers V."""
        v = self.ocv_with_T(soc, T) - I_app * self.R0(T) - V_rc
        return float(np.clip(v, 0.0, self.v_max + 0.5))

    # ------------------------------------------------------------------
    # State derivatives
    # ------------------------------------------------------------------
    def derivatives(self, soc, V_rc, T, I_app):
        """d/dt of [SOC, V_rc, T]."""
        I_sh = self.shuttle_current(soc, T)
        # shuttle always drains -> add to discharge current in SOC balance.
        # clamp SOC dynamics at the rails so it cannot run out of [0,1].
        I_drain = I_app + I_sh
        dsoc = -I_drain / (3600.0 * self.Q)
        if soc <= 0.0 and dsoc < 0.0:
            dsoc = 0.0
        if soc >= 1.0 and dsoc > 0.0:
            dsoc = 0.0

        # 1-RC relaxation (I>0 discharge charges the RC overpotential positive)
        dvrc = I_app / self.C1 - V_rc / (self.R1(T) * self.C1)

        # heat balance
        q_ohm = I_app**2 * self.R0(T) + V_rc**2 / self.R1(T)      # >= 0 irreversibles
        q_rev = I_app * T * self.dOCV_dT                          # >0 dOCV/dT -> endo on discharge
        # shuttle dissipates the OCV it shorts internally as heat (always >= 0)
        q_shuttle = I_sh * self.ocv(soc)
        q_cool = self.hA * (T - self.T_amb)
        dT = (q_ohm - q_rev + q_shuttle - q_cool) / (self.m_cell * self.cp_cell)
        return dsoc, dvrc, dT

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0, T0, dt, duration_s, V_rc0=0.0):
        """
        Integrate the coupled [SOC, V_rc, T] ODE with solve_ivp.

        Parameters
        ----------
        current_A : float or callable(t)   applied current [A] (I>0 discharge)
        soc0 : float                        initial SOC in [0,1]
        T0 : float                          initial temperature [K]
        dt : float                          output time step [s]
        duration_s : float                  total duration [s]
        V_rc0 : float                       initial RC overpotential [V]

        Returns
        -------
        dict of time-series arrays.
        """
        _I = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            soc, V_rc, T = y
            I = _I(t)
            dsoc, dvrc, dT = self.derivatives(soc, V_rc, T, I)
            return [dsoc, dvrc, dT]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [soc0, V_rc0, T0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        soc_out = np.clip(sol.y[0], 0.0, 1.0)
        vrc_out = sol.y[1]
        T_out = sol.y[2]
        N = len(t_out)

        voltage = np.zeros(N)
        ocv_arr = np.zeros(N)
        i_shuttle = np.zeros(N)
        eta_c = np.zeros(N)
        power = np.zeros(N)
        for i in range(N):
            I = _I(t_out[i])
            voltage[i] = self.terminal_voltage(soc_out[i], I, vrc_out[i], T_out[i])
            ocv_arr[i] = self.ocv_with_T(soc_out[i], T_out[i])
            i_shuttle[i] = self.shuttle_current(soc_out[i], T_out[i])
            eta_c[i] = self.coulombic_efficiency(I, soc_out[i], T_out[i])
            power[i] = voltage[i] * I

        return {
            "t": t_out,
            "soc": soc_out,
            "voltage": voltage,
            "ocv": ocv_arr,
            "temperature": T_out,
            "v_rc": vrc_out,
            "shuttle_current": i_shuttle,
            "coulombic_efficiency": eta_c,
            "power": power,
        }
