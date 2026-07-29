"""
EC029 -- Nickel-Metal Hydride (NiMH) Battery -- F2a Thevenin 2-RC Electrothermal Model

Physics-lumped (0D) first-principles model: a Thevenin equivalent-circuit model
(ECM) with two RC pairs, Coulomb-counted SOC, Arrhenius temperature dependence of
all resistances, a self-discharge leak, and a lumped thermal ODE that captures the
strongly exothermic NiMH overcharge regime (oxygen evolution at the positive
electrode followed by recombination at the negative electrode at top-of-charge).

State vector  y = [SOC, V_rc1, V_rc2, T]
    dSOC/dt   = -(I_net) / (3600 * Q_eff)        Coulomb counting (+ self-discharge)
    dV_rc1/dt = I/C1 - V_rc1/(R1(T)*C1)          fast charge-transfer RC
    dV_rc2/dt = I/C2 - V_rc2/(R2(T)*C2)          slow diffusion RC
    m*cp*dT/dt = Q_irrev + Q_rev + Q_recomb - hA*(T - T_amb)   energy balance

Terminal voltage (sign convention: I>0 discharge, I<0 charge):
    V_term = OCV(SOC) - I*R0(T) - V_rc1 - V_rc2

Resistances follow Arrhenius:
    R(T) = R_ref * exp( E_a/R_gas * (1/T - 1/T_ref) )

OCV(SOC) is a 5th-order polynomial giving the flat ~1.2 V NiMH plateau.

Overcharge / oxygen-recombination (NiMH-specific):
    Near full charge the positive electrode evolves O2 instead of storing charge.
    A logistic gate f_oc(SOC, charging) diverts a fraction of the charge current
    into the O2 evolution/recombination loop:  O2 + 2H2O + 4e- -> 4OH- (at the MH
    electrode). The chemical energy of the loop is dissipated as heat, making the
    cell strongly exothermic at top-of-charge while SOC stops rising (charge
    acceptance collapses). This is the classic NiMH "voltage/temperature roll-over"
    used for -dV/dT and dT/dt fast-charge termination.

References:
    Linden's Handbook of Batteries, 4th ed. (2011), ch. 30-31.
    Bernardi, Pawlikowski & Newman (1985), J. Electrochem. Soc. 132(1), 5-12 (energy balance).
    Bernardi & Carpenter (1995), J. Electrochem. Soc. 142(8), 2631-2642 (NiMH entropic heat).
    Khun et al. (2006), Electrochimica Acta 51, 2877-2887 (NiMH impedance / Arrhenius R).
    Hu, Li & Peng (2012), J. Power Sources 198, 359-367 (Thevenin RC ECM identification).
    Gao, Liu & Dougal (2002), IEEE Trans. Components & Packaging 25(3), 495-505 (lumped thermal).
    Notten & Hokkeling (1991) / Ten Haaf, NiMH oxygen-recombination overcharge mechanism.
"""

import numpy as np
from scipy.integrate import solve_ivp


class NiMH_F2a:
    """NiMH Thevenin 2-RC equivalent-circuit model with lumped thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_ref = u["capacity_ref"]["value"]            # Ah
        self.v_max = u["voltage_max"]["value"]             # V
        self.v_min = u["voltage_min"]["value"]             # V

        self.R0_ref = u["R0_ref"]["value"]                 # Ohm
        self.R1_ref = u["R1_ref"]["value"]                 # Ohm
        self.C1 = u["C1"]["value"]                         # F
        self.R2_ref = u["R2_ref"]["value"]                 # Ohm
        self.C2 = u["C2"]["value"]                         # F

        self.ocv_coeff = np.asarray(u["ocv_coefficients"]["value"], dtype=float)

        self.T_ref = u["T_ref"]["value"]                   # K
        self.E_a = u["E_a"]["value"]                       # J/mol
        self.R_gas = u["R_gas"]["value"]                   # J/(mol.K)
        self.dOCV_dT = u["dOCV_dT"]["value"]               # V/K (positive)

        self.k_sd = u["k_self_discharge"]["value"]         # 1/s
        self.E_a_sd = u["E_a_self"]["value"]               # J/mol

        self.soc_oc_onset = u["soc_oc_onset"]["value"]     # -
        self.f_oc_max = u["f_oc_max"]["value"]             # -
        self.dH_recomb = u["dH_recomb"]["value"]           # V (effective)

        self.m_cell = u["m_cell"]["value"]                 # kg
        self.cp_cell = u["cp_cell"]["value"]               # J/(kg.K)
        self.hA = u["hA_cell"]["value"]                    # W/K
        self.T_amb = u["T_amb"]["value"]                   # K

    # ------------------------------------------------------------------
    # Open-circuit voltage (flat NiMH plateau)
    # ------------------------------------------------------------------
    def ocv(self, soc, T=None):
        """Open-circuit voltage [V] as function of SOC (0..1), with entropic
        temperature correction OCV += dOCV/dT * (T - T_ref)."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([s ** i for i in range(len(self.ocv_coeff))], axis=-1)
        v = np.dot(powers, self.ocv_coeff)
        if T is not None:
            v = v + self.dOCV_dT * (np.asarray(T, dtype=float) - self.T_ref)
        return v

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
    # Self-discharge leak (Arrhenius-accelerated, fraction of capacity / s)
    # ------------------------------------------------------------------
    def self_discharge_rate(self, soc, T):
        """SOC leak rate [1/s] >= 0 (only discharges, scales with stored charge)."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        k = self.k_sd * np.exp(self.E_a_sd / self.R_gas * (1.0 / self.T_ref - 1.0 / np.asarray(T, dtype=float)))
        return k * s

    # ------------------------------------------------------------------
    # Overcharge oxygen-recombination gate
    # ------------------------------------------------------------------
    def overcharge_fraction(self, soc, current):
        """
        Fraction of (charge) current diverted to O2 evolution/recombination.
        Returns 0 unless charging (current<0) and SOC near/above onset.
        Logistic ramp from soc_oc_onset to ~1.0, saturating at f_oc_max.
        """
        if current >= 0:           # discharge -> no oxygen recombination
            return 0.0
        s = float(soc)
        # logistic centred slightly above onset; width ~0.03 SOC
        x = (s - (self.soc_oc_onset + 0.01)) / 0.02
        logistic = 1.0 / (1.0 + np.exp(-x))
        return self.f_oc_max * logistic

    # ------------------------------------------------------------------
    # Terminal voltage from state
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, current, T, V_rc1=0.0, V_rc2=0.0):
        """Terminal voltage [V]. I>0 discharge (V drops), I<0 charge (V rises)."""
        ocv = self.ocv(soc, T)
        v = ocv - current * self.R0(T) - V_rc1 - V_rc2
        return float(np.clip(v, self.v_min, self.v_max))

    # ------------------------------------------------------------------
    # Heat generation terms (W)
    # ------------------------------------------------------------------
    def heat_terms(self, soc, current, T, V_rc1, V_rc2):
        """
        Returns (Q_irrev, Q_rev, Q_recomb) in Watts.
        Q_irrev  : Joule heating in R0 + both RC resistors (always >= 0)
        Q_rev    : reversible entropic heat = -I * T * dOCV/dT
                   (NiMH dOCV/dT>0: discharge I>0 -> heat release; charge I<0 -> absorb)
        Q_recomb : exothermic O2 evolution+recombination at top-of-charge (>=0)
        """
        R0 = self.R0(T)
        # RC resistor currents: I_Rk = V_rck / Rk  (current through the resistor)
        I_R1 = V_rc1 / self.R1(T)
        I_R2 = V_rc2 / self.R2(T)
        Q_irrev = current ** 2 * R0 + I_R1 ** 2 * self.R1(T) + I_R2 ** 2 * self.R2(T)

        # reversible (entropic). Bernardi (1985): Q_rev = -I * T * dU/dT
        Q_rev = -current * T * self.dOCV_dT

        # overcharge recombination: diverted charge current * effective enthalpy V
        f_oc = self.overcharge_fraction(soc, current)
        I_recomb = f_oc * abs(current)        # A of charge current going to O2 loop
        Q_recomb = I_recomb * self.dH_recomb  # W, strongly exothermic
        return Q_irrev, Q_rev, Q_recomb

    # ------------------------------------------------------------------
    # State derivative
    # ------------------------------------------------------------------
    def _rhs(self, t, y, current_fn):
        soc, V_rc1, V_rc2, T = y
        I = current_fn(t)

        # Coulomb counting: net charge that actually enters/leaves storage.
        # On charge (I<0) the recombined fraction does NOT charge the cell.
        f_oc = self.overcharge_fraction(soc, I)
        I_store = I + f_oc * abs(I) if I < 0 else I   # reduce charging current
        dsoc_dt = -I_store / (3600.0 * self.Q_ref)
        # self-discharge always removes charge
        dsoc_dt -= self.self_discharge_rate(soc, T)

        # RC dynamics
        dV1_dt = I / self.C1 - V_rc1 / (self.R1(T) * self.C1)
        dV2_dt = I / self.C2 - V_rc2 / (self.R2(T) * self.C2)

        # thermal energy balance
        Q_irrev, Q_rev, Q_recomb = self.heat_terms(soc, I, T, V_rc1, V_rc2)
        Q_loss = self.hA * (T - self.T_amb)
        dT_dt = (Q_irrev + Q_rev + Q_recomb - Q_loss) / (self.m_cell * self.cp_cell)

        return [dsoc_dt, dV1_dt, dV2_dt, dT_dt]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.5, T0=None, dt=1.0, duration_s=600.0):
        """
        Simulate NiMH electrothermal dynamics.

        Parameters
        ----------
        current_A : float or callable(t)
            Load current [A]. I>0 discharge, I<0 charge.
        soc0 : float
            Initial state of charge (0..1, may start >1 for overcharge tests).
        T0 : float
            Initial temperature [K] (defaults to ambient).
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation time [s].

        Returns
        -------
        dict: t, soc, voltage, current, power, temperature, efficiency,
              heat (dict of irrev/rev/recomb/loss arrays), overcharge_fraction
        """
        if T0 is None:
            T0 = self.T_amb
        cur_fn = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [float(soc0), 0.0, 0.0, float(T0)]
        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, cur_fn),
            (0.0, duration_s), y0,
            t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9,
            max_step=max(dt, 1.0),
        )

        t = sol.t
        soc = np.clip(sol.y[0], 0.0, 1.0)
        V_rc1 = sol.y[1]
        V_rc2 = sol.y[2]
        T = sol.y[3]
        N = len(t)

        voltage = np.zeros(N)
        current = np.zeros(N)
        power = np.zeros(N)
        efficiency = np.zeros(N)
        f_oc = np.zeros(N)
        Q_irrev = np.zeros(N)
        Q_rev = np.zeros(N)
        Q_recomb = np.zeros(N)
        Q_loss = np.zeros(N)

        for i in range(N):
            I = cur_fn(t[i])
            current[i] = I
            voltage[i] = self.terminal_voltage(soc[i], I, T[i], V_rc1[i], V_rc2[i])
            power[i] = voltage[i] * I
            f_oc[i] = self.overcharge_fraction(soc[i], I)
            qi, qr, qc = self.heat_terms(soc[i], I, T[i], V_rc1[i], V_rc2[i])
            Q_irrev[i], Q_rev[i], Q_recomb[i] = qi, qr, qc
            Q_loss[i] = self.hA * (T[i] - self.T_amb)
            # round-trip-ish coulombic/voltaic efficiency proxy: V_term/OCV on
            # discharge, OCV/V_term on charge -> always in (0,1)
            ocv = self.ocv(soc[i], T[i])
            if abs(I) < 1e-9 or ocv <= 0:
                efficiency[i] = 1.0 - 1e-6
            elif I > 0:                       # discharge
                efficiency[i] = min(voltage[i] / ocv, 1.0 - 1e-9)
            else:                              # charge
                efficiency[i] = min(ocv / voltage[i], 1.0 - 1e-9)
            efficiency[i] = float(np.clip(efficiency[i], 1e-6, 1.0 - 1e-9))

        return {
            "t": t,
            "soc": soc,
            "voltage": voltage,
            "current": current,
            "power": power,
            "temperature": T,
            "efficiency": efficiency,
            "overcharge_fraction": f_oc,
            "heat": {
                "irreversible": Q_irrev,
                "reversible": Q_rev,
                "recombination": Q_recomb,
                "loss": Q_loss,
            },
        }
