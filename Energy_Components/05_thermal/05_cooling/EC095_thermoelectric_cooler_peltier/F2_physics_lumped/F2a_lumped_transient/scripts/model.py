"""
EC095 — Thermoelectric Cooler (Peltier) — F2a Physics-Lumped Transient

Two-node lumped first-principles model of a Bi2Te3 thermoelectric (Peltier)
module stack with cold-side and hot-side heat sinks, integrated in time with
scipy.integrate.solve_ivp.

Per-module steady thermoelectric heat balance (Rowe 2006 Ch.1; Goldsmid 2010):

    Peltier (Seebeck) pumping at cold junction   : alpha * I * T_c
    Peltier release at hot junction              : alpha * I * T_h
    Joule heating                                : I**2 * R  (split half/half
                                                   to each junction)
    Fourier conduction leak (hot -> cold)        : K * (T_h - T_c)

  =>  Cold-side heat absorbed (pumped FROM the cold reservoir):
          Q_c = alpha * I * T_c - 0.5 * I**2 * R - K * (T_h - T_c)
      Electrical work input:
          W_in = alpha * I * (T_h - T_c) + I**2 * R
      Hot-side heat rejected (1st-law energy conservation):
          Q_h = Q_c + W_in = alpha * I * T_h + 0.5 * I**2 * R - K*(T_h-T_c)
      Cooling coefficient of performance:
          COP_c = Q_c / W_in           (W_in > 0)

These are summed over N series modules (same I).

Material figure of merit (diagnostic):  ZT = alpha^2 / (R * K) * T.

Lumped transient energy balance on the two plate/sink nodes:

    C_c dT_c/dt = -Q_c(I,T_c,T_h) + Q_load + hA_c * (T_load - T_c)
    C_h dT_h/dt =  Q_h(I,T_c,T_h) - hA_h * (T_h - T_amb)

i.e. as the TEC pumps heat Q_c off the cold node it cools down; the rejected
Q_h warms the hot node, which sheds it to ambient through the hot sink.  An
external thermal load Q_load and a conductive coupling hA_c to a load reservoir
at T_load feed the cold node.

References:
    Rowe, D.M. (Ed.) (2006). CRC Handbook of Thermoelectrics, CRC Press.
    Goldsmid, H.J. (2010). Introduction to Thermoelectricity, Springer.
    Riffat, S.B., Ma, X. (2003). Appl. Thermal Eng. 23, 913-935.
"""

import numpy as np
from scipy.integrate import solve_ivp


class PeltierTEC_F2a:
    """Physics-lumped transient Peltier cooler (2-node thermal ODE)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = int(u["n_modules"]["value"])
        self.alpha = float(u["alpha_module"]["value"])
        self.R = float(u["R_module"]["value"])
        self.K = float(u["K_module"]["value"])
        self.I_max = float(u["I_max"]["value"])
        self.aux_W = float(u.get("auxiliary_power", {"value": 0.0})["value"])
        self.ZT_ref = float(u.get("ZT_ref", {"value": 0.71})["value"])

        self.C_cold = float(u["C_cold"]["value"])
        self.C_hot = float(u["C_hot"]["value"])
        self.hA_cold = float(u["hA_cold"]["value"])
        self.hA_hot = float(u["hA_hot"]["value"])

    # ------------------------------------------------------------------
    # Steady thermoelectric heat/work terms (absolute temperatures in K)
    # ------------------------------------------------------------------
    def cooling_power(self, I, Tc, Th):
        """Cold-side heat absorbed Q_c [W] over the N-module stack."""
        I = np.asarray(I, dtype=float)
        Tc = np.asarray(Tc, dtype=float)
        Th = np.asarray(Th, dtype=float)
        q = self.alpha * I * Tc - 0.5 * I * I * self.R - self.K * (Th - Tc)
        return self.N * q

    def electrical_input(self, I, Tc, Th):
        """Electrical work input W_in [W] over the stack (+ auxiliary)."""
        I = np.asarray(I, dtype=float)
        Tc = np.asarray(Tc, dtype=float)
        Th = np.asarray(Th, dtype=float)
        w = self.alpha * I * (Th - Tc) + I * I * self.R
        return self.N * w + self.aux_W

    def heat_rejection(self, I, Tc, Th):
        """Hot-side heat rejected Q_h = Q_c + W_in [W] (energy conservation)."""
        return self.cooling_power(I, Tc, Th) + self.electrical_input(I, Tc, Th)

    def cop(self, I, Tc, Th):
        """Cooling COP = Q_c / W_in.  Returns 0 where W_in <= 0 or Q_c <= 0."""
        Qc = self.cooling_power(I, Tc, Th)
        W = self.electrical_input(I, Tc, Th)
        Qc = np.asarray(Qc, dtype=float)
        W = np.asarray(W, dtype=float)
        good = (W > 1e-9) & (Qc > 0.0)
        return np.where(good, Qc / np.where(W > 1e-9, W, 1.0), 0.0)

    def carnot_cop(self, Tc, Th):
        """Carnot upper bound on cooling COP = Tc / (Th - Tc)."""
        Tc = np.asarray(Tc, dtype=float)
        Th = np.asarray(Th, dtype=float)
        dT = Th - Tc
        return np.where(dT > 1e-9, Tc / np.where(dT > 1e-9, dT, 1.0), np.inf)

    def zt(self, T):
        """Material figure of merit ZT = alpha^2 / (R*K) * T (per module)."""
        return (self.alpha ** 2) / (self.R * self.K) * np.asarray(T, dtype=float)

    def optimum_current_qc(self, Tc):
        """Current maximising Q_c at fixed Tc: dQc/dI = alpha*Tc - I*R = 0."""
        Tc = np.asarray(Tc, dtype=float)
        return np.minimum(self.alpha * Tc / self.R, self.I_max)

    def max_cooling_power(self, Tc, Th):
        """Q_c at the Q_c-optimal current (no I_max clip), per stack [W]."""
        I_opt = self.alpha * np.asarray(Tc, dtype=float) / self.R
        return self.cooling_power(I_opt, Tc, Th)

    def optimum_current_cop(self, Tc, Th):
        """Current maximising COP (Rowe 2006):
        I_optCOP = alpha*(Th-Tc) / ( R*(sqrt(1+Z*Tm) - 1) ),
        with Z = alpha^2/(R*K) (per module) and Tm = (Th+Tc)/2."""
        Tc = float(Tc)
        Th = float(Th)
        Z = (self.alpha ** 2) / (self.R * self.K)
        Tm = 0.5 * (Th + Tc)
        denom = self.R * (np.sqrt(1.0 + Z * Tm) - 1.0)
        if denom <= 1e-12:
            return self.I_max
        return min(self.alpha * (Th - Tc) / denom, self.I_max)

    # ------------------------------------------------------------------
    # Transient lumped 2-node ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, I_func, Q_load_func, T_load, T_amb):
        Tc, Th = y
        I = I_func(t)
        Q_load = Q_load_func(t)
        Qc = float(self.cooling_power(I, Tc, Th))
        Qh = float(self.heat_rejection(I, Tc, Th))
        dTc = (-Qc + Q_load + self.hA_cold * (T_load - Tc)) / self.C_cold
        dTh = (Qh - self.hA_hot * (Th - T_amb)) / self.C_hot
        return [dTc, dTh]

    def simulate(self, current_a, T_cold0_K, T_hot0_K, T_load_K, T_amb_K,
                 Q_load_W=0.0, dt=1.0, duration_s=600.0):
        """
        Integrate the 2-node lumped ODE with solve_ivp.

        current_a   : scalar A, or callable t->A (per module, series stack)
        T_cold0_K   : initial cold-plate temperature [K]
        T_hot0_K    : initial hot-plate temperature  [K]
        T_load_K    : cold-side load/reservoir temperature [K]
        T_amb_K     : ambient temperature [K]
        Q_load_W    : scalar or callable t->W external heat load on cold side
        dt          : output sample interval [s]
        duration_s  : total simulated time [s]
        """
        I_func = current_a if callable(current_a) else (lambda t: current_a)
        Q_func = Q_load_W if callable(Q_load_W) else (lambda t: Q_load_W)

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T_cold0_K, T_hot0_K],
            t_eval=t_eval, args=(I_func, Q_func, T_load_K, T_amb_K),
            method="LSODA", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        Tc = sol.y[0]
        Th = sol.y[1]
        I_arr = np.array([I_func(tt) for tt in t], dtype=float)
        Qload_arr = np.array([Q_func(tt) for tt in t], dtype=float)

        Qc = self.cooling_power(I_arr, Tc, Th)
        W = self.electrical_input(I_arr, Tc, Th)
        Qh = Qc + W
        cop = self.cop(I_arr, Tc, Th)
        cop_carnot = self.carnot_cop(Tc, Th)

        return {
            "t": t,
            "T_cold": Tc,
            "T_hot": Th,
            "T_cold_C": Tc - 273.15,
            "T_hot_C": Th - 273.15,
            "current": I_arr,
            "Q_cold": Qc,
            "Q_hot": Qh,
            "W_elec": W,
            "Q_load": Qload_arr,
            "cop": cop,
            "cop_carnot": cop_carnot,
            "dT": Th - Tc,
        }
