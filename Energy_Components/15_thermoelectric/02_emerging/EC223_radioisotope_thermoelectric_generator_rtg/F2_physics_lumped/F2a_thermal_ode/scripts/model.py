"""
EC223 -- Radioisotope Thermoelectric Generator (RTG) -- F2a Physics-Lumped

Physics-lumped (0D) model of an RTG: radioisotope decay heat drives a bank of
thermoelectric (Seebeck) couples; the hot-side block temperature is governed by
a lumped capacitance ODE balancing decay heat against thermoelectric conduction,
parasitic leakage, and T^4 radiator rejection.

--------------------------------------------------------------------------------
1. Radioisotope decay heat (Pu-238 alpha decay, first-order kinetics)
       Q_decay(t) = Q0 * exp(-ln(2) * t / t_half) = Q0 * exp(-t / tau)
       tau = t_half / ln(2)          (mean lifetime)
   For Pu-238 t_half = 87.7 yr. This is the only heat *source*.

2. Thermoelectric (TE) module -- single-couple physics summed over N couples
   (Ioffe / Rowe Handbook, Ch. 1 & 9). For a couple between T_h (hot junction)
   and T_c (cold junction) with effective Seebeck S, internal resistance R,
   thermal conductance K, driving a load resistance R_L:

       I   = S * (T_h - T_c) / (R + R_L)              Seebeck EMF / Ohm's law
       P_e = I^2 * R_L                                 electrical power to load
       Q_h = S * I * T_h - 0.5 * I^2 * R + K*(T_h-T_c) Peltier - half-Joule + Fourier
       Q_c = S * I * T_c + 0.5 * I^2 * R + K*(T_h-T_c)
   Energy conservation:  Q_h - Q_c = P_e  (exactly, by construction).
   At matched load (R_L = R) electrical power is maximised.

   Module efficiency:  eta_TE = P_e / Q_h, bounded above by the Carnot-ZT limit
       eta_max = eta_Carnot * (sqrt(1+ZT) - 1) / (sqrt(1+ZT) + T_c/T_h)
   with eta_Carnot = 1 - T_c/T_h. Always eta_TE < eta_Carnot.

3. Hot-side lumped thermal ODE (the F2 first-principles upgrade over F1):
       C_hot * dT_h/dt = Q_decay(t) - Q_h_total(T_h) - Q_parasitic
   where
       Q_h_total  = N_couples * Q_h(single couple)     heat pulled through TE legs
       Q_parasitic= K_hp * (T_h - T_c)                 structural insulation leak
   Integrated with scipy.integrate.solve_ivp over mission years (seconds).
   T_h relaxes to a quasi-steady value that slowly falls as Q_decay decays,
   so electrical power declines over the mission -- the defining RTG behaviour.

   Radiator rejection is on the COLD side (downstream of the couples): the
   waste heat reaching the cold junction, Q_c = N*Q_c_couple, is rejected to
   deep space by a Stefan-Boltzmann radiator,
       Q_rad = eps * sigma * A_rad * (T_c^4 - T_space^4),
   which is what holds the cold side near T_cold. radiator_balance() reports
   the residual Q_c - Q_rad (energy-conservation check on the cold side).

References:
    Rowe, D.M. (ed.) (1995, 2006). CRC Handbook of Thermoelectrics. CRC Press.
    Ioffe, A.F. (1957). Semiconductor Thermoelements and Thermoelectric Cooling.
    Bennett, G.L. (2006). Space nuclear power. AIAA 2006-4191.
    El-Genk, M.S. & Saber, H.H. (2005). Energy Convers. Mgmt. 46(7-8), 1083.
    NASA GPHS-RTG / MMRTG specifications, rps.nasa.gov.
"""

import numpy as np
from scipy.integrate import solve_ivp

LN2 = np.log(2.0)
SIGMA = 5.670374419e-8          # Stefan-Boltzmann constant [W/(m^2 K^4)]
SEC_PER_YEAR = 365.25 * 24.0 * 3600.0


class RTG_F2a:
    """RTG -- decay-heat-driven thermoelectric module with hot-side thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q0 = u["P_thermal_0_W"]["value"]          # W thermal at BOL
        self.t_half = u["t_half_years"]["value"]        # years
        self.tau_s = self.t_half / LN2 * SEC_PER_YEAR   # mean lifetime [s]

        self.N = u["n_couples"]["value"]                # number of TE couples
        self.S = u["seebeck_couple"]["value"]           # V/K per couple
        self.R = u["R_couple"]["value"]                 # ohm per couple
        self.K = u["K_couple"]["value"]                 # W/K per couple (Fourier)
        self.ZT = u["ZT"]["value"]                       # figure of merit

        self.T_hot_0 = u["T_hot_0_K"]["value"]          # K
        self.T_cold = u["T_cold_K"]["value"]            # K (cold junction / radiator interface)
        self.C_hot = u["C_hot_J_per_K"]["value"]        # J/K lumped hot-side capacitance
        self.eps = u["emissivity"]["value"]
        self.A_rad = u["A_radiator_m2"]["value"]        # m2
        self.T_space = u["T_space_K"]["value"]          # K
        self.K_hp = u["K_hot_to_cold_W_per_K"]["value"] # W/K parasitic leak

    # ------------------------------------------------------------------
    # 1. Decay heat
    # ------------------------------------------------------------------
    def decay_heat(self, t_years):
        """Decay thermal power Q(t) = Q0 * exp(-t/tau) [W]. t in years."""
        t = np.maximum(np.asarray(t_years, dtype=float), 0.0)
        return self.Q0 * np.exp(-LN2 * t / self.t_half)

    # ------------------------------------------------------------------
    # 2. Single-couple thermoelectric solution at a given (T_h, T_c)
    # ------------------------------------------------------------------
    def couple_state(self, T_h, T_c, R_load=None):
        """
        Thermoelectric couple operating point.

        Parameters
        ----------
        T_h, T_c : hot / cold junction temperatures [K]
        R_load   : load resistance per couple [ohm]; default = matched (R_load=R)

        Returns dict with per-couple I, P_e, Q_h, Q_c and module-level totals.
        """
        dT = max(T_h - T_c, 0.0)
        R_L = self.R if R_load is None else R_load

        I = self.S * dT / (self.R + R_L)               # A
        P_e = I * I * R_L                              # W electrical, per couple
        # Heat absorbed at hot junction: Peltier - half-Joule (returns to hot)
        # + Fourier conduction down the legs.
        Q_h = self.S * I * T_h - 0.5 * I * I * self.R + self.K * dT
        Q_c = self.S * I * T_c + 0.5 * I * I * self.R + self.K * dT

        eta = P_e / Q_h if Q_h > 0 else 0.0

        return {
            "I": I,
            "P_e_couple": P_e,
            "Q_h_couple": Q_h,
            "Q_c_couple": Q_c,
            "P_e_total": self.N * P_e,
            "Q_h_total": self.N * Q_h,
            "Q_c_total": self.N * Q_c,
            "eta_module": eta,
            "dT": dT,
        }

    # ------------------------------------------------------------------
    # Carnot and ZT-limited efficiency bounds
    # ------------------------------------------------------------------
    def eta_carnot(self, T_h, T_c):
        return 1.0 - T_c / T_h

    def eta_zt_max(self, T_h, T_c):
        """Maximum thermoelectric efficiency (Ioffe/Rowe), < eta_Carnot."""
        etac = self.eta_carnot(T_h, T_c)
        m = np.sqrt(1.0 + self.ZT)
        return etac * (m - 1.0) / (m + T_c / T_h)

    # ------------------------------------------------------------------
    # Radiator rejection (T^4) -- on the cold side
    # ------------------------------------------------------------------
    def Q_radiator(self, T):
        """Stefan-Boltzmann radiative rejection from the radiator at temp T [W]."""
        return self.eps * SIGMA * self.A_rad * (T**4 - self.T_space**4)

    def radiator_balance(self, T_h, T_c=None, R_load=None):
        """
        Cold-side energy-conservation check. The couples dump Q_c at the cold
        junction; the radiator must reject it at the cold-side temperature.
        Returns (Q_c, Q_rad, residual). |residual| ~ 0 means the radiator is
        correctly sized for the cold-side temperature.
        """
        T_c = self.T_cold if T_c is None else T_c
        cs = self.couple_state(T_h, T_c, R_load)
        Q_c = cs["Q_c_total"] + self.K_hp * max(T_h - T_c, 0.0)
        Q_rad = self.Q_radiator(T_c)
        return Q_c, Q_rad, Q_c - Q_rad

    # ------------------------------------------------------------------
    # 3. Hot-side thermal ODE  dT_h/dt
    # ------------------------------------------------------------------
    def _rhs(self, t_s, y, R_load):
        """ODE right-hand side; t_s in seconds, y=[T_h]."""
        T_h = y[0]
        T_c = self.T_cold
        t_years = t_s / SEC_PER_YEAR

        Q_in = self.Q0 * np.exp(-t_s / self.tau_s)     # decay heat (seconds form)
        cs = self.couple_state(T_h, T_c, R_load)
        Q_te = cs["Q_h_total"]                          # heat drawn through TE legs
        Q_par = self.K_hp * max(T_h - T_c, 0.0)         # parasitic insulation leak

        # Hot-block balance: decay source minus heat leaving through couples and
        # parasitic insulation. The cold-side radiator (T^4) rejects the
        # resulting Q_c and holds the cold junction near T_cold.
        dTdt = (Q_in - Q_te - Q_par) / self.C_hot
        return [dTdt]

    # ------------------------------------------------------------------
    # Driver: integrate over a mission and report electrical output
    # ------------------------------------------------------------------
    def simulate(self, mission_years=50.0, n_points=200, T_h0=None,
                 R_load=None, max_step_years=2.0):
        """
        Integrate the hot-side ODE over a mission and report the time series.

        Returns dict of numpy arrays:
            t_years, T_hot_K, Q_decay_W, P_electric_W, eta_module,
            eta_carnot, eta_zt_max, current_A, Q_radiator_W, power_fraction
        """
        T0 = self.T_hot_0 if T_h0 is None else T_h0
        t_end = mission_years * SEC_PER_YEAR
        t_eval = np.linspace(0.0, t_end, int(n_points))
        max_step = max_step_years * SEC_PER_YEAR

        sol = solve_ivp(
            self._rhs, (0.0, t_end), [T0], t_eval=t_eval,
            args=(R_load,), method="LSODA", rtol=1e-7, atol=1e-4,
            max_step=max_step,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        t_years = sol.t / SEC_PER_YEAR
        T_h = sol.y[0]

        # Post-process electrical quantities along the trajectory
        Q_decay = self.Q0 * np.exp(-sol.t / self.tau_s)
        P_e = np.empty_like(T_h)
        eta_mod = np.empty_like(T_h)
        I_arr = np.empty_like(T_h)
        eta_c = np.empty_like(T_h)
        eta_zt = np.empty_like(T_h)
        Q_rad = np.empty_like(T_h)

        for i, Th in enumerate(T_h):
            cs = self.couple_state(Th, self.T_cold, R_load)
            P_e[i] = cs["P_e_total"]
            eta_mod[i] = cs["eta_module"]
            I_arr[i] = cs["I"]
            eta_c[i] = self.eta_carnot(Th, self.T_cold)
            eta_zt[i] = self.eta_zt_max(Th, self.T_cold)
            Q_rad[i] = self.Q_radiator(Th)

        P0 = P_e[0] if P_e[0] > 0 else 1e-12
        return {
            "t_years": t_years,
            "T_hot_K": T_h,
            "Q_decay_W": Q_decay,
            "P_electric_W": P_e,
            "eta_module": eta_mod,
            "eta_carnot": eta_c,
            "eta_zt_max": eta_zt,
            "current_A": I_arr,
            "Q_radiator_W": Q_rad,
            "power_fraction": P_e / P0,
        }

    # ------------------------------------------------------------------
    # Quasi-steady-state hot temperature at a given mission time
    # ------------------------------------------------------------------
    def steady_T_hot(self, t_years, R_load=None):
        """Root-find the hot-side temperature that zeroes the ODE rhs at time t."""
        from scipy.optimize import brentq
        t_s = t_years * SEC_PER_YEAR
        f = lambda T: self._rhs(t_s, [T], R_load)[0]
        # Bracket: between cold side and a generous upper bound.
        lo, hi = self.T_cold + 1.0, 2000.0
        if f(lo) <= 0:    # already over-rejecting even at low T
            return lo
        return brentq(f, lo, hi, xtol=1e-3)
