"""
EC223 — Radioisotope Thermoelectric Generator (RTG) — F1b Multi-Layer TEG Model

Extends F1a (parametric decay + TEG degradation) with:

1. **Temperature-dependent SiGe material properties**:
   alpha(T) = alpha0 * (1 + a1*(T - T_ref))        Seebeck coefficient
   k(T)     = k0 * (1 + b1*(T - T_ref))             Thermal conductivity
   sigma(T) = sigma0 * (1 + c1*(T - T_ref))          Electrical conductivity
   ZT(T)    = alpha(T)^2 * sigma(T) * T / k(T)      Figure of merit

2. **Thermal resistance network**:
   Hot-side: T_hot_junction = T_source - P_thermal * R_th_hot
   Cold-side: T_cold_junction = T_sink + P_radiated * R_th_cold
   With iterative self-consistent solution (T_hot and T_cold are coupled
   to TEG power output via Peltier/Seebeck back-reaction).

3. **Correct first-law TEG power**:
   Uses Angist formula with ZT averaged over actual junction temperatures,
   not over fixed nominal temperatures:
   eta = eta_Carnot * (sqrt(1+ZT_avg) - 1) / (sqrt(1+ZT_avg) + T_c/T_h)
   P_electric = P_thermal * eta

4. **Contact degradation** (same floor as F1a but applied to thermal conductance):
   Degradation factor applied to R_thermal_hot (contact conductance decreases).

5. **Open-circuit voltage and matched-load current** at junction temperatures:
   V_oc = N * alpha_avg * (T_h - T_c)     [V]
   R_int = 2 * N * L / (sigma_avg * A) * (1 + r_contact)
   V_load = V_oc / 2 (matched load)
   I_mp = V_oc / (2 * R_int)
   P_max = V_oc^2 / (4 * R_int)   (matched load power)

References:
    Bennett, G.L. (2006). Space nuclear power. Acta Astronautica.
    El-Genk, M.S. & Saber, H.H. (2005). Energy Convers. Mgmt. 46(7-8), 1083.
    Fleurial, J-P. et al. (1997). Proc. 16th IECEC, paper 97450.
    Rowe, D.M. (ed.) (2006). Thermoelectrics Handbook. CRC Press.
"""

import numpy as np

ln2 = np.log(2.0)


def _trapz(y, x):
    """Backward-compatible trapz (numpy >=2 renamed to trapezoid)."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


class RTGF1b:
    """RTG with temperature-dependent SiGe TEG model and thermal resistance network."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P0 = u["P_thermal_0_W"]["value"]
        self.t_half = u["t_half_years"]["value"]
        self.T_hot_0 = u["T_hot_0_K"]["value"]
        self.T_cold_sink = u["T_cold_K"]["value"]

        # TEG material (SiGe)
        self.alpha0 = u["alpha0_SiGe"]["value"]
        self.k0 = u["k0_SiGe"]["value"]
        self.sigma0 = u["sigma0_SiGe"]["value"]
        self.T_ref = u["T_ref_K"]["value"]
        self.a1 = u["a1_SiGe"]["value"]
        self.b1 = u["b1_SiGe"]["value"]
        self.c1 = u["c1_SiGe"]["value"]

        # Geometry
        self.N = u["n_couples_inner"]["value"]
        self.L = u["L_element_m"]["value"]
        self.A = u["A_element_m2"]["value"]
        self.r_contact = u["contact_resistance_fraction"]["value"]

        # Thermal resistances
        self.R_hot_0 = u["R_thermal_hot_K_W"]["value"]   # K/W (hot side, per RTG aggregate)
        self.R_cold = u["R_thermal_cold_K_W"]["value"]   # K/W (cold radiator)

        # Degradation
        self.deg_rate = u["eta_teg_degradation_rate"]["value"]  # 1/year

    # ---- Material property functions ----

    def alpha(self, T):
        """Seebeck coefficient [V/K] at temperature T."""
        T = np.asarray(T, dtype=float)
        return self.alpha0 * (1.0 + self.a1 * (T - self.T_ref))

    def k_thermal(self, T):
        """Thermal conductivity [W/(m*K)] at T."""
        T = np.asarray(T, dtype=float)
        return np.maximum(self.k0 * (1.0 + self.b1 * (T - self.T_ref)), 0.5)

    def sigma_elec(self, T):
        """Electrical conductivity [S/m] at T."""
        T = np.asarray(T, dtype=float)
        return np.maximum(self.sigma0 * (1.0 + self.c1 * (T - self.T_ref)), 1e3)

    def zt_local(self, T):
        """Local ZT at T."""
        T = np.asarray(T, dtype=float)
        a = self.alpha(T)
        k = self.k_thermal(T)
        s = self.sigma_elec(T)
        return a ** 2 * s * T / (k + 1e-12)

    def zt_average(self, T_h, T_c):
        """Average ZT over temperature gradient using trapezoidal integration."""
        T_h_v = float(np.atleast_1d(T_h)[0]) if np.ndim(T_h) > 0 else float(T_h)
        T_c_v = float(np.atleast_1d(T_c)[0]) if np.ndim(T_c) > 0 else float(T_c)
        if abs(T_h_v - T_c_v) < 0.1:
            return float(self.zt_local((T_h_v + T_c_v) / 2.0))
        n_pts = 60
        T_arr = np.linspace(T_c_v, T_h_v, n_pts)
        zt_arr = self.zt_local(T_arr)
        return float(_trapz(zt_arr, T_arr) / (T_h_v - T_c_v))

    def _zt_average_array(self, T_h_arr, T_c_arr):
        """ZT average for array inputs (vectorised over pairs)."""
        T_h_arr = np.asarray(T_h_arr, dtype=float)
        T_c_arr = np.asarray(T_c_arr, dtype=float)
        shape = np.broadcast_shapes(T_h_arr.shape, T_c_arr.shape)
        T_h_arr = np.broadcast_to(T_h_arr, shape).copy()
        T_c_arr = np.broadcast_to(T_c_arr, shape).copy()
        result = np.empty(shape)
        for idx in np.ndindex(shape):
            result[idx] = self.zt_average(T_h_arr[idx], T_c_arr[idx])
        return result

    # ---- Thermal resistance model ----

    def _hot_junction_T(self, P_thermal, t_years):
        """Hot junction temperature after contact degradation.
        T_hj = T_source - P_thermal * R_hot(t)
        T_source is inferred from initial hot-junction temperature and the ratio
        P(t)/P(0): T_source(t) ~ T_hot_0 * (P(t)/P0)^0.25 (radiative heat balance).
        R_hot grows as contacts degrade (thermal resistance increases).
        """
        # Degradation: R_hot increases as thermal conductance decreases
        deg = np.maximum(1.0 - self.deg_rate * t_years, 0.5)
        R_hot = self.R_hot_0 / deg   # resistance increases as conductance falls

        # Source temperature from radiative balance (same scaling as F1a)
        T_source = self.T_hot_0 * (P_thermal / (self.P0 + 1e-12)) ** 0.25
        T_source = np.maximum(T_source, self.T_cold_sink + 10.0)

        T_hj = T_source - P_thermal * R_hot
        return np.maximum(T_hj, self.T_cold_sink + 5.0)

    def _cold_junction_T(self, P_radiated):
        """Cold junction temperature — cold-side thermal resistance to space radiator.
        T_cj = T_cold_sink + P_radiated * R_cold
        """
        return self.T_cold_sink + P_radiated * self.R_cold

    # ---- Module electrical properties ----

    def _module_resistance(self, T_avg):
        """Module internal resistance [ohm].
        R_int = 2 * N * L / (sigma * A) * (1 + r_contact)
        """
        sigma = self.sigma_elec(T_avg)
        R_elem = self.L / (sigma * self.A + 1e-30)
        return 2.0 * self.N * R_elem * (1.0 + self.r_contact)

    def _module_seebeck(self, T_avg):
        """Total module Seebeck coefficient [V/K]: alpha_module = N * alpha(T_avg)."""
        return self.N * self.alpha(T_avg)

    # ---- Self-consistent iteration ----

    def _solve_operating_point(self, P_thermal, t_years):
        """Iteratively solve for self-consistent T_hj, T_cj, P_electric.

        Iteration:
          1. Start with T_hj_guess from thermal resistance + decay T_source
          2. Compute T_cj from cold-side resistance (P_radiated = P_thermal - P_electric)
          3. Compute P_electric via ZT efficiency formula
          4. Update T_cj, repeat until convergence

        Returns T_hj, T_cj, P_electric, eta, ZT_avg, V_oc, I_mp, R_int
        """
        P_thermal = np.asarray(P_thermal, dtype=float)
        t_years = np.asarray(t_years, dtype=float)

        # Initial guesses
        T_hj = self._hot_junction_T(P_thermal, t_years)
        T_cj = np.full_like(T_hj, self.T_cold_sink + 10.0) if T_hj.ndim > 0 else self.T_cold_sink + 10.0

        for _ in range(12):  # converges in ~5 iterations
            T_avg = (T_hj + T_cj) / 2.0
            dT = T_hj - T_cj

            # ZT average
            if T_hj.ndim == 0:
                ZT = self.zt_average(T_hj, T_cj)
            else:
                ZT = self._zt_average_array(T_hj, T_cj)
            ZT = np.asarray(ZT, dtype=float)

            # Efficiency (Angist formula)
            eta_c = 1.0 - T_cj / T_hj
            sqrt_ZT = np.sqrt(1.0 + ZT)
            eta = eta_c * (sqrt_ZT - 1.0) / (sqrt_ZT + T_cj / T_hj)
            eta = np.clip(eta, 0.0, 0.6)

            P_electric = P_thermal * eta
            P_radiated = P_thermal - P_electric

            # Update cold-junction temperature
            T_cj_new = self._cold_junction_T(P_radiated)
            T_cj_new = np.maximum(T_cj_new, self.T_cold_sink)

            if np.max(np.abs(T_cj_new - T_cj)) < 0.05:
                T_cj = T_cj_new
                break
            T_cj = T_cj_new

        # Module electrical parameters at converged temperatures
        T_avg = (T_hj + T_cj) / 2.0
        dT = T_hj - T_cj
        alpha_mod = self._module_seebeck(T_avg)
        R_int = self._module_resistance(T_avg)
        V_oc = alpha_mod * dT
        I_mp = V_oc / (2.0 * R_int + 1e-12)  # matched load current
        P_max_circuit = V_oc ** 2 / (4.0 * R_int + 1e-12)

        return {
            "T_hj_K": T_hj,
            "T_cj_K": T_cj,
            "ZT_avg": ZT,
            "eta_teg": eta,
            "P_electric_W": P_electric,
            "P_max_circuit_W": P_max_circuit,
            "V_oc_V": V_oc,
            "I_mp_A": I_mp,
            "R_int_ohm": R_int,
            "eta_carnot": eta_c,
        }

    def compute(self, t_years):
        """
        Parameters
        ----------
        t_years : float or array — time since launch [years]

        Returns
        -------
        dict with all outputs
        """
        t = np.asarray(t_years, dtype=float)
        t = np.maximum(t, 0.0)

        # Thermal power from Pu-238 decay
        P_thermal = self.P0 * np.exp(-ln2 * t / self.t_half)

        op = self._solve_operating_point(P_thermal, t)

        fraction_thermal = P_thermal / self.P0
        power_fraction = op["P_electric_W"] / (self.P0 * 0.065 + 1e-12)  # normalize to ~BOL electric

        return {
            "P_thermal_W": P_thermal,
            "T_hj_K": op["T_hj_K"],
            "T_cj_K": op["T_cj_K"],
            "ZT_avg": op["ZT_avg"],
            "eta_teg": op["eta_teg"],
            "eta_carnot": op["eta_carnot"],
            "P_electric_W": op["P_electric_W"],
            "P_max_circuit_W": op["P_max_circuit_W"],
            "V_oc_V": op["V_oc_V"],
            "I_mp_A": op["I_mp_A"],
            "R_int_ohm": op["R_int_ohm"],
            "fraction_thermal_remaining": fraction_thermal,
            "power_fraction": power_fraction,
        }
