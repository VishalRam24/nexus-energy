"""
EC216 -- Thermoelectric Generator (TEG) -- F2a Coupled Thermal-Electrical Model

Coupled 1D steady-state energy balance (lumped to 0D) for a thermoelectric module.

Physics:
  Seebeck voltage:    V_oc = N * alpha_avg * (T_h - T_c)
  Electrical:         V = V_oc - I * R_int;  P = I * V
  Hot-side heat:      Q_h = N * [alpha(T_h)*T_h*I - 0.5*I^2*R_te + K*(T_h-T_c)]
  Cold-side heat:     Q_c = N * [alpha(T_c)*T_c*I + 0.5*I^2*R_te + K*(T_h-T_c)]
  Energy balance:     P_elec = Q_h - Q_c
  Contact resistances couple source/sink to junction temperatures iteratively.

Temperature-dependent Bi2Te3 properties:
  alpha(T), rho(T), kappa(T) via polynomial fits.

References:
    Rowe, D.M. (ed.) (2006). Thermoelectrics Handbook. CRC Press.
    Snyder, G.J. & Toberer, E.S. (2008). Nature Materials, 7, 105-114.
    Goldsmid, H.J. (2010). Introduction to Thermoelectricity. Springer.
"""

import numpy as np


class TEG_CoupledF2a:
    """Coupled thermal-electrical TEG model with iterative junction temperature solver."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = u["N_couples"]["value"]
        self.A_te = u["A_te_m2"]["value"]           # m2 per element
        self.L_te = u["L_te_m"]["value"]             # m
        self.T_ref = u["T_ref"]["value"]             # K

        # Polynomial coefficients for temperature-dependent properties
        self.alpha_c = np.array(u["alpha_coeffs"]["value"])   # [a0, a1, a2]
        self.rho_c = np.array(u["rho_coeffs"]["value"])       # [r0, r1, r2]
        self.kappa_c = np.array(u["kappa_coeffs"]["value"])   # [k0, k1, k2]

        # Contact resistances
        self.R_contact_hot = u["R_contact_hot"]["value"]      # K/W
        self.R_contact_cold = u["R_contact_cold"]["value"]    # K/W
        self.R_elec_contact = u["R_electrical_contact"]["value"]  # ohm

        # Solver settings
        self.max_iter = int(u["max_iterations"]["value"])
        self.tol = u["convergence_tol"]["value"]

    # ------------------------------------------------------------------
    # Temperature-dependent material properties
    # ------------------------------------------------------------------
    def alpha(self, T):
        """Seebeck coefficient [V/K] at temperature T [K] (per couple, p-n pair)."""
        dT = T - self.T_ref
        return self.alpha_c[0] + self.alpha_c[1] * dT + self.alpha_c[2] * dT**2

    def rho(self, T):
        """Electrical resistivity [ohm*m] at temperature T [K]."""
        dT = T - self.T_ref
        return self.rho_c[0] + self.rho_c[1] * dT + self.rho_c[2] * dT**2

    def kappa(self, T):
        """Thermal conductivity [W/(m*K)] at temperature T [K]."""
        dT = T - self.T_ref
        return self.kappa_c[0] + self.kappa_c[1] * dT + self.kappa_c[2] * dT**2

    # ------------------------------------------------------------------
    # Module-level lumped parameters at a given average temperature
    # ------------------------------------------------------------------
    def _R_internal(self, T_avg):
        """Total module electrical resistance [ohm].

        Each couple has 2 legs (p + n):  R = 2*N * rho(T)*L/A + R_contact
        """
        r = self.rho(T_avg)
        R_elem = r * self.L_te / self.A_te
        return 2.0 * self.N * R_elem + self.R_elec_contact

    def _K_thermal(self, T_avg):
        """Module thermal conductance [W/K].

        K = 2*N * kappa(T) * A / L
        """
        k = self.kappa(T_avg)
        return 2.0 * self.N * k * self.A_te / self.L_te

    def _R_te_single(self, T_avg):
        """Single-couple electrical resistance [ohm] (both legs)."""
        r = self.rho(T_avg)
        return 2.0 * r * self.L_te / self.A_te

    def _K_single(self, T_avg):
        """Single-couple thermal conductance [W/K] (both legs)."""
        k = self.kappa(T_avg)
        return 2.0 * k * self.A_te / self.L_te

    # ------------------------------------------------------------------
    # Core solver: given junction temps and load, compute all electrical/thermal
    # ------------------------------------------------------------------
    def _compute_at_junctions(self, T_h, T_c, R_load):
        """Compute electrical and thermal quantities for given junction temperatures.

        Uses alpha evaluated at T_h for Peltier heat at hot side and alpha at T_c
        for Peltier heat at cold side. The open-circuit voltage (Seebeck EMF) is
        computed as alpha_h*T_h - alpha_c*T_c to ensure exact energy conservation:
            P_elec = Q_h - Q_c.

        Returns dict with V_oc, I, V, P, Q_h, Q_c, R_int, efficiency.
        """
        T_avg = 0.5 * (T_h + T_c)
        dT = T_h - T_c

        alpha_h = self.alpha(T_h)
        alpha_c_ = self.alpha(T_c)

        R_int = self._R_internal(T_avg)
        R_te = self._R_te_single(T_avg)
        K_s = self._K_single(T_avg)

        # Open-circuit voltage (Seebeck EMF) — consistent with Peltier terms
        # V_oc = N * (alpha_h * T_h - alpha_c * T_c) ... but this mixes Seebeck
        # with Peltier. For a lumped model the standard consistent formulation:
        # V_oc = N * alpha_avg * dT   with Q_h, Q_c using alpha_avg too.
        alpha_avg = self.alpha(T_avg)
        V_oc = self.N * alpha_avg * dT

        # Current through load
        I = V_oc / (R_int + R_load) if (R_int + R_load) > 0 else 0.0

        # Terminal voltage and power
        V = I * R_load
        P = I * V

        # Hot-side heat absorption (per module) — use alpha_avg for consistency
        # Q_h = N * [alpha_avg * T_h * I - 0.5 * I^2 * R_te + K_s * dT]
        Q_h = self.N * (alpha_avg * T_h * I - 0.5 * I**2 * R_te + K_s * dT)

        # Cold-side heat rejection (per module)
        # Q_c = N * [alpha_avg * T_c * I + 0.5 * I^2 * R_te + K_s * dT]
        Q_c = self.N * (alpha_avg * T_c * I + 0.5 * I**2 * R_te + K_s * dT)

        # Efficiency
        eta = P / Q_h if Q_h > 1e-12 else 0.0

        # Carnot limit
        eta_carnot = 1.0 - T_c / T_h if T_h > T_c else 0.0

        return {
            "V_oc": V_oc,
            "I": I,
            "V": V,
            "P": P,
            "Q_h": Q_h,
            "Q_c": Q_c,
            "R_int": R_int,
            "efficiency": eta,
            "eta_carnot": eta_carnot,
            "T_h_junction": T_h,
            "T_c_junction": T_c,
        }

    # ------------------------------------------------------------------
    # Iterative coupled solver
    # ------------------------------------------------------------------
    def solve_steady_state(self, T_hot, T_cold, R_load):
        """Solve the coupled thermal-electrical system iteratively.

        Parameters
        ----------
        T_hot : float
            Heat source temperature [K].
        T_cold : float
            Heat sink temperature [K].
        R_load : float
            External load resistance [ohm].

        Returns
        -------
        dict with keys: P, V, I, efficiency, Q_h, Q_c, V_oc, R_int,
                        T_h_junction, T_c_junction, eta_carnot, converged, iterations.
        """
        T_hot = float(T_hot)
        T_cold = float(T_cold)
        R_load = float(R_load)

        if T_hot <= T_cold:
            return {
                "P": 0.0, "V": 0.0, "I": 0.0, "efficiency": 0.0,
                "Q_h": 0.0, "Q_c": 0.0, "V_oc": 0.0, "R_int": self._R_internal(T_hot),
                "T_h_junction": T_hot, "T_c_junction": T_cold,
                "eta_carnot": 0.0, "converged": True, "iterations": 0,
            }

        # Initial guess: junction temps = source/sink temps
        T_h = T_hot
        T_c = T_cold
        converged = False

        for it in range(1, self.max_iter + 1):
            res = self._compute_at_junctions(T_h, T_c, R_load)
            Q_h = res["Q_h"]
            Q_c = res["Q_c"]

            # Update junction temps via contact resistances
            # T_h_junction = T_source - Q_h * R_contact_hot
            # T_c_junction = T_sink  + Q_c * R_contact_cold
            T_h_new = T_hot - Q_h * self.R_contact_hot
            T_c_new = T_cold + Q_c * self.R_contact_cold

            # Clamp: hot junction must stay above cold junction
            if T_h_new <= T_c_new:
                T_h_new = 0.5 * (T_hot + T_cold) + 0.1
                T_c_new = 0.5 * (T_hot + T_cold) - 0.1

            # Check convergence
            err = max(abs(T_h_new - T_h), abs(T_c_new - T_c))

            # Under-relaxation for stability
            relax = 0.3
            T_h = T_h + relax * (T_h_new - T_h)
            T_c = T_c + relax * (T_c_new - T_c)

            if err < self.tol:
                converged = True
                break

        # Final computation at converged junction temps
        res = self._compute_at_junctions(T_h, T_c, R_load)
        res["converged"] = converged
        res["iterations"] = it

        return res

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def matched_load_resistance(self, T_hot, T_cold):
        """Internal resistance at average of source/sink temps (approximate matched load)."""
        T_avg = 0.5 * (T_hot + T_cold)
        return self._R_internal(T_avg)

    def iv_curve(self, T_hot, T_cold, N_points=100):
        """Compute I-V curve by sweeping load resistance.

        Returns dict with arrays: I, V, P, R_load.
        """
        R_int_approx = self.matched_load_resistance(T_hot, T_cold)
        # Sweep from short circuit (small R) to open circuit (large R)
        R_loads = np.geomspace(R_int_approx * 0.01, R_int_approx * 100, N_points)

        I_arr = np.zeros(N_points)
        V_arr = np.zeros(N_points)
        P_arr = np.zeros(N_points)

        for i, R_L in enumerate(R_loads):
            res = self.solve_steady_state(T_hot, T_cold, R_L)
            I_arr[i] = res["I"]
            V_arr[i] = res["V"]
            P_arr[i] = res["P"]

        return {"I": I_arr, "V": V_arr, "P": P_arr, "R_load": R_loads}

    def sweep_delta_T(self, T_cold, dT_array, R_load=None):
        """Sweep over temperature differences at fixed cold-side temp.

        If R_load is None, uses matched load at each operating point.

        Returns dict with arrays: dT, P, efficiency, V_oc, I, Q_h, Q_c.
        """
        dT_array = np.asarray(dT_array, dtype=float)
        n = len(dT_array)
        out = {k: np.zeros(n) for k in ["dT", "P", "efficiency", "V_oc", "I", "Q_h", "Q_c"]}
        out["dT"] = dT_array.copy()

        for i, dT in enumerate(dT_array):
            T_hot = T_cold + dT
            if R_load is None:
                RL = self.matched_load_resistance(T_hot, T_cold)
            else:
                RL = R_load
            res = self.solve_steady_state(T_hot, T_cold, RL)
            out["P"][i] = res["P"]
            out["efficiency"][i] = res["efficiency"]
            out["V_oc"][i] = res["V_oc"]
            out["I"][i] = res["I"]
            out["Q_h"][i] = res["Q_h"]
            out["Q_c"][i] = res["Q_c"]

        return out
