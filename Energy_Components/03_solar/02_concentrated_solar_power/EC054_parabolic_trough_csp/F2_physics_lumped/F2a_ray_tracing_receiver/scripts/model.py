"""
EC054 -- Parabolic Trough CSP -- F2a Lumped HCE Thermal Model

Steady-state energy balance on the Heat Collection Element (HCE):
an evacuated tube receiver (absorber tube inside a glass envelope)
at the focal line of a parabolic trough collector.

Physics:
    1. Solar energy absorbed by the absorber tube:
       Q_abs = DNI * cos(theta) * IAM(theta) * f_end * W * L
               * rho_mirror * gamma * tau_glass * alpha_abs

    2. Absorber-to-glass heat transfer (annulus):
       - Radiation:  Q_rad_ag = eps_eff * sigma * pi * D_abs * L * (T_abs^4 - T_glass^4)
       - Convection: negligible if vacuum intact; add free convection if degraded

    3. Glass-to-ambient heat transfer:
       - Wind convection: Q_conv_ga = h_wind * pi * D_glass * L * (T_glass - T_amb)
       - Radiation to sky: Q_rad_ga = eps_glass * sigma * pi * D_glass * L
                                       * (T_glass^4 - T_sky^4)

    4. Useful heat to HTF:
       Q_useful = m_dot * Cp_htf * (T_out - T_in)

    5. Energy balance solved iteratively:
       - On absorber: Q_abs = Q_useful + Q_loss_abs_to_glass
       - On glass:    Q_loss_abs_to_glass = Q_loss_glass_to_amb
       - T_abs and T_glass are unknowns; solved via fixed-point iteration

HTF properties (Therminol VP-1) from polynomial fits (Forristall 2003).

References:
    Forristall (2003), NREL/TP-550-34169.
    Burkholder & Kutscher (2009), NREL/TP-550-45633.
    Dudley et al. (1994), SAND94-1884.
"""

import numpy as np

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m2*K4)]


class ParabolicTroughF2a:
    """Lumped steady-state HCE thermal model for parabolic trough CSP."""

    def __init__(self, params: dict):
        u = params["unit"]
        htf = params["htf"]
        iam = params["iam_coefficients"]
        wind = params["wind"]

        # Geometry
        self.W = u["W_aperture"]["value"]
        self.L = u["L_collector"]["value"]
        self.f_L = u["focal_length"]["value"]
        self.D_abs_o = u["D_abs_outer"]["value"]
        self.D_abs_i = u["D_abs_inner"]["value"]
        self.D_glass_o = u["D_glass_outer"]["value"]
        self.D_glass_i = u["D_glass_inner"]["value"]

        # Optical / surface properties
        self.alpha_abs = u["alpha_abs"]["value"]
        self.eps_abs_ref = u["eps_abs_ref"]["value"]
        self.eps_glass = u["eps_glass"]["value"]
        self.rho_mirror = u["mirror_reflectivity"]["value"]
        self.gamma = u["intercept_factor"]["value"]
        self.tau_glass = u["glass_transmittance"]["value"]
        self.k_abs = u["k_abs"]["value"]
        self.vacuum_intact = u["vacuum_intact"]["value"]

        # IAM polynomial
        self.iam_c1 = iam["c1"]["value"]
        self.iam_c2 = iam["c2"]["value"]

        # Wind / ambient
        self.h_wind_default = wind["h_wind_default"]["value"]
        self.T_sky_offset = wind["T_sky_offset"]["value"]

        # HTF polynomial coefficients (Therminol VP-1, T in degC)
        # Cp [J/(kg*K)] = c0 + c1*T + c2*T^2
        self.Cp_c = np.array(htf["Cp_coeffs"])
        # mu [Pa*s] = c0 + c1*T  (simplified linear in operating range)
        self.mu_c = np.array(htf["mu_coeffs"])
        # rho [kg/m3] = c0 + c1*T + c2*T^2
        self.rho_c = np.array(htf["rho_coeffs"])
        # k_htf [W/(m*K)] = c0 + c1*T + c2*T^2
        self.k_htf_c = np.array(htf["k_htf_coeffs"])

        # Derived
        self.A_aperture = self.W * self.L
        # Effective annulus emissivity (concentric cylinders)
        # 1/eps_eff = 1/eps_abs + (D_abs/D_glass)*(1/eps_glass - 1)
        ratio = self.D_abs_o / self.D_glass_i
        self.eps_annulus = 1.0 / (1.0 / self.eps_abs_ref + ratio * (1.0 / self.eps_glass - 1.0))

    # ------------------------------------------------------------------ #
    # HTF properties (Therminol VP-1)
    # ------------------------------------------------------------------ #
    def htf_Cp(self, T_C):
        """Specific heat [J/(kg*K)] as function of temperature [degC]."""
        T = np.asarray(T_C, dtype=float)
        return self.Cp_c[0] + self.Cp_c[1] * T + self.Cp_c[2] * T**2

    def htf_rho(self, T_C):
        """Density [kg/m3]."""
        T = np.asarray(T_C, dtype=float)
        return self.rho_c[0] + self.rho_c[1] * T + self.rho_c[2] * T**2

    def htf_mu(self, T_C):
        """Dynamic viscosity [Pa*s] (simplified)."""
        T = np.asarray(T_C, dtype=float)
        return np.maximum(self.mu_c[0] + self.mu_c[1] * T, 1e-6)

    def htf_k(self, T_C):
        """Thermal conductivity [W/(m*K)]."""
        T = np.asarray(T_C, dtype=float)
        return self.k_htf_c[0] + self.k_htf_c[1] * T + self.k_htf_c[2] * T**2

    # ------------------------------------------------------------------ #
    # Optics
    # ------------------------------------------------------------------ #
    def iam(self, theta_deg):
        """Incidence Angle Modifier (Dudley et al. polynomial)."""
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        val = np.cos(theta_rad) + self.iam_c1 * theta + self.iam_c2 * theta**2
        return np.clip(val, 0.0, 1.0)

    def end_loss_factor(self, theta_deg):
        """End loss factor for finite collector length."""
        theta = np.asarray(theta_deg, dtype=float)
        theta_rad = np.radians(theta)
        tan_theta = np.where(theta < 85.0, np.tan(theta_rad), 100.0)
        f_end = 1.0 - self.f_L * tan_theta / self.L
        return np.clip(f_end, 0.0, 1.0)

    def Q_absorbed(self, dni, theta_deg):
        """
        Solar power absorbed by the absorber tube [W].

        Q_abs = DNI * cos(theta) * W * L * rho * gamma * tau * alpha * IAM * f_end
        Note: IAM already includes cos(theta), so we use IAM directly (not cos*IAM).
        Actually IAM = cos(theta) + correction terms, so it replaces cos(theta).
        """
        G = np.asarray(dni, dtype=float)
        iam_val = self.iam(theta_deg)
        f_end = self.end_loss_factor(theta_deg)
        eta_opt = self.rho_mirror * self.gamma * self.tau_glass * self.alpha_abs
        return G * self.W * self.L * eta_opt * iam_val * f_end  # W

    # ------------------------------------------------------------------ #
    # Heat transfer
    # ------------------------------------------------------------------ #
    def Q_annulus_radiation(self, T_abs_K, T_glass_K):
        """Radiation from absorber to glass envelope [W]."""
        return (self.eps_annulus * SIGMA * np.pi * self.D_abs_o * self.L
                * (T_abs_K**4 - T_glass_K**4))

    def Q_annulus_convection(self, T_abs_K, T_glass_K):
        """
        Free convection in annulus [W].
        If vacuum intact, return 0. Otherwise use a simplified correlation.
        """
        if self.vacuum_intact:
            return 0.0
        # Simplified: h_nat ~ 5 W/(m2*K) for degraded vacuum (air at low pressure)
        h_nat = 5.0
        return h_nat * np.pi * self.D_abs_o * self.L * (T_abs_K - T_glass_K)

    def Q_glass_convection(self, T_glass_K, T_amb_K, h_wind=None):
        """Wind convection from glass envelope to ambient [W]."""
        h = h_wind if h_wind is not None else self.h_wind_default
        return h * np.pi * self.D_glass_o * self.L * (T_glass_K - T_amb_K)

    def Q_glass_radiation(self, T_glass_K, T_sky_K):
        """Radiation from glass envelope to sky [W]."""
        return (self.eps_glass * SIGMA * np.pi * self.D_glass_o * self.L
                * (T_glass_K**4 - T_sky_K**4))

    def htf_internal_h(self, T_htf_C, m_dot):
        """
        Internal convection coefficient for HTF inside absorber tube [W/(m2*K)].
        Uses Dittus-Boelter correlation: Nu = 0.023 * Re^0.8 * Pr^0.4
        """
        T = np.asarray(T_htf_C, dtype=float)
        rho = self.htf_rho(T)
        mu = self.htf_mu(T)
        Cp = self.htf_Cp(T)
        k = self.htf_k(T)

        A_flow = np.pi / 4.0 * self.D_abs_i**2
        v = m_dot / (rho * A_flow)
        Re = rho * v * self.D_abs_i / mu
        Pr = Cp * mu / k

        # Dittus-Boelter (heating)
        Nu = 0.023 * np.maximum(Re, 1.0)**0.8 * np.maximum(Pr, 0.1)**0.4
        return Nu * k / self.D_abs_i

    # ------------------------------------------------------------------ #
    # Main solver: iterative energy balance
    # ------------------------------------------------------------------ #
    def solve(self, dni, theta_deg, T_htf_in_C, m_dot,
              T_amb_C=25.0, h_wind=None, max_iter=100, tol=0.1):
        """
        Solve the HCE energy balance for a single operating point.

        Parameters
        ----------
        dni          : Direct Normal Irradiance [W/m2]
        theta_deg    : Incidence angle [deg]
        T_htf_in_C   : HTF inlet temperature [degC]
        m_dot        : HTF mass flow rate [kg/s]
        T_amb_C      : Ambient temperature [degC]
        h_wind       : Wind convection coefficient [W/(m2*K)] (optional)
        max_iter     : Maximum iterations
        tol          : Temperature convergence tolerance [K]

        Returns
        -------
        dict with: Q_useful_W, Q_abs_W, Q_loss_W, T_htf_out_C, T_abs_C,
                   T_glass_C, eta_thermal, eta_optical, h_htf, converged
        """
        dni = float(dni)
        theta_deg = float(theta_deg)
        T_in = float(T_htf_in_C)
        m_dot = float(m_dot)
        T_amb = float(T_amb_C)
        h_w = float(h_wind) if h_wind is not None else self.h_wind_default

        T_amb_K = T_amb + 273.15
        T_sky_K = T_amb_K - self.T_sky_offset

        # Solar absorbed
        Q_abs = float(self.Q_absorbed(dni, theta_deg))

        if Q_abs <= 0.0 or m_dot <= 0.0:
            return {
                "Q_useful_W": 0.0, "Q_abs_W": Q_abs, "Q_loss_W": 0.0,
                "T_htf_out_C": T_in, "T_abs_C": T_in, "T_glass_C": T_amb,
                "eta_thermal": 0.0, "eta_optical": 0.0,
                "h_htf": 0.0, "converged": True,
            }

        # Initial guesses
        T_htf_avg_C = T_in + 10.0  # will refine
        T_abs_K = T_htf_avg_C + 273.15 + 5.0
        T_glass_K = T_amb_K + 20.0

        converged = False
        for iteration in range(max_iter):
            T_abs_K_old = T_abs_K
            T_glass_K_old = T_glass_K

            # HTF properties at average temperature
            Cp = float(self.htf_Cp(T_htf_avg_C))
            h_htf = float(self.htf_internal_h(T_htf_avg_C, m_dot))

            # Heat transfer from absorber to HTF
            # Q_useful = h_htf * pi * D_abs_i * L * (T_abs - T_htf_avg)
            # Q_useful = m_dot * Cp * (T_out - T_in)
            R_htf = 1.0 / (h_htf * np.pi * self.D_abs_i * self.L)

            # Absorber-to-glass losses
            Q_ag_rad = float(self.Q_annulus_radiation(T_abs_K, T_glass_K))
            Q_ag_conv = float(self.Q_annulus_convection(T_abs_K, T_glass_K))
            Q_ag = Q_ag_rad + Q_ag_conv

            # Glass-to-ambient losses
            Q_ga_conv = float(self.Q_glass_convection(T_glass_K, T_amb_K, h_w))
            Q_ga_rad = float(self.Q_glass_radiation(T_glass_K, T_sky_K))
            Q_ga = Q_ga_conv + Q_ga_rad

            # Energy balance on absorber: Q_abs = Q_useful + Q_ag
            Q_useful = Q_abs - Q_ag
            Q_useful = max(Q_useful, 0.0)

            # HTF outlet temperature
            if m_dot > 0 and Cp > 0:
                T_out_C = T_in + Q_useful / (m_dot * Cp)
            else:
                T_out_C = T_in

            # Update average HTF temperature
            T_htf_avg_C = (T_in + T_out_C) / 2.0

            # Update absorber temperature from HTF side
            # T_abs = T_htf_avg + Q_useful * R_htf (in K)
            T_htf_avg_K = T_htf_avg_C + 273.15
            T_abs_K_new = T_htf_avg_K + Q_useful * R_htf

            # Update glass temperature from glass energy balance
            # Q_ag = Q_ga  =>  solve for T_glass
            # Use linearized radiation for glass update:
            # Q_ga_conv = h_w * pi * D_glass_o * L * (T_glass - T_amb)
            # Q_ga_rad  ~ eps_glass * sigma * pi * D_glass_o * L * 4 * T_amb^3 * (T_glass - T_sky)
            # Simplified: T_glass from Q_ag = Q_ga
            h_rad_glass = self.eps_glass * SIGMA * 4.0 * T_amb_K**3
            h_total_glass = (h_w + h_rad_glass) * np.pi * self.D_glass_o * self.L
            if h_total_glass > 0:
                # Q_ag = h_total_glass * (T_glass - T_amb_K)  approximately
                T_glass_K_new = T_amb_K + Q_ag / h_total_glass
            else:
                T_glass_K_new = T_amb_K + 20.0

            # Relaxation for stability
            alpha_relax = 0.5
            T_abs_K = alpha_relax * T_abs_K_new + (1.0 - alpha_relax) * T_abs_K
            T_glass_K = alpha_relax * T_glass_K_new + (1.0 - alpha_relax) * T_glass_K

            # Ensure physical bounds
            T_abs_K = max(T_abs_K, T_htf_avg_K)
            T_glass_K = max(T_glass_K, T_amb_K)
            T_glass_K = min(T_glass_K, T_abs_K)

            if (abs(T_abs_K - T_abs_K_old) < tol and
                    abs(T_glass_K - T_glass_K_old) < tol):
                converged = True
                break

        # Final recompute with converged temperatures
        Q_ag_rad = float(self.Q_annulus_radiation(T_abs_K, T_glass_K))
        Q_ag_conv = float(self.Q_annulus_convection(T_abs_K, T_glass_K))
        Q_loss = Q_ag_rad + Q_ag_conv
        Q_useful = max(Q_abs - Q_loss, 0.0)

        if m_dot > 0 and Cp > 0:
            T_out_C = T_in + Q_useful / (m_dot * Cp)
        else:
            T_out_C = T_in

        # Efficiencies
        Q_incident = dni * self.A_aperture  # W
        if Q_incident > 0:
            eta_optical = Q_abs / Q_incident
            eta_thermal = Q_useful / Q_incident
        else:
            eta_optical = 0.0
            eta_thermal = 0.0

        return {
            "Q_useful_W": Q_useful,
            "Q_abs_W": Q_abs,
            "Q_loss_W": Q_loss,
            "T_htf_out_C": T_out_C,
            "T_abs_C": T_abs_K - 273.15,
            "T_glass_C": T_glass_K - 273.15,
            "eta_thermal": min(eta_thermal, 1.0),
            "eta_optical": min(eta_optical, 1.0),
            "h_htf": h_htf,
            "converged": converged,
        }

    def solve_array(self, dni_arr, theta_arr, T_in_arr, m_dot_arr,
                    T_amb_arr=None, h_wind_arr=None):
        """
        Vectorised wrapper: solve for arrays of operating points.
        Returns dict of arrays.
        """
        dni_arr = np.atleast_1d(np.asarray(dni_arr, dtype=float))
        theta_arr = np.atleast_1d(np.asarray(theta_arr, dtype=float))
        T_in_arr = np.atleast_1d(np.asarray(T_in_arr, dtype=float))
        m_dot_arr = np.atleast_1d(np.asarray(m_dot_arr, dtype=float))

        n = len(dni_arr)
        if T_amb_arr is None:
            T_amb_arr = np.full(n, 25.0)
        else:
            T_amb_arr = np.atleast_1d(np.asarray(T_amb_arr, dtype=float))
        if h_wind_arr is None:
            h_wind_arr = [None] * n
        else:
            h_wind_arr = np.atleast_1d(np.asarray(h_wind_arr, dtype=float))

        # Broadcast scalars
        if len(theta_arr) == 1:
            theta_arr = np.full(n, theta_arr[0])
        if len(T_in_arr) == 1:
            T_in_arr = np.full(n, T_in_arr[0])
        if len(m_dot_arr) == 1:
            m_dot_arr = np.full(n, m_dot_arr[0])
        if len(T_amb_arr) == 1:
            T_amb_arr = np.full(n, T_amb_arr[0])

        keys = ["Q_useful_W", "Q_abs_W", "Q_loss_W", "T_htf_out_C",
                "T_abs_C", "T_glass_C", "eta_thermal", "eta_optical",
                "h_htf", "converged"]
        results = {k: np.zeros(n) for k in keys}

        for i in range(n):
            hw = h_wind_arr[i] if h_wind_arr[i] is not None else None
            r = self.solve(dni_arr[i], theta_arr[i], T_in_arr[i],
                           m_dot_arr[i], T_amb_arr[i], hw)
            for k in keys:
                results[k][i] = r[k]

        return results
