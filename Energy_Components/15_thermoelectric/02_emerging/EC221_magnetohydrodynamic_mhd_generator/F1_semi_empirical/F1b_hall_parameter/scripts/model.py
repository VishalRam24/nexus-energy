"""
EC221 — MHD Generator — F1b Hall Parameter + T-Dependent Conductivity Model

Extends F1a (ideal Faraday MHD) with three corrections:

1. **Stagnation-enthalpy Q_in** (first-law correct):
   Q_in = rho * u * (cp * T + 0.5 * u^2) * w * h   [W]
   This is the true available enthalpy flux. Using 0.5*rho*u^3 (kinetic only)
   violates the first law for thermally-dominated plasma (cp*T >> 0.5*u^2).

2. **Hall parameter correction** to effective conductivity and power:
   In a Faraday channel with Hall parameter beta = omega_e * tau_e:
     sigma_eff = sigma / (1 + beta^2)
   The effective power density becomes:
     p = sigma_eff * u^2 * B^2 * K * (1 - K)
   The Hall current reduces both Jx (Faraday current) and increases J_y
   (Hall current). For a segmented Faraday channel, cross-field conduction
   is suppressed and sigma_eff replaces sigma.

   Optimal load factor with Hall effect (from generalized MHD analysis):
     K_opt = (1 - beta^2 * Kc / (1 + beta^2)) * 0.5
   For practical ranges: K_opt ≈ 0.5 is still a good approximation,
   but we expose K as an input so optimisation sweeps are possible.

3. **Temperature-dependent conductivity**:
   sigma(T) = sigma_0 * (T / T_ref)^exp_sigma
   Plasma conductivity rises with T (reduced Coulomb scattering).
   Exponent ~1.5 for Spitzer scaling in seeded plasma.

Combined efficiency metric:
   eta_electric = P_elec / Q_in      (first-law efficiency)
   eta_mhd      = K * (1 - K)        (ideal Faraday fraction — unchanged)

References:
    Rosa, R.J. (1987). Magnetohydrodynamic Energy Conversion. McGraw-Hill.
    Messerle, H.K. (1995). Magnetohydrodynamic Electrical Power Generation. Wiley.
    Miura, K. & Mori, M. (1985). AIAA J., Diagonal MHD channel analysis.
    Veefkind, A. (1977). Appl. Phys. Lett., Hall parameter effects on MHD efficiency.
"""

import numpy as np


class MHDF1b:
    """MHD Faraday generator — Hall parameter + T-dependent sigma + stagnation Q_in."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.sigma_0 = u["sigma_plasma_0"]["value"]    # S/m at T_ref
        self.T_ref = u["T_ref_K"]["value"]             # K
        self.exp_sigma = u["sigma_T_exp"]["value"]     # exponent ~1.5
        self.u0 = u["u_plasma"]["value"]               # m/s
        self.B0 = u["B_field"]["value"]                # T
        self.K0 = u["K_load"]["value"]                 # -
        self.beta0 = u["beta_hall"]["value"]           # Hall parameter
        self.L = u["channel_length"]["value"]          # m
        self.w = u["channel_width"]["value"]           # m
        self.h = u["channel_height"]["value"]          # m
        self.cp = u["cp_plasma"]["value"]              # J/(kg*K)
        self.rho = u["rho_plasma"]["value"]            # kg/m^3
        self.T_plasma_0 = u["T_plasma_K"]["value"]    # K
        self.V_ch = self.L * self.w * self.h           # m^3

    def sigma_at_T(self, T_K):
        """Temperature-dependent plasma conductivity [S/m].
        sigma(T) = sigma_0 * (T / T_ref)^exp_sigma
        Spitzer scaling for seeded combustion plasma: exp ~ 1.5
        """
        T = np.asarray(T_K, dtype=float)
        T = np.maximum(T, 500.0)   # physical floor
        return self.sigma_0 * (T / self.T_ref) ** self.exp_sigma

    def sigma_effective(self, sigma, beta):
        """Effective Faraday conductivity reduced by Hall effect.
        sigma_eff = sigma / (1 + beta^2)
        Derived from the full MHD conductivity tensor for a Faraday channel
        with open Hall circuit (segmented electrodes suppress Hall current).
        """
        sigma = np.asarray(sigma, dtype=float)
        beta_v = np.asarray(beta, dtype=float)
        return sigma / (1.0 + beta_v ** 2)

    def stagnation_enthalpy_flux(self, rho, u, T_K):
        """Stagnation enthalpy flux through channel cross-section [W].
        Q_in = rho * u * (cp * T + 0.5 * u^2) * w * h
        This is the correct first-law heat input — NOT 0.5*rho*u^3.
        For seeded combustion plasma at 2500K: cp*T ~ 3.0e6 J/kg >> 0.5*u^2 ~ 3.2e5 J/kg.
        """
        rho_v = np.asarray(rho, dtype=float)
        u_v = np.asarray(u, dtype=float)
        T_v = np.asarray(T_K, dtype=float)
        h_stag = self.cp * T_v + 0.5 * u_v ** 2   # stagnation specific enthalpy [J/kg]
        Q_in = rho_v * u_v * h_stag * self.w * self.h
        return np.maximum(Q_in, 1e-12)

    def compute(self, sigma_base, u, B, K, beta, T_plasma_K=None):
        """
        Parameters
        ----------
        sigma_base : float or array — base plasma conductivity at T_ref [S/m]
        u          : float or array — plasma velocity [m/s]
        B          : float or array — magnetic field [T]
        K          : float or array — load factor [-] (0=short, 1=open)
        beta       : float or array — Hall parameter = omega_e * tau_e
        T_plasma_K : float or array — plasma static temperature [K]; if None, use T_ref

        Returns
        -------
        dict with all outputs plus Hall-corrected and stagnation-enthalpy quantities
        """
        sigma_b = np.asarray(sigma_base, dtype=float)
        u_v = np.asarray(u, dtype=float)
        B_v = np.asarray(B, dtype=float)
        K_v = np.clip(np.asarray(K, dtype=float), 0.0, 1.0)
        beta_v = np.asarray(beta, dtype=float)

        if T_plasma_K is None:
            T_v = np.full_like(u_v, self.T_plasma_0)
        else:
            T_v = np.asarray(T_plasma_K, dtype=float)
            T_v = np.broadcast_to(T_v, np.broadcast_shapes(T_v.shape, u_v.shape)).copy() if T_v.ndim > 0 else T_v

        # --- Temperature-dependent conductivity ---
        sigma_T = self.sigma_at_T(T_v)
        # Use the T-corrected sigma (scales the base if T differs from T_ref)
        # If sigma_base is provided explicitly, it is the conductivity at T_ref;
        # we apply the T-correction on top.
        sigma_actual = sigma_b * (sigma_T / self.sigma_0)

        # --- Hall-corrected effective conductivity ---
        sigma_eff = self.sigma_effective(sigma_actual, beta_v)

        # --- Induced EMF ---
        EMF = u_v * B_v * self.h    # V (per unit length in u direction, total over h)

        # --- Power density with Hall correction [W/m^3] ---
        p_density = sigma_eff * u_v ** 2 * B_v ** 2 * K_v * (1.0 - K_v)

        # --- Total electrical power ---
        P_elec = p_density * self.V_ch

        # --- Current density (Faraday direction) ---
        J = sigma_eff * u_v * B_v * (1.0 - K_v)  # A/m^2

        # --- Hall current density [A/m^2] ---
        # J_Hall = beta * J_Faraday (perpendicular, in channel width direction)
        J_hall = beta_v * J

        # --- Stagnation enthalpy Q_in (first-law correct) ---
        Q_in = self.stagnation_enthalpy_flux(self.rho, u_v, T_v)

        # --- Efficiencies ---
        eta_mhd = K_v * (1.0 - K_v)           # ideal Faraday fraction (max 0.25)
        eta_hall = 1.0 / (1.0 + beta_v ** 2)  # Hall reduction factor
        eta_electric = P_elec / Q_in
        eta_electric = np.clip(eta_electric, 0.0, 1.0)

        # --- Optimal load factor for max power (Hall-corrected) ---
        # d/dK [sigma_eff * K * (1-K)] = 0 → K_opt = 0.5 regardless of beta
        # because sigma_eff is constant w.r.t. K; confirmed analytically.
        K_opt = np.full_like(u_v, 0.5) if u_v.ndim > 0 else 0.5

        return {
            "EMF_V": EMF,
            "J_Am2": J,
            "J_hall_Am2": J_hall,
            "sigma_actual_Sm": sigma_actual,
            "sigma_eff_Sm": sigma_eff,
            "power_density_Wm3": p_density,
            "power_elec_W": P_elec,
            "heat_input_stag_W": Q_in,
            "eta_mhd": eta_mhd,
            "eta_hall": eta_hall,
            "eta_electric": eta_electric,
            "K_optimal": K_opt,
        }
