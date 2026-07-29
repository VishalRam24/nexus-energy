"""
EC075 — Finned-Tube Heat Exchanger — F1b Fouling + Property Corrections

Extends F1a (cross-flow e-NTU) with:

1. Fouling resistance correction (TEMA):
       1/U_fouled = 1/(eta_o * U_clean) + Rf_air/eta_o + Rf_tube

   where eta_o is the overall surface efficiency accounting for fin efficiency.
   For finned-tube surfaces the air-side fouling is referenced to total external area.

2. Temperature-dependent property corrections via Sieder-Tate correlation:
       h_corr = h_ref * (Re/Re_ref)^n_Re * (Pr/Pr_ref)^n_Pr * (mu_bulk/mu_wall)^0.14

   Applied to update U_clean when flow rate or fluid temperature deviates from design.

3. Part-load LMTD correction:
       U_eff(m_dot) = U_clean * (m_dot / m_dot_ref)^n_Re
   accounts for reduced convective coefficient at lower flow rates.

Cross-flow effectiveness (one fluid unmixed, Incropera eq. 11.32):
    eps = 1 - exp( (NTU^0.22 / C_r) * (exp(-C_r * NTU^0.78) - 1) )

References:
    Incropera & DeWitt (2006). Fundamentals of Heat and Mass Transfer, ch.11.
    Kays & London (1984). Compact Heat Exchanger Design. McGraw-Hill.
    Shah & Sekulic (2003). Fundamentals of Heat Exchanger Design. Wiley.
    Webb (1994). Principles of Enhanced Heat Transfer. Wiley.
    TEMA Standards, 10th ed. (2007). Fouling resistance tables.
    Sieder & Tate (1936). Ind. Eng. Chem. 28(12), 1429-1435.
"""

import numpy as np


class FinnedTubeHXF1b:
    """Cross-flow finned-tube HX: e-NTU with fouling and property corrections."""

    def __init__(self, params: dict):
        u = params["unit"]
        fp = params["fluid_properties"]

        self.U_clean = u["U_clean"]["value"]              # W/m2K (overall, referred to ext area)
        self.A = u["A"]["value"]                          # m2 (total external finned area)
        self.eta_o = u["eta_fin"]["value"]                # Overall surface efficiency
        self.cp_h = u["cp_hot"]["value"]                  # J/kgK
        self.cp_c = u["cp_cold"]["value"]                 # J/kgK
        self.Rf_tube_default = u["Rf_tube_default"]["value"]
        self.Rf_air_default = u["Rf_air_default"]["value"]
        self.Re_ref = u["Re_ref"]["value"]
        self.Pr_ref = u["Pr_ref"]["value"]
        self.mu_ref = u["mu_ref"]["value"]
        self.n_Re = u["nusselt_exponent_Re"]["value"]
        self.n_Pr = u["nusselt_exponent_Pr"]["value"]

        self.n_mu = fp["mu_correction_exponent"]["value"]   # Sieder-Tate exponent
        self.T_design = fp["T_design_degC"]["value"]        # degC

    # ------------------------------------------------------------------
    # Property correction utilities
    # ------------------------------------------------------------------

    @staticmethod
    def water_viscosity(T_c):
        """
        Approximate dynamic viscosity of water (Pa.s) vs temperature (degC).
        Fit: mu ≈ 2.414e-5 * 10^(247.8 / (T_K - 140))  (Vogel equation)
        Valid 0-100 degC.
        """
        T_c = np.asarray(T_c, dtype=float)
        T_K = np.clip(T_c + 273.15, 274.0, 380.0)
        return 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))

    @staticmethod
    def water_prandtl(T_c):
        """
        Approximate Prandtl number of water vs temperature (degC).
        Fit to NIST data: Pr ~ 13.7 at 0C, 6.99 at 20C, 4.34 at 40C, 1.75 at 100C.
        Polynomial: Pr ≈ a0 + a1*T + a2*T^2
        """
        T_c = np.asarray(T_c, dtype=float)
        T_c = np.clip(T_c, 0.0, 100.0)
        return 13.7 - 0.119 * T_c + 4.8e-4 * T_c ** 2

    def U_effective(self, m_dot_hot, T_h_in, T_h_out_approx=None,
                    Rf_tube=None, Rf_air=None):
        """
        Effective U value accounting for:
          - Part-load Re scaling: U_h ~ (m_dot / m_dot_ref)^n_Re
          - Sieder-Tate viscosity/Prandtl correction
          - Fouling (tube side + air side via fin efficiency)

        U_fouled = 1 / (1/(eta_o * U_eff) + Rf_air/eta_o + Rf_tube)

        Args:
            m_dot_hot: Hot-side mass flow rate (kg/s)
            T_h_in: Hot-side inlet temperature (degC)
            T_h_out_approx: Approximate hot-side outlet (for wall temp estimate, degC).
                            If None, uses T_h_in as a conservative estimate.
            Rf_tube: Tube-side fouling resistance (m2K/W)
            Rf_air: Air-side fouling resistance (m2K/W)
        """
        if Rf_tube is None:
            Rf_tube = self.Rf_tube_default
        if Rf_air is None:
            Rf_air = self.Rf_air_default

        Rf_tube = np.asarray(Rf_tube, dtype=float)
        Rf_air = np.asarray(Rf_air, dtype=float)
        m_dot_h = np.asarray(m_dot_hot, dtype=float)
        T_h = np.asarray(T_h_in, dtype=float)

        # Part-load Re scaling factor (based on hot-side, dominant resistance for finned-tube)
        m_ref_est = 1.0  # kg/s (nominal design, simplified; actual Re proportional to m_dot)
        re_factor = np.maximum(m_dot_h / m_ref_est, 1e-3) ** self.n_Re

        # Sieder-Tate property correction: (mu_bulk/mu_wall)^0.14 * (Pr_bulk/Pr_ref)^n_Pr
        # Bulk at T_h_in; wall estimated midway (rough)
        T_wall = T_h - 5.0 if T_h_out_approx is None else (T_h + np.asarray(T_h_out_approx)) / 2.0
        mu_bulk = self.water_viscosity(T_h)
        mu_wall = self.water_viscosity(T_wall)
        Pr_bulk = self.water_prandtl(T_h)

        mu_corr = (mu_bulk / np.maximum(mu_wall, 1e-6)) ** self.n_mu
        pr_corr = (Pr_bulk / self.Pr_ref) ** self.n_Pr

        # Corrected clean U
        U_eff = self.U_clean * re_factor * mu_corr * pr_corr

        # Fouled: 1/U_fouled = 1/(eta_o * U_eff) + Rf_air/eta_o + Rf_tube
        R_total = 1.0 / (self.eta_o * U_eff) + Rf_air / self.eta_o + Rf_tube
        U_fouled = 1.0 / np.maximum(R_total, 1e-12)

        return U_fouled, U_eff

    # ------------------------------------------------------------------
    # Cross-flow effectiveness (one fluid unmixed)
    # ------------------------------------------------------------------

    @staticmethod
    def _effectiveness_crossflow(NTU, C_r):
        """
        Cross-flow e-NTU, one fluid unmixed (Incropera eq. 11.32):
            eps = 1 - exp((NTU^0.22 / C_r) * (exp(-C_r * NTU^0.78) - 1))
        Degenerates to parallel/counter flow for C_r -> 0.
        """
        NTU = np.asarray(NTU, dtype=float)
        C_r = np.asarray(C_r, dtype=float)

        C_r_safe = np.maximum(C_r, 1e-10)
        inner = np.exp(-C_r_safe * NTU ** 0.78) - 1.0
        eps = 1.0 - np.exp((NTU ** 0.22 / C_r_safe) * inner)
        return np.clip(eps, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main predict method
    # ------------------------------------------------------------------

    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                Rf_tube=None, Rf_air=None):
        """
        Parameters
        ----------
        T_h_in, T_c_in   : float or array [degC]
        m_dot_hot         : float or array [kg/s] — hot-side (liquid)
        m_dot_cold        : float or array [kg/s] — cold-side (air)
        Rf_tube           : float or array [m2K/W] — tube-side fouling
        Rf_air            : float or array [m2K/W] — air-side fouling

        Returns
        -------
        dict : Q_kw, T_h_out, T_c_out, effectiveness, ntu, U_fouled,
               U_effective_clean, effectiveness_reduction, cleanliness_factor
        """
        T_h_in = np.asarray(T_h_in, dtype=float)
        T_c_in = np.asarray(T_c_in, dtype=float)
        m_dot_h = np.asarray(m_dot_hot, dtype=float)
        m_dot_c = np.asarray(m_dot_cold, dtype=float)

        C_h = m_dot_h * self.cp_h
        C_c = m_dot_c * self.cp_c
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        zero_flow = C_min < 1e-10
        C_min_s = np.where(zero_flow, 1.0, C_min)
        C_max_s = np.where(zero_flow, 1.0, C_max)
        C_h_s = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_s = np.where(C_c < 1e-10, 1.0, C_c)

        C_r = C_min_s / C_max_s

        # Get effective U and fouled U
        U_fouled, U_eff = self.U_effective(m_dot_h, T_h_in,
                                           T_h_out_approx=None,
                                           Rf_tube=Rf_tube, Rf_air=Rf_air)

        # Clean U for comparison (no fouling, design flow)
        U_clean_only, _ = self.U_effective(m_dot_h, T_h_in,
                                           T_h_out_approx=None,
                                           Rf_tube=0.0, Rf_air=0.0)

        NTU_fouled = U_fouled * self.A / C_min_s
        NTU_clean = U_clean_only * self.A / C_min_s

        eps_fouled = self._effectiveness_crossflow(NTU_fouled, C_r)
        eps_clean = self._effectiveness_crossflow(NTU_clean, C_r)

        dT_max = T_h_in - T_c_in
        Q_W = eps_fouled * C_min * dT_max
        Q_W = np.maximum(Q_W, 0.0)

        T_h_out = T_h_in - Q_W / C_h_s
        T_c_out = T_c_in + Q_W / C_c_s

        eps_reduction = np.where(
            eps_clean > 1e-10,
            (eps_clean - eps_fouled) / eps_clean,
            0.0,
        )

        cleanliness_factor = np.where(U_clean_only > 1e-10,
                                       U_fouled / U_clean_only, 1.0)

        # Zero-flow override
        Q_W = np.where(zero_flow, 0.0, Q_W)
        T_h_out = np.where(zero_flow, T_h_in, T_h_out)
        T_c_out = np.where(zero_flow, T_c_in, T_c_out)
        eps_fouled = np.where(zero_flow, 0.0, eps_fouled)
        NTU_fouled = np.where(zero_flow, 0.0, NTU_fouled)

        return {
            "Q_kw": Q_W / 1000.0,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "effectiveness": eps_fouled,
            "ntu": NTU_fouled,
            "U_fouled": U_fouled,
            "U_effective_clean": U_clean_only,
            "effectiveness_reduction": eps_reduction,
            "cleanliness_factor": cleanliness_factor,
        }
