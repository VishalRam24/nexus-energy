"""
EC049 — Multi-Junction Concentrator PV (CPV) — F1b Two-Diode + Thermal Model

Two-diode equation extended with:
  1. Concentration-ratio scaling of photocurrent: I_ph = C * (G / G_ref) * I_ph_ref
  2. Logarithmic Voc gain with concentration: Voc ~ n1*Vt*ln(C)
  3. Faiman NOCT thermal model adapted for CPV (higher NOCT due to concentration)
  4. Series resistance penalty at high current (dominant at high C)

Physics:
    C = G_direct / G_ref          (concentration ratio)
    I_ph = C * I_ph_ref           (photocurrent scales linearly with C)
    I = I_ph - I_01*(exp((V+I*Rs)/(n1*Vt))-1) - I_02*(exp((V+I*Rs)/(n2*Vt))-1) - (V+I*Rs)/Rsh
    T_cell = T_amb + C * G_ref * (NOCT_CPV - 20) / (800 * C_noct)

Multi-junction context (GaInP/GaInAs/Ge):
  - Three junctions in series; limiting junction governs short-circuit current
  - Top sub-cell (GaInP, Eg~1.85 eV): governs Jsc in most CPV conditions
  - Two-diode model with n1=1 (diffusion) and n2=2 (recombination)
  - Much higher Voc than Si (~2.6 V per cell at 1-sun, scales logarithmically with C)
  - Efficiency peaks 35-45% at 200-500× concentration

References:
    Araki & Yamaguchi (2003). "Two-diode equivalent circuit parameters for real CPV."
    Solar Energy Materials and Solar Cells 75(3), 457-466.
    Yamaguchi et al. (2005). "Multi-junction III-V solar cells." Solar Energy 79(1), 78-85.
    King et al. (2012). "Solar cell generations over 40% efficiency." Prog. Photovolt. 20, 801.
    Cotal et al. (2009). "III-V multijunction solar cells for concentrating photovoltaics."
    Energy & Environ. Sci. 2, 174-192.
"""

import numpy as np
from scipy.optimize import brentq


class MJCPVf1b:
    """Multi-junction CPV — two-diode + Faiman thermal model."""

    def __init__(self, params: dict):
        mod = params["module"]

        self.N_s = mod["cells_in_series"]["value"]
        self.area = mod["lens_area"]["value"]           # m2 (primary optics area)
        self.I_ph_ref = mod["I_ph_ref"]["value"]        # A at 1-sun STC (bottom junction limited)
        self.I_01 = mod["I_01"]["value"]
        self.I_02 = mod["I_02"]["value"]
        self.n1 = mod["n1"]["value"]
        self.n2 = mod["n2"]["value"]
        self.Rs = mod["Rs"]["value"]
        self.Rsh_ref = mod["Rsh_ref"]["value"]
        self.T_ref_c = mod["T_ref"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]       # A/K
        self.Eg_eff = mod["Eg_eff"]["value"]            # Effective/average bandgap [eV]
        self.optical_efficiency = mod["optical_efficiency"]["value"]  # lens + optics

        # Thermal
        self.NOCT_cpv = mod["NOCT_cpv"]["value"]        # degC (at reference C_noct concentration)
        self.C_noct = mod["C_noct"]["value"]            # concentration at NOCT test

        # Physical constants
        self.k = 1.380649e-23
        self.q = 1.602176634e-19
        self.G_ref = 1000.0  # W/m2 DNI at STC

    def concentration_ratio(self, dni):
        """
        Geometric concentration ratio.
        C = eta_opt * A_lens / A_cell, simplified as proportional to DNI.
        For prediction, we express C = dni / G_ref * C_geo
        but in this model we treat dni as the effective irradiance delivered to cell:
        G_cell = optical_efficiency * dni
        and keep C = G_cell / G_ref for scaling.
        """
        G = np.asarray(dni, dtype=float)
        return np.maximum(G / self.G_ref, 0.0)

    def cell_temperature(self, dni, T_amb_c):
        """
        Modified Faiman NOCT model for CPV.
        Higher heat density at concentration → NOCT scales with C/C_noct.
        T_cell = T_amb + C * (G_ref / C_noct) * (NOCT_cpv - 20) / 800

        RATIONALE: At reference concentration C_noct, this recovers the standard
        NOCT formula. At higher C, the thermal load scales proportionally.
        Reference: Cotal et al. (2009) and Yamaguchi et al. (2005).
        """
        G = np.asarray(dni, dtype=float)
        T_amb = np.asarray(T_amb_c, dtype=float)
        C = self.concentration_ratio(G)
        dT_noct_ref = (self.NOCT_cpv - 20.0) / 800.0
        T_cell = T_amb + C * self.G_ref * dT_noct_ref / self.C_noct
        return T_cell

    def _thermal_voltage(self, T_kelvin):
        return self.N_s * self.k * T_kelvin / self.q

    def _calc_params(self, dni, cell_temp_c):
        G = np.atleast_1d(np.asarray(dni, dtype=float))
        T_c = np.atleast_1d(np.asarray(cell_temp_c, dtype=float))
        T_K = T_c + 273.15
        T_ref_K = self.T_ref_c + 273.15

        # Concentration ratio
        C = np.maximum(G / self.G_ref, 0.0)

        # Photocurrent: scales linearly with concentration (DNI delivered to cell)
        I_ph = C * (self.I_ph_ref + self.alpha_sc * (T_c - self.T_ref_c))
        I_ph = np.maximum(I_ph, 0.0)

        # Thermal voltage
        Vt = self._thermal_voltage(T_K)

        # Saturation currents: temperature scaling
        temp_ratio = T_K / T_ref_K
        exp_factor = (self.Eg_eff * self.q / self.k) * (1.0 / T_ref_K - 1.0 / T_K)
        I_01 = self.I_01 * temp_ratio**3 * np.exp(exp_factor)
        I_02 = self.I_02 * temp_ratio**(3.0 / 2.0) * np.exp(exp_factor / 2.0)

        # Shunt resistance: inversely proportional to concentration
        Rsh = self.Rsh_ref * (1.0 / np.maximum(C, 0.001))

        return I_ph, I_01, I_02, Vt, Rsh

    def _iv_equation(self, I, V, I_ph, I_01, I_02, Vt, Rsh):
        Vd = V + I * self.Rs
        return (I_ph
                - I_01 * (np.exp(np.clip(Vd / (self.n1 * Vt), -100, 100)) - 1.0)
                - I_02 * (np.exp(np.clip(Vd / (self.n2 * Vt), -100, 100)) - 1.0)
                - Vd / Rsh - I)

    def _solve_I_at_V(self, V, I_ph, I_01, I_02, Vt, Rsh):
        try:
            I = brentq(self._iv_equation, -0.1, float(I_ph) + 0.1,
                       args=(V, I_ph, I_01, I_02, Vt, Rsh), xtol=1e-10)
            return max(I, 0.0)
        except (ValueError, RuntimeError):
            return 0.0

    def mpp(self, dni, T_amb_c):
        """
        Maximum power point for CPV cell/module.

        Parameters
        ----------
        dni       : W/m2, Direct Normal Irradiance incident on primary optics
        T_amb_c   : degC, ambient temperature
        """
        G = np.atleast_1d(np.asarray(dni, dtype=float))
        T_amb = np.atleast_1d(np.asarray(T_amb_c, dtype=float))
        if G.shape != T_amb.shape:
            G, T_amb = np.broadcast_arrays(G, T_amb)

        T_cell = self.cell_temperature(G, T_amb)

        v_mp = np.zeros_like(G)
        i_mp = np.zeros_like(G)
        p_mp = np.zeros_like(G)
        v_oc = np.zeros_like(G)
        i_sc = np.zeros_like(G)

        I_ph_arr, I_01_arr, I_02_arr, Vt_arr, Rsh_arr = self._calc_params(G, T_cell)

        for idx in range(G.size):
            g = G.flat[idx]
            if g < 1.0:
                continue

            iph = float(I_ph_arr.flat[idx])
            i01 = float(I_01_arr.flat[idx])
            i02 = float(I_02_arr.flat[idx])
            vt = float(Vt_arr.flat[idx])
            rsh = float(Rsh_arr.flat[idx])

            i_sc_val = self._solve_I_at_V(0.0, iph, i01, i02, vt, rsh)

            try:
                v_oc_est = self.n1 * vt * np.log(max(iph / max(i01, 1e-30), 1.0) + 1.0)
                v_oc_val = brentq(
                    lambda v: self._iv_equation(0.0, v, iph, i01, i02, vt, rsh),
                    0.0, v_oc_est * 1.5, xtol=1e-8)
            except (ValueError, RuntimeError):
                v_oc_val = 0.0

            n_pts = 200
            V_sweep = np.linspace(0, v_oc_val, n_pts)
            I_sweep = np.array([self._solve_I_at_V(v, iph, i01, i02, vt, rsh) for v in V_sweep])
            P_sweep = V_sweep * I_sweep
            idx_max = np.argmax(P_sweep)

            v_mp.flat[idx] = V_sweep[idx_max]
            i_mp.flat[idx] = I_sweep[idx_max]
            p_mp.flat[idx] = P_sweep[idx_max]
            v_oc.flat[idx] = v_oc_val
            i_sc.flat[idx] = i_sc_val

        fill_factor = np.where(v_oc * i_sc > 0, p_mp / (v_oc * i_sc), 0.0)

        return {
            "i_mp": np.squeeze(i_mp),
            "v_mp": np.squeeze(v_mp),
            "p_mp": np.squeeze(p_mp),
            "i_sc": np.squeeze(i_sc),
            "v_oc": np.squeeze(v_oc),
            "fill_factor": np.squeeze(fill_factor),
            "T_cell_c": np.squeeze(T_cell),
            "concentration_ratio": np.squeeze(np.maximum(G / self.G_ref, 0.0)),
        }

    def efficiency(self, dni, T_amb_c):
        """
        System efficiency based on primary optics (lens) area.
        eta = P_mp / (G * A_lens * optical_efficiency)
        """
        result = self.mpp(dni, T_amb_c)
        G = np.asarray(dni, dtype=float)
        G_on_cell = G * self.optical_efficiency
        return np.where(G_on_cell > 0, result["p_mp"] / (G_on_cell * self.area), 0.0)
