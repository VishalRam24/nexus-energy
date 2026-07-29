"""
EC048 — Perovskite Solar Cell — F1b Two-Diode + Hysteresis Model

Two-diode equation for perovskite with hysteresis correction:
    P_actual = P_two_diode * (1 - h_factor * |dG/dt| / G_ref)

The hysteresis factor accounts for ion migration in the perovskite lattice
that causes forward/reverse scan mismatch. Under rapidly changing irradiance,
the actual power deviates from steady-state predictions.

Parameters are area-normalised (Ohm*cm2) for the single-cell geometry.

References:
    Tress (2017). "Maximum Efficiency and Open-Circuit Voltage of Perovskite
    Solar Cells." J. Phys. Chem. Lett., 8, 3106-3114.
    Miyano et al. (2016). J. Phys. Chem. Lett., 7, 2199-2202.
"""

import numpy as np
from scipy.optimize import brentq


class PerovskitePVF1b:
    """Two-diode perovskite model with hysteresis correction."""

    def __init__(self, params: dict):
        mod = params["module"]

        self.N_s = mod["N_s"]["value"]
        self.area = mod["area"]["value"]            # m2
        self.area_cm2 = self.area * 1e4             # cm2
        self.I_ph_ref = mod["I_ph_ref"]["value"]    # A
        self.I_01 = mod["I_01"]["value"]            # A
        self.I_02 = mod["I_02"]["value"]            # A
        self.n1 = mod["n1"]["value"]
        self.n2 = mod["n2"]["value"]
        # Convert area-normalised resistance to absolute
        self.Rs = mod["Rs"]["value"] / self.area_cm2   # Ohm
        self.Rsh = mod["Rsh"]["value"] / self.area_cm2  # Ohm (at STC, per cell)
        self.T_ref_c = mod["T_ref"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.Eg = mod["Eg"]["value"]
        self.h_factor = mod["h_factor"]["value"]
        self.G_ref = mod["G_ref"]["value"]

        # Physical constants
        self.k = 1.380649e-23
        self.q = 1.602176634e-19

    def _thermal_voltage(self, T_kelvin):
        """Module thermal voltage Vt = N_s * k * T / q."""
        return self.N_s * self.k * T_kelvin / self.q

    def _calc_params(self, irradiance, cell_temp_c):
        """Calculate two-diode parameters at operating conditions."""
        G = np.atleast_1d(np.asarray(irradiance, dtype=float))
        T_c = np.atleast_1d(np.asarray(cell_temp_c, dtype=float))
        T_K = T_c + 273.15
        T_ref_K = self.T_ref_c + 273.15

        I_ph = (G / self.G_ref) * (self.I_ph_ref + self.alpha_sc * (T_c - self.T_ref_c))

        Vt = self._thermal_voltage(T_K)
        exp_factor = (self.Eg * self.q / self.k) * (1.0 / T_ref_K - 1.0 / T_K)
        temp_ratio = T_K / T_ref_K

        I_01 = self.I_01 * temp_ratio**3 * np.exp(exp_factor)
        I_02 = self.I_02 * temp_ratio**(3.0 / 2.0) * np.exp(exp_factor / 2.0)

        Rsh = self.Rsh * (self.G_ref / np.maximum(G, 1.0))

        return I_ph, I_01, I_02, Vt, Rsh

    def _iv_equation(self, I, V, I_ph, I_01, I_02, Vt, Rsh):
        """Implicit I-V equation: f(I) = 0."""
        Vd = V + I * self.Rs
        return (I_ph - I_01 * (np.exp(np.clip(Vd / (self.n1 * Vt), -100, 100)) - 1.0)
                - I_02 * (np.exp(np.clip(Vd / (self.n2 * Vt), -100, 100)) - 1.0)
                - Vd / Rsh - I)

    def _solve_I_at_V(self, V, I_ph, I_01, I_02, Vt, Rsh):
        """Solve for current at given voltage."""
        try:
            I = brentq(self._iv_equation, -0.01, float(I_ph) + 0.01,
                       args=(V, I_ph, I_01, I_02, Vt, Rsh), xtol=1e-12)
            return max(I, 0.0)
        except (ValueError, RuntimeError):
            return 0.0

    def _hysteresis_factor(self, irradiance_rate):
        """Hysteresis derating: 1 - h_factor * |dG/dt| / G_ref. Clipped to [0.5, 1.0]."""
        dGdt = np.asarray(irradiance_rate, dtype=float)
        factor = 1.0 - self.h_factor * np.abs(dGdt) / self.G_ref
        return np.clip(factor, 0.5, 1.0)

    def mpp(self, irradiance, cell_temp_c, irradiance_rate=0.0):
        """Maximum power point with hysteresis correction."""
        G = np.atleast_1d(np.asarray(irradiance, dtype=float))
        T = np.atleast_1d(np.asarray(cell_temp_c, dtype=float))
        dGdt = np.atleast_1d(np.asarray(irradiance_rate, dtype=float))

        if G.shape != T.shape:
            G, T = np.broadcast_arrays(G, T)
        dGdt = np.broadcast_to(dGdt, G.shape)

        v_mp = np.zeros_like(G)
        i_mp = np.zeros_like(G)
        p_mp = np.zeros_like(G)
        v_oc = np.zeros_like(G)
        i_sc = np.zeros_like(G)

        I_ph_arr, I_01_arr, I_02_arr, Vt_arr, Rsh_arr = self._calc_params(G, T)

        for idx in range(G.size):
            g = G.flat[idx]
            if g < 1.0:
                continue

            iph = float(I_ph_arr.flat[idx])
            i01 = float(I_01_arr.flat[idx])
            i02 = float(I_02_arr.flat[idx])
            vt = float(Vt_arr.flat[idx])
            rsh = float(Rsh_arr.flat[idx])

            # Short-circuit current
            i_sc_val = self._solve_I_at_V(0.0, iph, i01, i02, vt, rsh)

            # Open-circuit voltage
            try:
                v_oc_est = self.n1 * vt * np.log(max(iph / max(i01, 1e-30), 1.0) + 1.0)
                v_oc_val = brentq(
                    lambda v: self._iv_equation(0.0, v, iph, i01, i02, vt, rsh),
                    0.0, v_oc_est * 1.5, xtol=1e-10)
            except (ValueError, RuntimeError):
                v_oc_val = 0.0

            # MPP search
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

        # Apply hysteresis correction to power
        hyst = self._hysteresis_factor(dGdt)
        p_mp_actual = p_mp * hyst

        # Hysteresis index: fractional deviation from steady-state
        hysteresis_index = 1.0 - hyst

        # Efficiency
        efficiency = np.where(G > 0, p_mp_actual / (G * self.area), 0.0)

        return {
            "i_mp": np.squeeze(i_mp),
            "v_mp": np.squeeze(v_mp),
            "p_mp": np.squeeze(p_mp_actual),
            "i_sc": np.squeeze(i_sc),
            "v_oc": np.squeeze(v_oc),
            "efficiency": np.squeeze(efficiency),
            "hysteresis_index": np.squeeze(hysteresis_index),
        }
