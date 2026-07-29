"""
EC044 — Monocrystalline Silicon PV — F1b Two-Diode Model

Two-diode equation with separate diffusion and recombination currents:
    I = I_ph - I_01*(exp((V+I*Rs)/(n1*Vt))-1) - I_02*(exp((V+I*Rs)/(n2*Vt))-1) - (V+I*Rs)/Rsh

where Vt = N_s * k * T / q (thermal voltage of the module).

Advantages over F1a single-diode:
    - Better accuracy at low irradiance (< 200 W/m2)
    - Physically separates diffusion (n1=1) and recombination (n2=2) mechanisms
    - More accurate fill factor prediction across operating range

References:
    Ishaque et al. (2011). "An improved modeling method to determine the model
    parameters of photovoltaic (PV) modules using differential evolution (DE)."
    Solar Energy, 85(9), 2349-2359.
"""

import numpy as np
from scipy.optimize import brentq


class MonoSiPVF1b:
    """Two-diode PV model for monocrystalline silicon."""

    def __init__(self, params: dict):
        mod = params["module"]

        self.N_s = mod["cells_in_series"]["value"]
        self.area = mod["area"]["value"]
        self.I_ph_ref = mod["I_ph_ref"]["value"]
        self.I_01 = mod["I_01"]["value"]
        self.I_02 = mod["I_02"]["value"]
        self.n1 = mod["n1"]["value"]
        self.n2 = mod["n2"]["value"]
        self.Rs = mod["Rs"]["value"]
        self.Rsh = mod["Rsh"]["value"]
        self.T_ref_c = mod["T_ref"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.Eg = mod["Eg"]["value"]

        # Physical constants
        self.k = 1.380649e-23       # Boltzmann [J/K]
        self.q = 1.602176634e-19    # electron charge [C]
        self.G_ref = 1000.0         # STC irradiance [W/m2]

    def _thermal_voltage(self, T_kelvin):
        """Module thermal voltage Vt = N_s * k * T / q."""
        return self.N_s * self.k * T_kelvin / self.q

    def _calc_params(self, irradiance, cell_temp_c):
        """Calculate two-diode parameters at operating conditions."""
        G = np.atleast_1d(np.asarray(irradiance, dtype=float))
        T_c = np.atleast_1d(np.asarray(cell_temp_c, dtype=float))
        T_K = T_c + 273.15
        T_ref_K = self.T_ref_c + 273.15

        # Photocurrent scales with irradiance and temperature
        I_ph = (G / self.G_ref) * (self.I_ph_ref + self.alpha_sc * (T_c - self.T_ref_c))

        # Saturation currents scale with temperature (Boltzmann relation)
        Vt_ref = self._thermal_voltage(T_ref_K)
        Vt = self._thermal_voltage(T_K)

        temp_ratio = T_K / T_ref_K
        exp_factor = (self.Eg * self.q / self.k) * (1.0 / T_ref_K - 1.0 / T_K)

        I_01 = self.I_01 * temp_ratio**3 * np.exp(exp_factor)
        I_02 = self.I_02 * temp_ratio**(3.0 / 2.0) * np.exp(exp_factor / 2.0)

        # Shunt resistance inversely proportional to irradiance
        Rsh = self.Rsh * (self.G_ref / np.maximum(G, 1.0))

        return I_ph, I_01, I_02, Vt, Rsh

    def _iv_equation(self, I, V, I_ph, I_01, I_02, Vt, Rsh):
        """Implicit I-V equation: f(I) = 0."""
        Vd = V + I * self.Rs
        return (I_ph - I_01 * (np.exp(Vd / (self.n1 * Vt)) - 1.0)
                - I_02 * (np.exp(Vd / (self.n2 * Vt)) - 1.0)
                - Vd / Rsh - I)

    def _solve_I_at_V(self, V, I_ph, I_01, I_02, Vt, Rsh):
        """Solve for current at given voltage using Brent's method."""
        try:
            I = brentq(self._iv_equation, -1.0, float(I_ph) + 1.0,
                       args=(V, I_ph, I_01, I_02, Vt, Rsh), xtol=1e-10)
            return max(I, 0.0)
        except (ValueError, RuntimeError):
            return 0.0

    def iv_curve(self, irradiance, cell_temp_c, n_points=100):
        """Generate I-V curve for a single operating condition."""
        G = float(irradiance)
        T = float(cell_temp_c)

        if G < 1.0:
            return np.zeros(n_points), np.zeros(n_points)

        I_ph, I_01, I_02, Vt, Rsh = self._calc_params(G, T)
        I_ph, I_01, I_02, Vt, Rsh = float(I_ph), float(I_01), float(I_02), float(Vt), float(Rsh)

        # Estimate Voc as upper bound for voltage sweep
        V_oc_est = self.n1 * Vt * np.log(I_ph / I_01 + 1.0)
        V = np.linspace(0, V_oc_est * 1.05, n_points)
        I = np.array([self._solve_I_at_V(v, I_ph, I_01, I_02, Vt, Rsh) for v in V])
        return V, I

    def mpp(self, irradiance, cell_temp_c):
        """Maximum power point and key I-V parameters."""
        G = np.atleast_1d(np.asarray(irradiance, dtype=float))
        T = np.atleast_1d(np.asarray(cell_temp_c, dtype=float))

        # Broadcast
        if G.shape != T.shape:
            G, T = np.broadcast_arrays(G, T)

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

            # Short-circuit current (V=0)
            i_sc_val = self._solve_I_at_V(0.0, iph, i01, i02, vt, rsh)

            # Open-circuit voltage (I=0) — solve V such that I=0
            try:
                v_oc_val = brentq(
                    lambda v: self._iv_equation(0.0, v, iph, i01, i02, vt, rsh),
                    0.0, self.n1 * vt * np.log(iph / i01 + 1.0) * 1.2,
                    xtol=1e-8)
            except (ValueError, RuntimeError):
                v_oc_val = 0.0

            # MPP search: sweep voltage and find max power
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

        # Fill factor
        denom = v_oc * i_sc
        fill_factor = np.where(denom > 0, p_mp / denom, 0.0)

        return {
            "i_mp": np.squeeze(i_mp),
            "v_mp": np.squeeze(v_mp),
            "p_mp": np.squeeze(p_mp),
            "i_sc": np.squeeze(i_sc),
            "v_oc": np.squeeze(v_oc),
            "fill_factor": np.squeeze(fill_factor),
        }

    def efficiency(self, irradiance, cell_temp_c):
        """Module efficiency = P_mp / (G * A_module)."""
        result = self.mpp(irradiance, cell_temp_c)
        G = np.asarray(irradiance, dtype=float)
        return np.where(G > 0, result["p_mp"] / (G * self.area), 0.0)
