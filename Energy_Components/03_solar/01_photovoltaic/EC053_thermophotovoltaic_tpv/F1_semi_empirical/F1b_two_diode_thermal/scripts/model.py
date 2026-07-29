"""
EC053 — Thermophotovoltaic (TPV) — F1b Two-Diode + Thermal Model

Two-diode equation with:
  1. Thermal emitter spectrum effect: effective current scales with emitter temperature
     via sub-bandgap filtered photon flux (Planck spectrum convolved with bandgap filter)
  2. Cell thermal model: cell temperature from conductive cooling via heat sink
  3. Emitter temperature dependence of photocurrent: I_ph ~ f(T_emitter)

TPV physics:
    Unlike solar PV (where G is direct irradiance), TPV converts thermal radiation from
    a hot emitter. The photon flux depends on emitter temperature and spectral emissivity.

    For an ideal blackbody emitter at T_emitter with photocell bandgap Eg:
        Q_above_bandgap ∝ ∫[Eg/h to ∞] B(ν, T_emitter) dν

    In F1b, this is approximated as:
        I_ph = I_ph_ref * (Q_above_bg(T_emitter) / Q_above_bg(T_emitter_ref))

    where Q_above_bg is computed via exponential integral approximation of the
    above-bandgap Planck photon flux.

    Cell temperature: TPV cells are cooled by contact with heat sink.
        T_cell = T_heatsink + P_absorbed * R_thermal
    where P_absorbed is the total irradiance (including sub-bandgap).

References:
    Coutts (1999). "A review of progress in thermophotovoltaic generation."
    Renewable & Sustainable Energy Reviews 3(2), 77-184.
    Bauer (2011). "Thermophotovoltaics: basic principles and critical aspects of
    system design." Springer.
    Stephens & Cody (1977). "Optical reflectance and transmission of a textured surface."
    Phys. Rev. B 15(11), 5080.
    Datas & Algora (2010). "Analytical modelling of monolithic interconnected modules."
    Prog. Photovolt. 18(7), 537-552.
"""

import numpy as np
from scipy.optimize import brentq


class TPVf1b:
    """TPV cell — two-diode + emitter temperature scaling + heat sink thermal model."""

    # Physical constants
    h_planck = 6.62607015e-34   # J·s
    k_B = 1.380649e-23          # J/K
    c_light = 2.99792458e8      # m/s
    q = 1.602176634e-19         # C

    def __init__(self, params: dict):
        mod = params["module"]

        self.N_s = mod["cells_in_series"]["value"]
        self.area = mod["cell_area"]["value"]           # m2 per cell
        self.I_ph_ref = mod["I_ph_ref"]["value"]        # A at T_emitter_ref
        self.I_01 = mod["I_01"]["value"]
        self.I_02 = mod["I_02"]["value"]
        self.n1 = mod["n1"]["value"]
        self.n2 = mod["n2"]["value"]
        self.Rs = mod["Rs"]["value"]
        self.Rsh_ref = mod["Rsh_ref"]["value"]
        self.T_cell_ref = mod["T_cell_ref"]["value"]    # degC (cell at reference)
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.Eg = mod["Eg"]["value"]                    # eV (cell bandgap)
        self.T_emitter_ref = mod["T_emitter_ref"]["value"]  # K (1500 K typical)

        # Thermal model
        self.R_thermal = mod["R_thermal"]["value"]      # K/W (cell-to-heatsink resistance)
        self.T_heatsink = mod["T_heatsink"]["value"]    # degC

        # Physical constants (instance copies)
        self.k = self.k_B
        self.k_eV = self.k_B / self.q  # eV/K

    def _above_bandgap_photon_flux(self, T_emitter_K):
        """
        Approximate above-bandgap photon flux from blackbody emitter.
        Uses exponential approximation valid when Eg >> kT_emitter:
        N_ph ∝ (2*pi/(h^3*c^2)) * (kT)^3 * [x^2 + 2x + 2] * exp(-x)
        where x = Eg / (k*T_emitter)

        This is the Wien approximation of the Planck integral above Eg.
        Valid for x = Eg/kT > 1 (i.e., T_emitter < ~17400 K for Eg=1.5eV).
        """
        T = np.asarray(T_emitter_K, dtype=float)
        T = np.maximum(T, 300.0)  # avoid division by zero
        x = self.Eg / (self.k_eV * T)  # dimensionless energy
        x = np.maximum(x, 0.01)  # clip to avoid overflow

        # Wien approximation: integral ∝ (x^2 + 2x + 2) * exp(-x)
        flux = (x**2 + 2*x + 2) * np.exp(-np.minimum(x, 700.0))
        return flux

    def _photocurrent_scale(self, T_emitter_K):
        """Scale photocurrent with emitter temperature relative to reference."""
        flux = self._above_bandgap_photon_flux(T_emitter_K)
        flux_ref = self._above_bandgap_photon_flux(self.T_emitter_ref)
        return np.maximum(flux / np.maximum(flux_ref, 1e-300), 0.0)

    def cell_temperature(self, T_emitter_K, irradiance_total=None):
        """
        Cell temperature from thermal resistance model.
        T_cell = T_heatsink + Q_absorbed * R_thermal
        Q_absorbed is approximated from total irradiance or emitter temperature.
        """
        T_e = np.asarray(T_emitter_K, dtype=float)
        T_hs = self.T_heatsink  # degC

        # Approximate total irradiance from Stefan-Boltzmann (emitter area = cell area assumption)
        sigma = 5.670374419e-8
        Q_absorbed = sigma * T_e**4 * self.area if irradiance_total is None else irradiance_total
        T_cell = T_hs + Q_absorbed * self.R_thermal
        return T_cell

    def _thermal_voltage(self, T_K):
        return self.N_s * self.k * T_K / self.q

    def _calc_params(self, T_emitter_K, T_cell_c):
        T_e = np.atleast_1d(np.asarray(T_emitter_K, dtype=float))
        T_c = np.atleast_1d(np.asarray(T_cell_c, dtype=float))
        T_K = T_c + 273.15
        T_ref_K = self.T_cell_ref + 273.15

        # Scale photocurrent with emitter temperature
        scale = self._photocurrent_scale(T_e)
        I_ph = scale * (self.I_ph_ref + self.alpha_sc * (T_c - self.T_cell_ref))
        I_ph = np.maximum(I_ph, 0.0)

        Vt = self._thermal_voltage(T_K)
        temp_ratio = T_K / T_ref_K
        exp_factor = (self.Eg * self.q / self.k) * (1.0 / T_ref_K - 1.0 / T_K)

        I_01 = self.I_01 * temp_ratio**3 * np.exp(exp_factor)
        I_02 = self.I_02 * temp_ratio**(3.0/2.0) * np.exp(exp_factor / 2.0)
        Rsh = self.Rsh_ref * (np.maximum(scale, 0.001) / 1.0)  # scales with flux

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

    def mpp(self, T_emitter_K, T_heatsink_c=None):
        """
        Maximum power point for TPV cell.

        Parameters
        ----------
        T_emitter_K  : K, emitter temperature
        T_heatsink_c : degC, heat sink temperature (uses default if None)
        """
        T_e = np.atleast_1d(np.asarray(T_emitter_K, dtype=float))
        if T_heatsink_c is not None:
            T_hs_c = np.asarray(T_heatsink_c, dtype=float)
        else:
            T_hs_c = np.full_like(T_e, self.T_heatsink)

        T_cell_c = T_hs_c + np.maximum(0.0, T_e - 1000.0) * self.R_thermal * 1e-3
        # Simplified: T_cell scales mildly with emitter T above 1000K
        T_cell_c = np.clip(T_cell_c, self.T_cell_ref, 80.0)

        v_mp = np.zeros_like(T_e)
        i_mp = np.zeros_like(T_e)
        p_mp = np.zeros_like(T_e)
        v_oc = np.zeros_like(T_e)
        i_sc = np.zeros_like(T_e)

        I_ph_arr, I_01_arr, I_02_arr, Vt_arr, Rsh_arr = self._calc_params(T_e, T_cell_c)

        for idx in range(T_e.size):
            te = T_e.flat[idx]
            if te < 500.0:  # below 500K emitter, negligible above-Eg flux
                continue

            iph = float(I_ph_arr.flat[idx])
            i01 = float(I_01_arr.flat[idx])
            i02 = float(I_02_arr.flat[idx])
            vt = float(Vt_arr.flat[idx])
            rsh = float(Rsh_arr.flat[idx])

            if iph < 1e-15:
                continue

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
            "T_cell_c": np.squeeze(T_cell_c),
        }

    def efficiency(self, T_emitter_K, T_heatsink_c=None):
        """
        TPV electrical efficiency = P_mp / (sigma * T_emitter^4 * cell_area).
        Based on total blackbody power incident on cell area.
        """
        result = self.mpp(T_emitter_K, T_heatsink_c)
        sigma = 5.670374419e-8
        T_e = np.asarray(T_emitter_K, dtype=float)
        Q_bb = sigma * T_e**4 * self.area
        return np.where(Q_bb > 0, result["p_mp"] / Q_bb, 0.0)
