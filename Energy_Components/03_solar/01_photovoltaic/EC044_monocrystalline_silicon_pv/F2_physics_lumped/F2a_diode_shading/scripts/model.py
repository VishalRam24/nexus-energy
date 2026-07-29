"""
EC044 -- Monocrystalline Silicon PV -- F2a Diode + Partial Shading

Cell-level single-diode model with bypass diodes and string mismatch.
Each of N_s cells can receive different irradiance (partial shading).
Bypass diodes activate when a cell is reverse-biased.

Single-cell I-V:
    I = I_L(G,T) - I_o(T)*[exp((V+I*Rs)/(n*Vt)) - 1] - (V+I*Rs)/Rsh

Module: N_s cells in series. For a given module current I, each cell
develops a voltage V_k. If V_k < V_bypass, bypass diode activates.

Reference:
    De Soto et al. (2006), Solar Energy, 80, 78-88
    Bishop (1988), Solar Cells, 25, 73-89
    PVMismatch library concept
"""

import numpy as np
from scipy.optimize import brentq


class PV_DiodeShading_F2a:
    """PV module with cell-level single-diode + bypass diodes."""

    k_B = 1.381e-23    # Boltzmann J/K
    q = 1.602e-19       # electron charge C

    def __init__(self, params: dict):
        mod = params["module"]
        self.N_s = mod["N_s"]["value"]
        self.I_L_ref = mod["I_L_ref"]["value"]
        self.I_o_ref = mod["I_o_ref"]["value"]
        self.R_s = mod["R_s_cell"]["value"]
        self.R_sh = mod["R_sh_cell"]["value"]
        self.n = mod["n_diode"]["value"]
        self.E_g = mod["E_g"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.V_bypass = mod["V_bypass"]["value"]
        self.T_ref = mod["T_ref"]["value"]
        self.G_ref = mod["G_ref"]["value"]

    # ------------------------------------------------------------------
    # Temperature-dependent parameters
    # ------------------------------------------------------------------
    def _cell_params(self, G, T_K):
        """Compute I_L and I_o for a cell at irradiance G and temperature T_K."""
        dT = T_K - self.T_ref
        I_L = (self.I_L_ref + self.alpha_sc * dT) * (G / self.G_ref)
        I_L = max(I_L, 0.0)

        # Saturation current temperature dependence
        I_o = self.I_o_ref * (T_K / self.T_ref) ** 3 * np.exp(
            (self.E_g * self.q / self.k_B) * (1.0 / self.T_ref - 1.0 / T_K)
        )
        return I_L, I_o

    def _Vt(self, T_K):
        """Thermal voltage for single cell."""
        return self.n * self.k_B * T_K / self.q

    # ------------------------------------------------------------------
    # Single-cell voltage for a given current I
    # ------------------------------------------------------------------
    def cell_voltage(self, I, G, T_K):
        """
        Solve for cell voltage given module current I, cell irradiance G, temp T_K.
        Returns cell voltage (can be negative if cell is shaded).
        """
        I_L, I_o = self._cell_params(G, T_K)
        Vt = self._Vt(T_K)

        if G <= 0 and I <= 0:
            return 0.0

        # Solve: I = I_L - I_o*(exp((V+I*Rs)/(n*Vt))-1) - (V+I*Rs)/Rsh
        # Rearrange to find V
        def residual(V):
            Vd = V + I * self.R_s
            I_diode = I_o * (np.exp(np.clip(Vd / Vt, -50, 50)) - 1.0)
            I_sh = Vd / self.R_sh
            return I_L - I_diode - I_sh - I

        # Search range: from reverse bias to open circuit
        V_lo = -5.0
        V_hi = 0.8

        try:
            # Check if signs differ
            f_lo = residual(V_lo)
            f_hi = residual(V_hi)
            if f_lo * f_hi > 0:
                # Current exceeds capability -- cell is in bypass
                return self.V_bypass if I > I_L else V_hi
            V = brentq(residual, V_lo, V_hi, xtol=1e-9)
        except (ValueError, RuntimeError):
            V = self.V_bypass

        return V

    # ------------------------------------------------------------------
    # Module I-V with per-cell irradiance
    # ------------------------------------------------------------------
    def module_voltage(self, I, irradiance_per_cell, T_K):
        """
        Total module voltage for current I.
        Each cell has its own irradiance. Bypass diodes clip at V_bypass.
        """
        V_total = 0.0
        for G_cell in irradiance_per_cell:
            V_cell = self.cell_voltage(I, G_cell, T_K)
            # Bypass diode: if cell voltage goes below V_bypass, diode conducts
            V_total += max(V_cell, self.V_bypass)
        return V_total

    # ------------------------------------------------------------------
    # Sweep I-V curve
    # ------------------------------------------------------------------
    def iv_curve(self, irradiance_per_cell, temperature_degC, N_points=200):
        """
        Generate full module I-V and P-V curves.

        Parameters
        ----------
        irradiance_per_cell : array-like of length N_s [W/m2]
        temperature_degC : float
        N_points : int

        Returns
        -------
        dict with I, V, P arrays and MPP info
        """
        T_K = temperature_degC + 273.15
        G_arr = np.asarray(irradiance_per_cell)
        assert len(G_arr) == self.N_s, f"Need {self.N_s} irradiance values, got {len(G_arr)}"

        # Estimate Isc (max photocurrent among cells)
        I_sc_est = max(
            (self.I_L_ref + self.alpha_sc * (T_K - self.T_ref)) * (g / self.G_ref)
            for g in G_arr
        )
        I_sc_est = max(I_sc_est, 0.1)

        I_sweep = np.linspace(0.0, I_sc_est * 1.02, N_points)
        V_sweep = np.zeros(N_points)
        P_sweep = np.zeros(N_points)

        for k, I in enumerate(I_sweep):
            V_sweep[k] = self.module_voltage(I, G_arr, T_K)
            P_sweep[k] = V_sweep[k] * I

        # Find MPP
        idx_mpp = np.argmax(P_sweep)
        P_mp = P_sweep[idx_mpp]
        V_mp = V_sweep[idx_mpp]
        I_mp = I_sweep[idx_mpp]

        # Count local maxima in P-V (for partial shading detection)
        local_max = 0
        for k in range(1, N_points - 1):
            if P_sweep[k] > P_sweep[k - 1] and P_sweep[k] > P_sweep[k + 1]:
                local_max += 1

        # Shading loss: compare with uniform irradiance case
        G_max = np.max(G_arr)
        if G_max > 0:
            V_uniform = np.zeros(N_points)
            P_uniform = np.zeros(N_points)
            G_uniform = np.full(self.N_s, G_max)
            for k, I in enumerate(I_sweep):
                V_uniform[k] = self.module_voltage(I, G_uniform, T_K)
                P_uniform[k] = V_uniform[k] * I
            P_mp_uniform = np.max(P_uniform)
            shading_loss_pct = (1.0 - P_mp / max(P_mp_uniform, 1e-6)) * 100.0
        else:
            shading_loss_pct = 0.0

        return {
            "I": I_sweep,
            "V": V_sweep,
            "P": P_sweep,
            "P_mp": P_mp,
            "V_mp": V_mp,
            "I_mp": I_mp,
            "num_local_maxima": local_max,
            "shading_loss_pct": max(shading_loss_pct, 0.0),
        }
