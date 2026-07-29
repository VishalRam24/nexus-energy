"""
EC139 -- Salinity Gradient Blue Energy (PRO) -- F1b Membrane Resistance Model

Extends F1a (Gibbs energy density) with:
  1. Membrane transport model: A-B solution-diffusion framework
  2. Concentration polarization (CP): both internal (ICP) and external (ECP)
  3. Temperature dependence of osmotic pressure and diffusivity
  4. Hydraulic pressure optimisation for maximum power density
  5. Net power accounting for pump parasitic work

Basis: per m³ FRESHWATER feed (Phase 7 convention; Yip & Elimelech 2012).
All energy densities are in kWh per m³ of freshwater permeated.

--- Model Equations ---

Osmotic pressure (Van't Hoff, T-dependent):
    Pi(T) = nu * R * T * dC_mol        [Pa]

Temperature correction on diffusivity (Stokes-Einstein approximation):
    D(T) = D_ref * (T / T_ref) * (mu_ref / mu(T))
    Using mu(T) = 2.414e-5 * 10^(247.8 / (T - 140)) [Pa*s]

Effective concentration at membrane surfaces (CP model):
    Internal CP (feed side, porous support):
        C_fs_eff = C_fw * exp(J_w / (D/S))   [ICP, dilutive mode]
    External CP (draw side, boundary layer):
        C_ds_eff = C_sw * exp(-J_w / k_d)     [ECP, dilutive mode in draw]

Water flux (A-B model, iterative):
    J_w = A_w * (dPi_eff - dP)         [m/s]
    dPi_eff = Pi(C_ds_eff) - Pi(C_fs_eff)

Permeate flux (per membrane area):
    J_w [m/s] = A_w [m/s/Pa] * (dPi_eff - dP) [Pa]

Salt flux:
    J_s = B * dC_m       [g/(m2*s)]

Power density (per unit membrane area):
    W_d = J_w * dP       [W/m2]

Net specific energy (per m³ freshwater permeated):
    w_gross = dP * (J_w * A_mem) / Q_fw_perm   [Pa * m/s = W/m2]
    w_net   = (w_gross * eta_turbine - w_pump) / J_PER_KWH

References:
    Yip, N.Y. & Elimelech, M. (2012). Environ. Sci. Technol. 46, 5230-5239.
    Achilli, A. & Childress, A.E. (2010). Desalination 261, 205-211.
    Loeb, S. et al. (1997). J. Membr. Sci. 129, 243-249.
    Straub, A.P. et al. (2016). Nature Energy 1, 16090.
"""

import numpy as np

_R         = 8.314        # J/(mol*K)
_J_PER_KWH = 3.6e6        # J/kWh
_T_REF     = 298.15       # K (25 degC reference)
_MU_REF    = 8.9e-4       # Pa*s at 25 degC (water viscosity)


def _water_viscosity(T_K):
    """Dynamic viscosity of water [Pa*s] using Vogel-Fulcher-Tammann fit."""
    T_K = np.asarray(T_K, dtype=float)
    return 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))


def _osmotic_pressure_pa(C_g_L, T_K, M_NaCl, nu):
    """Van't Hoff osmotic pressure [Pa] at temperature T_K."""
    C_mol_m3 = np.asarray(C_g_L, dtype=float) / M_NaCl * 1000.0
    return nu * _R * T_K * C_mol_m3


class SalinityGradientPROF1b:
    """
    PRO salinity gradient model with concentration polarization and
    temperature-dependent membrane transport.

    All energy outputs are PER m³ of freshwater actually permeated
    (Yip & Elimelech 2012 Phase 7 convention).
    """

    def __init__(self, params: dict):
        s = params["system"]
        m = params["membrane"]

        self.C_sw_default = s["C_seawater_g_per_L"]["value"]
        self.C_fw_default = s["C_freshwater_g_per_L"]["value"]
        self.T_degC_default = s["T_degC"]["value"]
        self.Q_feed       = s["Q_feed_m3_per_s"]["value"]
        self.recovery     = s["recovery_ratio"]["value"]
        self.M_NaCl       = s["M_NaCl_g_per_mol"]["value"]
        self.nu           = s["nu_NaCl"]["value"]

        self.A_mem        = m["A_m2"]["value"]
        # Convert A_w from L/(m2*h*bar) to m/(s*Pa)
        self.A_w          = m["A_w_L_m2_h_bar"]["value"] * 1e-3 / 3600 / 1e5
        # Convert B from g/(m2*h) to kg/(m2*s)  [then /1000 for mass fraction]
        self.B            = m["B_g_m2_h"]["value"] / 1000 / 3600
        self.S            = m["S_m"]["value"]
        self.D_ref        = m["D_NaCl_m2_s"]["value"]
        self.k_f          = m["k_f_m_s"]["value"]
        self.k_d          = m["k_d_m_s"]["value"]
        self.dP_bar_default = m["delta_pressure_bar"]["value"]
        self.eta_turbine  = m["eta_turbine"]["value"]
        self.eta_pump     = m["eta_pump"]["value"]
        self.eta_px       = m["eta_pressure_exchanger"]["value"]

    # ------------------------------------------------------------------
    # Temperature-dependent diffusivity
    # ------------------------------------------------------------------

    def diffusivity(self, T_K):
        """D(T) = D_ref * (T/T_ref) * (mu_ref/mu(T)) [m2/s]."""
        mu_T = _water_viscosity(T_K)
        return self.D_ref * (T_K / _T_REF) * (_MU_REF / mu_T)

    # ------------------------------------------------------------------
    # Concentration polarization
    # ------------------------------------------------------------------

    def cp_concentrations(self, C_sw, C_fw, J_w, T_K):
        """
        Effective concentrations at membrane surfaces after CP correction.

        ICP (dilutive, feed side porous support):
            C_fs_eff = C_fw * exp(J_w * S / D(T))

        ECP (dilutive, draw side boundary layer):
            C_ds_eff = C_sw * exp(-J_w / k_d)

        Args:
            C_sw, C_fw: concentrations [g/L]
            J_w:        water flux [m/s]
            T_K:        temperature [K]

        Returns:
            C_fs_eff, C_ds_eff [g/L]
        """
        D = self.diffusivity(T_K)
        # Internal CP: dilutive mode — feed concentration at active layer is higher
        C_fs_eff = np.asarray(C_fw, dtype=float) * np.exp(J_w * self.S / D)
        # External CP on draw side: dilutive — effective draw concentration lower
        C_ds_eff = np.asarray(C_sw, dtype=float) * np.exp(-J_w / self.k_d)
        return C_fs_eff, C_ds_eff

    # ------------------------------------------------------------------
    # Iterative water flux solver
    # ------------------------------------------------------------------

    def water_flux(self, C_sw, C_fw, dP_Pa, T_K, n_iter=15):
        """
        Compute water flux J_w [m/s] iteratively accounting for CP.

        Algorithm:
            1. Initial guess: J_w = A_w * (Pi_bulk - dP)
            2. Compute CP concentrations at J_w
            3. Recompute osmotic pressure difference
            4. Update J_w = A_w * (dPi_eff - dP)
            5. Repeat until convergence

        Returns:
            J_w [m/s], C_fs_eff [g/L], C_ds_eff [g/L]
        """
        C_sw = np.asarray(C_sw, dtype=float)
        C_fw = np.asarray(C_fw, dtype=float)

        Pi_sw = _osmotic_pressure_pa(C_sw, T_K, self.M_NaCl, self.nu)
        Pi_fw = _osmotic_pressure_pa(C_fw, T_K, self.M_NaCl, self.nu)
        dPi_bulk = Pi_sw - Pi_fw

        # Initial guess: no CP
        J_w = np.maximum(self.A_w * (dPi_bulk - dP_Pa), 0.0)

        for _ in range(n_iter):
            C_fs_eff, C_ds_eff = self.cp_concentrations(C_sw, C_fw, J_w, T_K)
            Pi_ds = _osmotic_pressure_pa(C_ds_eff, T_K, self.M_NaCl, self.nu)
            Pi_fs = _osmotic_pressure_pa(C_fs_eff, T_K, self.M_NaCl, self.nu)
            dPi_eff = Pi_ds - Pi_fs
            J_w_new = np.maximum(self.A_w * (dPi_eff - dP_Pa), 0.0)
            if np.max(np.abs(J_w_new - J_w)) < 1e-12:
                break
            J_w = J_w_new

        C_fs_eff, C_ds_eff = self.cp_concentrations(C_sw, C_fw, J_w, T_K)
        return J_w, C_fs_eff, C_ds_eff

    # ------------------------------------------------------------------
    # Power density
    # ------------------------------------------------------------------

    def power_density_W_m2(self, C_sw=None, C_fw=None, dP_bar=None, T_degC=None):
        """
        Gross power density [W/m2 membrane area].

        W_d = J_w * dP

        Optimal hydraulic pressure is approximately dP_opt = dPi_eff / 2
        (maximum power condition; Yip & Elimelech 2012).
        """
        C_sw = self.C_sw_default if C_sw is None else np.asarray(C_sw, dtype=float)
        C_fw = self.C_fw_default if C_fw is None else np.asarray(C_fw, dtype=float)
        T_K  = (self.T_degC_default if T_degC is None else float(T_degC)) + 273.15
        dP_Pa = (self.dP_bar_default if dP_bar is None else float(dP_bar)) * 1e5

        J_w, _, _ = self.water_flux(C_sw, C_fw, dP_Pa, T_K)
        return J_w * dP_Pa

    # ------------------------------------------------------------------
    # Energy outputs per m³ freshwater (Phase 7 basis)
    # ------------------------------------------------------------------

    def net_energy_kwh_per_m3_fw(self, C_sw=None, C_fw=None,
                                  dP_bar=None, T_degC=None):
        """
        Net energy density [kWh per m³ freshwater permeated].

        Per-membrane:
            Q_fw_perm = J_w * A_mem          [m3/s]
            P_gross   = J_w * dP * A_mem      [W]
            P_turbine = P_gross * eta_turbine  [W]
            P_pump    = (Q_fw_perm * dP / eta_pump) * (1 - eta_px)   [W]
            P_net     = P_turbine - P_pump     [W]
            w_net     = P_net / Q_fw_perm / J_PER_KWH   [kWh/m3_fw]

        Pump energy is partially recovered by the pressure exchanger (eta_px).

        Returns:
            dict: J_w_m_s, dPi_eff_bar, power_density_W_m2, net_energy_kwh_per_m3,
                  power_kw, cp_factor_ICP, cp_factor_ECP
        """
        C_sw = self.C_sw_default if C_sw is None else np.asarray(C_sw, dtype=float)
        C_fw = self.C_fw_default if C_fw is None else np.asarray(C_fw, dtype=float)
        T_K  = (self.T_degC_default if T_degC is None else float(T_degC)) + 273.15
        dP_Pa = (self.dP_bar_default if dP_bar is None else float(dP_bar)) * 1e5

        J_w, C_fs_eff, C_ds_eff = self.water_flux(C_sw, C_fw, dP_Pa, T_K)

        # Effective osmotic pressure difference
        Pi_ds = _osmotic_pressure_pa(C_ds_eff, T_K, self.M_NaCl, self.nu)
        Pi_fs = _osmotic_pressure_pa(C_fs_eff, T_K, self.M_NaCl, self.nu)
        dPi_eff_Pa = Pi_ds - Pi_fs

        # Gross power density [W/m2]
        W_d = J_w * dP_Pa

        # Flows
        Q_fw_perm = J_w * self.A_mem     # m3/s permeated

        # Power
        P_gross   = W_d * self.A_mem                        # W
        P_turbine = P_gross * self.eta_turbine               # W

        # Pump power to pressurise draw side, partially recovered by PX
        P_pump_raw = Q_fw_perm * dP_Pa / self.eta_pump       # W
        P_pump_net = P_pump_raw * (1.0 - self.eta_px)        # W after PX recovery

        P_net = P_turbine - P_pump_net                       # W

        # Per m³ freshwater permeated
        with np.errstate(divide='ignore', invalid='ignore'):
            w_net = np.where(Q_fw_perm > 0,
                             P_net / Q_fw_perm / _J_PER_KWH,
                             0.0)

        # CP factors for diagnostics
        cp_icp = np.where(C_fw > 0, C_fs_eff / C_fw, 1.0)
        cp_ecp = np.where(C_sw > 0, C_ds_eff / C_sw, 1.0)

        return {
            "J_w_m_s":               J_w,
            "dPi_eff_bar":           dPi_eff_Pa / 1e5,
            "power_density_W_m2":    W_d,
            "net_energy_kwh_per_m3": w_net,
            "power_kw":              P_net / 1000.0,
            "cp_factor_ICP":         cp_icp,
            "cp_factor_ECP":         cp_ecp,
        }

    # ------------------------------------------------------------------
    # Optimal pressure
    # ------------------------------------------------------------------

    def optimal_pressure_bar(self, C_sw=None, C_fw=None, T_degC=None,
                              n_scan=50):
        """
        Find hydraulic pressure that maximises net power density.
        Scans dP from 0 to 0.95 * Pi_bulk and returns argmax.

        Yip & Elimelech (2012) show W_d is maximised near dP = dPi_eff / 2,
        but CP shifts this optimum downward.
        """
        C_sw = self.C_sw_default if C_sw is None else float(C_sw)
        C_fw = self.C_fw_default if C_fw is None else float(C_fw)
        T_K  = (self.T_degC_default if T_degC is None else float(T_degC)) + 273.15

        Pi_bulk = _osmotic_pressure_pa(C_sw, T_K, self.M_NaCl, self.nu)
        dP_scan = np.linspace(0.5e5, 0.95 * Pi_bulk, n_scan)  # Pa

        Wd_arr = np.array([
            self.net_energy_kwh_per_m3_fw(C_sw, C_fw, dP_bar=dP/1e5, T_degC=T_K-273.15)
            ["power_density_W_m2"]
            for dP in dP_scan
        ])
        return float(dP_scan[np.argmax(Wd_arr)] / 1e5)
