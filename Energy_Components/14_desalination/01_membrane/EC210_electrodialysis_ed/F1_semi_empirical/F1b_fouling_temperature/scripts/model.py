"""
EC210 — Electrodialysis (ED) — F1b Current Density + Donnan Equilibrium + Temperature Model

Extends F1a with:
  1. Current density vs desalination rate: J_ion = i * t_± / (z * F) [mol/(m2*s)]
     Higher current density → faster desalination but lower current efficiency
     (limiting current density i_lim = z*F*D*C_bulk / (delta*t_±)).
  2. Donnan equilibrium temperature dependence:
     K_D(T) = exp(-z*F*(phi_D)/(R*T)) — Donnan potential decreases at higher T,
     leading to reduced co-ion exclusion and lower selectivity.
  3. Membrane resistance aging: R_mem(t) = R0 * (1 + k_age * t/8760)
     Increased resistance → higher voltage drop → higher SEC.
  4. Stack efficiency at part-load current density:
     eta_current(i) = 1 - (i/i_lim)^2 * (1-eta_0)
     Near limiting current: efficiency drops sharply.

Physics references:
    Strathmann, H. (2010). Desalination, 264(3), 268-288.
    Pilat, B. (2001). Desalination, 139(1-3), 385-392.
    Campione, A. et al. (2018). Desalination, 434, 121-160.
"""

import numpy as np

R_GAS = 8.314   # J/(mol*K)
FARADAY = 96485.0  # C/mol


class EDF1b:
    """Electrodialysis — current density + Donnan T-dependence + membrane aging model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.i_design       = u["i_design"]["value"]        # A/m2 design current density
        self.i_lim_ref      = u["i_lim_ref"]["value"]       # A/m2 limiting current density at ref T
        self.n_cell_pairs   = u["n_cell_pairs"]["value"]     # number of cell pairs
        self.A_mem          = u["A_mem_m2"]["value"]        # membrane area per pair [m2]
        self.t_transport    = u["t_transport"]["value"]      # counter-ion transport number
        self.z_ion          = u["z_ion"]["value"]            # ion charge number
        self.R0_mem         = u["R0_mem"]["value"]           # Ohm*m2 fresh membrane resistance
        self.k_age_R        = u["k_age_R"]["value"]          # 1/year resistance increase rate
        self.eta_0          = u["eta_0"]["value"]            # current efficiency at low i/i_lim
        self.phi_D_ref      = u["phi_D_ref"]["value"]        # V Donnan potential at T_ref
        self.T_ref_K        = u["T_ref_K"]["value"]          # K reference temperature
        self.C_feed_ref     = u["C_feed_ref_mol_m3"]["value"]  # mol/m3 reference feed concentration
        self.V_cell         = u["V_cell"]["value"]           # V applied voltage per cell pair

    # ------------------------------------------------------------------ #
    #  Aging and temperature factors
    # ------------------------------------------------------------------ #

    def _membrane_resistance(self, operating_hours):
        """Membrane area resistance (Ohm*m2) after aging.
        R(t) = R0 * (1 + k_age * t_years)
        """
        t_years = np.asarray(operating_hours, dtype=float) / 8760.0
        return self.R0_mem * (1.0 + self.k_age_R * t_years)

    def _ilim_temperature(self, T_degC, C_feed_mol_m3):
        """Temperature-dependent limiting current density (A/m2).
        i_lim = k_mass_transfer * C_bulk, and k_mass ∝ D(T)/delta.
        D(T) ≈ D_ref * (T/T_ref) — simplified Stokes-Einstein.
        """
        T_K = np.asarray(T_degC, dtype=float) + 273.15
        C   = np.asarray(C_feed_mol_m3, dtype=float)
        # Limiting current scales with D(T) and concentration
        T_factor = T_K / self.T_ref_K
        C_factor = C / self.C_feed_ref
        return self.i_lim_ref * T_factor * C_factor

    def _donnan_selectivity(self, T_degC):
        """Donnan selectivity factor (lower at higher T = worse co-ion exclusion).
        From Donnan equilibrium: K_D = exp(-z*F*phi_D/(R*T))
        phi_D here is fixed (geometric property of membrane fixed charge density).
        Higher T → smaller K_D → reduced selectivity.
        Selectivity factor S = (K_D / K_D_ref) clipped [0.5, 1.0].
        """
        T_K   = np.asarray(T_degC, dtype=float) + 273.15
        # K_D(T) / K_D(T_ref) = exp(-z*F*phi_D/R * (1/T - 1/T_ref))
        exp_arg = -(self.z_ion * FARADAY * self.phi_D_ref / R_GAS) * (1.0 / T_K - 1.0 / self.T_ref_K)
        return np.clip(np.exp(exp_arg), 0.5, 1.5)

    # ------------------------------------------------------------------ #
    #  Performance calculations
    # ------------------------------------------------------------------ #

    def current_efficiency(self, current_density, T_degC, C_feed_mol_m3):
        """Current efficiency (dimensionless).
        eta = eta_0 * (1 - (i/i_lim)^2 * (1-eta_0))
        Drops near limiting current density.
        """
        i      = np.asarray(current_density, dtype=float)
        i_lim  = self._ilim_temperature(T_degC, C_feed_mol_m3)
        ratio2 = np.clip((i / i_lim) ** 2, 0.0, 1.0)
        eta    = self.eta_0 * (1.0 - ratio2 * (1.0 - self.eta_0))
        return np.clip(eta, 0.1, self.eta_0)

    def desalination_rate_mol_s(self, current_density, T_degC, C_feed_mol_m3):
        """Ion removal rate [mol/s] by the ED stack.
        J_ion = N_cp * A * i * eta / (z * F)
        """
        i     = np.asarray(current_density, dtype=float)
        eta   = self.current_efficiency(current_density, T_degC, C_feed_mol_m3)
        sel   = self._donnan_selectivity(T_degC)
        J_mol_s = self.n_cell_pairs * self.A_mem * i * eta * sel / (self.z_ion * FARADAY)
        return np.clip(J_mol_s, 0.0, None)

    def sec_kwh_m3(self, current_density, T_degC, C_feed_mol_m3, operating_hours,
                   flow_rate_m3_h):
        """Specific energy consumption [kWh/m3 product].
        SEC = (N_cp * i^2 * R_mem * A) / (Q_product * 3600)
        Plus voltage drop: P = N_cp * i * (V_applied) * A
        """
        i       = np.asarray(current_density, dtype=float)
        Q_m3h   = np.asarray(flow_rate_m3_h, dtype=float)
        Q_m3s   = Q_m3h / 3600.0
        R_mem   = self._membrane_resistance(operating_hours)

        # Total stack power [W]
        # V_stack = N_cp * (V_applied + i * R_mem * 2)  [2 membranes per cell pair: AEM + CEM]
        V_stack = self.n_cell_pairs * (self.V_cell + i * R_mem * 2.0)
        I_total = i * self.A_mem          # total current [A]
        P_watts = V_stack * I_total

        # SEC = P / (Q_product) [W / (m3/s)] = J/m3, convert to kWh/m3
        Q_m3s_safe = np.clip(Q_m3s, 1e-9, None)
        sec = P_watts / Q_m3s_safe / 3.6e6
        return np.clip(sec, 0.1, 30.0)

    def salinity_reduction_pct(self, current_density, T_degC, C_feed_mol_m3,
                                flow_rate_m3_h):
        """Salinity reduction (%) in a single pass.
        dC/C = J_ion / (Q_vol * C_feed)  [simplified CSTR approximation]
        Actual: plug-flow gives dC/dt = -J_ion * A / V_ch.
        """
        J_mol_s = self.desalination_rate_mol_s(current_density, T_degC, C_feed_mol_m3)
        Q_m3s   = np.asarray(flow_rate_m3_h, dtype=float) / 3600.0
        C       = np.asarray(C_feed_mol_m3, dtype=float)
        Q_safe  = np.clip(Q_m3s, 1e-9, None)
        C_safe  = np.clip(C, 1.0, None)
        reduction = J_mol_s / (Q_safe * C_safe) * 100.0
        return np.clip(reduction, 0.0, 95.0)

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, current_density, T_feed_degC, C_feed_mol_m3,
                flow_rate_m3_h, operating_hours):
        """Full computation returning all outputs.

        Parameters
        ----------
        current_density     : A/m2    — applied current density (0–300 A/m2)
        T_feed_degC         : degC    — feed water temperature
        C_feed_mol_m3       : mol/m3  — feed NaCl concentration (seawater ~600 mol/m3)
        flow_rate_m3_h      : m3/h    — product flow rate
        operating_hours     : hours   — cumulative hours (for membrane aging)

        Returns
        -------
        dict with desalination_rate_mol_s, salinity_reduction_pct, current_efficiency,
                  sec_kwh_m3, donnan_selectivity_factor
        """
        desalt_rate = self.desalination_rate_mol_s(current_density, T_feed_degC,
                                                    C_feed_mol_m3)
        sal_red     = self.salinity_reduction_pct(current_density, T_feed_degC,
                                                   C_feed_mol_m3, flow_rate_m3_h)
        eta_curr    = self.current_efficiency(current_density, T_feed_degC, C_feed_mol_m3)
        sec         = self.sec_kwh_m3(current_density, T_feed_degC, C_feed_mol_m3,
                                      operating_hours, flow_rate_m3_h)
        donnan_sel  = self._donnan_selectivity(T_feed_degC)

        return {
            "desalination_rate_mol_s":    desalt_rate,
            "salinity_reduction_pct":     sal_red,
            "current_efficiency":         eta_curr,
            "sec_kwh_m3":                 sec,
            "donnan_selectivity_factor":  donnan_sel,
        }
