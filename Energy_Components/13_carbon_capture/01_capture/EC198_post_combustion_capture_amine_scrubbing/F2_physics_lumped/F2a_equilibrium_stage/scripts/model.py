"""
EC198 -- Post-Combustion Capture (Amine Scrubbing) -- F2a Equilibrium Stage Model

Lumped equilibrium-stage model of absorber and stripper columns for MEA-based
post-combustion CO2 capture.  Replaces the simple capture-rate correlations
(F1a) with a first-principles thermodynamic model based on CO2-MEA equilibrium
chemistry.

Physics
-------
1. CO2-MEA equilibrium (simplified Kent-Eisenberg):
       CO2 + 2 MEA  <-->  MEACOO-  +  MEAH+     (carbamate formation)
       Equilibrium loading  alpha = mol_CO2 / mol_MEA
       VLE:  P_CO2* = exp(A - B/T) * (alpha / (1 - 2*alpha))^2

2. Absorber  --  counter-current, N equilibrium stages with Murphree efficiency:
       Flue gas enters bottom, lean solvent enters top.
       At each stage:  mass balance  +  VLE equilibrium * stage efficiency.
       Rich loading alpha_rich set by equilibrium with inlet CO2.
       Capture rate = (CO2_in - CO2_out) / CO2_in.

3. Stripper  --  heated to T_reb ~ 120 C to release CO2:
       Q_reb = Q_sensible + Q_reaction + Q_vaporization
       Q_sensible   = m_solv * Cp * (T_strip - T_abs) * (1 - eta_HX)
       Q_reaction   = n_CO2_stripped * dH_abs
       Q_vaporization = steam_ratio * m_CO2 * dH_vap
       Cross-heat exchanger recovers 70-85% of sensible heat.

4. Energy penalty:
       SRD = Q_reb / m_CO2_captured   [GJ/tCO2]   (typical 3.5-4.0 for 30 wt% MEA)
       Electrical penalty for blower, pumps.

Reference
---------
    Kent, R.L. & Eisenberg, B. (1976). Hydrocarbon Processing, 55(2), 87-90.
    Abu-Zahra, M.R.M. et al. (2007). Int. J. Greenhouse Gas Control, 1(1), 37-46.
"""

import numpy as np


class AmineCapture_F2a:
    """
    Equilibrium-stage model for MEA-based post-combustion CO2 capture.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, params: dict):
        u = params["unit"]

        # Molecular weights
        self.MW_CO2   = u["MW_CO2"]["value"] / 1000.0       # kg/mol
        self.MW_air   = u["MW_air"]["value"] / 1000.0       # kg/mol
        self.MW_MEA   = u["MEA_molar_mass"]["value"] / 1000.0  # kg/mol
        self.MW_water = u["water_molar_mass"]["value"] / 1000.0

        # Solvent
        self.MEA_wt = u["MEA_concentration_wt"]["value"] / 100.0  # fraction
        self.lean_loading = u["lean_loading"]["value"]
        self.rich_loading_max = u["rich_loading_max"]["value"]

        # Column sizing
        self.N_abs = int(u["absorber_stages"]["value"])
        self.N_strip = int(u["stripper_stages"]["value"])

        # Murphree stage efficiency (accounts for mass transfer limitations)
        self.murphree_eff = u["murphree_efficiency"]["value"]

        # Temperatures / pressures
        self.T_abs   = u["absorber_T_in_gas"]["value"]      # K
        self.T_solv  = u["absorber_T_solvent_in"]["value"]   # K
        self.P_abs   = u["absorber_P"]["value"]              # Pa
        self.T_reb   = u["stripper_T_reboiler"]["value"]     # K
        self.P_strip = u["stripper_P"]["value"]              # Pa

        # Flue gas defaults
        self.y_CO2_default = u["flue_gas_CO2_fraction"]["value"]
        self.fg_flow_default = u["flue_gas_flow"]["value"]   # kg/s

        # VLE parameters  (Kent-Eisenberg simplified)
        self.VLE_A = u["VLE_A"]["value"]
        self.VLE_B = u["VLE_B"]["value"]

        # Thermodynamic
        self.dH_abs   = u["delta_H_abs"]["value"]            # J/mol CO2
        self.Cp_solv  = u["Cp_solvent"]["value"]              # J/(kg.K)
        self.dH_vap   = u["delta_H_vap_water"]["value"]       # J/kg
        self.steam_ratio = u["steam_ratio"]["value"]           # kg_steam/kg_CO2

        # Cross-heat exchanger efficiency (rich/lean heat recovery)
        self.eta_HX = u["cross_hx_efficiency"]["value"]

        # Electrical
        self.elec_specific = u["electricity_specific"]["value"]  # GJ/tCO2

        # Precompute MEA mole fraction in solvent
        self._x_MEA = (self.MEA_wt / self.MW_MEA) / (
            self.MEA_wt / self.MW_MEA + (1.0 - self.MEA_wt) / self.MW_water
        )
        self._MW_solv = (self._x_MEA * self.MW_MEA +
                         (1.0 - self._x_MEA) * self.MW_water)

    # ------------------------------------------------------- VLE correlation
    def co2_equilibrium_pressure(self, alpha, T):
        """
        Equilibrium partial pressure of CO2 over MEA solution [Pa].

        P_CO2* = exp(A - B/T) * (alpha / (1 - 2*alpha))^2

        Parameters
        ----------
        alpha : float or array  --  CO2 loading (mol CO2 / mol MEA)
        T     : float           --  temperature [K]
        """
        alpha = np.asarray(alpha, dtype=float)
        alpha = np.clip(alpha, 1e-6, 0.499)
        ratio = alpha / (1.0 - 2.0 * alpha)
        P_star = np.exp(self.VLE_A - self.VLE_B / T) * ratio ** 2
        return P_star  # Pa

    def equilibrium_loading(self, P_CO2, T):
        """
        Invert VLE to find loading alpha given P_CO2 [Pa] and T [K].

        From  P = K * (alpha / (1 - 2*alpha))^2 ,  solve for alpha:
            r = sqrt(P / K),   alpha = r / (1 + 2*r)
        """
        P_CO2 = np.asarray(P_CO2, dtype=float)
        K = np.exp(self.VLE_A - self.VLE_B / T)
        K = max(K, 1e-30)
        r = np.sqrt(np.maximum(P_CO2 / K, 0.0))
        alpha = r / (1.0 + 2.0 * r)
        return np.clip(alpha, 0.0, self.rich_loading_max)

    # ------------------------------------------------- absorber stage model
    def absorber(self, y_CO2_in, L_G, N_stages=None, T_abs=None):
        """
        Counter-current absorber with N equilibrium stages and Murphree
        stage efficiency.

        Parameters
        ----------
        y_CO2_in : float  --  mole fraction CO2 in flue gas at absorber inlet
        L_G      : float  --  liquid-to-gas molar ratio (mol solvent / mol gas)
        N_stages : int    --  number of equilibrium stages (default: self.N_abs)
        T_abs    : float  --  absorber temperature [K] (default: self.T_abs)

        Returns
        -------
        dict with keys:
            capture_rate, y_CO2_out, rich_loading, stage_loadings, stage_y_CO2
        """
        if N_stages is None:
            N_stages = self.N_abs
        if T_abs is None:
            T_abs = self.T_abs

        N_stages = max(int(N_stages), 1)
        alpha_lean = self.lean_loading
        E_mv = self.murphree_eff  # Murphree vapor-phase efficiency

        x_MEA = self._x_MEA

        # Stage-wise calculation (top = stage 0, bottom = stage N-1)
        # Gas flows upward: stage N-1 is gas inlet
        # Liquid flows downward: stage 0 is liquid inlet (lean)
        y = np.zeros(N_stages + 1)     # y[i] = CO2 mole fraction leaving stage i upward
        alpha = np.zeros(N_stages + 1)  # alpha[i] = loading leaving stage i downward

        # Boundary conditions
        alpha[0] = alpha_lean         # lean solvent enters top (stage 0)
        y[N_stages] = y_CO2_in        # flue gas enters bottom (stage N)

        # Initialize with linear profile
        for i in range(N_stages + 1):
            frac = float(i) / N_stages
            y[i] = y_CO2_in * frac * 0.5
            alpha[i] = alpha_lean + (0.45 - alpha_lean) * frac

        alpha[0] = alpha_lean
        y[N_stages] = y_CO2_in

        for _iteration in range(120):
            alpha_old = alpha.copy()
            y_old = y.copy()

            # Forward sweep (top to bottom): update alpha from mass balance
            for j in range(1, N_stages + 1):
                # Overall mass balance around stages 0..j:
                # L * x_MEA * (alpha[j] - alpha[0]) = G * (y[j] - y[0])
                alpha[j] = alpha_lean + (y[j] - y[0]) / (L_G * x_MEA + 1e-30)
                alpha[j] = np.clip(alpha[j], alpha_lean, self.rich_loading_max)

            # Backward sweep (bottom to top): update y from equilibrium + efficiency
            for j in range(N_stages - 1, -1, -1):
                # Murphree efficiency: y[j] = y[j+1] - E_mv * (y[j+1] - y_eq[j])
                # where y_eq is the equilibrium vapor composition at stage j
                alpha_stage = 0.5 * (alpha[j] + alpha[j + 1])
                P_CO2_eq = self.co2_equilibrium_pressure(alpha_stage, T_abs)
                y_eq = P_CO2_eq / self.P_abs
                y_eq = np.clip(y_eq, 0.0, y_CO2_in)

                # With Murphree efficiency
                y[j] = y[j + 1] - E_mv * (y[j + 1] - y_eq)
                y[j] = max(y[j], 0.0)

            # Re-enforce boundaries
            alpha[0] = alpha_lean
            y[N_stages] = y_CO2_in

            # Check convergence
            if (np.max(np.abs(alpha - alpha_old)) < 1e-8 and
                    np.max(np.abs(y - y_old)) < 1e-10):
                break

        y_CO2_out = float(y[0])
        rich_loading = float(alpha[N_stages])

        # Capture rate
        capture_rate = 1.0 - y_CO2_out / (y_CO2_in + 1e-30)
        capture_rate = float(np.clip(capture_rate, 0.0, 1.0))

        return {
            "capture_rate": capture_rate,
            "y_CO2_out": y_CO2_out,
            "rich_loading": rich_loading,
            "stage_loadings": alpha.tolist(),
            "stage_y_CO2": y.tolist(),
        }

    # ---------------------------------------------------- stripper / reboiler
    def stripper_duty(self, rich_loading, L_G, flue_gas_kg_s, y_CO2_in,
                      capture_rate, T_abs=None):
        """
        Compute reboiler duty for the stripper.

        Includes cross-heat exchanger (rich/lean heat recovery) which reduces
        the sensible heat component.

        Parameters
        ----------
        rich_loading   : float  --  rich solvent loading (mol CO2 / mol MEA)
        L_G            : float  --  liquid/gas molar ratio
        flue_gas_kg_s  : float  --  flue gas mass flow [kg/s]
        y_CO2_in       : float  --  CO2 mole fraction in flue gas
        capture_rate   : float  --  fraction of CO2 captured
        T_abs          : float  --  absorber temperature [K]

        Returns
        -------
        dict with Q_sensible, Q_reaction, Q_vaporization, Q_reboiler [MW],
             SRD [GJ/tCO2], CO2_captured_kg_s
        """
        if T_abs is None:
            T_abs = self.T_abs

        # CO2 captured (kg/s)
        MW_flue = y_CO2_in * self.MW_CO2 + (1.0 - y_CO2_in) * self.MW_air
        mass_frac_CO2 = y_CO2_in * self.MW_CO2 / MW_flue
        CO2_in_kg_s = flue_gas_kg_s * mass_frac_CO2
        CO2_captured_kg_s = CO2_in_kg_s * capture_rate

        # Moles CO2 captured per second
        n_CO2_s = CO2_captured_kg_s / self.MW_CO2  # mol/s

        # Solvent flow rate
        n_gas = flue_gas_kg_s / MW_flue  # mol/s
        n_liquid = L_G * n_gas            # mol/s total liquid

        # Solvent mass flow (kg/s)
        m_solv_kg_s = n_liquid * self._MW_solv

        # Delta-T for sensible heat (reduced by cross-heat exchanger)
        dT = (self.T_reb - T_abs) * (1.0 - self.eta_HX)

        # Q_sensible  [W]
        Q_sens = m_solv_kg_s * self.Cp_solv * dT

        # Q_reaction  [W]  = n_CO2 * dH_abs
        Q_rxn = n_CO2_s * self.dH_abs

        # Q_vaporization  [W]  = steam_ratio * CO2_kg_s * dH_vap
        Q_vap = self.steam_ratio * CO2_captured_kg_s * self.dH_vap

        # Total reboiler duty [W]
        Q_reb = Q_sens + Q_rxn + Q_vap

        # Convert to MW
        Q_reb_MW = Q_reb / 1e6
        Q_sens_MW = Q_sens / 1e6
        Q_rxn_MW = Q_rxn / 1e6
        Q_vap_MW = Q_vap / 1e6

        # Specific reboiler duty [GJ/tCO2]
        # Q_reb [W] / CO2 [kg/s]  = J/kg  -> GJ/t = J/kg * 1e-9 * 1e3 = 1e-6
        if CO2_captured_kg_s > 1e-10:
            SRD = (Q_reb / CO2_captured_kg_s) * 1e-6  # GJ/tCO2
        else:
            SRD = 0.0

        return {
            "Q_sensible_MW": float(Q_sens_MW),
            "Q_reaction_MW": float(Q_rxn_MW),
            "Q_vaporization_MW": float(Q_vap_MW),
            "Q_reboiler_MW": float(Q_reb_MW),
            "SRD_GJ_per_tCO2": float(SRD),
            "CO2_captured_kg_s": float(CO2_captured_kg_s),
            "solvent_flow_kg_s": float(m_solv_kg_s),
        }

    # ---------------------------------------------------- electricity demand
    def electricity_demand(self, CO2_captured_kg_s):
        """
        Electrical consumption (MW) for fans, pumps, CO2 compression.

        Returns
        -------
        dict with electricity_MW, electricity_GJ_per_tCO2
        """
        # GJ/tCO2 * kg/s  ->  MW
        # 1 GJ/t * 1 kg/s = 1e9 J / 1e3 kg * 1 kg/s = 1e6 J/s = 1 MW
        elec_MW = self.elec_specific * CO2_captured_kg_s
        return {
            "electricity_MW": float(elec_MW),
            "electricity_GJ_per_tCO2": float(self.elec_specific),
        }

    # -------------------------------------------------------- full compute
    def compute(self, y_CO2_in=None, L_G=2.5, flue_gas_kg_s=None,
                N_stages=None, T_abs=None):
        """
        Full system computation: absorber + stripper + energy.

        Parameters
        ----------
        y_CO2_in      : float  --  CO2 mole fraction in flue gas
        L_G           : float  --  liquid/gas molar ratio
        flue_gas_kg_s : float  --  flue gas mass flow [kg/s]
        N_stages      : int    --  absorber equilibrium stages
        T_abs         : float  --  absorber temperature [K]

        Returns
        -------
        dict with all results
        """
        if y_CO2_in is None:
            y_CO2_in = self.y_CO2_default
        if flue_gas_kg_s is None:
            flue_gas_kg_s = self.fg_flow_default
        if T_abs is None:
            T_abs = self.T_abs

        # Absorber
        abs_result = self.absorber(y_CO2_in, L_G, N_stages=N_stages, T_abs=T_abs)

        # Stripper
        strip_result = self.stripper_duty(
            abs_result["rich_loading"], L_G, flue_gas_kg_s, y_CO2_in,
            abs_result["capture_rate"], T_abs=T_abs,
        )

        # Electricity
        elec_result = self.electricity_demand(strip_result["CO2_captured_kg_s"])

        # Total energy
        total_energy_MW = strip_result["Q_reboiler_MW"] + elec_result["electricity_MW"]
        if strip_result["CO2_captured_kg_s"] > 1e-10:
            total_SRD = (total_energy_MW * 1e6 / strip_result["CO2_captured_kg_s"]) * 1e-6
        else:
            total_SRD = 0.0

        return {
            # Absorber
            "capture_rate": abs_result["capture_rate"],
            "y_CO2_out": abs_result["y_CO2_out"],
            "rich_loading": abs_result["rich_loading"],
            "lean_loading": self.lean_loading,
            # Stripper
            "Q_sensible_MW": strip_result["Q_sensible_MW"],
            "Q_reaction_MW": strip_result["Q_reaction_MW"],
            "Q_vaporization_MW": strip_result["Q_vaporization_MW"],
            "Q_reboiler_MW": strip_result["Q_reboiler_MW"],
            "SRD_GJ_per_tCO2": strip_result["SRD_GJ_per_tCO2"],
            # Flows
            "CO2_captured_kg_s": strip_result["CO2_captured_kg_s"],
            "CO2_captured_t_per_year": strip_result["CO2_captured_kg_s"] * 3600 * 8760 / 1000.0,
            "solvent_flow_kg_s": strip_result["solvent_flow_kg_s"],
            # Electricity
            "electricity_MW": elec_result["electricity_MW"],
            # Totals
            "total_energy_MW": total_energy_MW,
            "total_specific_energy_GJ_per_tCO2": total_SRD,
            # Stage profiles
            "stage_loadings": abs_result["stage_loadings"],
            "stage_y_CO2": abs_result["stage_y_CO2"],
        }
