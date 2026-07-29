"""
EC201 -- Direct Air Capture (DAC) Solid Sorbent -- F2a TSA Cycle Model

Lumped-parameter Temperature Swing Adsorption (TSA) cycle model for
amine-on-silica sorbent (Climeworks-style) direct air capture.

Physics modelled:
  1. Langmuir adsorption isotherm with van 't Hoff temperature dependence
  2. Working capacity = q(T_ads, P_CO2) - q(T_des, P_CO2_des)
  3. Sensible heat for sorbent temperature swing (with heat recovery)
  4. Heat of adsorption (desorption enthalpy)
  5. Fan power for air flow through sorbent bed
  6. Cycle timing and productivity

References:
    Fasihi et al. (2019). J. Cleaner Production, 224, 957-980.
    Sinha, A. et al. (2017). Ind. Eng. Chem. Res., 56(3), 750-764.
    Wurzbacher et al. (2012). Energy Environ. Sci., 5, 7874.
"""

import numpy as np


class TSACycleModel:
    """Solid-sorbent DAC -- lumped TSA cycle model."""

    def __init__(self, params: dict):
        u = params["unit"]

        # Sorbent properties
        s = u["sorbent"]
        self.q_max = s["q_max"]["value"]               # mmol/g
        self.K_ads_ref = s["K_ads_ref"]["value"]        # 1/bar at T_ref
        self.T_ref_K = s["T_ref_K"]["value"]            # K
        self.delta_H_ads = s["delta_H_ads"]["value"]    # kJ/mol (magnitude)
        self.Cp_sorbent = s["Cp_sorbent"]["value"]      # J/(kg*K)
        self.m_sorbent = s["m_sorbent"]["value"]        # kg

        # Cycle conditions
        c = u["cycle"]
        self.T_ads_degC = c["T_ads_degC"]["value"]
        self.T_des_degC = c["T_des_degC"]["value"]
        self.P_CO2_amb_kPa = c["P_CO2_ambient_kPa"]["value"]
        self.P_vac_atm = c["P_vac_atm"]["value"]
        self.t_ads_s = c["t_ads_s"]["value"]
        self.t_des_s = c["t_des_s"]["value"]
        self.heat_recovery = c["heat_recovery"]["value"]

        # Bed geometry
        b = u["bed"]
        self.porosity = b["porosity"]["value"]
        self.A_bed = b["cross_section_m2"]["value"]     # m2
        self.v_air = b["v_air_m_s"]["value"]            # m/s
        self.delta_P_bed = b["delta_P_bed_Pa"]["value"] # Pa

        # Fan
        self.eta_fan = u["fan"]["eta_fan"]["value"]

        # Constants
        self.R = u["constants"]["R"]["value"]           # J/(mol*K)
        self.M_CO2 = u["constants"]["M_CO2"]["value"]   # g/mol

    # ------------------------------------------------------------------
    # Langmuir isotherm with van 't Hoff temperature dependence
    # ------------------------------------------------------------------

    def langmuir_K(self, T_K):
        """Langmuir affinity constant at temperature T_K.

        For exothermic adsorption (dH_ads > 0 by convention here):
            K(T) = K_ref * exp[ (dH/R) * (1/T - 1/T_ref) ]

        Higher T -> (1/T - 1/T_ref) < 0 -> exponent < 0 -> K decreases.
        This correctly models reduced adsorption affinity at higher T.
        """
        T_K = np.asarray(T_K, dtype=float)
        # delta_H_ads in kJ/mol -> J/mol for consistency with R in J/(mol*K)
        dH_J = self.delta_H_ads * 1000.0
        exponent = (dH_J / self.R) * (1.0 / T_K - 1.0 / self.T_ref_K)
        return self.K_ads_ref * np.exp(exponent)

    def loading(self, T_K, P_CO2_kPa):
        """Equilibrium CO2 loading from Langmuir isotherm.

        q = q_max * K * P / (1 + K * P)

        Parameters
        ----------
        T_K : float or array
            Temperature [K]
        P_CO2_kPa : float or array
            CO2 partial pressure [kPa]

        Returns
        -------
        q : float or array
            Equilibrium loading [mmol CO2 / g sorbent]
        """
        T_K = np.asarray(T_K, dtype=float)
        P_bar = np.asarray(P_CO2_kPa, dtype=float) / 100.0  # kPa -> bar
        K = self.langmuir_K(T_K)
        return self.q_max * K * P_bar / (1.0 + K * P_bar)

    # ------------------------------------------------------------------
    # Working capacity
    # ------------------------------------------------------------------

    def working_capacity(self, T_ads_degC=None, T_des_degC=None,
                         P_CO2_ads_kPa=None, P_vac_atm=None):
        """Working capacity delta_q [mmol/g] = q_ads - q_des.

        During desorption, the sorbent chamber is sealed and heated while
        vacuum is applied. CO2 desorbs and is pumped out continuously.
        The effective CO2 partial pressure over the sorbent at the END of
        desorption (residual loading) is low -- approximately the ambient
        CO2 partial pressure scaled by the vacuum ratio, since the vacuum
        pump maintains a low total pressure and CO2 is a small fraction of
        the remaining gas (mostly steam from humidity).

        Effective P_CO2_des ~ P_vac * P_CO2_ambient / P_atm
        This gives a low residual CO2 partial pressure during desorption,
        which combined with high T produces low residual loading.
        """
        T_ads_K = (T_ads_degC if T_ads_degC is not None
                   else self.T_ads_degC) + 273.15
        T_des_K = (T_des_degC if T_des_degC is not None
                   else self.T_des_degC) + 273.15
        P_CO2_ads = (P_CO2_ads_kPa if P_CO2_ads_kPa is not None
                     else self.P_CO2_amb_kPa)
        P_vac = (P_vac_atm if P_vac_atm is not None
                 else self.P_vac_atm)

        # Adsorption: ambient air with dilute CO2
        q_ads = self.loading(T_ads_K, P_CO2_ads)

        # Desorption: vacuum-assisted with steam purge.
        # Effective CO2 partial pressure is low: P_vac * x_CO2_residual.
        # Use P_CO2_des = P_vac * P_CO2_ambient / 1.0 atm (scaling)
        P_CO2_des_kPa = P_vac * self.P_CO2_amb_kPa
        q_des = self.loading(T_des_K, P_CO2_des_kPa)

        return q_ads - q_des

    # ------------------------------------------------------------------
    # CO2 captured per cycle
    # ------------------------------------------------------------------

    def co2_per_cycle_kg(self, T_ads_degC=None, T_des_degC=None,
                         P_CO2_ads_kPa=None, P_vac_atm=None):
        """CO2 captured per cycle [kg].

        CO2 = m_sorbent * delta_q * M_CO2
        Units: kg * (mmol/g) * (g/mol) = kg * 1e-3 mol/g * g/mol
             = kg * 1e-3 [dimensionless mass-ratio]
        More carefully:
          delta_q [mmol/g] * m_sorbent [kg] * 1e3 [g/kg] = mmol total
          mmol * M_CO2 [g/mol] / 1e6 [mg->kg conversion via mmol->mol / g->kg]
          = m_sorbent * delta_q * 1e3 * M_CO2 / 1e6  [kg]
        """
        dq = self.working_capacity(T_ads_degC, T_des_degC,
                                   P_CO2_ads_kPa, P_vac_atm)
        # Clip negative working capacity to zero
        dq = np.maximum(dq, 0.0)
        return self.m_sorbent * dq * 1e3 * self.M_CO2 / 1e6

    # ------------------------------------------------------------------
    # Energy requirements
    # ------------------------------------------------------------------

    def thermal_energy_per_cycle_kJ(self, T_ads_degC=None, T_des_degC=None,
                                    P_CO2_ads_kPa=None, P_vac_atm=None):
        """Total thermal energy per cycle [kJ].

        Q = Q_sensible + Q_desorption
        Q_sensible = m_sorbent * Cp * dT * (1 - heat_recovery)
        Q_desorption = CO2_captured_mol * delta_H_ads
        """
        T_ads = T_ads_degC if T_ads_degC is not None else self.T_ads_degC
        T_des = T_des_degC if T_des_degC is not None else self.T_des_degC
        dT = T_des - T_ads

        # Sensible heat [kJ] (with heat recovery)
        Q_sensible = (self.m_sorbent * self.Cp_sorbent * dT
                      * (1.0 - self.heat_recovery) / 1000.0)

        # Desorption enthalpy [kJ]
        co2_kg = self.co2_per_cycle_kg(T_ads_degC, T_des_degC,
                                       P_CO2_ads_kPa, P_vac_atm)
        co2_mol = co2_kg * 1000.0 / self.M_CO2 * 1000.0  # kg -> g -> mol
        # Wait: co2_kg * 1e3 [g/kg] / M_CO2 [g/mol] = mol
        co2_mol = co2_kg * 1000.0 / self.M_CO2
        Q_desorption = co2_mol * self.delta_H_ads  # kJ

        return Q_sensible + Q_desorption

    def fan_energy_per_cycle_kJ(self):
        """Electrical (fan) energy per cycle [kJ].

        E_fan = delta_P * V_dot * t_ads / eta_fan
        Only during adsorption phase (no air flow during desorption).
        """
        V_dot = self.A_bed * self.v_air  # m3/s
        # Power [W] = delta_P [Pa] * V_dot [m3/s] / eta_fan
        P_fan_W = self.delta_P_bed * V_dot / self.eta_fan
        # Energy [kJ] = Power [W] * time [s] / 1000
        return P_fan_W * self.t_ads_s / 1000.0

    # ------------------------------------------------------------------
    # Cycle metrics
    # ------------------------------------------------------------------

    def cycle_time_s(self):
        """Total cycle time [s] = adsorption + desorption."""
        return self.t_ads_s + self.t_des_s

    def productivity_kg_h(self, T_ads_degC=None, T_des_degC=None,
                          P_CO2_ads_kPa=None, P_vac_atm=None):
        """CO2 capture productivity [kg CO2 / h]."""
        co2 = self.co2_per_cycle_kg(T_ads_degC, T_des_degC,
                                    P_CO2_ads_kPa, P_vac_atm)
        t_cycle_h = self.cycle_time_s() / 3600.0
        return co2 / t_cycle_h

    def specific_thermal_energy_GJ_tCO2(self, T_ads_degC=None, T_des_degC=None,
                                         P_CO2_ads_kPa=None, P_vac_atm=None):
        """Specific thermal energy [GJ / tonne CO2]."""
        Q_kJ = self.thermal_energy_per_cycle_kJ(T_ads_degC, T_des_degC,
                                                P_CO2_ads_kPa, P_vac_atm)
        co2_kg = self.co2_per_cycle_kg(T_ads_degC, T_des_degC,
                                       P_CO2_ads_kPa, P_vac_atm)
        co2_kg = np.maximum(co2_kg, 1e-12)
        # kJ per kg -> GJ per tonne: * 1e-6 * 1e3 = * 1e-3
        return Q_kJ / co2_kg * 1e-3

    def specific_electrical_energy_GJ_tCO2(self, T_ads_degC=None, T_des_degC=None,
                                            P_CO2_ads_kPa=None, P_vac_atm=None):
        """Specific electrical energy [GJ / tonne CO2]."""
        E_kJ = self.fan_energy_per_cycle_kJ()
        co2_kg = self.co2_per_cycle_kg(T_ads_degC, T_des_degC,
                                       P_CO2_ads_kPa, P_vac_atm)
        co2_kg = np.maximum(co2_kg, 1e-12)
        return E_kJ / co2_kg * 1e-3

    def compute(self, T_ads_degC=None, T_des_degC=None,
                P_CO2_ads_kPa=None, P_vac_atm=None):
        """Full cycle computation returning all outputs.

        Parameters
        ----------
        T_ads_degC : float, optional
            Adsorption temperature [degC]. Default from parameters.
        T_des_degC : float, optional
            Desorption temperature [degC]. Default from parameters.
        P_CO2_ads_kPa : float, optional
            Ambient CO2 partial pressure [kPa]. Default from parameters.
        P_vac_atm : float, optional
            Vacuum pressure during desorption [atm]. Default from parameters.

        Returns
        -------
        dict with cycle metrics
        """
        args = (T_ads_degC, T_des_degC, P_CO2_ads_kPa, P_vac_atm)

        T_ads = T_ads_degC if T_ads_degC is not None else self.T_ads_degC
        T_des = T_des_degC if T_des_degC is not None else self.T_des_degC

        P_CO2_ads = P_CO2_ads_kPa if P_CO2_ads_kPa is not None else self.P_CO2_amb_kPa
        P_vac = P_vac_atm if P_vac_atm is not None else self.P_vac_atm
        q_ads = self.loading(T_ads + 273.15, P_CO2_ads)
        # Desorption: same model as working_capacity()
        P_CO2_des_kPa = P_vac * self.P_CO2_amb_kPa
        q_des = self.loading(T_des + 273.15, P_CO2_des_kPa)
        dq = self.working_capacity(*args)
        co2_kg = self.co2_per_cycle_kg(*args)
        Q_th_kJ = self.thermal_energy_per_cycle_kJ(*args)
        E_fan_kJ = self.fan_energy_per_cycle_kJ()
        SEC_th = self.specific_thermal_energy_GJ_tCO2(*args)
        SEC_el = self.specific_electrical_energy_GJ_tCO2(*args)
        prod = self.productivity_kg_h(*args)

        return {
            "q_ads_mmol_g": float(q_ads),
            "q_des_mmol_g": float(q_des),
            "working_capacity_mmol_g": float(dq),
            "co2_per_cycle_kg": float(co2_kg),
            "thermal_energy_per_cycle_kJ": float(Q_th_kJ),
            "fan_energy_per_cycle_kJ": float(E_fan_kJ),
            "specific_thermal_GJ_tCO2": float(SEC_th),
            "specific_electrical_GJ_tCO2": float(SEC_el),
            "total_SEC_GJ_tCO2": float(SEC_th + SEC_el),
            "productivity_kg_CO2_h": float(prod),
            "cycle_time_s": float(self.cycle_time_s()),
            "T_ads_degC": float(T_ads),
            "T_des_degC": float(T_des),
        }
