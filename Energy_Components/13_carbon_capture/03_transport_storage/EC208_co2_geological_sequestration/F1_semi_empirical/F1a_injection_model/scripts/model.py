"""
EC208 — CO2 Geological Sequestration — F1a Injection Model

Two sub-models:

A. INJECTION RATE (Darcy radial flow):
   Based on the radial Darcy equation for steady-state injection into a porous reservoir:

       Q_reservoir [m³/s] = (2π k h (P_well - P_res)) / (μ ln(r_e/r_w) + S)

   where:
       k   = permeability [m²]  (1 mD = 9.869e-16 m²)
       h   = reservoir thickness [m]
       P_well = bottomhole injection pressure [Pa]
       P_res  = reservoir pressure [Pa]
       μ   = CO2 viscosity [Pa·s]
       r_e = drainage radius [m]
       r_w = wellbore radius [m]
       S   = skin factor [-]

   Bottomhole pressure from wellhead:
       P_bh = P_wh + rho_CO2 * g * depth

   Mass injection rate:
       m_dot [kg/s] = Q_reservoir [m³/s] * rho_CO2 [kg/m³]

B. STORAGE CAPACITY:
   Total theoretical CO2 storage:
       V_pore = area × thickness × porosity   [m³]
       M_storable = V_pore × E × rho_CO2_inplace   [kg or tCO2]

   where E = storage efficiency factor (1–4% for saline aquifer, IPCC 2005)
         rho_CO2_inplace = CO2 density at reservoir T,P conditions

References:
    van der Meer, L.G.H. (1993). Energy Convers. Mgmt 34, 959-966.
    IPCC (2005). Special Report on CCS. Chapter 5: Underground Geological Storage.
    Benson, S. & Cole, D.R. (2008). Elements, 4, 325-331.
"""

import numpy as np


class CO2SequestrationF1a:
    """CO2 geological injection and storage capacity model."""

    G = 9.81  # m/s²
    MD_TO_M2 = 9.869233e-16  # 1 mD → m²

    def __init__(self, params: dict):
        r = params["reservoir"]
        c = params["co2"]

        self.depth = r["depth_m"]["value"]
        self.thickness = r["thickness_m"]["value"]
        self.area_km2 = r["area_km2"]["value"]
        self.porosity = r["porosity"]["value"]
        self.k_mD = r["permeability_mD"]["value"]
        self.P_res = r["P_reservoir_bar"]["value"] * 1e5  # Pa
        self.T_res = r["T_reservoir_K"]["value"]
        self.eff = r["storage_efficiency"]["value"]
        self.skin = r["skin_factor"]["value"]
        self.r_w = r["wellbore_radius_m"]["value"]
        self.r_e = r["drainage_radius_m"]["value"]

        self.rho_inj = c["rho_injection"]["value"]      # kg/m³ at injection conditions
        self.mu_inj = c["viscosity_injection"]["value"]  # Pa·s
        self.P_wh_default = c["P_injection_wellhead"]["value"] * 1e5  # Pa

    def bottomhole_pressure_pa(self, P_wellhead_bar=None, depth_m=None):
        """Bottomhole pressure [Pa] = wellhead + hydrostatic CO2 column."""
        P_wh = (self.P_wh_default if P_wellhead_bar is None
                else np.asarray(P_wellhead_bar, dtype=float) * 1e5)
        d = self.depth if depth_m is None else np.asarray(depth_m, dtype=float)
        return P_wh + self.rho_inj * self.G * d

    def injection_rate_m3_per_s(self, P_wellhead_bar=None, depth_m=None,
                                  k_mD=None, h_m=None, mu_Pa_s=None):
        """
        Volumetric injection rate at reservoir conditions [m³/s].
        Uses steady-state radial Darcy flow.
        """
        P_bh = self.bottomhole_pressure_pa(P_wellhead_bar, depth_m)
        k = (self.k_mD if k_mD is None else np.asarray(k_mD, dtype=float)) * self.MD_TO_M2
        h = self.thickness if h_m is None else np.asarray(h_m, dtype=float)
        mu = self.mu_inj if mu_Pa_s is None else np.asarray(mu_Pa_s, dtype=float)

        ln_term = np.log(self.r_e / self.r_w) + self.skin
        dP = np.maximum(P_bh - self.P_res, 0.0)
        Q = (2.0 * np.pi * k * h * dP) / (mu * ln_term)
        return Q  # m³/s at reservoir conditions

    def injection_rate_kg_per_s(self, P_wellhead_bar=None, depth_m=None,
                                 k_mD=None, h_m=None, mu_Pa_s=None):
        """Mass injection rate [kg/s]."""
        Q = self.injection_rate_m3_per_s(P_wellhead_bar, depth_m, k_mD, h_m, mu_Pa_s)
        return Q * self.rho_inj

    def injection_rate_tco2_per_day(self, P_wellhead_bar=None, depth_m=None,
                                     k_mD=None, h_m=None, mu_Pa_s=None):
        """Injection rate [tCO2/day]."""
        m_kg_s = self.injection_rate_kg_per_s(P_wellhead_bar, depth_m, k_mD, h_m, mu_Pa_s)
        return m_kg_s * 86400.0 / 1000.0

    def storage_capacity_pore_volume_m3(self, area_km2=None, thickness_m=None, porosity=None):
        """Total pore volume [m³]."""
        A = (self.area_km2 if area_km2 is None else np.asarray(area_km2, dtype=float)) * 1e6
        h = self.thickness if thickness_m is None else np.asarray(thickness_m, dtype=float)
        phi = self.porosity if porosity is None else np.asarray(porosity, dtype=float)
        return A * h * phi

    def storage_capacity_tco2(self, area_km2=None, thickness_m=None, porosity=None,
                               storage_efficiency=None, rho_co2_inplace_kg_m3=None):
        """
        Effective CO2 storage capacity [tCO2].
        M = V_pore × E × rho_CO2_inplace
        """
        V_pore = self.storage_capacity_pore_volume_m3(area_km2, thickness_m, porosity)
        E = (self.eff if storage_efficiency is None
             else np.asarray(storage_efficiency, dtype=float))
        rho = (self.rho_inj if rho_co2_inplace_kg_m3 is None
               else np.asarray(rho_co2_inplace_kg_m3, dtype=float))
        return V_pore * E * rho / 1000.0  # kg → tonnes

    def years_to_fill(self, injection_rate_kg_s, area_km2=None, thickness_m=None,
                       porosity=None, storage_efficiency=None):
        """How many years at the given injection rate to fill the reservoir."""
        capacity_t = self.storage_capacity_tco2(area_km2, thickness_m, porosity, storage_efficiency)
        m_kg_s = np.asarray(injection_rate_kg_s, dtype=float)
        m_t_per_year = m_kg_s * 86400.0 * 365.25 / 1000.0
        return capacity_t / np.maximum(m_t_per_year, 1e-9)
