"""
EC087 -- Biomass Boiler -- F2a Combustion + Flue-Gas Balance + Lumped Thermal-Mass ODE

Physics-lumped (0D) first-principles upgrade of the F1 semi-empirical boiler.
The same device (50 kW modulating wood-pellet boiler) is now resolved as a coupled
combustion / flue-gas / thermal-inertia system integrated in time with
scipy.integrate.solve_ivp.

PRESERVED from F1 (EC087 memory): biomass moisture -> LHV ratio coupling.
    LHV_eff = LHV_dry * (1 - w) - h_fg * w          [kJ/kg, wet basis]
    Higher moisture both lowers the energy per kg of as-fired fuel AND adds
    water vapour to the flue stream, raising stack loss -> lower efficiency.
    (Obernberger & Thek 2008; Jenkins et al. 1998)

----------------------------------------------------------------------------
1. Combustion (mass + energy)
   Fuel-feed first-order lag (grate / auger dynamics):
       d(mf)/dt = (mf_demand(PLR) - mf) / tau_fuel
   Heat release (combustion efficiency eta_comb accounts for unburnt carbon):
       Q_comb = eta_comb * mf * LHV_eff                       [kW]

2. Combustion air + flue-gas mass balance (stoichiometry, Basu 2013)
       m_air  = lambda * AFR_stoich * mf_dry                  (mf_dry = (1-w)*mf)
       m_flue = mf_dry * (1 + lambda*AFR_stoich) + w*mf       (products + air + steam)

3. Flue-gas (stack) sensible-heat loss
       T_flue = T_air + (T_flue_full - T_air) * (0.35 + 0.65*PLR)
       Q_stack = m_flue * cp_flue * (T_flue - T_air)          [kW]

4. Lumped thermal-mass ODE (boiler water + metal block, single node)
       C * dT/dt = Q_to_water - Q_useful - Q_casing
       C        = m_water*cp_water + m_metal*cp_metal          [kJ/K]
       Q_to_water = Q_comb - Q_stack          (heat into the block after stack loss)
       Q_casing   = UA_loss * (T - T_air)     (standby / casing loss)
       Q_useful   = UA_water * (T - T_return) (delivered to the load/return water),
                    clipped >= 0; this is the heat extracted by the circulation loop.

   Useful (water-side) efficiency:
       eta = Q_useful / (mf * LHV_eff)        in (0,1), lower at high moisture.

References:
    Basu, P. (2013) Biomass Gasification, Pyrolysis and Torrefaction, 2nd ed.,
        Academic Press. (combustion stoichiometry, AFR, heat release)
    van Loo, S. & Koppejan, J. (2008) The Handbook of Biomass Combustion and
        Co-firing, Earthscan. (excess air, flue cp, combustion efficiency)
    Obernberger, I. & Thek, G. (2008) The Pellet Handbook, Earthscan.
        (moisture -> effective LHV, flue moisture)
    Jenkins, B.M. et al. (1998) Prog. Energy Comb. Sci. 24, 47-81.
    Incropera, DeWitt (2007) Fundamentals of Heat and Mass Transfer. (cp metal)
"""

import numpy as np
from scipy.integrate import solve_ivp


class BiomassBoilerF2a:
    """Biomass boiler: combustion + flue-gas energy balance + lumped thermal-mass ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated   = u["Q_rated"]["value"]                       # kW
        self.LHV_dry   = u["LHV_dry_MJ_kg"]["value"] * 1000.0        # kJ/kg
        self.moisture  = u["moisture_content"]["value"]              # [0,1] wet basis
        self.h_fg      = u["h_fg_water"]["value"]                    # kJ/kg
        self.afr       = u["stoich_afr"]["value"]                    # -
        self.lam       = u["excess_air_ratio"]["value"]             # -
        self.cp_flue   = u["cp_flue"]["value"]                       # kJ/(kg.K)
        self.cp_water  = u["cp_water"]["value"]                      # kJ/(kg.K)
        self.m_water   = u["m_water"]["value"]                       # kg
        self.m_metal   = u["m_metal"]["value"]                       # kg
        self.cp_metal  = u["cp_metal"]["value"]                      # kJ/(kg.K)
        self.T_flue_full = u["T_flue_full"]["value"]                 # degC
        self.T_air     = u["T_air"]["value"]                         # degC
        self.UA_loss   = u["UA_loss"]["value"]                       # kW/K
        self.UA_water  = u["UA_water"]["value"]                      # kW/K
        self.eta_comb  = u["combustion_efficiency"]["value"]         # -
        self.PLR_min   = u["PLR_min"]["value"]                       # -
        self.tau_fuel  = u["tau_fuel"]["value"]                      # s

        # Lumped heat capacity of the boiler block [kJ/K]
        self.C = self.m_water * self.cp_water + self.m_metal * self.cp_metal

        # Effective (wet-basis) LHV -- PRESERVED moisture coupling
        self._LHV_eff = self.effective_lhv(self.moisture)

    # ------------------------------------------------------------------
    # Fuel properties (moisture -> LHV coupling)
    # ------------------------------------------------------------------
    def effective_lhv(self, w):
        """Moisture-corrected wet-basis LHV [kJ/kg]. Obernberger & Thek (2008)."""
        return self.LHV_dry * (1.0 - w) - self.h_fg * w

    @property
    def LHV_eff_kj_kg(self):
        return self._LHV_eff

    # ------------------------------------------------------------------
    # Combustion: fuel demand and heat release
    # ------------------------------------------------------------------
    def fuel_demand_kg_s(self, PLR):
        """As-fired (wet) fuel mass flow demanded at a given PLR [kg/s].

        Sized so that combustion heat release at full load matches the rated
        thermal output divided by a nominal full-load water-side efficiency.
        """
        PLR = float(np.clip(PLR, 0.0, 1.0))
        if PLR < self.PLR_min and PLR > 0.0:
            PLR = self.PLR_min
        # Full-load fuel: Q_rated / (eta_nom_design); use combustion eff + a
        # design useful fraction so the block reaches steady state near Q_rated.
        # m_full chosen from heat release: eta_comb * m_full * LHV_eff = Q_rated / eta_water_design
        eta_water_design = 0.88
        m_full = (self.Q_rated / eta_water_design) / (self.eta_comb * max(self._LHV_eff, 1.0))
        return PLR * m_full

    def heat_release_kw(self, m_fuel):
        """Combustion heat release [kW] from as-fired fuel flow [kg/s]."""
        return self.eta_comb * m_fuel * self._LHV_eff

    # ------------------------------------------------------------------
    # Flue-gas mass + stack loss
    # ------------------------------------------------------------------
    def flue_mass_flow_kg_s(self, m_fuel):
        """Flue-gas mass flow [kg/s]: dry products + excess air + fuel moisture."""
        m_dry = (1.0 - self.moisture) * m_fuel
        m_water_vap = self.moisture * m_fuel
        return m_dry * (1.0 + self.lam * self.afr) + m_water_vap

    def flue_temp_c(self, PLR):
        """Flue-gas exit temperature [degC], scales with load."""
        PLR = float(np.clip(PLR, 0.0, 1.0))
        PLR_eff = max(PLR, self.PLR_min) if PLR > 0 else 0.0
        return self.T_air + (self.T_flue_full - self.T_air) * (0.35 + 0.65 * PLR_eff)

    def stack_loss_kw(self, m_fuel, PLR):
        """Sensible flue-gas (stack) heat loss [kW]."""
        m_flue = self.flue_mass_flow_kg_s(m_fuel)
        T_flue = self.flue_temp_c(PLR)
        return m_flue * self.cp_flue * (T_flue - self.T_air)

    # ------------------------------------------------------------------
    # Air mass flow (for reporting / mass conservation checks)
    # ------------------------------------------------------------------
    def air_mass_flow_kg_s(self, m_fuel):
        m_dry = (1.0 - self.moisture) * m_fuel
        return self.lam * self.afr * m_dry

    # ------------------------------------------------------------------
    # Lumped thermal-mass ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, plr_fn, T_return_c):
        """State y = [T_block (degC), m_fuel (kg/s)]."""
        T, m_fuel = y
        PLR = float(plr_fn(t))

        m_demand = self.fuel_demand_kg_s(PLR)
        dmf_dt = (m_demand - m_fuel) / self.tau_fuel       # kg/s per s

        Q_comb = self.heat_release_kw(m_fuel)              # kW
        Q_stack = self.stack_loss_kw(m_fuel, PLR)          # kW
        Q_to_water = max(Q_comb - Q_stack, 0.0)            # kW into block
        Q_casing = self.UA_loss * (T - self.T_air)         # kW
        # Useful heat delivered to circulation loop (only when block above return)
        Q_useful = max(self.UA_water * (T - T_return_c), 0.0)

        dT_dt = (Q_to_water - Q_useful - Q_casing) / self.C  # degC/s
        return [dT_dt, dmf_dt]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, PLR, T_water_init_K=333.15, T_return_K=323.15,
                 dt=2.0, duration_s=1200.0):
        """Integrate the coupled fuel-lag + thermal-mass ODE.

        PLR: scalar or callable PLR(t) in [0,1].
        Returns dict of time-series arrays (SI-ish engineering units).
        """
        if callable(PLR):
            plr_fn = PLR
        else:
            plr_val = float(PLR)
            plr_fn = lambda t: plr_val

        T0_c = T_water_init_K - 273.15
        T_return_c = T_return_K - 273.15
        m0 = self.fuel_demand_kg_s(plr_fn(0.0))

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T0_c, m0],
            t_eval=t_eval, args=(plr_fn, T_return_c),
            method="RK45", rtol=1e-6, atol=1e-8, max_step=dt,
        )

        T = sol.y[0]
        m_fuel = np.maximum(sol.y[1], 0.0)
        tt = sol.t

        PLR_arr = np.array([float(plr_fn(t)) for t in tt])
        Q_comb = self.eta_comb * m_fuel * self._LHV_eff
        m_flue = np.array([self.flue_mass_flow_kg_s(mf) for mf in m_fuel])
        m_air = np.array([self.air_mass_flow_kg_s(mf) for mf in m_fuel])
        T_flue = np.array([self.flue_temp_c(p) for p in PLR_arr])
        Q_stack = m_flue * self.cp_flue * (T_flue - self.T_air)
        Q_casing = self.UA_loss * (T - self.T_air)
        Q_useful = np.maximum(self.UA_water * (T - T_return_c), 0.0)

        fuel_energy = m_fuel * self._LHV_eff               # kW (LHV basis)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(fuel_energy > 1e-6, Q_useful / fuel_energy, 0.0)
        eta = np.clip(np.nan_to_num(eta), 0.0, 1.0)

        return {
            "t": tt,
            "T_water": T + 273.15,             # K
            "T_water_C": T,                    # degC
            "PLR": PLR_arr,
            "m_fuel": m_fuel,                  # kg/s as-fired
            "m_air": m_air,                    # kg/s
            "m_flue": m_flue,                  # kg/s
            "T_flue_C": T_flue,                # degC
            "Q_comb": Q_comb,                  # kW heat release
            "Q_stack": Q_stack,                # kW flue loss
            "Q_casing": Q_casing,              # kW casing/standby loss
            "Q_useful": Q_useful,              # kW delivered to water
            "efficiency": eta,                 # - (water-side, LHV basis)
            "LHV_eff": self._LHV_eff,          # kJ/kg
        }
