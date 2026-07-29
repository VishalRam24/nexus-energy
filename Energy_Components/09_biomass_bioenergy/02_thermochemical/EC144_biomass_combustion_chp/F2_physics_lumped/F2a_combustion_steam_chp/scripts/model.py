"""
EC144 -- Biomass Combustion CHP -- F2a Combustion + Steam-Cycle (Physics-Lumped)

Physics-lumped (0D) first-principles model of a biomass back-pressure
steam-turbine combined-heat-and-power plant.

Energy chain (first principles at each operating point):

  1. Fuel chemistry --- effective lower heating value of WET fuel:
         LHV_eff = LHV_dry * (1 - M) - h_fg * M          [kJ/kg wet]
     (M = moisture mass fraction, wet basis; van Loo & Koppejan 2008.)
     Moisture both dilutes the dry fuel and consumes latent heat to
     evaporate the bound water -> LHV_eff decreases monotonically with M.

  2. Combustion energy balance --- the heat released to the working fluid
     is the fuel input minus the two dominant boiler losses:
         Q_fuel  = m_fuel * LHV_eff
         m_flue  = m_fuel * [ (1-M)(1 + lambda * AFR_st) + M ]   (mass cons.)
         Q_flue  = m_flue * cp_flue * (T_flue - T_amb)           (sensible stack loss)
         Q_rad   = f_rad * Q_fuel                                (radiation/unburnt)
         eta_boiler = (Q_fuel - Q_flue - Q_rad) / Q_fuel
     Excess air (lambda > 1) raises m_flue and therefore the stack loss,
     lowering boiler efficiency -- the classic biomass trade-off
     (van Loo & Koppejan 2008, Ch. 7).

  3. Lumped boiler thermal transient (the ODE, integrated with solve_ivp):
         m_b * cp_b * dT/dt = Q_to_boiler(t) - Q_steam_demand - UA*(T - T_amb)
     A single boiler node temperature T(t) tracks how fast the steam
     drum heats up / cools down as fuel input changes -- the F2 upgrade
     over the algebraic F1 model.

  4. Back-pressure steam-turbine CHP split (Moran & Shapiro 2018; cogeneration):
         Carnot ceiling  eta_carnot = 1 - T_back / T_steam
         w_turbine       = eta_isen * eta_carnot         (work fraction of useful heat)
         P_el  = Q_useful * w_turbine * eta_mech_gen      (electricity)
         Q_th  = Q_useful - P_el / eta_mech_gen           (recovered process heat = turbine exhaust)
     so  eta_electrical < eta_total = eta_el + eta_th < 1  by construction,
     and the electrical conversion never exceeds the Carnot limit.

References:
    van Loo, S. & Koppejan, J. (2008). The Handbook of Biomass Combustion
        and Co-firing. Earthscan/IEA Bioenergy Task 32.
    Obernberger, I. & Thek, G. (2008). The Pellet Handbook. Earthscan.
    Moran, M.J., Shapiro, H.N. et al. (2018). Fundamentals of Engineering
        Thermodynamics, 9th ed. Wiley. (Rankine / cogeneration cycles)
    EN 303-5:2012. Heating boilers -- Solid fuel heating boilers.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BiomassCombustionCHP_F2a:
    """Biomass combustion back-pressure steam CHP, physics-lumped with thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated_fuel = u["Q_rated_fuel_kw"]["value"]          # kW_fuel
        self.LHV_dry = u["biomass_LHV_dry_MJ_kg"]["value"] * 1000.0  # kJ/kg
        self.h_fg = u["h_fg_water_kJ_kg"]["value"]                  # kJ/kg
        self.afr_st = u["stoich_afr"]["value"]                      # kg/kg
        self.lam = u["excess_air_ratio"]["value"]                   # -
        self.cp_flue = u["cp_flue_kJ_kgK"]["value"]                 # kJ/(kg.K)
        self.cp_steam = u["cp_steam_kJ_kgK"]["value"]               # kJ/(kg.K)
        self.T_flue_full = u["T_flue_full_degC"]["value"]           # degC
        self.T_amb_degC = u["T_ambient_degC"]["value"]              # degC
        self.f_rad = u["radiation_loss_frac"]["value"]              # -
        self.T_steam = u["T_steam_K"]["value"]                      # K
        self.T_set = u["T_boiler_setpoint_K"]["value"]             # K
        self.T_back = u["T_back_pressure_K"]["value"]               # K
        self.eta_isen = u["eta_isentropic"]["value"]               # -
        self.eta_mg = u["eta_mech_gen"]["value"]                    # -
        self.m_b = u["m_boiler_kg"]["value"]                        # kg
        self.cp_b = u["cp_boiler_kJ_kgK"]["value"]                  # kJ/(kg.K)
        self.UA = u["UA_loss_kW_K"]["value"]                        # kW/K

    # ------------------------------------------------------------------
    # 1. Fuel chemistry -- effective LHV of wet fuel
    # ------------------------------------------------------------------
    def lhv_effective(self, moisture_fraction):
        """Effective LHV of wet fuel [kJ/kg]. van Loo & Koppejan (2008)."""
        M = float(np.clip(moisture_fraction, 0.0, 0.95))
        return max(0.0, self.LHV_dry * (1.0 - M) - self.h_fg * M)

    # ------------------------------------------------------------------
    # 2. Combustion / boiler energy balance
    # ------------------------------------------------------------------
    def flue_temperature(self, PLR):
        """Stack temperature [degC] rises with load (lower share at part load)."""
        PLR = float(np.clip(PLR, 0.0, 1.0))
        return self.T_amb_degC + (self.T_flue_full - self.T_amb_degC) * (0.35 + 0.65 * PLR)

    def boiler_efficiency(self, PLR, moisture_fraction):
        """
        Combustion-to-steam efficiency = 1 - flue_loss_frac - radiation_loss_frac.
        Mass conservation on the flue stream; excess air & moisture raise the loss.
        """
        LHV_eff = self.lhv_effective(moisture_fraction)
        if LHV_eff <= 1.0:
            return 0.0
        M = float(np.clip(moisture_fraction, 0.0, 0.95))
        # flue mass per kg wet fuel: dry combustion products + excess air + evaporated water
        m_flue_per_fuel = (1.0 - M) * (1.0 + self.lam * self.afr_st) + M
        T_flue = self.flue_temperature(PLR)
        q_flue_per_kg = m_flue_per_fuel * self.cp_flue * (T_flue - self.T_amb_degC)  # kJ/kg fuel
        flue_frac = q_flue_per_kg / LHV_eff
        eta = 1.0 - flue_frac - self.f_rad
        return float(np.clip(eta, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 3. Lumped boiler thermal transient ODE
    # ------------------------------------------------------------------
    def _q_to_boiler(self, PLR_func, moisture_fraction, t):
        """Useful heat delivered to the boiler node [kW] at time t."""
        PLR = PLR_func(t) if callable(PLR_func) else PLR_func
        PLR = float(np.clip(PLR, 0.0, 1.0))
        eta_b = self.boiler_efficiency(PLR, moisture_fraction)
        return PLR * self.Q_rated_fuel * eta_b

    def simulate_thermal(self, PLR_func, moisture_fraction, T0_K=None,
                         dt=10.0, duration_s=3600.0):
        """
        Integrate the lumped boiler node temperature with scipy.solve_ivp.

            m_b cp_b dT/dt = Q_in(t) - Q_steam_extract - UA (T - T_amb)

        Steam extraction is modelled as proportional control toward the
        setpoint so the node settles near T_set at steady fuel input.
        """
        if T0_K is None:
            T0_K = self.T_amb_degC + 273.15
        T_amb_K = self.T_amb_degC + 273.15
        mc = self.m_b * self.cp_b  # kJ/K
        # proportional steam-extraction gain [kW/K] -> sets thermal time constant
        k_extract = max(self.Q_rated_fuel / 50.0, self.UA * 5.0)

        def rhs(t, y):
            T = y[0]
            Q_in = self._q_to_boiler(PLR_func, moisture_fraction, t)
            # steam extraction grows with superheat above the back-pressure
            # delivery temperature -> node settles near setpoint at steady input
            Q_extract = k_extract * max(0.0, T - self.T_back)
            Q_loss = self.UA * (T - T_amb_K)
            dTdt = (Q_in - Q_extract - Q_loss) / mc  # K/s
            return [dTdt]

        t_eval = np.arange(0.0, duration_s + dt, dt)
        sol = solve_ivp(rhs, (0.0, duration_s), [T0_K], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-3, max_step=dt)
        return {"t": sol.t, "T_boiler_K": sol.y[0]}

    # ------------------------------------------------------------------
    # 4. CHP split -- back-pressure steam turbine (steady operating point)
    # ------------------------------------------------------------------
    def carnot_efficiency(self):
        """Carnot ceiling for the back-pressure turbine [-]."""
        return 1.0 - self.T_back / self.T_steam

    def predict_steady(self, PLR, moisture_fraction):
        """
        Full steady-state operating point.

        Returns dict with fuel input, useful heat, electrical & thermal
        outputs, and the efficiency breakdown (all energy-conserving).
        """
        PLR = float(np.clip(PLR, 0.0, 1.0))
        LHV_eff = self.lhv_effective(moisture_fraction)
        Q_fuel = PLR * self.Q_rated_fuel                      # kW (LHV basis)
        eta_b = self.boiler_efficiency(PLR, moisture_fraction)
        Q_useful = Q_fuel * eta_b                              # kW heat to steam

        eta_carnot = self.carnot_efficiency()
        w_turbine = self.eta_isen * eta_carnot                # useful-heat -> shaft work fraction
        P_shaft = Q_useful * w_turbine                        # kW
        P_el = P_shaft * self.eta_mg                           # kW electrical
        # back-pressure: heat not converted to shaft work leaves as usable process heat
        Q_th = Q_useful - P_shaft                             # kW recovered heat

        eta_el = P_el / Q_fuel if Q_fuel > 0 else 0.0
        eta_th = Q_th / Q_fuel if Q_fuel > 0 else 0.0
        eta_total = eta_el + eta_th
        p2h = P_el / Q_th if Q_th > 1e-9 else 0.0

        flue_loss = Q_fuel * (1.0 - eta_b - self.f_rad) if eta_b > 0 else 0.0
        rad_loss = Q_fuel * self.f_rad

        return {
            "fuel_input_kw": Q_fuel,
            "useful_heat_kw": Q_useful,
            "P_electrical_kw": P_el,
            "Q_thermal_kw": Q_th,
            "flue_loss_kw": flue_loss,
            "radiation_loss_kw": rad_loss,
            "eta_boiler": eta_b,
            "eta_electrical": eta_el,
            "eta_thermal": eta_th,
            "eta_total_chp": eta_total,
            "eta_carnot": eta_carnot,
            "power_to_heat_ratio": p2h,
            "LHV_eff_MJ_kg": LHV_eff / 1000.0,
        }
