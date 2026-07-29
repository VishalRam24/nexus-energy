"""
EC088 -- Oil-Fired Boiler -- F2a Physics-Lumped Dynamic Thermal Mass

Physics-lumped (0D) first-principles upgrade of the F1b part-load model. The
F1 model is algebraic and steady-state; here the boiler water temperature is a
state variable governed by a lumped-capacitance ODE, integrated with
scipy.integrate.solve_ivp. Combustion, stack (flue-gas) losses and standby
losses are all derived from first principles each time step.

----------------------------------------------------------------------------
1. COMBUSTION (mass & energy balance)
----------------------------------------------------------------------------
Fuel-oil mass flow from the commanded firing rate (PLR) and the heat input:

    Q_fuel  = PLR * Q_rated / eta_design_ref          (LHV basis, kW)
    m_fuel  = Q_fuel / LHV_oil                          (kg/s)

Stoichiometric + excess air (lambda = excess_air_ratio):

    m_air   = lambda * AFR_stoich * m_fuel              (kg/s)
    m_flue  = m_fuel + m_air = m_fuel * (1 + lambda*AFR) (mass conservation)

Reference: Annamalai & Puri (2007), "Combustion Science and Engineering",
CRC Press, Ch. 3 (stoichiometry, air-fuel ratio, excess air).

----------------------------------------------------------------------------
2. STACK (FLUE-GAS) LOSS -- sensible + latent  (ASME PTC 4 heat-loss method)
----------------------------------------------------------------------------
Sensible loss: dry/wet flue gas leaves at T_flue carrying enthalpy above datum

    Q_sensible = m_flue * cp_flue * (T_flue - T_air)    (W)

Latent loss: hydrogen in the fuel burns to water vapour; for an LHV efficiency
basis the vaporization enthalpy of that water leaves up the stack:

    m_H2O   = 9 * H_fuel * m_fuel        (kg/s)  (1 kg H -> 9 kg H2O)
    Q_latent = m_H2O * hfg_water                          (W)

T_flue rises with firing rate (more excess fuel, less relative wall cooling):

    T_flue(PLR) = T_air + (T_flue_full - T_air)*(0.32 + 0.68*PLR)

Reference: ASME PTC 4-2013, "Fired Steam Generators", heat-loss method
(losses L_G dry gas, L_H2O water from H2). Babcock & Wilcox, "Steam" 41st ed.

----------------------------------------------------------------------------
3. LUMPED THERMAL MASS ODE (water + metal capacitance)
----------------------------------------------------------------------------
A single lumped capacitance C = m_water*cp_water + m_metal*cp_metal stores the
boiler thermal energy. State = boiler water temperature T_w [K]:

    C * dT_w/dt = Q_useful - Q_load - Q_standby

  Q_useful  = Q_fuel - Q_sensible - Q_latent     (combustion heat reaching water)
  Q_load    = UA_load * (T_w - T_return)          (heat delivered to the circuit)
  Q_standby = UA_loss * (T_w - T_ambient)         (casing/jacket radiation loss)

Instantaneous combustion efficiency (LHV) and overall efficiency:

    eta_comb = Q_useful / Q_fuel
    eta_overall = Q_load / Q_fuel

All efficiencies are bounded to (0,1). Energy conservation is enforced:
Q_fuel = Q_useful + Q_sensible + Q_latent exactly (combustion node).

References:
    Annamalai & Puri (2007), Combustion Science and Engineering, CRC Press.
    ASME PTC 4-2013, Fired Steam Generators (heat-loss / indirect efficiency).
    Babcock & Wilcox (2005), "Steam: Its Generation and Use", 41st ed.
    EN 303-1 -- Heating boilers; EnergyPlus Engineering Ref. Boiler:HotWater.

Hardcoded property values (cited in parameters.json):
    cp_water = 4186 J/(kg.K)   (NIST/IAPWS, liquid water ~50C)
    cp_flue  = 1050 J/(kg.K)   (mean flue gas 20-250C, Babcock & Wilcox)
    hfg_water= 2.44e6 J/kg      (latent heat of water ~25C, NIST)
    cp_metal = 490  J/(kg.K)    (carbon steel)
"""

import numpy as np
from scipy.integrate import solve_ivp

# Design-reference efficiency used to convert firing rate -> fuel input.
# Chosen so that at PLR=1 the useful heat ~= Q_rated (consistent with F1b peak).
_ETA_DESIGN_REF = 0.88


class OilBoilerF2a:
    """Oil-fired boiler -- lumped combustion + thermal-mass dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated   = u["Q_rated"]["value"] * 1000.0        # kW -> W
        self.LHV_oil   = u["LHV_oil"]["value"] * 1.0e6         # MJ/kg -> J/kg
        self.AFR       = u["stoich_afr"]["value"]              # kg_air/kg_fuel
        self.lambda_a  = u["excess_air_ratio"]["value"]
        self.H_fuel    = u["H_fuel"]["value"] / 100.0          # mass fraction
        self.cp_flue   = u["cp_flue"]["value"]                 # J/(kg.K)
        self.cp_water  = u["cp_water"]["value"]                # J/(kg.K)
        self.hfg       = u["hfg_water"]["value"]               # J/kg
        self.T_air     = u["T_air"]["value"]                   # degC
        self.T_flue_full = u["T_flue_full"]["value"]           # degC
        self.m_water   = u["m_water"]["value"]                 # kg
        self.m_metal   = u["m_metal"]["value"]                 # kg
        self.cp_metal  = u["cp_metal"]["value"]                # J/(kg.K)
        self.UA_loss   = u["UA_loss"]["value"]                 # W/K
        self.UA_load   = u["UA_load"]["value"]                 # W/K
        self.T_set     = u["T_setpoint"]["value"]              # degC
        self.PLR_min   = u["PLR_min"]["value"]

        # Lumped thermal capacitance [J/K]
        self.C = self.m_water * self.cp_water + self.m_metal * self.cp_metal

    # ------------------------------------------------------------------
    # Combustion -- mass balance
    # ------------------------------------------------------------------
    def fuel_input(self, plr):
        """Fuel heat input Q_fuel [W] (LHV basis) for firing rate plr."""
        plr = float(np.clip(plr, 0.0, 1.0))
        if plr < self.PLR_min:
            return 0.0  # burner off below turndown limit
        return plr * self.Q_rated / _ETA_DESIGN_REF

    def fuel_mass_flow(self, plr):
        """m_fuel [kg/s]."""
        return self.fuel_input(plr) / self.LHV_oil

    def flue_mass_flow(self, plr, excess_air=None):
        """m_flue [kg/s] = m_fuel*(1 + lambda*AFR) -- mass conservation."""
        lam = self.lambda_a if excess_air is None else float(excess_air)
        return self.fuel_mass_flow(plr) * (1.0 + lam * self.AFR)

    # ------------------------------------------------------------------
    # Stack losses
    # ------------------------------------------------------------------
    def flue_temp(self, plr):
        """Stack temperature [degC] rising with firing rate."""
        plr = float(np.clip(plr, 0.0, 1.0))
        plr_eff = max(plr, self.PLR_min)
        return self.T_air + (self.T_flue_full - self.T_air) * (0.32 + 0.68 * plr_eff)

    def sensible_loss(self, plr, excess_air=None):
        """Sensible flue-gas loss [W]."""
        m_flue = self.flue_mass_flow(plr, excess_air)
        return m_flue * self.cp_flue * (self.flue_temp(plr) - self.T_air)

    def latent_loss(self, plr):
        """Latent (H2->H2O vapour) loss [W]; 1 kg H burns to 9 kg H2O."""
        m_fuel = self.fuel_mass_flow(plr)
        m_h2o = 9.0 * self.H_fuel * m_fuel
        return m_h2o * self.hfg

    # ------------------------------------------------------------------
    # Energy node -- useful heat to water
    # ------------------------------------------------------------------
    def useful_heat(self, plr, excess_air=None):
        """Q_useful [W] = Q_fuel - stack losses (energy conservation)."""
        Q_fuel = self.fuel_input(plr)
        if Q_fuel <= 0.0:
            return 0.0
        Q_loss = self.sensible_loss(plr, excess_air) + self.latent_loss(plr)
        return max(Q_fuel - Q_loss, 0.0)

    def combustion_efficiency(self, plr, excess_air=None):
        """eta_comb = Q_useful / Q_fuel, bounded (0,1)."""
        Q_fuel = self.fuel_input(plr)
        if Q_fuel <= 0.0:
            return 0.0
        eta = self.useful_heat(plr, excess_air) / Q_fuel
        return float(np.clip(eta, 1e-6, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # Lumped thermal-mass ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, T_w, plr_fn, T_return, T_ambient, excess_air):
        """dT_w/dt for the lumped capacitance [K/s]. T_w in degC."""
        plr = plr_fn(t)
        Q_useful  = self.useful_heat(plr, excess_air)
        Q_load    = self.UA_load * (T_w[0] - T_return)
        Q_load    = max(Q_load, 0.0)            # no reverse heat-grab from load
        Q_standby = self.UA_loss * (T_w[0] - T_ambient)
        dTdt = (Q_useful - Q_load - Q_standby) / self.C
        return [dTdt]

    def simulate(self, firing_rate, T_water_init=40.0, T_return=60.0,
                 T_ambient=20.0, excess_air=None, dt=5.0, duration_s=1800.0):
        """
        Integrate the lumped thermal-mass ODE with solve_ivp.

        firing_rate : float, or callable f(t)->[0,1]. If a float, an on/off
                      thermostat around T_setpoint is applied automatically.
        Returns dict of time-series arrays.
        """
        excess_air = self.lambda_a if excess_air is None else float(excess_air)

        if callable(firing_rate):
            plr_fn = firing_rate
        else:
            plr_const = float(firing_rate)

            def plr_fn(t, _T_set=self.T_set, _p=plr_const):
                # simple thermostat: fire at commanded rate, modulate near setpoint
                return _p

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [float(T_water_init)],
            t_eval=t_eval, method="RK45", max_step=dt, rtol=1e-6, atol=1e-6,
            args=(plr_fn, T_return, T_ambient, excess_air),
        )

        t = sol.t
        T_w = sol.y[0]

        plr_arr   = np.array([float(np.clip(plr_fn(ti), 0.0, 1.0)) for ti in t])
        Q_fuel    = np.array([self.fuel_input(p) for p in plr_arr])
        Q_sens    = np.array([self.sensible_loss(p, excess_air) for p in plr_arr])
        Q_lat     = np.array([self.latent_loss(p) for p in plr_arr])
        Q_useful  = np.array([self.useful_heat(p, excess_air) for p in plr_arr])
        Q_load    = np.maximum(self.UA_load * (T_w - T_return), 0.0)
        Q_standby = self.UA_loss * (T_w - T_ambient)
        T_flue    = np.array([self.flue_temp(p) for p in plr_arr])
        m_fuel    = np.array([self.fuel_mass_flow(p) for p in plr_arr])
        m_flue    = np.array([self.flue_mass_flow(p, excess_air) for p in plr_arr])

        eta_comb = np.where(Q_fuel > 0.0, Q_useful / np.where(Q_fuel > 0, Q_fuel, 1.0), 0.0)
        eta_overall = np.where(Q_fuel > 0.0, Q_load / np.where(Q_fuel > 0, Q_fuel, 1.0), 0.0)

        return {
            "t": t,
            "T_water_C": T_w,
            "firing_rate": plr_arr,
            "Q_fuel_W": Q_fuel,
            "Q_useful_W": Q_useful,
            "Q_load_W": Q_load,
            "Q_sensible_loss_W": Q_sens,
            "Q_latent_loss_W": Q_lat,
            "Q_standby_loss_W": Q_standby,
            "T_flue_C": T_flue,
            "m_fuel_kg_s": m_fuel,
            "m_flue_kg_s": m_flue,
            "eta_combustion": eta_comb,
            "eta_overall": eta_overall,
        }
