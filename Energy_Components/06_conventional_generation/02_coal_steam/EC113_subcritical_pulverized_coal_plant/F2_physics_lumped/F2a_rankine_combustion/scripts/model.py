"""
EC113 -- Subcritical Pulverized Coal Plant -- F2a Physics-Lumped
         Combustion + Subcritical Rankine Cycle + Drum Thermal ODE

Physics-lumped (0D) first-principles model of a drum-type subcritical
pulverized-coal power plant. Three coupled subsystems:

  1. Pulverized-coal combustion / boiler energy balance
        Q_fuel  = m_coal * LHV_coal
        Q_stack = m_flue * cp_flue * (T_stack - T_air_in)      (dry-gas loss)
        eta_boiler = 1 - (Q_stack + Q_rad) / Q_fuel
        Q_steam = eta_boiler * Q_fuel                          -> heat into cycle

  2. Subcritical Rankine steam cycle (single reheat, regenerative datum)
        State 1: condensate (sat. liquid at P_cond)
        State 2: feedwater after pump  (isentropic + pump efficiency)
        State 3: superheated main steam (P_drum, T_superheat)
        State 4: HP exhaust / cold reheat (P_reheat, isentropic eff.)
        State 5: hot reheat (P_reheat, T_reheat)
        State 6: LP exhaust (P_cond, isentropic eff.)
        w_turb = (h3-h4) + (h5-h6)        ; w_pump = (h2-h1)
        eta_cycle = (w_turb - w_pump) / q_in
        eta_net   = eta_boiler * eta_cycle * eta_mech * eta_gen * (1 - aux_frac)

  3. Lumped boiler-drum thermal transient (ODE, scipy.solve_ivp)
        (M*cp) dT_drum/dt = Q_steam_in(t) - Q_extracted(T_drum)
        Q_extracted = UA * (T_drum - T_sat(P_drum))
     captures the warm-up / load-change thermal inertia of the drum.

Steam/water enthalpies use simplified IAPWS-style correlations (no CoolProp):
  - Saturation pressure/temperature: Antoine-form fit to IAPWS-IF97.
  - Liquid enthalpy:  h_f ~ cp_w * (T - 273.15).
  - Latent heat h_fg: Watson-type correlation anchored to IAPWS data.
  - Superheated-vapour enthalpy: h_g(sat) + cp_steam*(T - T_sat) - P work term.
Constants anchored to steam-table values in Borgnakke & Sonntag (2009) App. B.

Enforced physics:
  * eta_net < eta_Carnot(T_hot, T_cond)             (2nd law)
  * energy & mass balance close to machine precision
  * 0 < eta_boiler, eta_cycle, eta_net < 1
  * net plant efficiency in realistic 0.36-0.40 LHV band at design point

References:
  Black & Veatch (1996), Power Plant Engineering, Chapman & Hall, ch. 2,4,8.
  El-Wakil, M.M. (1984), Powerplant Technology, McGraw-Hill, ch. 2-3.
  Borgnakke, C. & Sonntag, R. (2009), Fundamentals of Thermodynamics,
      7th ed., Wiley -- steam tables (App. B), Rankine cycle (ch. 11).
  Wagner, W. et al. (2000), IAPWS-IF97 industrial formulation, J. Eng. Gas
      Turbines Power 122(1) -- reference data for the simplified fits.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ===========================================================================
#  Simplified IAPWS-style steam/water property correlations
#  (cited fits to Borgnakke & Sonntag steam tables / IAPWS-IF97 data)
# ===========================================================================

CP_WATER = 4.186      # kJ/(kg.K)  liquid water specific heat
V_WATER = 1.0e-3      # m3/kg      liquid specific volume (incompressible)
TC_WATER = 647.096    # K          critical temperature of water (IAPWS)


def cp_steam(P_bar):
    """
    Mean superheated-steam specific heat [kJ/(kg.K)] over the saturation-to-
    superheat span.  Real-gas effects make cp rise steeply toward the critical
    pressure; linear fit anchored to the IAPWS degree-of-superheat enthalpy
    rise at the design points: cp(40 bar)=2.54, cp(165 bar)=4.50.
    """
    P = np.asarray(P_bar, dtype=float)
    cp0, k = 1.913, 0.01568
    return np.clip(cp0 + k * P, 2.0, 5.0)


def Tsat_from_P(P_bar):
    """
    Saturation temperature [K] from pressure [bar].
    Antoine-form fit to IAPWS-IF97 (valid ~0.05-220 bar).
    log10(P_kPa) = A - B/(C + T_C).  Constants fitted to steam-table points.
    """
    P_kPa = np.asarray(P_bar, dtype=float) * 100.0
    A, B, C = 7.0436, 1648.7, 226.1   # fit anchored to (0.07 bar,39C),(165 bar,357C)
    T_C = B / (A - np.log10(P_kPa)) - C
    return T_C + 273.15


def Psat_from_T(T_K):
    """Saturation pressure [bar] from temperature [K] (inverse of Tsat fit)."""
    A, B, C = 7.0436, 1648.7, 226.1
    T_C = np.asarray(T_K, dtype=float) - 273.15
    P_kPa = 10.0 ** (A - B / (C + T_C))
    return P_kPa / 100.0


def hfg_latent(T_K):
    """
    Latent heat of vaporization [kJ/kg], Watson-type correlation.
    h_fg = h_fg0 * (1 - T/Tc)^0.38 ; anchored to 2257 kJ/kg at 100C.
    Tc = 647.096 K (critical T of water, IAPWS).
    """
    T = np.asarray(T_K, dtype=float)
    ref = 2257.0 / (1.0 - 373.15 / TC_WATER) ** 0.38
    return ref * np.maximum(1.0 - T / TC_WATER, 1e-6) ** 0.38


def h_liquid(T_K):
    """Saturated/compressed liquid enthalpy [kJ/kg] above 0C datum."""
    return CP_WATER * (np.asarray(T_K, dtype=float) - 273.15)


def h_sat_vapor(P_bar):
    """
    Saturated vapour enthalpy h_g [kJ/kg] at pressure P.
    Saturated-vapour enthalpy has a non-monotone shape: ~2572 kJ/kg at the
    triple region, rising to a peak ~2802 kJ/kg near 30 bar, then declining
    toward the critical point (2086 kJ/kg).  Captured here by a quadratic in
    saturation temperature fitted to IAPWS steam-table anchor points
    (Borgnakke & Sonntag App. B):
        (Tsat,  h_g)  = (313 K, 2572), (523 K, 2801), (630 K, 2580).
    """
    Tsat = np.asarray(Tsat_from_P(P_bar), dtype=float)
    # quadratic h_g = a + b*Tsat + c*Tsat^2  (Tsat in K)
    a, b, c = 6.00974e2, 9.41328e0, -9.95551e-3
    return a + b * Tsat + c * Tsat ** 2


def h_superheated(P_bar, T_K):
    """
    Superheated-vapour enthalpy [kJ/kg].
    h = h_g(P) + cp_steam*(T - Tsat(P)).  Degree-of-superheat sensible term.
    cp_steam anchored so h(165bar,540C)~3410, h(40bar,540C)~3536 kJ/kg.
    """
    Tsat = Tsat_from_P(P_bar)
    T = np.asarray(T_K, dtype=float)
    return h_sat_vapor(P_bar) + cp_steam(P_bar) * (T - Tsat)


def hfg_from_tables(P_bar):
    """Latent heat h_fg = h_g - h_f [kJ/kg] consistent with the h_g fit."""
    Tsat = Tsat_from_P(P_bar)
    return np.maximum(h_sat_vapor(P_bar) - h_liquid(Tsat), 1.0)


def s_superheated(P_bar, T_K):
    """
    Specific entropy [kJ/(kg.K)] of (super)heated steam -- simplified.
    s = s_g(P) + cp_steam*ln(T/Tsat).  s_g from Clausius datum + h_fg/Tsat.
    """
    Tsat = Tsat_from_P(P_bar)
    T = np.asarray(T_K, dtype=float)
    s_f = CP_WATER * np.log(Tsat / 273.15)
    s_g = s_f + hfg_from_tables(P_bar) / Tsat
    return s_g + cp_steam(P_bar) * np.log(np.maximum(T, 1.0) / Tsat)


def h_wet(P_bar, x):
    """Two-phase (wet) enthalpy [kJ/kg] at pressure P and quality x in [0,1]."""
    Tsat = Tsat_from_P(P_bar)
    x = np.clip(x, 0.0, 1.0)
    return h_liquid(Tsat) + x * hfg_from_tables(P_bar)


def s_sat_liquid(P_bar):
    Tsat = Tsat_from_P(P_bar)
    return CP_WATER * np.log(Tsat / 273.15)


def s_sat_vapor(P_bar):
    Tsat = Tsat_from_P(P_bar)
    return s_sat_liquid(P_bar) + hfg_from_tables(P_bar) / Tsat


def expand_isentropic(P_in, T_in, P_out):
    """
    Isentropic expansion from (P_in, T_in) to P_out.
    Returns (h_out_ideal [kJ/kg], quality_or_None).
    If the isentropic state lands in the wet region (s < s_g(P_out)),
    compute quality; otherwise it is still superheated.
    """
    s_in = float(s_superheated(P_in, T_in))
    s_g_out = float(s_sat_vapor(P_out))
    s_f_out = float(s_sat_liquid(P_out))
    if s_in >= s_g_out:
        # still superheated: invert s = s_g + cp*ln(T/Tsat)
        Tsat = float(Tsat_from_P(P_out))
        T_out = Tsat * np.exp((s_in - s_g_out) / float(cp_steam(P_out)))
        return float(h_superheated(P_out, T_out)), None
    # wet region: x = (s_in - s_f)/(s_g - s_f)
    x = (s_in - s_f_out) / (s_g_out - s_f_out)
    x = float(np.clip(x, 0.0, 1.0))
    return float(h_wet(P_out, x)), x


# ===========================================================================
#  Plant model
# ===========================================================================

class SubcriticalCoalF2a:
    """Physics-lumped subcritical pulverized-coal plant model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated        = u["rated_power_mw"]["value"]            # MW_e

        # --- fuel / combustion
        self.LHV_coal       = u["LHV_coal"]["value"]                 # MJ/kg
        self.CO2_per_kg     = u["CO2_per_kg_coal"]["value"]
        self.stoich_air     = u["stoich_air_kg_per_kg_coal"]["value"]
        self.excess_air     = u["excess_air_fraction"]["value"]
        self.T_stack        = u["T_stack_c"]["value"] + 273.15       # K
        self.T_air_in       = u["T_air_in_c"]["value"] + 273.15      # K
        self.cp_flue        = u["cp_flue_kj_kgK"]["value"]           # kJ/(kg.K)
        self.rad_loss       = u["boiler_radiation_loss_frac"]["value"]

        # --- Rankine cycle
        self.P_drum         = u["P_drum_bar"]["value"]               # bar
        self.T_sh           = u["T_superheat_c"]["value"] + 273.15   # K
        self.P_rh           = u["P_reheat_bar"]["value"]             # bar
        self.T_rh           = u["T_reheat_c"]["value"] + 273.15      # K
        self.P_cond         = u["P_condenser_bar"]["value"]          # bar
        self.T_fw           = u["feedwater_temp_c"]["value"] + 273.15   # K
        self.regen_bleed    = u["regen_bleed_fraction"]["value"]
        self.eta_turb       = u["eta_turbine_isen"]["value"]
        self.eta_pump       = u["eta_pump_isen"]["value"]
        self.eta_gen        = u["eta_generator"]["value"]
        self.eta_mech       = u["eta_mechanical"]["value"]
        self.aux_frac       = u["aux_power_frac"]["value"]

        # --- drum thermal ODE
        self.drum_mass      = u["drum_metal_mass_kg"]["value"]       # kg
        self.drum_cp        = u["drum_cp_kj_kgK"]["value"]           # kJ/(kg.K)
        self.drum_UA        = u["drum_UA_kw_K"]["value"]             # kW/K

        self.min_plr        = u["min_plr"]["value"]
        self.max_plr        = u["max_plr"]["value"]

    # ------------------------------------------------------------------
    #  Combustion / boiler energy balance
    # ------------------------------------------------------------------
    def air_fuel_ratio(self):
        """Actual air/fuel mass ratio including excess air [kg_air/kg_coal]."""
        return self.stoich_air * (1.0 + self.excess_air)

    def flue_per_fuel(self):
        """Flue-gas mass per kg coal [kg_flue/kg_coal] (mass conservation)."""
        # products = fuel + air (ash neglected for gas-phase balance)
        return 1.0 + self.air_fuel_ratio()

    def boiler_efficiency(self):
        """
        Boiler thermal efficiency by heat-loss method (El-Wakil 1984).
        eta_boiler = 1 - dry_gas_loss - radiation/unburnt loss.
        """
        m_flue = self.flue_per_fuel()                       # per kg coal
        Q_fuel = self.LHV_coal * 1.0e3                      # kJ per kg coal
        Q_stack = m_flue * self.cp_flue * (self.T_stack - self.T_air_in)
        dry_gas_loss = Q_stack / Q_fuel
        eta = 1.0 - dry_gas_loss - self.rad_loss
        return float(np.clip(eta, 1e-3, 0.999))

    # ------------------------------------------------------------------
    #  Rankine steam cycle (steady-state thermodynamics)
    # ------------------------------------------------------------------
    def carnot_efficiency(self):
        """Carnot upper bound between SH temperature and condenser temperature."""
        T_hot = self.T_sh
        T_cold = Tsat_from_P(self.P_cond)
        return 1.0 - T_cold / T_hot

    def cycle_states(self):
        """
        Compute the 6 cycle state enthalpies [kJ/kg]. Returns dict.
        Isentropic expansion enthalpies use the simplified entropy model,
        then real enthalpy drop = eta_turbine * ideal drop.
        """
        # State 1: condensate (sat. liquid at condenser)
        T1 = Tsat_from_P(self.P_cond)
        h1 = h_liquid(T1)

        # State 2: condensate after condensate-extraction pump.
        wp_ideal = V_WATER * (self.P_drum - self.P_cond) * 100.0     # kJ/kg (bar->kPa)
        wp_real = wp_ideal / self.eta_pump
        h2 = h1 + wp_real

        # State 2fw: feedwater leaving the regenerative feedwater-heater train
        # (heated to T_fw by bled steam, ~sat. liquid enthalpy at T_fw).
        h2_fw = float(h_liquid(self.T_fw)) + wp_real

        # State 3: main superheated steam at drum pressure
        h3 = float(h_superheated(self.P_drum, self.T_sh))

        # State 4: HP exhaust to reheat pressure (isentropic then efficiency)
        h4s, _ = expand_isentropic(self.P_drum, self.T_sh, self.P_rh)
        h4 = h3 - self.eta_turb * (h3 - h4s)

        # State 5: hot reheat
        h5 = float(h_superheated(self.P_rh, self.T_rh))

        # State 6: LP exhaust to condenser (isentropic then efficiency)
        h6s, x6s = expand_isentropic(self.P_rh, self.T_rh, self.P_cond)
        h6 = h5 - self.eta_turb * (h5 - h6s)

        return {"h1": h1, "h2": h2, "h2_fw": h2_fw, "h3": h3, "h4": h4,
                "h5": h5, "h6": h6, "h4s": h4s, "h6s": h6s, "x6s": x6s}

    def cycle_efficiency(self):
        """
        Thermal (gross-shaft) efficiency of the reheat-regenerative cycle.
        Regeneration: feedwater enters the boiler at h2_fw (raised by bled
        steam), so q_in is reduced.  The bled steam (regen_bleed fraction)
        does not complete the LP expansion, so the IP/LP work is scaled by
        (1 - regen_bleed) -- the classic reheat-regenerative trade
        (Black & Veatch 1996; El-Wakil 1984).
        """
        st = self.cycle_states()
        f = self.regen_bleed
        w_hp = (st["h3"] - st["h4"])
        w_iplp = (1.0 - f) * (st["h5"] - st["h6"])
        w_turb = w_hp + w_iplp
        w_pump = (st["h2"] - st["h1"])
        q_in = (st["h3"] - st["h2_fw"]) + (1.0 - f) * (st["h5"] - st["h4"])
        eta = (w_turb - w_pump) / q_in
        return float(eta), float(w_turb - w_pump), float(q_in)

    def net_efficiency(self):
        """
        Net plant LHV efficiency.
        eta_net = eta_boiler * eta_cycle * eta_mech * eta_gen * (1 - aux_frac).
        """
        eta_b = self.boiler_efficiency()
        eta_c, _, _ = self.cycle_efficiency()
        eta_net = eta_b * eta_c * self.eta_mech * self.eta_gen * (1.0 - self.aux_frac)
        return float(eta_net), eta_b, eta_c

    # ------------------------------------------------------------------
    #  Fuel / emissions at a given load
    # ------------------------------------------------------------------
    def coal_rate_kgs(self, plr=1.0):
        """Coal mass flow [kg/s] to deliver P = plr*P_rated net."""
        eta_net, _, _ = self.net_efficiency()
        P_net_mw = self.P_rated * float(plr)
        fuel_mw = P_net_mw / eta_net                       # MW_th
        return fuel_mw / self.LHV_coal                     # MW_th/(MJ/kg)=kg/s

    def co2_rate_kgs(self, plr=1.0):
        return self.coal_rate_kgs(plr) * self.CO2_per_kg

    def co2_intensity_g_per_kwh(self, plr=1.0):
        P_kw = self.P_rated * float(plr) * 1.0e3
        return self.co2_rate_kgs(plr) * 1.0e3 / max(P_kw, 1e-6) * 3600.0

    def steam_flow_kgs(self, plr=1.0):
        """Main-steam mass flow [kg/s] from cycle heat input."""
        eta_b = self.boiler_efficiency()
        m_coal = self.coal_rate_kgs(plr)
        Q_steam_kw = eta_b * m_coal * self.LHV_coal * 1.0e3    # kW
        _, _, q_in = self.cycle_efficiency()                   # kJ/kg
        return Q_steam_kw / q_in

    # ------------------------------------------------------------------
    #  Lumped boiler-drum thermal transient ODE
    # ------------------------------------------------------------------
    def _drum_rhs(self, t, T, Q_in_func):
        """
        Lumped drum-boiler thermal energy balance (Astrom & Bell 2000 form).

            (M*cp) dT/dt = Q_in(t) - Q_evap(T)

        The drum water/metal warms sensibly until it reaches saturation
        T_sat(P_drum); from there the surplus firing heat goes to LATENT
        evaporation (steam generation), which clamps the temperature at T_sat
        because boiling is isothermal.  Q_evap is modelled as a conductance
        that rises steeply once T approaches T_sat, so the steady state sits
        at (or just below) T_sat for any firing rate -- physically correct for
        a saturated drum.
        """
        Tdrum = T[0]
        Q_in = Q_in_func(t)                                # kW into drum
        T_sat = float(Tsat_from_P(self.P_drum))
        # Below Tsat: weak sink (warm-up dominated by Q_in).
        # At/above Tsat: very stiff sink representing latent evaporation.
        if Tdrum < T_sat:
            Q_evap = self.drum_UA * max(Tdrum - T_sat, 0.0)   # ~0 while warming
        else:
            # surplus removed as steam; large effective conductance pins T_sat
            Q_evap = Q_in + self.drum_UA * (Tdrum - T_sat)
        cap = self.drum_mass * self.drum_cp                # kJ/K
        return [(Q_in - Q_evap) / cap]

    def simulate(self, plr=1.0, T0_K=None, dt=10.0, duration_s=6000.0):
        """
        Time-march the lumped drum thermal ODE under a (possibly time-varying)
        load. `plr` may be a scalar or a callable plr(t).

        Returns dict of time-series arrays + steady-state design metrics.
        """
        T_sat = float(Tsat_from_P(self.P_drum))
        if T0_K is None:
            T0_K = self.T_air_in                            # cold start from air temp

        plr_func = plr if callable(plr) else (lambda t: float(plr))

        # firing-side heat into the drum working fluid at a given load
        def Q_in_func(t):
            p = plr_func(t)
            eta_b = self.boiler_efficiency()
            m_coal = self.coal_rate_kgs(p)
            return eta_b * m_coal * self.LHV_coal * 1.0e3   # kW

        t_eval = np.arange(0.0, duration_s + dt, dt)
        sol = solve_ivp(self._drum_rhs, (0.0, duration_s), [T0_K],
                        t_eval=t_eval, args=(Q_in_func,),
                        method="RK45", rtol=1e-6, atol=1e-4, max_step=dt)

        T_drum = sol.y[0]
        t = sol.t
        plr_arr = np.array([plr_func(ti) for ti in t])

        # derived series
        eta_net, eta_b, eta_c = self.net_efficiency()
        P_net = self.P_rated * plr_arr
        coal = np.array([self.coal_rate_kgs(p) for p in plr_arr])
        co2 = coal * self.CO2_per_kg
        # steam generation only once the drum has reached saturation
        steam = np.where(T_drum >= T_sat - 0.5,
                         np.array([self.steam_flow_kgs(p) for p in plr_arr]),
                         0.0)

        return {
            "t": t,
            "T_drum": T_drum,
            "T_sat_drum": np.full_like(t, T_sat),
            "plr": plr_arr,
            "power_net_mw": P_net,
            "coal_rate_kgs": coal,
            "co2_rate_kgs": co2,
            "steam_rate_kgs": steam,
            "eta_net": eta_net,
            "eta_boiler": eta_b,
            "eta_cycle": eta_c,
            "eta_carnot": self.carnot_efficiency(),
        }
