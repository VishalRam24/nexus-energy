"""
EC214 -- Mechanical Vapor Compression (MVC) Desalination -- F2a Physics-Lumped

Single-effect evaporator in which the vapor boiled off the brine is mechanically
compressed (raising its saturation temperature) and routed back to the evaporator
tubes as the *heating* steam. This is a vapor-driven heat pump: the only energy
input is the compressor shaft work (electricity); there is no external steam.

================================  PHYSICS  ===================================

Brine boils at the evaporator temperature  T_b  [K]. Because of dissolved salt
the vapor leaves at the saturation temperature of pure water  T_v = T_b - BPE,
where BPE is the boiling-point elevation. To reuse this vapor as heating steam
it must condense *hotter* than the boiling brine, so the compressor lifts its
saturation temperature to

    T_steam = T_v + dT_lift                       (dT_lift = net tube ΔT)

The compressor pressure ratio follows from the Clausius-Clapeyron saturation
curve between T_v and T_steam.

1) Compressor work (per kg vapor) -- ideal gas / polytropic relation
   (El-Dessouky & Ettouney 2002, Fundamentals of Salt Water Desalination, Ch.6):

       w_s = cp_v * T_v * ( (P_s/P_v)^((g-1)/g) - 1 )      [kJ/kg]  isentropic
       w   = w_s / (eta_comp * eta_motor)                  actual electrical

   The compressed vapor is desuperheated; its latent heat at T_steam drives
   evaporation of fresh brine -- the heat-pump effect.

2) Evaporator UA heat transfer (the reused steam condenses on the tubes):

       Q_evap = UA * (T_steam - T_b)                       [kW]

   and the latent heat balance sets the distillate (vapor) production rate

       m_dist = Q_evap / hfg(T_b)                           [kg/s]

3) Specific electric energy consumption (SEC):

       SEC = (m_dist * w) / m_dist  -> w  in kJ/kg
       SEC[kWh/m3] = w[kJ/kg] * rho_dist / 3600

   plus a small sensible-heat term for raising feed to boiling (after the
   regenerative pre-heater recovers most of it). Real MVC SEC ~ 7-12 kWh/m3
   (Veza 1995; El-Dessouky & Ettouney 2002).

4) Lumped transient (0D ODE, integrated with scipy.solve_ivp):

   Thermal energy of the evaporator (brine holdup + metal):
       (M_b*cp_b + M_m*cp_m) dT_b/dt = Q_evap - m_dist*hfg - Q_feed_sensible
   Liquid level (mass balance on the sump):
       A_lvl*rho_b dL/dt = m_feed - m_dist - m_brine_blowdown

   At steady operation Q_evap ≈ m_dist*hfg and feed = dist + blowdown, so
   dT_b/dt -> 0 and dL/dt -> 0 (conservation enforced and tested).

Boiling-point elevation (Sharqawy, Lienhard & Zubair 2010, Desal. Water Treat.
16:354-380) and seawater cp/rho from the same source. Latent heat hfg(T) and
saturation pressure from IAPWS-IF97 / Antoine-type fits (NIST).

References:
    El-Dessouky, H.T. & Ettouney, H.M. (2002). Fundamentals of Salt Water
        Desalination. Elsevier. Chapter 6 (Single-Effect Mechanical Vapor
        Compression).
    Veza, J.M. (1995). Mechanical vapour compression desalination plants -- A
        case study. Desalination 101:1-10.
    Sharqawy, M.H., Lienhard, J.H., Zubair, S.M. (2010). Thermophysical
        properties of seawater: a review. Desal. Water Treat. 16:354-380.
"""

import numpy as np
from scipy.integrate import solve_ivp

T0_C = 273.15  # 0 degC in K


class MVC_F2a:
    """Single-effect MVC evaporator -- physics-lumped, heat-pump vapor cycle."""

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["constants"]

        # design / operating
        self.cap_m3_day = u["capacity_m3_day"]["value"]
        self.T_brine_C = u["T_brine_C"]["value"]
        self.dT_lift = u["delta_T_lift_C"]["value"]
        self.sal_ppm = u["feed_salinity_ppm"]["value"]
        self.recovery = u["recovery"]["value"]
        self.eta_comp = u["compressor_efficiency"]["value"]
        self.eta_motor = u["motor_efficiency"]["value"]
        self.UA = u["UA_kW_K"]["value"]                 # kW/K
        self.M_b = u["evap_holdup_kg"]["value"]
        self.M_m = u["evap_metal_mass_kg"]["value"]
        self.A_lvl = u["level_area_m2"]["value"]
        self.T_feed_C = u["T_feed_C"]["value"]
        self.preheat_eff = u["feed_preheat_eff"]["value"]

        # properties
        self.cp_b = c["cp_brine_kJ_kgK"]["value"]
        self.cp_m = c["cp_metal_kJ_kgK"]["value"]
        self.rho_b = c["rho_brine_kg_m3"]["value"]
        self.hfg_ref = c["hfg_ref_kJ_kg"]["value"]      # at 60 C
        self.cp_v = c["cp_vapor_kJ_kgK"]["value"]
        self.gamma = c["gamma_vapor"]["value"]
        self.R_v = c["R_vapor_kJ_kgK"]["value"]

    # ---------------------------------------------------------------- props
    def hfg(self, T_C):
        """Latent heat of vaporization of water [kJ/kg] (Watson/IAPWS fit).
        Anchored at 2358 kJ/kg @ 60 C; ~2257 @ 100 C, ~2454 @ 20 C."""
        # Linear-ish around operating band (valid 20-100 C, err < 0.5%)
        return self.hfg_ref - 2.45 * (T_C - 60.0)

    def p_sat_kPa(self, T_C):
        """Saturation pressure of water [kPa] -- Antoine (NIST, 1-100 C).
        log10(P_bar) = A - B/(C+T). Constants for water (Bridgeman & Aldrich)."""
        A, B, C = 5.40221, 1838.675, 241.263  # T in degC, P in bar
        P_bar = 10.0 ** (A - B / (C + T_C))
        return P_bar * 100.0  # bar -> kPa

    def bpe(self, T_C=None):
        """Boiling-point elevation [K] for seawater
        (El-Dessouky & Ettouney 2002, Eq. for BPE; salinity X in weight %).
        Gives ~0.50 K at 42 g/kg, 60 C; rises with salinity and T."""
        if T_C is None:
            T_C = self.T_brine_C
        X = self.sal_ppm / 10000.0  # ppm -> weight percent
        A = 8.325e-2 + 1.883e-4 * T_C + 4.02e-6 * T_C ** 2
        B = -7.625e-4 + 9.02e-5 * T_C - 5.2e-7 * T_C ** 2
        Cc = 1.522e-4 - 3.0e-6 * T_C - 3.0e-8 * T_C ** 2
        bpe = A * X + B * X ** 2 + Cc * X ** 3
        return max(bpe, 0.0)

    # ------------------------------------------------------- temperatures
    def temperatures(self, T_brine_C=None, dT_lift=None):
        """Return (T_brine, T_vapor, T_steam) [degC].
        T_vapor = T_brine - BPE (pure-water sat temp of evolved vapor).
        T_steam = T_vapor + dT_lift (compressor lifts it above brine)."""
        Tb = self.T_brine_C if T_brine_C is None else T_brine_C
        dT = self.dT_lift if dT_lift is None else dT_lift
        bpe = self.bpe(Tb)
        Tv = Tb - bpe
        Ts = Tv + dT
        return Tb, Tv, Ts

    # ----------------------------------------------------- compressor work
    def compressor_work(self, T_brine_C=None, dT_lift=None):
        """Actual electrical compressor work per kg vapor [kJ/kg].
        Isentropic polytropic compression of low-pressure steam between the
        saturation pressures at T_vapor and T_steam, divided by efficiencies."""
        Tb, Tv, Ts = self.temperatures(T_brine_C, dT_lift)
        P_v = self.p_sat_kPa(Tv)
        P_s = self.p_sat_kPa(Ts)
        pr = P_s / P_v
        Tv_K = Tv + T0_C
        cp_v = self.cp_v
        # isentropic enthalpy rise (ideal-gas, El-Dessouky & Ettouney 2002)
        w_s = cp_v * Tv_K * (pr ** ((self.gamma - 1.0) / self.gamma) - 1.0)
        w_act = w_s / (self.eta_comp * self.eta_motor)
        return w_act, w_s, pr

    # ---------------------------------------------------------------- SEC
    def specific_energy(self, T_brine_C=None, dT_lift=None):
        """Specific electric energy consumption [kWh/m3].
        Compressor work per kg distillate + residual feed sensible heat
        (after regenerative pre-heat). 1 m3 distillate ~ 1000 kg."""
        w_act, _, _ = self.compressor_work(T_brine_C, dT_lift)
        Tb, _, _ = self.temperatures(T_brine_C, dT_lift)
        # The regenerative pre-heater recovers heat from the hot distillate and
        # hot brine, raising the feed to within a small terminal pinch of T_brine.
        # Only this residual pinch heating is a *net* electrical (compressor) load
        # via additional vapor demand. (El-Dessouky & Ettouney 2002, Ch.6.)
        feed_per_dist = 1.0 / self.recovery        # kg feed per kg distillate
        dT_pinch = (Tb - self.T_feed_C) * (1.0 - self.preheat_eff)  # residual K
        q_sens = feed_per_dist * self.cp_b * dT_pinch  # kJ per kg distillate
        w_total = w_act + q_sens                       # kJ/kg distillate
        rho_dist = 1000.0                              # kg/m3 fresh water
        sec_kWh_m3 = w_total * rho_dist / 3600.0
        return sec_kWh_m3

    def gor(self, T_brine_C=None, dT_lift=None):
        """Equivalent gained output ratio = latent heat reused / work in.
        For MVC: GOR_equiv = hfg / w_act (heat-pump amplification)."""
        w_act, _, _ = self.compressor_work(T_brine_C, dT_lift)
        Tb, _, _ = self.temperatures(T_brine_C, dT_lift)
        return self.hfg(Tb) / w_act

    # ------------------------------------------------------ steady design
    def design_point(self):
        """Steady-state design distillate, power, UA-implied dT."""
        m_dist = self.cap_m3_day * 1000.0 / 86400.0   # kg/s
        Tb, Tv, Ts = self.temperatures()
        Q_evap = m_dist * self.hfg(Tb)                 # kW (latent demand)
        w_act, _, pr = self.compressor_work()
        P_elec = m_dist * w_act                        # kW
        sec = self.specific_energy()
        return {
            "m_dist_kg_s": m_dist,
            "Q_evap_kW": Q_evap,
            "P_elec_kW": P_elec,
            "SEC_kWh_m3": sec,
            "GOR_equiv": self.gor(),
            "pressure_ratio": pr,
            "T_brine_C": Tb, "T_vapor_C": Tv, "T_steam_C": Ts,
            "BPE_C": self.bpe(),
        }

    # ----------------------------------------------------- transient ODE
    def _rhs(self, t, y, m_feed, T_steam_set):
        """State y = [T_brine_C, level_m]. Lumped energy + mass balance.

        The compressor holds its discharge saturation temperature at
        T_steam_set (fixed pressure ratio); the condensing steam delivers
        Q_steam = UA*(T_steam_set - T_brine) to the tubes. Part of this heat
        boils brine (latent, producing distillate), the remainder is net
        sensible storage. As T_brine rises toward T_steam_set the driving
        ΔT -> dT_lift-BPE, Q -> design, giving a stable first-order approach.
        """
        Tb, L = y
        # heat delivered by condensing reused (compressed) steam on the tubes
        Q_steam = max(self.UA * (T_steam_set - Tb), 0.0)   # kW
        hfg = self.hfg(Tb)
        # feed sensible load (raise pre-heated feed to current brine temp)
        T_feed_pre = self.T_feed_C + self.preheat_eff * (Tb - self.T_feed_C)
        Q_feed = m_feed * self.cp_b * max(Tb - T_feed_pre, 0.0)  # kW
        # Boiling only occurs once brine reaches its saturation (boiling) temp,
        # T_boil = T_steam_set - dT_lift + BPE = design T_brine. Below that the
        # net steam heat raises sensible temperature; at/above it the surplus
        # becomes latent and produces distillate (vapor). This boiling-fraction
        # switch is what makes T_brine approach the design boiling point.
        T_boil = T_steam_set - (self.dT_lift - self.bpe(Tb))
        Cth = self.M_b * self.cp_b + self.M_m * self.cp_m  # kJ/K  capacitance
        if Tb < T_boil - 1e-6:
            # sub-cooled: all surplus heat is sensible, no boiling yet
            m_dist = 0.0
            dTb_dt = (Q_steam - Q_feed) / Cth
        else:
            # boiling: surplus steam heat over feed load becomes latent
            Q_latent = max(Q_steam - Q_feed, 0.0)
            m_dist = Q_latent / hfg                         # kg/s vapor
            # clamp temperature at boiling point (saturation); tiny restoring term
            dTb_dt = (Q_steam - m_dist * hfg - Q_feed) / Cth - 0.05 * (Tb - T_boil)
        # mass / level balance on the sump: feed in, distillate + blowdown out
        m_blow = max(m_feed - m_dist, 0.0)                 # kg/s brine reject
        dL_dt = (m_feed - m_dist - m_blow) / (self.A_lvl * self.rho_b)  # m/s
        return [dTb_dt, dL_dt]

    def simulate(self, T0_brine_C=None, level0_m=1.0, duration_s=600.0,
                 dt=5.0, dT_lift=None, m_feed=None):
        """Integrate the lumped transient with scipy.solve_ivp.
        Returns dict of time series + scalar steady metrics."""
        if T0_brine_C is None:
            T0_brine_C = self.T_brine_C - 10.0   # cold-ish start
        if dT_lift is None:
            dT_lift = self.dT_lift
        # design feed rate to hit target distillate at steady state
        m_dist_des = self.cap_m3_day * 1000.0 / 86400.0
        if m_feed is None:
            m_feed = m_dist_des / self.recovery
        # compressor discharge saturation temp held at design steam temp
        _, _, T_steam_set = self.temperatures(self.T_brine_C, dT_lift)

        t_eval = np.arange(0.0, duration_s + dt, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T0_brine_C, level0_m],
            t_eval=t_eval, args=(m_feed, T_steam_set),
            method="RK45", rtol=1e-6, atol=1e-8, max_step=dt,
        )
        Tb = sol.y[0]
        L = sol.y[1]
        # derived series (mirror the rhs definitions, incl. boiling switch)
        Q_steam = np.maximum(self.UA * (T_steam_set - Tb), 0.0)
        hfg = self.hfg(Tb)
        T_feed_pre = self.T_feed_C + self.preheat_eff * (Tb - self.T_feed_C)
        Q_feed = m_feed * self.cp_b * np.maximum(Tb - T_feed_pre, 0.0)
        T_boil = T_steam_set - (self.dT_lift - np.array([self.bpe(tb) for tb in Tb]))
        boiling = Tb >= (T_boil - 1e-6)          # no distillate while sub-cooled
        Q_evap = np.where(boiling, np.maximum(Q_steam - Q_feed, 0.0), 0.0)
        m_dist = Q_evap / hfg
        w_act = np.array([self.compressor_work(tb, dT_lift)[0] for tb in Tb])
        P_elec = m_dist * w_act
        sec = np.where(m_dist > 1e-9, P_elec / np.maximum(m_dist, 1e-9)
                       * 1000.0 / 3600.0, 0.0)  # kWh/m3

        return {
            "t": sol.t,
            "T_brine_C": Tb,
            "level_m": L,
            "Q_evap_kW": Q_evap,
            "m_dist_kg_s": m_dist,
            "distillate_m3_day": m_dist * 86400.0 / 1000.0,
            "P_elec_kW": P_elec,
            "SEC_kWh_m3": sec,
            "success": sol.success,
        }
