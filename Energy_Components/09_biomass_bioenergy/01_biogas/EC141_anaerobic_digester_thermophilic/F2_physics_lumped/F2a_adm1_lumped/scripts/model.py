"""
EC141 -- Anaerobic Digester (Thermophilic) -- F2a Simplified ADM1 Kinetics + Thermal

Physics-lumped 0D model of a continuously-stirred tank reactor (CSTR) operating
at thermophilic (~55 degC) conditions. Couples a reduced 3-step biochemical
kinetics model (a simplified ADM1) with a lumped digester thermal energy balance.
Time integration via scipy.integrate.solve_ivp.

------------------------------------------------------------------------------
Biochemical model (state vector in COD units, kgCOD/m3, plus biomass)
------------------------------------------------------------------------------
States (concentrations in the liquid phase):
    Xc  : particulate / composite substrate            [kgCOD/m3]
    Ss  : soluble (mono/oligomer) substrate            [kgCOD/m3]
    Sa  : volatile fatty acids (VFA, acetate-eq.)      [kgCOD/m3]
    Xaci: acidogenic biomass                           [kgCOD/m3]
    Xmet: (acetoclastic) methanogenic biomass          [kgCOD/m3]

Three conversion steps (ADM1 reduced, Batstone 2002):
  1. Hydrolysis/disintegration (first-order):
        r_hyd = k_hyd * Xc
  2. Acidogenesis (Monod growth of acidogens on Ss):
        mu_aci = mu_max_aci * Ss / (Ks_aci + Ss)
        r_aci  = mu_aci * Xaci                           (substrate->VFA+biomass)
  3. Methanogenesis (Monod growth of methanogens on VFA, with VFA
     non-competitive self-inhibition -- Angelidaki 1993):
        I_vfa  = KI_vfa / (KI_vfa + Sa)
        mu_met = mu_max_met * Sa / (Ks_met + Sa) * I_vfa
        r_met  = mu_met * Xmet                           (VFA->CH4+biomass)

CSTR mass balances with dilution rate D = Q_in / V_liq:
    dXc/dt   = D*(Xc_in - Xc)   - r_hyd
    dSs/dt   = D*(Ss_in - Ss)   + r_hyd        - r_aci/Y_aci
    dSa/dt   = D*(Sa_in - Sa)   + (1-Y_aci)*r_aci/Y_aci - r_met/Y_met
    dXaci/dt = -D*Xaci          + r_aci        - kdec*Xaci
    dXmet/dt = -D*Xmet          + r_met        - kdec*Xmet

  where r_*/Y gives the substrate consumption (growth r_* is in biomass-COD/d).
  Influent particulate Xc_in carries the feed COD; Ss_in=Sa_in=0 by default.

COD -> methane (Batstone 2002): COD removed to gas is the COD flux that leaves
as CH4, i.e. the methanogenic substrate flux not retained as biomass:
    COD_to_CH4 = (1 - Y_met) * r_met / Y_met        [kgCOD/(m3.d)]
    Q_CH4 = f_ch4_cod * V_liq * COD_to_CH4          [m3CH4/day]   (0.35 m3/kgCOD STP)
    Q_biogas = Q_CH4 / CH4_fraction

------------------------------------------------------------------------------
Thermal model (lumped digester energy balance, single capacitance):
------------------------------------------------------------------------------
    rho*cp*V * dT/dt = Q_heater - UA*(T - T_amb) - rho*cp*Q_in*(T - T_feed)
A proportional heater holds the thermophilic setpoint:
    Q_heater = max(0, Kp * (T_set - T) + Q_loss_ss)   (feed-forward + P term)
The *heating demand* (energy that must be supplied) is reported as the steady
heater duty needed to overcome wall loss + sensible feed heating at setpoint.

References:
    Batstone, D.J. et al. (2002). Anaerobic Digestion Model No.1 (ADM1).
        Water Sci. Technol. 45(10), 65-73.
    Angelidaki, I., Ellegaard, L., Ahring, B.K. (1993). A mathematical model for
        dynamic simulation of anaerobic digestion of complex substrates: focusing
        on ammonia inhibition. Biotechnol. Bioeng. 42, 159-166.
    Angelidaki, I. et al. (2009). Water Sci. Technol. 59(5), 927.
    Donoso-Bravo, A. et al. (2011). Model selection, identification and validation
        in anaerobic digestion. Water Res. 45, 5347-5364.
"""

import numpy as np
from scipy.integrate import solve_ivp


class AnaerobicDigesterThermophilicF2a:
    """Thermophilic AD CSTR: reduced ADM1 kinetics coupled to a lumped thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Reactor
        self.V_liq = float(u["V_liq"]["value"])              # m3

        # Hydrolysis
        self.k_hyd = float(u["k_hyd"]["value"])              # 1/day

        # Acidogenesis (Monod)
        self.mu_max_aci = float(u["mu_max_aci"]["value"])    # 1/day
        self.Ks_aci = float(u["Ks_aci"]["value"])            # kgCOD/m3
        self.Y_aci = float(u["Y_aci"]["value"])              # -

        # Methanogenesis (Monod + inhibition)
        self.mu_max_met = float(u["mu_max_met"]["value"])    # 1/day
        self.Ks_met = float(u["Ks_met"]["value"])            # kgCOD/m3
        self.Y_met = float(u["Y_met"]["value"])              # -
        self.KI_vfa = float(u["KI_vfa"]["value"])            # kgCOD/m3

        # Decay
        self.kdec = float(u["kdec"]["value"])                # 1/day

        # Gas
        self.f_ch4_cod = float(u["f_ch4_cod"]["value"])      # m3CH4/kgCOD
        self.CH4_fraction = float(u["CH4_fraction"]["value"])
        self.LHV_CH4 = float(u["LHV_methane_kWh_m3"]["value"])

        # Thermal
        self.rho = float(u["rho_slurry"]["value"])           # kg/m3
        self.cp = float(u["cp_slurry"]["value"])             # J/(kg.K)
        self.UA = float(u["UA_loss"]["value"])               # W/K
        self.T_set = float(u["T_set_degC"]["value"]) + 273.15
        self.T_amb = float(u["T_ambient_degC"]["value"]) + 273.15
        self.T_feed = float(u["T_feed_degC"]["value"]) + 273.15

    # ------------------------------------------------------------------
    # Kinetics helpers
    # ------------------------------------------------------------------
    def monod(self, S, Ks):
        """Monod saturation factor S/(Ks+S), clipped non-negative."""
        S = max(S, 0.0)
        return S / (Ks + S)

    def inhibition_vfa(self, Sa):
        """Non-competitive VFA self-inhibition KI/(KI+Sa) (Angelidaki 1993)."""
        Sa = max(Sa, 0.0)
        return self.KI_vfa / (self.KI_vfa + Sa)

    def reaction_rates(self, Xc, Ss, Sa, Xaci, Xmet):
        """Return process rates (kgCOD/(m3.day)) for the three steps + growths."""
        r_hyd = self.k_hyd * max(Xc, 0.0)
        mu_aci = self.mu_max_aci * self.monod(Ss, self.Ks_aci)
        r_aci = mu_aci * max(Xaci, 0.0)                       # acidogen growth
        mu_met = (self.mu_max_met * self.monod(Sa, self.Ks_met)
                  * self.inhibition_vfa(Sa))
        r_met = mu_met * max(Xmet, 0.0)                       # methanogen growth
        return r_hyd, r_aci, r_met

    # ------------------------------------------------------------------
    # Biochemical CSTR derivatives
    # ------------------------------------------------------------------
    def dydt_bio(self, y, D, Xc_in, Ss_in, Sa_in):
        """Derivatives of [Xc, Ss, Sa, Xaci, Xmet] (per day)."""
        Xc, Ss, Sa, Xaci, Xmet = y
        r_hyd, r_aci, r_met = self.reaction_rates(Xc, Ss, Sa, Xaci, Xmet)

        # substrate consumption = growth / yield
        s_aci = r_aci / self.Y_aci           # Ss consumed by acidogens
        s_met = r_met / self.Y_met           # Sa consumed by methanogens

        dXc = D * (Xc_in - Xc) - r_hyd
        dSs = D * (Ss_in - Ss) + r_hyd - s_aci
        # COD-balanced: fraction (1-Y_aci) of consumed Ss is routed to VFA
        dSa = D * (Sa_in - Sa) + (1.0 - self.Y_aci) * s_aci - s_met
        dXaci = -D * Xaci + r_aci - self.kdec * Xaci
        dXmet = -D * Xmet + r_met - self.kdec * Xmet
        return np.array([dXc, dSs, dSa, dXaci, dXmet])

    def ch4_production(self, y):
        """Instantaneous CH4 and biogas flow [m3/day] from methanogenic COD flux."""
        Xc, Ss, Sa, Xaci, Xmet = y
        _, _, r_met = self.reaction_rates(Xc, Ss, Sa, Xaci, Xmet)
        # COD leaving as CH4 = substrate consumed minus biomass retained
        cod_to_ch4 = (1.0 - self.Y_met) * r_met / self.Y_met   # kgCOD/(m3.d)
        Q_ch4 = self.f_ch4_cod * self.V_liq * cod_to_ch4       # m3CH4/day
        Q_ch4 = max(Q_ch4, 0.0)
        Q_biogas = Q_ch4 / self.CH4_fraction if self.CH4_fraction > 0 else 0.0
        return Q_ch4, Q_biogas

    # ------------------------------------------------------------------
    # Thermal model
    # ------------------------------------------------------------------
    def heating_demand_W(self, T, Q_in_m3day):
        """Steady heater duty [W] to hold T against wall loss + feed sensible load."""
        Q_in_s = Q_in_m3day / 86400.0                          # m3/s
        q_wall = self.UA * (T - self.T_amb)                     # W
        q_feed = self.rho * self.cp * Q_in_s * (T - self.T_feed)  # W
        return max(q_wall + q_feed, 0.0)

    def dTdt(self, T, Q_in_m3day, Kp=5000.0):
        """Temperature derivative [K/s] with proportional + feed-forward heater."""
        Q_loss_ss = self.heating_demand_W(self.T_set, Q_in_m3day)  # feed-forward
        Q_heater = max(Kp * (self.T_set - T) + Q_loss_ss, 0.0)
        Q_in_s = Q_in_m3day / 86400.0
        q_wall = self.UA * (T - self.T_amb)
        q_feed = self.rho * self.cp * Q_in_s * (T - self.T_feed)
        C = self.rho * self.cp * self.V_liq                    # J/K
        return (Q_heater - q_wall - q_feed) / C

    # ------------------------------------------------------------------
    # Coupled time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, S_in_COD, Q_in, T0_degC=55.0, duration_days=60.0,
                 dt_days=0.5, y0=None):
        """
        Simulate the thermophilic AD CSTR.

        Parameters
        ----------
        S_in_COD : float
            Influent particulate substrate concentration [kgCOD/m3].
        Q_in : float
            Volumetric feed rate [m3/day] (sets dilution D=Q_in/V_liq, 1/HRT).
        T0_degC : float
            Initial digester temperature [degC].
        duration_days : float
            Simulation horizon [days].
        dt_days : float
            Output sampling step [days].
        y0 : array-like or None
            Initial [Xc, Ss, Sa, Xaci, Xmet]; defaults to a seeded reactor.

        Returns
        -------
        dict of time-series arrays.
        """
        D = Q_in / self.V_liq                                  # 1/day (= 1/HRT)
        Xc_in, Ss_in, Sa_in = float(S_in_COD), 0.0, 0.0

        if y0 is None:
            # Seeded inoculum: some biomass present, substrate near feed level
            y0_bio = [S_in_COD * 0.5, 0.1, 0.1, 1.0, 1.0]
        else:
            y0_bio = list(y0)
        T0 = T0_degC + 273.15
        y_init = y0_bio + [T0]

        # thermal time-constant (s) is ~hours; bio time-constant is ~days.
        # Integrate both in SI-consistent "day" time for bio, convert T rate.
        SEC_PER_DAY = 86400.0

        def rhs(t, y):
            ybio = y[:5]
            T = y[5]
            dbio = self.dydt_bio(ybio, D, Xc_in, Ss_in, Sa_in)   # per day
            dT_s = self.dTdt(T, Q_in)                            # per second
            dT_day = dT_s * SEC_PER_DAY                          # per day
            return np.concatenate([dbio, [dT_day]])

        t_eval = np.arange(0.0, duration_days + dt_days * 0.5, dt_days)
        t_eval = t_eval[t_eval <= duration_days]

        sol = solve_ivp(
            rhs, (0.0, duration_days), y_init,
            t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9,
        )

        t = sol.t
        Xc, Ss, Sa, Xaci, Xmet = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4]
        T = sol.y[5]
        N = len(t)

        Q_ch4 = np.zeros(N)
        Q_biogas = np.zeros(N)
        heat_W = np.zeros(N)
        for i in range(N):
            yi = [Xc[i], Ss[i], Sa[i], Xaci[i], Xmet[i]]
            Q_ch4[i], Q_biogas[i] = self.ch4_production(yi)
            heat_W[i] = self.heating_demand_W(T[i], Q_in)

        energy_kWh_day = Q_ch4 * self.LHV_CH4

        return {
            "t": t,                                  # days
            "Xc": Xc, "Ss": Ss, "Sa_VFA": Sa,        # kgCOD/m3
            "Xaci": Xaci, "Xmet": Xmet,              # kgCOD/m3
            "temperature": T,                        # K
            "temperature_degC": T - 273.15,          # degC
            "Q_CH4_m3_day": Q_ch4,                   # m3/day
            "Q_biogas_m3_day": Q_biogas,             # m3/day
            "energy_kWh_day": energy_kWh_day,        # kWh/day
            "heating_demand_W": heat_W,              # W
            "CH4_fraction": self.CH4_fraction,
            "dilution_rate": D,                      # 1/day
            "HRT_days": (1.0 / D) if D > 0 else np.inf,
        }
