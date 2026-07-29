"""
EC151 -- Dry Steam Geothermal Plant -- F2a Physics-Lumped Steam-Turbine Expansion

Physics-lumped (0D) first-principles model of a dry-steam geothermal plant.
Dry saturated (or slightly superheated) steam is piped directly from the
reservoir to a steam turbine, expands isentropically (with isentropic
efficiency eta_s) down to condenser pressure, and the net shaft work is
converted to electricity. Net power and the geothermal *utilization
efficiency* follow from the steam enthalpy drop. Non-condensable gas (NCG,
mainly CO2) carries a parasitic gas-extraction work penalty.

A lumped wellhead/turbine transient is integrated with scipy.solve_ivp:
    state x1 = m_dot      (steam mass-flow to turbine, kg/s)
    state x2 = T_casing   (turbine/casing lumped metal temperature, K)

    tau_wh   * d(m_dot)/dt   = m_dot_target(P_wh) - m_dot
    tau_th   * d(T_casing)/dt = T_steam_in(P_wh) - T_casing

The mass-flow first-order lag represents wellhead choke + steam-line
capacitance; the thermal lag represents turbine-casing thermal inertia.
At every instant the (quasi-steady) thermodynamic expansion is evaluated
from the current m_dot, giving the electrical power trajectory.

Thermodynamics (the steam expansion), at each evaluation:
    h_in   = h_g(P_wh) [+ cp_steam * T_superheat   if superheated]
    s_in   = s_g(P_wh)
    h2s    = enthalpy at P_cond, s = s_in           (isentropic end state)
    dh_s   = h_in - h2s                              (isentropic drop)
    dh_act = eta_s * dh_s                            (actual drop)
    w_turb = dh_act                                  (kJ/kg)
    P_gross = m_dot_steam * w_turb * eta_mechgen
    P_ncg   = m_dot_ncg   * w_ncg_extraction         (parasitic)
    P_net   = P_gross - P_ncg

Utilization (2nd-law / functional) efficiency uses the maximum available
work = enthalpy drop to the dead state; here we use the practical plant
metric eta_util = w_net / (h_in - h_f(P_cond)) i.e. work divided by the
heat carried by the steam above the condensate (DiPippo's "utilization
efficiency" form). Carnot bound uses T_sat(P_wh) and T_sat(P_cond).

Steam property correlations are HARD-CODED simplified fits to IAPWS-IF97
on the saturation line over 0.005-1.5 MPa (the dry-steam operating band),
so the model needs no CoolProp / pyXsteam. Fits below reproduce IAPWS
steam-table values to within a few tenths of a percent across this band.

References
----------
DiPippo, R. (2015). Geothermal Power Plants: Principles, Applications,
    Case Studies and Environmental Impact, 4th ed., Ch. 7 (Dry Steam
    Power Plants). Butterworth-Heinemann.
Wagner, W. & Kretzschmar, H.-J. (2008). International Steam Tables
    (IAPWS-IF97), 2nd ed., Springer. (source of the property anchor data)
Moran, Shapiro, Boettner & Bailey (2018). Fundamentals of Engineering
    Thermodynamics, 9th ed., Wiley. (Rankine/turbine expansion framework)
"""

import numpy as np
from scipy.integrate import solve_ivp


class DrySteamGeothermalF2a:
    """Dry-steam geothermal plant -- physics-lumped steam-turbine model."""

    # Reference dead-state / condensate properties
    CP_SUPERHEAT = 2.10        # kJ/(kg.K)  approx cp of low-pressure steam

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_wh = u["P_wh_design"]["value"]            # MPa
        self.P_cond = u["P_cond"]["value"]               # MPa
        self.T_superheat = u["T_superheat"]["value"]     # K
        self.m_dot_design = u["m_dot_steam_design"]["value"]  # kg/s
        self.eta_s = u["eta_isentropic"]["value"]        # -
        self.eta_mechgen = u["eta_mech_gen"]["value"]    # -
        self.x_ncg = u["ncg_mass_fraction"]["value"]     # -
        self.w_ncg = u["w_ncg_extraction"]["value"]      # kJ/kg_ncg
        self.tau_wh = u["tau_wellhead"]["value"]         # s
        self.tau_th = u["tau_thermal"]["value"]          # s

    # ------------------------------------------------------------------
    # Hard-coded simplified IAPWS-IF97 saturation correlations
    # Valid 0.005-1.5 MPa (the dry-steam operating band).
    # Anchored to International Steam Tables (Wagner & Kretzschmar 2008).
    # ------------------------------------------------------------------
    @staticmethod
    def Tsat(P_MPa):
        """Saturation temperature [K] vs pressure [MPa].

        Antoine-type fit to IAPWS-IF97 saturation line.
        P in MPa -> convert to kPa for the correlation.
        """
        P_kPa = np.asarray(P_MPa, dtype=float) * 1000.0
        P_kPa = np.clip(P_kPa, 1.0, 2000.0)
        # Antoine (water): log10(P_kPa) = A - B/(C + T_C)
        # Constants least-squares fit to IAPWS-IF97 saturation line,
        # 5 kPa - 1.5 MPa; reproduces T_sat to within ~0.03 degC.
        A, B, C = 7.07198629, 1657.56128293, 227.1947111
        T_C = B / (A - np.log10(P_kPa)) - C
        return T_C + 273.15

    @staticmethod
    def hf_sat(P_MPa):
        """Saturated-liquid (condensate) enthalpy h_f [kJ/kg] vs P [MPa].

        Polynomial fit to IAPWS-IF97 saturated liquid line, 0.005-1.5 MPa.
        Reproduces steam-table h_f within ~0.3%.
        """
        T = DrySteamGeothermalF2a.Tsat(P_MPa)
        Tc = T - 273.15
        # Fit to IAPWS-IF97 saturated liquid line (kJ/kg), err < ~2 kJ/kg.
        return 4.12606004 * Tc + 6.37551434e-4 * Tc**2

    @staticmethod
    def hg_sat(P_MPa):
        """Saturated-vapour (dry steam) enthalpy h_g [kJ/kg] vs P [MPa].

        Fit to IAPWS-IF97 saturated vapour line, 0.005-1.5 MPa.
        h_g rises gently from ~2514 (5 kPa) to a max ~2792 then falls.
        Reproduces steam-table h_g within ~0.2%.
        """
        P = np.clip(np.asarray(P_MPa, dtype=float), 0.005, 1.5)
        lnP = np.log(P)
        # Quadratic in ln(P) fit to IAPWS-IF97 saturated vapour line
        # (kJ/kg), 5 kPa - 1.5 MPa; err < ~5 kJ/kg (<0.2%).
        return 2776.38273 + 45.2140975 * lnP + 0.802146204 * lnP**2

    @staticmethod
    def hfg_sat(P_MPa):
        """Latent heat of vaporisation h_fg [kJ/kg] = h_g - h_f."""
        return (DrySteamGeothermalF2a.hg_sat(P_MPa)
                - DrySteamGeothermalF2a.hf_sat(P_MPa))

    @staticmethod
    def sf_sat(P_MPa):
        """Saturated-liquid entropy s_f [kJ/(kg.K)] vs P [MPa]."""
        T = DrySteamGeothermalF2a.Tsat(P_MPa)
        # s_f ~ cp_liq * ln(T/T0) (Gibbs, incompressible liquid)
        return 4.186 * np.log(T / 273.15)

    @staticmethod
    def sg_sat(P_MPa):
        """Saturated-vapour entropy s_g [kJ/(kg.K)] vs P [MPa].

        s_g = s_f + h_fg / T_sat   (Clausius, phase change at T_sat).
        Thermodynamically consistent with the h-fits above.
        """
        T = DrySteamGeothermalF2a.Tsat(P_MPa)
        return (DrySteamGeothermalF2a.sf_sat(P_MPa)
                + DrySteamGeothermalF2a.hfg_sat(P_MPa) / T)

    # ------------------------------------------------------------------
    # Inlet steam state
    # ------------------------------------------------------------------
    def inlet_enthalpy(self, P_wh, T_superheat=None):
        """Turbine-inlet steam enthalpy [kJ/kg]."""
        if T_superheat is None:
            T_superheat = self.T_superheat
        h = self.hg_sat(P_wh) + self.CP_SUPERHEAT * T_superheat
        return h

    def inlet_entropy(self, P_wh, T_superheat=None):
        """Turbine-inlet steam entropy [kJ/(kg.K)]."""
        if T_superheat is None:
            T_superheat = self.T_superheat
        Tsat = self.Tsat(P_wh)
        s = self.sg_sat(P_wh)
        if T_superheat > 0:
            # ideal-gas superheat: ds = cp ln(T/Tsat)
            s = s + self.CP_SUPERHEAT * np.log((Tsat + T_superheat) / Tsat)
        return s

    # ------------------------------------------------------------------
    # Isentropic expansion to condenser pressure
    # ------------------------------------------------------------------
    def expansion_endstate(self, P_wh, P_cond, T_superheat=None):
        """Isentropic + actual end states for expansion P_wh -> P_cond.

        Returns dict with h_in, s_in, x2s (exit quality), h2s (isentropic
        exit enthalpy), dh_isentropic, dh_actual, h2_actual.
        """
        h_in = self.inlet_enthalpy(P_wh, T_superheat)
        s_in = self.inlet_entropy(P_wh, T_superheat)

        sf2 = self.sf_sat(P_cond)
        sg2 = self.sg_sat(P_cond)
        hf2 = self.hf_sat(P_cond)
        hfg2 = self.hfg_sat(P_cond)

        # Quality at isentropic exit (wet region typical for dry-steam plants)
        x2s = (s_in - sf2) / (sg2 - sf2)
        x2s = np.clip(x2s, 0.0, 1.0)
        h2s = hf2 + x2s * hfg2

        dh_s = h_in - h2s                       # isentropic drop
        dh_act = self.eta_s * dh_s              # actual drop
        h2_act = h_in - dh_act
        return {
            "h_in": h_in, "s_in": s_in,
            "x2s": x2s, "h2s": h2s,
            "dh_isentropic": dh_s, "dh_actual": dh_act,
            "h2_actual": h2_act,
        }

    # ------------------------------------------------------------------
    # Power & efficiency (quasi-steady, at given flow)
    # ------------------------------------------------------------------
    def specific_work(self, P_wh=None, P_cond=None, T_superheat=None):
        """Net turbine specific work [kJ/kg_steam] after mech/gen losses."""
        P_wh = self.P_wh if P_wh is None else P_wh
        P_cond = self.P_cond if P_cond is None else P_cond
        es = self.expansion_endstate(P_wh, P_cond, T_superheat)
        return es["dh_actual"] * self.eta_mechgen

    def carnot_efficiency(self, P_wh=None, P_cond=None):
        """Carnot bound between T_sat(P_wh) and T_sat(P_cond)."""
        P_wh = self.P_wh if P_wh is None else P_wh
        P_cond = self.P_cond if P_cond is None else P_cond
        Th = self.Tsat(P_wh)
        Tc = self.Tsat(P_cond)
        return 1.0 - Tc / Th

    def power(self, m_dot_steam, P_wh=None, P_cond=None, T_superheat=None,
              x_ncg=None):
        """Net & gross electrical power [kW] and efficiencies at given flow.

        m_dot_steam : total steam-phase mass flow to turbine [kg/s]
                      (NCG fraction rides along; NCG does negligible work).
        Returns dict.
        """
        P_wh = self.P_wh if P_wh is None else P_wh
        P_cond = self.P_cond if P_cond is None else P_cond
        x_ncg = self.x_ncg if x_ncg is None else x_ncg

        es = self.expansion_endstate(P_wh, P_cond, T_superheat)
        w_turb = es["dh_actual"] * self.eta_mechgen          # kJ/kg
        m_dot_steam = np.asarray(m_dot_steam, dtype=float)

        # NCG: rides with the stream; needs gas-extraction (vacuum) work.
        m_dot_ncg = m_dot_steam * x_ncg / max(1.0 - x_ncg, 1e-9)
        m_dot_h2o = m_dot_steam * (1.0 - x_ncg)              # working steam

        P_gross = m_dot_h2o * w_turb                          # kW (kJ/s)
        P_parasitic = m_dot_ncg * self.w_ncg                  # kW
        P_net = P_gross - P_parasitic

        # Heat carried by steam above condensate (utilization denominator)
        h_in = es["h_in"]
        hf_cond = self.hf_sat(P_cond)
        Q_steam = m_dot_h2o * (h_in - hf_cond)                # kW

        eta_util = np.where(Q_steam > 0, P_net / Q_steam, 0.0)
        eta_carnot = self.carnot_efficiency(P_wh, P_cond)
        eta_2nd = np.where(eta_carnot > 0, eta_util / eta_carnot, 0.0)

        return {
            "P_gross_kW": P_gross,
            "P_net_kW": P_net,
            "P_parasitic_kW": P_parasitic,
            "w_specific_kJ_kg": w_turb,
            "Q_steam_kW": Q_steam,
            "eta_utilization": eta_util,
            "eta_carnot": eta_carnot,
            "eta_2nd_law": eta_2nd,
            "m_dot_ncg_kgs": m_dot_ncg,
            "h_in_kJ_kg": h_in,
            "h2_actual_kJ_kg": es["h2_actual"],
            "x2_isentropic": es["x2s"],
        }

    # ------------------------------------------------------------------
    # Lumped wellhead / turbine transient ODE (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def _mdot_target(self, P_wh):
        """Quasi-steady steam flow vs wellhead pressure (choked-flow ~ P)."""
        return self.m_dot_design * (P_wh / self.P_wh)

    def simulate(self, P_wh_input, P_cond=None, T_superheat=None,
                 m_dot0=None, dt=1.0, duration_s=300.0):
        """Integrate the lumped wellhead/turbine transient.

        P_wh_input : float OR callable(t)->P_wh [MPa]  (wellhead pressure)
        Returns time series dict (t, m_dot, T_casing, P_net_kW, ...).
        """
        P_cond = self.P_cond if P_cond is None else P_cond
        T_sh = self.T_superheat if T_superheat is None else T_superheat

        if callable(P_wh_input):
            P_wh_fn = P_wh_input
        else:
            P_wh_fn = lambda t: float(P_wh_input)

        m0 = self._mdot_target(P_wh_fn(0.0)) if m_dot0 is None else m_dot0
        T0 = self.Tsat(P_wh_fn(0.0)) + T_sh

        def rhs(t, x):
            m_dot, T_casing = x
            P_wh = P_wh_fn(t)
            m_target = self._mdot_target(P_wh)
            T_steam_in = self.Tsat(P_wh) + T_sh
            dmdt = (m_target - m_dot) / self.tau_wh
            dTdt = (T_steam_in - T_casing) / self.tau_th
            return [dmdt, dTdt]

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(rhs, (0.0, duration_s), [m0, T0],
                        t_eval=t_eval, method="RK45",
                        rtol=1e-7, atol=1e-9, max_step=dt)

        t = sol.t
        m_dot = sol.y[0]
        T_casing = sol.y[1]

        P_wh_arr = np.array([P_wh_fn(tt) for tt in t])
        P_net = np.zeros_like(t)
        P_gross = np.zeros_like(t)
        eta_util = np.zeros_like(t)
        eta_carnot = np.zeros_like(t)
        for i in range(len(t)):
            r = self.power(m_dot[i], P_wh_arr[i], P_cond, T_sh)
            P_net[i] = r["P_net_kW"]
            P_gross[i] = r["P_gross_kW"]
            eta_util[i] = r["eta_utilization"]
            eta_carnot[i] = r["eta_carnot"]

        return {
            "t": t,
            "m_dot": m_dot,
            "T_casing": T_casing,
            "P_wh_MPa": P_wh_arr,
            "P_net_kW": P_net,
            "P_gross_kW": P_gross,
            "eta_utilization": eta_util,
            "eta_carnot": eta_carnot,
        }
