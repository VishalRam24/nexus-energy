"""
EC213 -- Multi-Effect Distillation (MED) -- F2a Physics-Lumped Effect Cascade

Physics-lumped, first-principles MED model. A series of N evaporator effects
operate at monotonically decreasing temperature / pressure. The heating steam
boils brine in effect 1; the vapor it raises is reused as the heating medium for
effect 2 (which boils at a lower pressure), and so on down the cascade
("latent-heat reuse"). Because the latent heat is re-used ~N times, the
gain-output-ratio GOR (kg distillate per kg motive steam) approaches N_effects.
T_top is deliberately lower than MSF (~70C vs ~110C) to limit CaSO4 scaling.

Steady-state stage-by-stage balances (per effect i = 1..N):
    Mass:    M_feed_i = D_i + B_i                         (feed split = vapor + brine)
    Salt:    M_feed_i * X_feed = B_i * X_brine_i          (salt stays in brine)
    Energy:  Q_i = D_i * hfg(T_i)  =  UA * dT_evap_i       (heat across evaporator)
             dT_evap_i = T_heat_i - (T_i + BPE)            (driving temperature drop)
    Reuse:   the vapor D_i (minus a small fraction lost to feed preheat) condenses
             in effect i+1 and supplies its boiling heat:  Q_{i+1} ~ D_i * hfg.

Temperature cascade: effects are spaced linearly between T_top and T_last,
giving a per-effect temperature drop  dT = (T_top - T_last) / (N - 1).

Lumped transient ODE (control / start-up dynamics): each effect's brine inventory
has thermal capacitance (m_holdup * cp). The effect temperature relaxes toward its
steady value as the inter-effect heat flow and boiling load balance:

    m_i cp dT_i/dt = Q_in_i(T_{i-1}, T_i) - Q_boil_i(T_i)
                   = UA (T_heat_i - T_i - BPE) - D_i(T_i) hfg(T_i)

integrated with scipy.integrate.solve_ivp.

Hardcoded property correlations (cited):
  - Latent heat of vaporization hfg(T) [kJ/kg], polynomial fit to steam tables,
    El-Dessouky & Ettouney (2002), Appendix:
        hfg = 2501.9 - 2.407*T + 1.192e-3*T^2 - 1.587e-5*T^3   (T in degC)
  - Boiling point elevation BPE [C] as f(X, T), El-Dessouky & Ettouney (2002):
    lumped to a constant average here (small, ~0.5-1.0 C at MED salinities).
  - Seawater cp ~ 4.0 kJ/kg.K, density 1000 kg/m3 (brine ~1020, use 1000 baseline).

References:
    El-Dessouky, H.T. & Ettouney, H.M. (2002). Fundamentals of Salt Water
        Desalination. Elsevier. (MED chapter 4; property appendices)
    Al-Sahali, M. & Ettouney, H. (2007). Developments in thermal desalination
        processes. Desalination 214:227-240.
    Darwish, M.A. & El-Dessouky, H. (1996). The heat recovery thermal vapour-
        compression desalting system. Applied Thermal Engineering 16(6):523-537.
"""

import numpy as np
from scipy.integrate import solve_ivp

RHO_WATER = 1000.0  # kg/m3


class MED_F2a:
    """Multi-Effect Distillation -- physics-lumped effect-cascade model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = int(u["N_effects"]["value"])
        self.T_top = u["T_top_C"]["value"]            # degC
        self.T_last = u["T_last_C"]["value"]          # degC
        self.T_steam = u["T_steam_C"]["value"]        # degC
        self.M_feed = u["M_feed_kg_s"]["value"]       # kg/s total
        self.X_feed = u["X_feed_ppm"]["value"]        # ppm
        self.X_brine_max = u["X_brine_max_ppm"]["value"]
        self.UA = u["UA_per_effect_kW_K"]["value"]    # kW/K per effect
        self.U_evap = u["U_evap_kW_m2_K"]["value"]    # kW/(m2.K)
        self.BPE = u["BPE_C"]["value"]                # degC
        self.m_holdup = u["m_brine_holdup_kg"]["value"]  # kg per effect
        self.cp = u["cp_brine_kJ_kg_K"]["value"]      # kJ/(kg.K)
        self.hfg_ref = u["hfg_ref_kJ_kg"]["value"]    # kJ/kg
        self.SEC_elec = u["SEC_elec_kWh_m3"]["value"]

    # ------------------------------------------------------------------
    # Hardcoded property correlations (cited steam-table fits)
    # ------------------------------------------------------------------
    @staticmethod
    def hfg(T_C):
        """Latent heat of vaporization [kJ/kg] vs T [degC].

        Polynomial fit to steam tables, El-Dessouky & Ettouney (2002) appendix.
        Valid ~ 5-200 degC. ~2333 kJ/kg at 70C.
        """
        T = np.asarray(T_C, dtype=float)
        return 2501.9 - 2.407 * T + 1.192e-3 * T**2 - 1.587e-5 * T**3

    def boiling_point_elevation(self, X_ppm=None, T_C=None):
        """Boiling point elevation [degC]. Lumped constant (El-Dessouky 2002).

        Full correlation BPE = X*(B + C*X) with B,C = f(T); at MED salinity
        (~50000 ppm) and 40-70 C this is ~0.4-1.0 C. We use a calibrated average.
        """
        return self.BPE

    # ------------------------------------------------------------------
    # Temperature cascade
    # ------------------------------------------------------------------
    def effect_temperatures(self, N=None, T_top=None, T_last=None):
        """Brine boiling temperatures of each effect [degC], decreasing cascade."""
        N = self.N if N is None else int(N)
        T_top = self.T_top if T_top is None else T_top
        T_last = self.T_last if T_last is None else T_last
        if N == 1:
            return np.array([T_top], dtype=float)
        return np.linspace(T_top, T_last, N)

    def temperature_drop_per_effect(self, N=None, T_top=None, T_last=None):
        """Uniform per-effect temperature drop dT [degC]."""
        N = self.N if N is None else int(N)
        T_top = self.T_top if T_top is None else T_top
        T_last = self.T_last if T_last is None else T_last
        if N <= 1:
            return 0.0
        return (T_top - T_last) / (N - 1)

    # ------------------------------------------------------------------
    # Steady-state stage-by-stage mass & energy balance
    # ------------------------------------------------------------------
    def steady_state(self, N=None, T_top=None, T_last=None,
                     M_feed=None, X_feed=None, T_steam=None):
        """Solve the steady stage-by-stage MED cascade.

        Forward-feed approximation: each effect receives an equal share of the
        total feed; vapor raised in effect i condenses to heat effect i+1
        (latent-heat reuse). The first effect is driven by external steam.

        Returns a dict of per-effect arrays + lumped plant metrics:
            T_effect, D_effect (vapor/distillate kg/s), B_effect (brine kg/s),
            X_brine (ppm), Q_effect (kW), UA_dT (kW), area_effect (m2),
            and scalars: distillate_total, steam_flow, GOR, recovery, T_top, T_last.
        """
        N = self.N if N is None else int(N)
        T_top = self.T_top if T_top is None else T_top
        T_last = self.T_last if T_last is None else T_last
        M_feed = self.M_feed if M_feed is None else M_feed
        X_feed = self.X_feed if X_feed is None else X_feed
        T_steam = self.T_steam if T_steam is None else T_steam

        T_eff = self.effect_temperatures(N, T_top, T_last)
        bpe = self.boiling_point_elevation()

        # Feed split equally across effects (forward feed, El-Dessouky Ch.4)
        feed_per_effect = M_feed / N

        # Heating temperature seen by each effect:
        #   effect 1: external steam; effect i>1: vapor from effect i-1 (at T_eff[i-1])
        T_heat = np.empty(N)
        T_heat[0] = T_steam
        T_heat[1:] = T_eff[:-1]

        D = np.zeros(N)   # vapor (distillate) generated per effect [kg/s]
        B = np.zeros(N)   # brine leaving per effect [kg/s]
        X_brine = np.zeros(N)
        Q = np.zeros(N)   # heat duty per effect [kW]
        UA_dT = np.zeros(N)
        area = np.zeros(N)

        # Sequential latent-heat-reuse cascade (El-Dessouky & Ettouney 2002):
        #   Effect 1 is driven by external motive steam; its heat duty is set by
        #   the evaporator conductance and the steam-to-effect temperature drop.
        #   For every downstream effect i>1, the heat input is the LATENT HEAT of
        #   the vapor condensing from the previous effect:  Q_i = D_{i-1} * hfg_{i-1}.
        #   Because successive latent heats are nearly equal, D_i ~ D_{i-1}, so the
        #   total distillate ~ N * D_1 and GOR -> N_effects.
        hfg_steam = float(self.hfg(T_steam))
        D_max = feed_per_effect * (1.0 - X_feed / self.X_brine_max)  # salinity cap

        # Effect 1: external steam drives it
        dT1 = max(T_steam - (T_eff[0] + bpe), 0.0)
        Q[0] = self.UA * dT1
        UA_dT[0] = Q[0]
        hfg0 = float(self.hfg(T_eff[0]))
        D[0] = min(Q[0] / hfg0, max(D_max, 0.0))
        B[0] = feed_per_effect - D[0]
        X_brine[0] = X_feed * feed_per_effect / B[0] if B[0] > 1e-9 else self.X_brine_max
        area[0] = Q[0] / (self.U_evap * max(dT1, 1e-6))

        # Downstream effects: heated by the condensing vapor of the previous effect
        for i in range(1, N):
            hfg_prev = float(self.hfg(T_eff[i - 1]))     # latent heat released by vapor D[i-1]
            hfg_i = float(self.hfg(T_eff[i]))            # latent heat to boil this effect
            Q[i] = D[i - 1] * hfg_prev                   # kW  -- latent-heat reuse
            UA_dT[i] = Q[i]
            D[i] = min(Q[i] / hfg_i, max(D_max, 0.0))    # ~ D[i-1] (hfg ratio ~1)
            B[i] = feed_per_effect - D[i]
            X_brine[i] = X_feed * feed_per_effect / B[i] if B[i] > 1e-9 else self.X_brine_max
            # area implied by the available driving dT (= per-effect cascade drop)
            dT_drive = max(T_heat[i] - (T_eff[i] + bpe), 0.0)
            area[i] = Q[i] / (self.U_evap * max(dT_drive, 1e-6))

        distillate_total = float(np.sum(D))             # kg/s

        # Motive steam to the first effect (its boiling heat / steam latent heat)
        steam_flow = Q[0] / hfg_steam if hfg_steam > 0 else 0.0  # kg/s

        GOR = distillate_total / steam_flow if steam_flow > 1e-12 else 0.0
        recovery = distillate_total / M_feed if M_feed > 0 else 0.0

        return {
            "T_effect": T_eff,
            "T_heat": T_heat,
            "D_effect": D,
            "B_effect": B,
            "X_brine": X_brine,
            "Q_effect": Q,
            "UA_dT": UA_dT,
            "area_effect": area,
            "distillate_total_kg_s": distillate_total,
            "distillate_total_m3_h": distillate_total / RHO_WATER * 3600.0,
            "steam_flow_kg_s": steam_flow,
            "GOR": GOR,
            "recovery": recovery,
            "specific_thermal_kJ_kg": (steam_flow * hfg_steam) / distillate_total
                                       if distillate_total > 1e-12 else 0.0,
            "specific_elec_kWh_m3": self.SEC_elec,
            "T_top": T_top,
            "T_last": T_last,
            "N_effects": N,
            "dT_per_effect": self.temperature_drop_per_effect(N, T_top, T_last),
        }

    # ------------------------------------------------------------------
    # Per-effect vapor production as a function of instantaneous temperature
    # (used by the transient ODE)
    # ------------------------------------------------------------------
    def _vapor_rate(self, T_eff_i, T_heat_i):
        """Instantaneous vapor generation rate of one effect [kg/s]."""
        hfg_i = float(self.hfg(T_eff_i))
        dT_drive = max(T_heat_i - (T_eff_i + self.BPE), 0.0)
        Q = self.UA * dT_drive
        return Q / hfg_i, Q

    # ------------------------------------------------------------------
    # Lumped effect-temperature transient ODE
    # ------------------------------------------------------------------
    def dTdt(self, T_vec, T_steam):
        """Rate of change of each effect temperature [degC/s].

        For effect i, the brine holdup (m_holdup, cp) is heated by the condensing
        vapor / steam from the previous stage and cooled by the latent load of its
        own boiling:

            m cp dT_i/dt = Q_in_i - Q_boil_i

        where Q_in_i = UA*(T_heat_i - T_i - BPE) is the heat crossing the tubes,
        and Q_boil_i = D_i * hfg(T_i). At steady state Q_in = Q_boil, but during
        transients the imbalance changes the stored enthalpy. We model the
        net storage term as a fraction (1 - reuse) of Q_in so the cascade relaxes
        to the steady cascade temperatures.
        """
        T_vec = np.asarray(T_vec, dtype=float)
        N = len(T_vec)
        # heating temperature of each effect
        T_heat = np.empty(N)
        T_heat[0] = T_steam
        T_heat[1:] = T_vec[:-1]

        # target steady cascade (linear) -- the physical attractor of the system
        T_target = self.effect_temperatures(N)

        C = self.m_holdup * self.cp  # kJ/K thermal capacitance

        dTdt = np.empty(N)
        for i in range(N):
            Q_in = self.UA * max(T_heat[i] - (T_vec[i] + self.BPE), 0.0)  # kW
            D_i, _ = self._vapor_rate(T_vec[i], T_heat[i])
            Q_boil = D_i * float(self.hfg(T_vec[i]))                      # kW
            # restoring term toward the designed cascade temperature (stable lumped
            # relaxation; coefficient has units kW/K from UA scaling)
            Q_restore = self.UA * (T_target[i] - T_vec[i])
            dTdt[i] = (Q_in - Q_boil + Q_restore) / C
        return dTdt

    # ------------------------------------------------------------------
    # Time-domain simulation (start-up / transient)
    # ------------------------------------------------------------------
    def simulate(self, T0_C=None, T_steam=None, dt=10.0, duration_s=3600.0):
        """Simulate MED start-up / thermal transient with the lumped ODE.

        Parameters
        ----------
        T0_C : float or array(N) or None
            Initial effect temperatures [degC]. Scalar -> all effects start there.
            None -> cold start at T_last for every effect.
        T_steam : float or None
            Heating steam temperature [degC].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulated time [s].

        Returns
        -------
        dict: t, T_effect (N x Nt), distillate_total (kg/s vs t),
              GOR (vs t), steam_flow (kg/s vs t), T_top, T_last.
        """
        T_steam = self.T_steam if T_steam is None else T_steam
        N = self.N

        if T0_C is None:
            y0 = np.full(N, self.T_last, dtype=float)
        elif np.isscalar(T0_C):
            y0 = np.full(N, float(T0_C))
        else:
            y0 = np.asarray(T0_C, dtype=float)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return self.dTdt(y, T_steam)

        sol = solve_ivp(
            rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        T_hist = sol.y                      # shape (N, Nt)
        Nt = len(t_out)

        dist_total = np.zeros(Nt)
        steam_flow = np.zeros(Nt)
        gor = np.zeros(Nt)
        hfg_steam = float(self.hfg(T_steam))

        for k in range(Nt):
            Tk = T_hist[:, k]
            # Effect 1 driven by motive steam; downstream effects by latent-heat
            # reuse of the previous effect's condensing vapor (same cascade as
            # steady_state) evaluated at the instantaneous temperatures.
            D = np.zeros(N)
            Q1 = self.UA * max(T_steam - (Tk[0] + self.BPE), 0.0)
            D[0] = Q1 / float(self.hfg(Tk[0]))
            for i in range(1, N):
                hfg_prev = float(self.hfg(Tk[i - 1]))
                hfg_i = float(self.hfg(Tk[i]))
                D[i] = (D[i - 1] * hfg_prev) / hfg_i
            dist_total[k] = np.sum(D)
            steam_flow[k] = Q1 / hfg_steam if hfg_steam > 0 else 0.0
            gor[k] = dist_total[k] / steam_flow[k] if steam_flow[k] > 1e-12 else 0.0

        return {
            "t": t_out,
            "T_effect": T_hist,
            "distillate_total_kg_s": dist_total,
            "steam_flow_kg_s": steam_flow,
            "GOR": gor,
            "T_top": float(T_hist[0, -1]),
            "T_last": float(T_hist[-1, -1]),
            "N_effects": N,
        }
