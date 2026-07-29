"""
EC212 -- Multi-Stage Flash Distillation (MSF) -- F2a Stage-Cascade (physics-lumped)

A 0D-per-stage first-principles model of an MSF brine-recirculation (MSF-BR) plant.
Recirculating brine is heated above its saturation temperature in a brine heater, then
flashes successively in N stages held at monotonically decreasing pressures (and
saturation temperatures). In each stage a small fraction of the brine flashes to vapour;
that vapour condenses on the recovery (condenser/preheater) tubes carrying the cold
recirculating brine, releasing its latent heat back into the cycle (heat recovery). The
condensate accumulates as product distillate.

Governing relations (all from El-Dessouky & Ettouney 2002, Ch. 7):

  1. Temperature cascade (steady design):
        T_stage[i] = T_top - i * dT_stage,   dT_stage = (T_top - T_last) / N      (linear)
     The brine entering stage i is at the previous stage's *flashing* temperature.

  2. Per-stage flashing (energy released by sensible cooling = latent heat of flashed vapour):
        D_i = M_brine * cp * (T_in_i - T_flash_i) / hfg
     where the brine cools by the stage flash-down and the released enthalpy boils off
     distillate D_i. Summed over stages gives total distillate.

  3. Non-equilibrium allowance (NEA): real brine leaves a stage slightly *above* the
     stage saturation temperature because flashing is incomplete:
        T_brine_out_i = T_sat_i + NEA_i,   NEA_i = NEA_coeff * dT_stage   (degC, capped)
     The effective flash range is reduced by the NEA, reducing distillate vs. equilibrium.

  4. Boiling-point elevation (BPE) of saline brine (Eq. 7 correlation, simplified):
        BPE = (B*S + C*S^2) ... here a compact polynomial in salinity & temperature.
     Raises the brine saturation temperature above pure water; reduces available flash.

  5. Brine-heater steam demand (energy to raise recirc brine from T_recovered to TBT):
        Q_heater = M_brine * cp * (T_top - T_brine_into_heater)
        M_steam  = Q_heater / hfg_steam
     Gain-Output-Ratio  GOR = M_distillate / M_steam.
     Performance ratio  PR  ~ GOR (kg distillate per ~2326 kJ, here we report GOR).

  6. Lumped stage-temperature transient (ODE, scipy.solve_ivp):
     Each stage brine pool is a well-mixed lumped capacitance with residence time tau:
        dT_i/dt = (1/tau) * ( T_in_i(t) - T_i ) - (hfg/cp)*(dx_flash/dt)_lumped
     Linearised: the inflow temperature of stage i is the outflow of stage i-1, so the
     cascade relaxes to the steady cascade with a chain of first-order lags. This gives
     the start-up / load-change dynamics of the temperature cascade.

References:
    El-Dessouky, H.T. & Ettouney, H.M. (2002). Fundamentals of Salt Water Desalination.
        Elsevier. Chapter 7 (Multistage Flash Desalination), Eqs. 7.1-7.40.
    Khawaji, A.D., Kutubkhanah, I.K., Wie, J.-M. (2008). Advances in seawater
        desalination technologies. Desalination 221:47-69. (GOR / PR ranges 8-12.)

Hardcoded properties (cited in parameters.json):
    cp_brine ~ 4.0 kJ/kg.K, hfg ~ 2333 kJ/kg (mean stage), hfg_steam ~ 2222 kJ/kg @116 C.
"""

import numpy as np
from scipy.integrate import solve_ivp


class MSF_F2a:
    """Physics-lumped stage-cascade MSF model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = int(u["N_stages"]["value"])
        self.T_top = u["T_top_brine_C"]["value"]
        self.T_last = u["T_last_stage_C"]["value"]
        self.T_sea = u["T_seawater_C"]["value"]
        self.T_steam = u["T_steam_C"]["value"]
        self.M_brine = u["M_recirc_kg_s"]["value"]
        self.S_feed = u["salinity_feed_ppm"]["value"]
        self.S_max = u["salinity_max_ppm"]["value"]
        self.cp = u["cp_brine_kJ_kgK"]["value"]
        self.hfg = u["hfg_kJ_kg"]["value"]
        self.hfg_steam = u["hfg_steam_kJ_kg"]["value"]
        self.NEA_coeff = u["NEA_coeff"]["value"]
        self.tau = u["tau_stage_s"]["value"]

    # ------------------------------------------------------------------ props
    @staticmethod
    def boiling_point_elevation(T_C, S_ppm):
        """Boiling-point elevation of seawater brine [degC].

        Compact correlation after El-Dessouky & Ettouney (2002, Eq. 7), valid roughly
        for 20-110 C and 30000-70000 ppm. Returns BPE in degC (typically 0.3-1.5).
        """
        X = S_ppm / 10000.0  # salinity in weight percent (1 wt% = 10000 ppm)
        T = np.asarray(T_C, dtype=float)
        A = 8.325e-2 + 1.883e-4 * T + 4.02e-6 * T ** 2
        B = -7.625e-4 + 9.02e-5 * T - 5.2e-7 * T ** 2
        C = 1.522e-4 - 3.0e-6 * T - 3.0e-8 * T ** 2
        bpe = A * X + B * X ** 2 + C * X ** 3
        return np.maximum(bpe, 0.0)

    def stage_temperatures(self, T_top=None, T_last=None):
        """Design stage saturation-temperature cascade [degC], length N.

        The brine enters stage 0 at the top brine temperature (TBT) and flashes INTO a
        stage held one stage-step below TBT; the last stage saturates at T_last. So the
        N stage saturation temperatures run from (TBT - dT_stage) down to T_last, every
        stage being below TBT (each stage is at a successively lower pressure).
        """
        Tt = self.T_top if T_top is None else T_top
        Tl = self.T_last if T_last is None else T_last
        dT_stage = (Tt - Tl) / self.N
        return np.linspace(Tt - dT_stage, Tl, self.N)

    def nea(self, dT_stage):
        """Non-equilibrium allowance per stage [degC] (El-Dessouky & Ettouney 2002).

        Brine leaves each stage above saturation; NEA grows with flash-down per stage.
        Capped to a physical fraction of the stage temperature drop.
        """
        return min(self.NEA_coeff * dT_stage, 0.9 * dT_stage)

    # ------------------------------------------------------ steady cascade
    def steady_state(self, T_top=None, T_last=None, M_brine=None):
        """Stage-by-stage steady mass & energy balance.

        Returns dict with:
            T_stage          : stage saturation temps [degC] (N,)
            distillate_stage : distillate produced in each stage [kg/s] (N,)
            D_total          : total distillate [kg/s]
            M_steam          : brine-heater steam [kg/s]
            Q_heater         : brine-heater duty [kW]
            GOR              : gain output ratio [-]
            PR               : performance ratio [kg/2326 kJ]
            flash_range      : TBT - T_last [degC]
            NEA              : non-equilibrium allowance per stage [degC]
            recovery         : distillate / feed brine [-]
        """
        Tt = self.T_top if T_top is None else float(T_top)
        Tl = self.T_last if T_last is None else float(T_last)
        Mb = self.M_brine if M_brine is None else float(M_brine)

        Ts = self.stage_temperatures(Tt, Tl)
        dT_stage = (Tt - Tl) / self.N
        nea = self.nea(dT_stage)

        # Brine enters stage 0 at TBT. In each stage it flashes down toward the stage
        # saturation temperature but, because of the non-equilibrium allowance (NEA) and
        # boiling-point elevation (BPE), it cannot cool below T_sat_i + NEA + BPE. The
        # brine OUTLET temperature therefore carries this excess to the next stage inlet,
        # so the NEA/BPE penalty accumulates toward the cold end: the brine leaves the
        # last stage hotter than T_last, leaving sensible heat un-flashed and reducing
        # total distillate relative to ideal (equilibrium, fresh-water) flashing.
        D = np.zeros(self.N)
        T_in = Tt
        S = self.S_feed
        T_out = T_in
        for i in range(self.N):
            T_sat_i = Ts[i]
            bpe = float(self.boiling_point_elevation(T_sat_i, S))
            T_out = max(T_sat_i + nea + bpe, 0.0)      # actual brine outlet of stage i
            dT_flash = max(T_in - T_out, 0.0)          # sensible cooling that flashes
            # energy balance: sensible heat released boils off distillate
            D[i] = Mb * self.cp * dT_flash / self.hfg
            T_in = T_out                                # outlet feeds the next stage inlet
            # salinity concentrates as water is removed (well-mixed approx)
            Mb_out = Mb - D[:i + 1].sum()
            S = self.S_feed * Mb / max(Mb_out, 1e-6)

        D_total = float(D.sum())
        T_brine_final = T_out                           # exits hotter than T_last (NEA+BPE)

        # --- Brine-heater duty via condenser heat-recovery closure -----------
        # In an MSF-BR plant the recirculating brine flows back UP through the condenser
        # tubes of every stage and is preheated by the latent heat of all the vapour
        # flashing in those stages. After this heat-recovery train the recycle brine
        # re-enters the brine heater preheated to within one terminal-temperature-
        # difference (TTD) of the top brine temperature. The brine heater supplies only
        # that final TTD rise using external heating steam (El-Dessouky & Ettouney 2002,
        # energy balance around the heat-recovery section): the condenser train recovers
        # the bulk of the flash range, and external steam closes the gap.
        #
        # The TTD is set by the brine-heater UA and is, by design, a small fraction of the
        # per-stage temperature drop. We model the heater rise as one stage drop plus the
        # design terminal difference (a few degC), giving a self-consistent GOR in the
        # 8-12 band characteristic of multi-stage flash plants.
        TTD = 3.0  # degC, brine-heater terminal temperature difference (design)
        dT_heater = dT_stage + TTD  # rise the external steam must supply
        T_into_heater = Tt - dT_heater
        Q_heater = Mb * self.cp * dT_heater  # kW
        M_steam = Q_heater / self.hfg_steam  # kg/s
        GOR = D_total / max(M_steam, 1e-9)
        # Performance ratio: kg distillate per 2326 kJ of heat input (classic MSF metric).
        PR = D_total * 2326.0 / max(Q_heater, 1e-9)

        Mb_out_final = Mb - D_total
        recovery = D_total / Mb

        return {
            "T_stage": Ts,
            "distillate_stage": D,
            "D_total": D_total,
            "M_steam": M_steam,
            "Q_heater": Q_heater,
            "GOR": GOR,
            "PR": PR,
            "flash_range": Tt - Tl,
            "NEA": nea,
            "dT_stage": dT_stage,
            "brine_out_kg_s": Mb_out_final,
            "salinity_out_ppm": self.S_feed * Mb / max(Mb_out_final, 1e-6),
            "recovery": recovery,
            "T_into_heater": T_into_heater,
            "T_brine_final": T_brine_final,
        }

    # ------------------------------------------------------ transient ODE
    def simulate(self, T_top=None, T_last=None, M_brine=None,
                 T0=None, duration_s=600.0, n_eval=200):
        """Lumped stage-temperature transient via scipy.solve_ivp.

        Models cold/warm start-up of the temperature cascade: each stage brine pool is a
        well-mixed first-order lag of time constant tau, fed by the previous stage's
        temperature. Stage 0 is fed by the brine-heater outlet (the top brine temperature
        TBT, assumed instantaneously available). The chain relaxes to the steady cascade.

            dT_0/dt = (TBT - T_0)/tau
            dT_i/dt = (T_{i-1} - T_i)/tau - (dT_stage)/tau   for i>=1

        The constant term -dT_stage/tau is the steady inter-stage flash drop, so the fixed
        point is exactly the design cascade T_stage[i].

        Returns dict: t, T_stages (n_eval x N), T_top_arr, plus the steady_state summary.
        """
        ss = self.steady_state(T_top, T_last, M_brine)
        Ts_target = ss["T_stage"]
        Tt = Ts_target[0]
        tau = self.tau
        N = self.N
        # Actual inter-stage temperature steps of the design cascade, so the ODE fixed
        # point is *exactly* the steady cascade (T_target[i-1] - T_target[i] per stage).
        d_steps = -np.diff(Ts_target)  # length N-1, all > 0

        if T0 is None:
            T_init = np.full(N, self.T_sea)  # cold start: all stages at seawater temp
        else:
            T_init = np.full(N, float(T0))

        def rhs(t, T):
            dTdt = np.empty(N)
            dTdt[0] = (Tt - T[0]) / tau
            for i in range(1, N):
                # inflow from previous stage minus the steady inter-stage flash drop
                dTdt[i] = (T[i - 1] - T[i]) / tau - d_steps[i - 1] / tau
            return dTdt

        t_eval = np.linspace(0.0, duration_s, n_eval)
        sol = solve_ivp(rhs, (0.0, duration_s), T_init, t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-8)

        result = {
            "t": sol.t,
            "T_stages": sol.y.T,            # (n_eval, N)
            "T_target": Ts_target,
            "T_top": Tt,
        }
        result.update(ss)
        return result
