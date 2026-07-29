"""
EC075 -- Finned-Tube Heat Exchanger -- F2a Physics-Lumped (Transient)
=====================================================================

Air-to-liquid cross-flow finned-tube heat exchanger discretised into N
stream-wise control volumes (CVs). Each CV carries three lumped states:

    T_h[i]  hot stream (liquid water) temperature
    T_c[i]  cold stream (air) temperature
    T_w[i]  tube + fin wall (metal matrix) temperature

Per-CV first-law energy balances (lumped, 0D-per-CV -> 1D chain):

    (m_h cp_h / N) dT_h/dt = mdot_h cp_h (T_h_up - T_h)            (advection in)
                             - (UA_h / N)(T_h - T_w)               (to wall)

    (m_c cp_c / N) dT_c/dt = mdot_c cp_c (T_c_up - T_c)            (advection in)
                             + (eta_o UA_c / N)(T_w - T_c)         (from wall, finned)

    (m_w cp_w / N) dT_w/dt = (UA_h / N)(T_h - T_w)                 (from liquid)
                             - (eta_o UA_c / N)(T_w - T_c)         (to air, finned)

where the wall is the thermal-mass coupling between streams. The air-side
("cold") path is the dominant resistance and carries the fin (overall surface)
efficiency eta_o per Incropera eq. 11.3, q = eta_o (hA)(T_b - T_inf).

UA split. The overall UA = U*A (reused from the F1a e-NTU model) is split into a
liquid-side conductance UA_h and a finned air-side conductance UA_c in series:

    1/UA = 1/UA_h + 1/(eta_o UA_c)

with the air side taking 'air_side_fraction' f of the total resistance:

    eta_o UA_c = UA / f        UA_h = UA / (1 - f)

This guarantees the steady-state series conductance equals UA, so the steady
state reproduces the cross-flow effectiveness-NTU result exactly (in the
single-CV limit) and converges to it as N grows for the multi-CV chain.

Cross-flow handling. For a cross-flow coil the air sweeps across the tube bank
while the liquid flows along the tubes. We model the liquid as advected through
the CV chain (counter index) and the air as locally entering each CV at the
common inlet T_c_in and exchanging with that CV's wall, which is the standard
"one fluid unmixed" cross-flow lumping (Kays & London 1984). Outlet air is the
flow-weighted mean of the per-CV air outlets.

Properties hardcoded (no CoolProp), cited in parameters.json:
    water  cp=4180 J/kgK, rho=992 kg/m3   (Incropera Table A.6, ~40 C)
    air    cp=1006 J/kgK, rho=1.1614 kg/m3 (Incropera Table A.4, 300 K)
    Al     cp=900  J/kgK                   (Incropera Table A.1, 300 K)

Integrator: scipy.integrate.solve_ivp (LSODA, stiff-capable).

References:
    Incropera & DeWitt (2006). Fundamentals of Heat and Mass Transfer, 6th ed.,
        ch. 3 (fin/surface efficiency) and ch. 11 (HX, e-NTU).
    Kays, W.M. & London, A.L. (1984). Compact Heat Exchanger Design, 3rd ed.
"""

import numpy as np
from scipy.integrate import solve_ivp


class FinnedTubeHXF2a:
    """Lumped multi-CV transient model of a finned-tube air-to-liquid HX."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U = float(u["U_overall"]["value"])          # W/m2K
        self.A = float(u["A"]["value"])                  # m2
        self.f_air = float(u["air_side_fraction"]["value"])  # -
        self.eta_o = float(u["fin_efficiency"]["value"])     # -
        self.cp_h = float(u["cp_hot"]["value"])          # J/kgK water
        self.rho_h = float(u["rho_hot"]["value"])        # kg/m3 water
        self.cp_c = float(u["cp_cold"]["value"])         # J/kgK air
        self.rho_c = float(u["rho_cold"]["value"])       # kg/m3 air
        self.cp_w = float(u["cp_wall"]["value"])         # J/kgK metal
        self.m_w = float(u["m_wall"]["value"])           # kg metal
        self.V_h = float(u["V_hot"]["value"])            # m3 liquid
        self.V_c = float(u["V_cold"]["value"])           # m3 air
        self.N = int(u["N_cv"]["value"])                 # CV count

        # Total overall conductance (reused from F1a e-NTU).
        self.UA = self.U * self.A                        # W/K

        # Series split of UA into liquid-side and finned air-side conductances.
        #   eta_o * UA_c = UA / f_air   (air side carries fraction f of resistance)
        #   UA_h         = UA / (1 - f_air)
        self.UA_c_eff = self.UA / self.f_air             # effective air-side (already incl. eta_o)
        self.UA_h = self.UA / (1.0 - self.f_air)         # liquid-side
        # Per-CV conductances
        self.gh = self.UA_h / self.N                     # W/K per CV liquid<->wall
        self.gc = self.UA_c_eff / self.N                 # W/K per CV wall<->air (finned)
        # Cached at simulate(): effectiveness-corrected per-CV conductances so the
        # well-mixed lumped node reproduces the true exponential within-CV approach
        # (LMTD), removing the staged-mixing bias and matching e-NTU. See _rhs.
        self._ghe = None     # geff liquid (set per flow in simulate)
        self._gce = None     # geff air

        # Lumped thermal masses (J/K)
        self.Ch_tot = self.rho_h * self.V_h * self.cp_h
        self.Cc_tot = self.rho_c * self.V_c * self.cp_c
        self.Cw_tot = self.m_w * self.cp_w

    # ---------------------------------------------------------------- helpers
    def fin_efficiency(self):
        """Overall surface (fin) efficiency eta_o used on the air side."""
        return self.eta_o

    def _rhs(self, t, y, Ch_cap, Cc_cap, Cw_cap, mh_cp, mc_cp_cv, Th_in, Tc_in):
        """ODE right-hand side. State vector y = [T_h(0..N-1), T_c, T_w]."""
        N = self.N
        Th = y[0:N]
        Tc = y[N:2 * N]
        Tw = y[2 * N:3 * N]

        # Upstream liquid temperature for each CV (plug flow along tubes).
        Th_up = np.empty(N)
        Th_up[0] = Th_in
        Th_up[1:] = Th[:-1]

        # Air enters every CV at the common inlet (one-fluid-unmixed cross-flow).
        Tc_up = Tc_in

        # Heat transfer driven by the CV INLET temperature with an
        # effectiveness-corrected conductance geff = W*(1-exp(-g/W)). This makes
        # the within-CV temperature follow the true exponential approach to the
        # wall (LMTD), so the lumped well-mixed nodes reproduce the analytic
        # cross-flow e-NTU result instead of the staged-mixing under-prediction.
        q_hw = self._ghe * (Th_up - Tw)     # liquid -> wall  (>0 when liquid hotter)
        q_wc = self._gce * (Tw - Tc_up)     # wall -> air     (>0 when wall hotter)

        # Node temperature represents the CV OUTLET. Advection carries enthalpy
        # in at W*(Tup - T); the wall exchange removes/adds q.
        dTh = (mh_cp * (Th_up - Th) - q_hw) / Ch_cap
        dTc = (mc_cp_cv * (Tc_up - Tc) + q_wc) / Cc_cap
        dTw = (q_hw - q_wc) / Cw_cap

        return np.concatenate([dTh, dTc, dTw])

    # ----------------------------------------------------------- e-NTU ref
    @staticmethod
    def _effectiveness_crossflow(NTU, C_r):
        """Cross-flow, one fluid unmixed (Incropera eq. 11.32)."""
        C_r = max(float(C_r), 1e-10)
        inner = np.exp(-C_r * NTU ** 0.78) - 1.0
        eps = 1.0 - np.exp((NTU ** 0.22 / C_r) * inner)
        return float(np.clip(eps, 0.0, 1.0))

    def steady_state_entu(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold):
        """Analytic e-NTU steady state (reference the transient must approach)."""
        C_h = m_dot_hot * self.cp_h
        C_c = m_dot_cold * self.cp_c
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        if C_min < 1e-12:
            return {"Q_kw": 0.0, "T_h_out": T_h_in, "T_c_out": T_c_in,
                    "effectiveness": 0.0, "NTU": 0.0}
        C_r = C_min / C_max
        NTU = self.UA / C_min
        eps = self._effectiveness_crossflow(NTU, C_r)
        Q = eps * C_min * (T_h_in - T_c_in)
        return {
            "Q_kw": Q / 1000.0,
            "T_h_out": T_h_in - Q / C_h,
            "T_c_out": T_c_in + Q / C_c,
            "effectiveness": eps,
            "NTU": NTU,
        }

    # ------------------------------------------------------------- simulate
    def simulate(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                 duration_s=600.0, n_out=120, T0=None):
        """
        Integrate the transient HX from an initial uniform field to (toward)
        steady state under constant inlet conditions and flow rates.

        Returns dict of time-series arrays plus final outlet quantities.
        """
        N = self.N
        mh_cp = m_dot_hot * self.cp_h          # full liquid stream (series chain)
        mc_cp = m_dot_cold * self.cp_c         # full air stream (split N ways)
        mc_cp_cv = mc_cp / N                   # per-CV parallel air path

        Ch_cap = self.Ch_tot / N
        Cc_cap = self.Cc_tot / N
        Cw_cap = self.Cw_tot / N

        # Effectiveness-corrected per-CV conductances: geff = W*(1-exp(-g/W)).
        # W is the per-CV capacity rate of each stream (liquid full chain, air
        # split N ways). As g/W -> 0 (fine grid) geff -> g; the correction
        # removes the well-mixed-node bias so steady state matches e-NTU.
        self._ghe = mh_cp * (1.0 - np.exp(-self.gh / mh_cp))
        self._gce = mc_cp_cv * (1.0 - np.exp(-self.gc / mc_cp_cv))

        # Initial condition: whole device at cold inlet temperature unless given.
        if T0 is None:
            T0 = T_c_in
        y0 = np.full(3 * N, float(T0))

        t_eval = np.linspace(0.0, duration_s, int(n_out))
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            method="LSODA", t_eval=t_eval,
            args=(Ch_cap, Cc_cap, Cw_cap, mh_cp, mc_cp_cv, T_h_in, T_c_in),
            rtol=1e-7, atol=1e-7,
        )

        Th = sol.y[0:N, :]
        Tc = sol.y[N:2 * N, :]
        Tw = sol.y[2 * N:3 * N, :]

        # Outlet liquid = last CV in the chain.
        T_h_out = Th[-1, :]
        # Outlet air = mean over CVs (parallel air paths, equal split).
        T_c_out = Tc.mean(axis=0)

        # Heat duty from each stream's enthalpy change.
        Q_h = mh_cp * (T_h_in - T_h_out)            # W
        Q_c = mc_cp * (T_c_out - T_c_in)            # W

        # Effectiveness = actual / maximum-possible duty. This is rigorously the
        # e-NTU effectiveness only at steady state; during the cold-start
        # transient the wall thermal mass stores energy so the hot stream's
        # apparent duty Q_h can momentarily exceed C_min*(T_h_in-T_c_in). We use
        # the steady, conserved duty (mean of the two streams) and clip to [0, 1]
        # so the reported effectiveness respects the thermodynamic limit at all t.
        C_min = min(mh_cp, mc_cp)
        denom = C_min * (T_h_in - T_c_in)
        Q_bal = 0.5 * (Q_h + Q_c)
        if abs(denom) > 1e-12:
            eps = np.clip(Q_bal / denom, 0.0, 1.0)
        else:
            eps = np.zeros_like(Q_h)

        return {
            "t": sol.t,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "T_wall_mean": Tw.mean(axis=0),
            "Q_hot_kw": Q_h / 1000.0,
            "Q_cold_kw": Q_c / 1000.0,
            "Q_kw": Q_h / 1000.0,
            "effectiveness": eps,
            "T_h_profile": Th,
            "T_c_profile": Tc,
            "success": bool(sol.success),
        }

    def steady_outputs(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                       duration_s=1800.0):
        """Run long enough to reach steady state; return scalar outlet dict."""
        r = self.simulate(T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                          duration_s=duration_s, n_out=60)
        return {
            "Q_kw": float(r["Q_kw"][-1]),
            "T_h_out": float(r["T_h_out"][-1]),
            "T_c_out": float(r["T_c_out"][-1]),
            "effectiveness": float(r["effectiveness"][-1]),
            "Q_imbalance_kw": float(abs(r["Q_hot_kw"][-1] - r["Q_cold_kw"][-1])),
        }
