"""
EC076 -- Regenerative Heat Exchanger -- F2a Physics-Lumped Regenerator Matrix

Periodic-flow / rotary regenerator. A thermal-storage matrix is alternately
heated by a hot gas blow and cooled by a cold gas blow. In a rotary wheel the
matrix rotates between the hot and cold ducts; here we model one matrix
"slug" that experiences a hot blow of duration P then a cold blow of duration
P (= 30 / rpm seconds each for a wheel split 50/50 between ducts).

Governing physics (Kays & London 1984, ch.5; Shah & Sekulic 2003, ch.5):

The matrix is discretized into N axial nodes (in the gas-flow direction).
Within a single blow the gas is assumed quasi-steady (gas thermal capacity
<< matrix capacity, the usual regenerator assumption), so the per-node gas
temperature follows a plug-flow attenuation while the matrix wall stores /
releases energy.

  Per-node matrix energy balance (ODE integrated in time):
      (M_w cp_w / N) dT_w,i/dt = hA/N * (T_g,i_mean - T_w,i)
  where T_g,i_mean is the mean gas temperature seen by node i during the blow.

  Gas marching along the matrix (steady within a blow), node i (1..N):
      C * (T_g,i_in - T_g,i_out) = (hA/N) * (T_g,i_mean - T_w,i)
  with NTU_node = (hA/N)/C and the log-mean / single-node exponential closure:
      T_g,out = T_w + (T_g,in - T_w) * exp(-NTU_node)
      T_g,mean = T_w + (T_g,in - T_w) * (1 - exp(-NTU_node)) / NTU_node
  so the heat picked up by node i is
      q_i = C * (T_g,in - T_g,out) = C * (T_g,in - T_w) * (1 - exp(-NTU_node)).

These per-node ODEs (state = N matrix temperatures) are integrated with
scipy.integrate.solve_ivp over many hot/cold blow cycles until the matrix
reaches a *periodic* steady state (cyclic equilibrium). The time-averaged
outlet gas temperatures over the converged hot and cold blows give the
regenerator outlet states and the effectiveness.

Closed-form cross-check -- Coppage & London (1953) reduced ε-NTU_o theory
for a balanced/symmetric regenerator:

      NTU_o = (1 / C_min) * 1 / (1/(hA)_h + 1/(hA)_c)      (overall, "modified")
      eps_cf = (1 - exp(-NTU_o (1 - Cr))) / (1 - Cr exp(-NTU_o (1 - Cr)))   (counterflow)
      Cr* = (M_w cp_w) * rev_rate / C_min     (matrix capacity-rate ratio)
      eps = eps_cf * (1 - 1 / (9 Cr*^1.93))    (Kays & London matrix-capacity correction)

The correction term shows the hallmark regenerator behaviour: effectiveness
RISES monotonically with matrix capacity-rate ratio Cr* toward the counterflow
recuperator limit as Cr* -> infinity.

Hardcoded air properties (no CoolProp): cp = 1006 J/(kg.K), rho = 1.184 kg/m3
at 25 C / 1 atm -- Cengel & Ghajar (2015) Heat and Mass Transfer, Table A-15.

References
---------
Coppage, J.E. & London, A.L. (1953). "The periodic-flow regenerator -- a
    summary of design theory." Trans. ASME 75:779-787.
Kays, W.M. & London, A.L. (1984). Compact Heat Exchangers, 3rd ed., ch.5.
Shah, R.K. & Sekulic, D.P. (2003). Fundamentals of Heat Exchanger Design,
    Wiley, ch.5 (Thermal Design Theory for Regenerators).
Cengel, Y.A. & Ghajar, A.J. (2015). Heat and Mass Transfer, 5th ed., Table A-15.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Hardcoded air properties at 25 C, 1 atm -- Cengel & Ghajar (2015) Table A-15.
AIR_CP = 1006.0       # J/(kg.K)
AIR_RHO = 1.184       # kg/m3


class RegeneratorF2a:
    """Physics-lumped periodic-flow / rotary regenerator with an N-node matrix."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = int(u["N_nodes"]["value"])
        self.C_h = float(u["C_h"]["value"])
        self.C_c = float(u["C_c"]["value"])
        self.hA_h = float(u["hA_h"]["value"])
        self.hA_c = float(u["hA_c"]["value"])
        self.M_matrix = float(u["M_matrix"]["value"])
        self.cp_matrix = float(u["cp_matrix"]["value"])
        self.rpm = float(u["rpm"]["value"])
        self.air_cp = float(u.get("air_cp", {}).get("value", AIR_CP))
        self.air_rho = float(u.get("air_rho", {}).get("value", AIR_RHO))

    # ------------------------------------------------------------------ #
    #  Closed-form Coppage-London / Kays-London effectiveness            #
    # ------------------------------------------------------------------ #
    def ntu_overall(self, C_h=None, C_c=None, hA_h=None, hA_c=None):
        """Overall (modified) NTU_o referenced to C_min (Shah & Sekulic 2003)."""
        C_h = self.C_h if C_h is None else C_h
        C_c = self.C_c if C_c is None else C_c
        hA_h = self.hA_h if hA_h is None else hA_h
        hA_c = self.hA_c if hA_c is None else hA_c
        C_min = min(C_h, C_c)
        UA = 1.0 / (1.0 / hA_h + 1.0 / hA_c)   # series convective conductance
        return UA / C_min

    def matrix_capacity_ratio(self, C_h=None, C_c=None, rpm=None):
        """Cr* = (M_w cp_w * rev_rate) / C_min  (matrix capacity-rate ratio)."""
        C_h = self.C_h if C_h is None else C_h
        C_c = self.C_c if C_c is None else C_c
        rpm = self.rpm if rpm is None else rpm
        C_min = min(C_h, C_c)
        rev_rate = rpm / 60.0          # rev/s
        C_r_matrix = self.M_matrix * self.cp_matrix * rev_rate
        return C_r_matrix / C_min

    @staticmethod
    def _eps_counterflow(NTU, Cr):
        if abs(Cr - 1.0) < 1e-6:
            return NTU / (NTU + 1.0)
        ex = np.exp(-NTU * (1.0 - Cr))
        return (1.0 - ex) / (1.0 - Cr * ex)

    def effectiveness_correlation(self, C_h=None, C_c=None, hA_h=None,
                                  hA_c=None, rpm=None):
        """Coppage-London regenerator effectiveness with matrix-capacity
        correction (Kays & London 1984, eq. 5-... ; Shah & Sekulic 2003)."""
        C_h = self.C_h if C_h is None else C_h
        C_c = self.C_c if C_c is None else C_c
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        Cr = C_min / C_max
        NTU_o = self.ntu_overall(C_h, C_c, hA_h, hA_c)
        eps_cf = self._eps_counterflow(NTU_o, Cr)
        Cr_star = self.matrix_capacity_ratio(C_h, C_c, rpm)
        # Kays & London matrix-capacity-rate correction factor (>=0, ->1):
        corr = 1.0 - 1.0 / (9.0 * Cr_star ** 1.93) if Cr_star > 0 else 0.0
        corr = max(0.0, min(1.0, corr))
        eps = eps_cf * corr
        return float(np.clip(eps, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  N-node matrix ODE simulation to periodic steady state             #
    # ------------------------------------------------------------------ #
    def _blow_marching(self, Tw, T_in, C, hA):
        """Given matrix node temps Tw (length N) and inlet gas T_in for one
        blow direction, march the steady gas through the N nodes and return
        per-node mean gas temperature, node heat q_i [W], and outlet gas T."""
        N = self.N
        NTU_node = (hA / N) / C
        a = 1.0 - np.exp(-NTU_node)              # fraction of (T_in-Tw) exchanged
        # mean-gas weighting (log-mean closure); guard NTU_node->0
        if NTU_node > 1e-9:
            w_mean = (1.0 - np.exp(-NTU_node)) / NTU_node
        else:
            w_mean = 1.0
        Tg_mean = np.empty(N)
        q = np.empty(N)
        Tg = T_in
        for i in range(N):
            dT = Tg - Tw[i]
            Tg_mean[i] = Tw[i] + dT * w_mean
            q[i] = C * dT * a                    # heat from gas -> node (W)
            Tg = Tg - dT * a                     # gas outlet of node i
        return Tg_mean, q, Tg

    def _matrix_rhs(self, t, Tw, phase, T_h_in, T_c_in):
        """ODE RHS: dTw/dt for the N matrix nodes during 'hot' or 'cold' blow."""
        N = self.N
        Cw_node = self.M_matrix * self.cp_matrix / N   # J/K per node
        if phase == "hot":
            Tg_mean, q, _ = self._blow_marching(Tw, T_h_in, self.C_h, self.hA_h)
        else:
            # cold blow: gas enters at the *opposite* face -> reverse node order
            Tw_rev = Tw[::-1]
            Tg_mean, q, _ = self._blow_marching(Tw_rev, T_c_in, self.C_c, self.hA_c)
            q = q[::-1]
        return q / Cw_node    # dTw/dt (q>0 heats node)

    def _blow_outlet(self, Tw_traj, t_traj, phase, T_h_in, T_c_in):
        """Time-average the outlet gas temperature over a blow given the matrix
        temperature trajectory Tw_traj (shape N x M) at times t_traj."""
        M = Tw_traj.shape[1]
        Tg_out = np.empty(M)
        for k in range(M):
            Tw = Tw_traj[:, k]
            if phase == "hot":
                _, _, Tg = self._blow_marching(Tw, T_h_in, self.C_h, self.hA_h)
            else:
                _, _, Tg = self._blow_marching(Tw[::-1], T_c_in, self.C_c, self.hA_c)
            Tg_out[k] = Tg
        # time average over the blow
        if M > 1:
            return np.trapz(Tg_out, t_traj) / (t_traj[-1] - t_traj[0])
        return Tg_out[0]

    def simulate(self, T_h_in, T_c_in, n_cycles=60, n_eval=40,
                 C_h=None, C_c=None, rpm=None):
        """Integrate the N-node matrix through hot/cold blow cycles with
        solve_ivp until periodic steady state.

        Returns dict with converged time-averaged outlet temperatures,
        effectiveness (ODE-derived and correlation), heat duty, and the
        cycle history of effectiveness.
        """
        if C_h is not None:
            self.C_h = float(C_h)
        if C_c is not None:
            self.C_c = float(C_c)
        if rpm is not None:
            self.rpm = float(rpm)

        N = self.N
        rev_rate = self.rpm / 60.0
        # Blow period: a 50/50 split wheel spends half a revolution in each duct
        P = 0.5 / rev_rate if rev_rate > 0 else 1.0   # seconds per blow

        C_min = min(self.C_h, self.C_c)
        dT_max = T_h_in - T_c_in

        # initial matrix profile: linear between the two inlet temps
        Tw = np.linspace(T_h_in, T_c_in, N)

        eps_history = []
        Th_out_avg = Tc_out_avg = 0.0
        prev_eps = -1.0
        for cyc in range(n_cycles):
            # --- hot blow ---
            t_eval_h = np.linspace(0.0, P, n_eval)
            sol_h = solve_ivp(self._matrix_rhs, (0.0, P), Tw,
                              args=("hot", T_h_in, T_c_in),
                              t_eval=t_eval_h, method="RK45",
                              rtol=1e-6, atol=1e-6, max_step=P / 4.0)
            Tw = sol_h.y[:, -1]
            Th_out_avg = self._blow_outlet(sol_h.y, sol_h.t, "hot",
                                           T_h_in, T_c_in)
            # --- cold blow ---
            t_eval_c = np.linspace(0.0, P, n_eval)
            sol_c = solve_ivp(self._matrix_rhs, (0.0, P), Tw,
                              args=("cold", T_h_in, T_c_in),
                              t_eval=t_eval_c, method="RK45",
                              rtol=1e-6, atol=1e-6, max_step=P / 4.0)
            Tw = sol_c.y[:, -1]
            Tc_out_avg = self._blow_outlet(sol_c.y, sol_c.t, "cold",
                                           T_h_in, T_c_in)

            # effectiveness from cold-stream temperature rise (q_actual/q_max)
            if abs(dT_max) > 1e-12:
                q_cold = self.C_c * (Tc_out_avg - T_c_in)
                eps = q_cold / (C_min * dT_max)
            else:
                eps = 0.0
            eps_history.append(eps)

            if abs(eps - prev_eps) < 1e-7 and cyc > 5:
                break
            prev_eps = eps

        eps_ode = float(np.clip(eps_history[-1], 0.0, 1.0))
        q_cold = self.C_c * (Tc_out_avg - T_c_in)        # W
        q_hot = self.C_h * (T_h_in - Th_out_avg)         # W

        return {
            "T_h_out": float(Th_out_avg),
            "T_c_out": float(Tc_out_avg),
            "effectiveness_ode": eps_ode,
            "effectiveness_correlation": self.effectiveness_correlation(),
            "Q_kW": float(q_cold / 1000.0),
            "Q_hot_kW": float(q_hot / 1000.0),
            "NTU_o": self.ntu_overall(),
            "Cr_star": self.matrix_capacity_ratio(),
            "blow_period_s": float(P),
            "n_cycles_run": len(eps_history),
            "eps_history": np.asarray(eps_history),
            "matrix_profile_final": Tw.copy(),
        }
