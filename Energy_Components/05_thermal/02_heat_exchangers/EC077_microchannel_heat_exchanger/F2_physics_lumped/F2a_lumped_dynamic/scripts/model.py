"""
EC077 -- Microchannel Heat Exchanger (MCHX) -- F2a Lumped Transient Model

Physics-lumped (0D-per-node, 1D-along-flow) transient model. The core is
discretised into N control volumes along the flow direction. Each control
volume carries THREE energy-balance ODE states: hot fluid node, cold fluid
node, and the metal separating wall node. Counterflow arrangement (cold
node index runs opposite to hot).

Per control-volume energy balances (lumped capacitance, plug flow):

    Hot fluid i:
        (m_h_cv * cp_h) dTh_i/dt
            = mdot_h * cp_h * (Th_{i-1} - Th_i)        (advection in - out)
              - h_h * A_cv * (Th_i - Tw_i)             (convection to wall)

    Cold fluid j (counterflow, j enters at node N-1):
        (m_c_cv * cp_c) dTc_j/dt
            = mdot_c * cp_c * (Tc_{j+1} - Tc_j)
              + h_c * A_cv * (Tw_i - Tc_j)

    Wall node i:
        (m_w_cv * cp_w) dTw_i/dt
            = h_h * A_cv * (Th_i - Tw_i)
              - h_c * A_cv * (Tw_i - Tc_j)
              + axial wall conduction (k_wall) between neighbouring wall nodes

Heat-transfer coefficient (microchannel laminar regime):
    Re = rho * u * Dh / mu  (typically << 2300 for liquid microchannels)
    Nu = 4.36 (fully developed, constant heat flux, circular duct;
               Shah & London 1978; Incropera eq. 8.53)
    h  = Nu * k_fluid / Dh
    Because Dh < 1 mm, h is very large (compact, high UA/volume) --
    the defining microchannel feature (Kandlikar & Grande 2003).

Pressure drop (laminar, fully developed; Darcy friction factor f = 64/Re):
    dP = f * (L/Dh) * (rho * u^2 / 2),   f = 64/Re   (Incropera eq. 8.19)
    Small Dh => notable pressure drop even at modest velocity.

Steady state of this ODE system reproduces the e-NTU effectiveness
(verified in test_model.py against the closed-form counterflow relation).

References:
    Kandlikar, S.G. & Grande, W.J. (2003). "Evolution of Microchannel Flow
        Passages." Heat Transfer Engineering, 24(1), 3-17.
    Incropera, F.P. & DeWitt, D.P. (2006). Fundamentals of Heat and Mass
        Transfer, 6th ed., Wiley. (Nu eq. 8.53, friction eq. 8.19,
        e-NTU eq. 11.29, property tables A.4 / A.6.)
    Shah, R.K. & London, A.L. (1978). Laminar Flow Forced Convection in Ducts.
    Webb, R.L. & Kim, N.-H. (2005). Principles of Enhanced Heat Transfer.
"""

import numpy as np
from scipy.integrate import solve_ivp


class MicrochannelHX_F2a:
    """Lumped transient microchannel HX -- N-CV two-stream + wall ODEs."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = int(u["N_cv"]["value"])

        # Geometry
        self.L = u["L_channel"]["value"]            # m
        self.Dh = u["Dh"]["value"]                  # m
        self.n_ch = u["n_channels"]["value"]        # parallel channels
        self.t_wall = u["wall_thickness"]["value"]  # m

        # Wall material
        self.k_wall = u["k_wall"]["value"]          # W/(m.K)
        self.m_wall = u["m_wall"]["value"]          # kg total
        self.cp_wall = u["cp_wall"]["value"]        # J/(kg.K)

        self.Nu = u["Nu_laminar"]["value"]          # -

        # Default flows
        self.mdot_h0 = u["mdot_h"]["value"]         # kg/s
        self.mdot_c0 = u["mdot_c"]["value"]         # kg/s

        # Hot fluid props (default liquid water)
        self.cp_h = u["cp_h"]["value"]
        self.rho_h = u["rho_h"]["value"]
        self.mu_h = u["mu_h"]["value"]
        self.k_h = u["k_fluid_h"]["value"]

        # Cold fluid props (default liquid water)
        self.cp_c = u["cp_c"]["value"]
        self.rho_c = u["rho_c"]["value"]
        self.mu_c = u["mu_c"]["value"]
        self.k_c = u["k_fluid_c"]["value"]

        # Air props (selectable)
        self.air = {
            "cp": u["air_cp"]["value"], "rho": u["air_rho"]["value"],
            "mu": u["air_mu"]["value"], "k": u["air_k"]["value"],
        }

        self.T_h_in0 = u["T_h_in"]["value"]
        self.T_c_in0 = u["T_c_in"]["value"]

        # ---- Derived per-control-volume geometry ----------------------
        # cross-sectional area of one circular microchannel
        A_ch = np.pi * (self.Dh / 2.0) ** 2
        self.A_cross = A_ch * self.n_ch             # total flow area per stream [m2]
        # wetted perimeter -> heat-transfer surface area of the full core
        perim = np.pi * self.Dh
        self.A_surface = perim * self.L * self.n_ch  # m2 (per stream side)
        self.A_cv = self.A_surface / self.N          # m2 per control volume

        # fluid hold-up volumes per control volume
        V_stream = self.A_cross * self.L             # m3 of fluid in one stream
        self.V_cv = V_stream / self.N                # m3 per control volume

        # wall mass and axial-conduction geometry per node
        self.m_w_cv = self.m_wall / self.N           # kg per wall node
        self.dx = self.L / self.N                    # m node length
        # effective axial conduction cross-section of the metal wall
        self.A_wall_axial = perim * self.t_wall * self.n_ch  # m2

    # ------------------------------------------------------------------
    # Property selection
    # ------------------------------------------------------------------
    def _props(self, stream):
        if stream == "air":
            return self.air["cp"], self.air["rho"], self.air["mu"], self.air["k"]
        if stream == "hot":
            return self.cp_h, self.rho_h, self.mu_h, self.k_h
        return self.cp_c, self.rho_c, self.mu_c, self.k_c

    # ------------------------------------------------------------------
    # Convective coefficient -- microchannel laminar Nu correlation
    # ------------------------------------------------------------------
    def reynolds(self, mdot, stream="hot"):
        cp, rho, mu, k = self._props(stream)
        u_vel = mdot / (rho * self.A_cross)          # m/s mean velocity
        return rho * u_vel * self.Dh / mu

    def htc(self, stream="hot"):
        """Convective coefficient h = Nu * k / Dh  [W/(m2.K)] (Incropera 8.53)."""
        cp, rho, mu, k = self._props(stream)
        return self.Nu * k / self.Dh

    def pressure_drop(self, mdot, stream="hot"):
        """Laminar fully-developed pressure drop [Pa], f = 64/Re (Incropera 8.19)."""
        cp, rho, mu, k = self._props(stream)
        u_vel = mdot / (rho * self.A_cross)
        Re = max(self.reynolds(mdot, stream), 1e-6)
        f = 64.0 / Re
        return f * (self.L / self.Dh) * (rho * u_vel ** 2) / 2.0

    def UA(self, mdot_h=None, mdot_c=None, hot_stream="hot", cold_stream="cold"):
        """Overall UA [W/K] from series hot-film, wall, cold-film resistances."""
        h_h = self.htc(hot_stream)
        h_c = self.htc(cold_stream)
        # wall conduction resistance across separating wall (thin -> small)
        R_wall = self.t_wall / (self.k_wall * self.A_surface)
        R = 1.0 / (h_h * self.A_surface) + R_wall + 1.0 / (h_c * self.A_surface)
        return 1.0 / R

    # ------------------------------------------------------------------
    # e-NTU reference (counterflow) -- for steady-state benchmarking
    # ------------------------------------------------------------------
    def epsilon_ntu_counterflow(self, mdot_h, mdot_c,
                                hot_stream="hot", cold_stream="cold"):
        cp_h = self._props(hot_stream)[0]
        cp_c = self._props(cold_stream)[0]
        C_h = mdot_h * cp_h
        C_c = mdot_c * cp_c
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        C_r = C_min / C_max
        NTU = self.UA(mdot_h, mdot_c, hot_stream, cold_stream) / C_min
        if abs(C_r - 1.0) < 1e-9:
            eps = NTU / (1.0 + NTU)
        else:
            num = 1.0 - np.exp(-NTU * (1.0 - C_r))
            den = 1.0 - C_r * np.exp(-NTU * (1.0 - C_r))
            eps = num / den
        return float(np.clip(eps, 0.0, 1.0)), NTU, C_min, C_max

    # ------------------------------------------------------------------
    # ODE right-hand side
    # State vector y = [Th_0..Th_{N-1}, Tc_0..Tc_{N-1}, Tw_0..Tw_{N-1}]
    # ------------------------------------------------------------------
    def _rhs(self, t, y, mdot_h, mdot_c, T_h_in, T_c_in,
             hot_stream, cold_stream):
        N = self.N
        Th = y[0:N]
        Tc = y[N:2 * N]
        Tw = y[2 * N:3 * N]

        cp_h, rho_h, _, _ = self._props(hot_stream)
        cp_c, rho_c, _, _ = self._props(cold_stream)

        h_h = self.htc(hot_stream)
        h_c = self.htc(cold_stream)

        # thermal capacitances per CV
        Cap_h = rho_h * self.V_cv * cp_h
        Cap_c = rho_c * self.V_cv * cp_c
        Cap_w = self.m_w_cv * self.cp_wall

        UA_cv_h = h_h * self.A_cv     # hot-film conductance per CV
        UA_cv_c = h_c * self.A_cv     # cold-film conductance per CV

        dTh = np.zeros(N)
        dTc = np.zeros(N)
        dTw = np.zeros(N)

        # axial wall conduction conductance between adjacent nodes [W/K]
        k_axial = self.k_wall * self.A_wall_axial / self.dx

        # Hot stream flows 0 -> N-1 ; cold (counterflow) flows N-1 -> 0
        for i in range(N):
            # --- hot upstream temperature ---
            Th_up = T_h_in if i == 0 else Th[i - 1]
            adv_h = mdot_h * cp_h * (Th_up - Th[i])
            conv_h = UA_cv_h * (Th[i] - Tw[i])
            dTh[i] = (adv_h - conv_h) / Cap_h

            # --- cold stream: counterflow, cold node i pairs with wall node i,
            #     cold enters at i = N-1 (upstream is i+1) ---
            Tc_up = T_c_in if i == N - 1 else Tc[i + 1]
            adv_c = mdot_c * cp_c * (Tc_up - Tc[i])
            conv_c = UA_cv_c * (Tw[i] - Tc[i])
            dTc[i] = (adv_c + conv_c) / Cap_c

            # --- wall node ---
            q_from_hot = UA_cv_h * (Th[i] - Tw[i])
            q_to_cold = UA_cv_c * (Tw[i] - Tc[i])
            q_axial = 0.0
            if i > 0:
                q_axial += k_axial * (Tw[i - 1] - Tw[i])
            if i < N - 1:
                q_axial += k_axial * (Tw[i + 1] - Tw[i])
            dTw[i] = (q_from_hot - q_to_cold + q_axial) / Cap_w

        return np.concatenate([dTh, dTc, dTw])

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, T_h_in=None, T_c_in=None, mdot_h=None, mdot_c=None,
                 dt=1.0, duration_s=120.0,
                 hot_stream="hot", cold_stream="cold",
                 T_init=None):
        """
        Integrate the 3N coupled energy-balance ODEs to (optionally) steady state.

        Returns dict with time series of outlet temps, duty Q, effectiveness,
        plus per-node final profiles and pressure drops.
        """
        T_h_in = self.T_h_in0 if T_h_in is None else T_h_in
        T_c_in = self.T_c_in0 if T_c_in is None else T_c_in
        mdot_h = self.mdot_h0 if mdot_h is None else mdot_h
        mdot_c = self.mdot_c0 if mdot_c is None else mdot_c

        N = self.N
        if T_init is None:
            T_init = T_c_in  # cold start
        y0 = np.full(3 * N, T_init, dtype=float)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="BDF",   # stiff (fast film, slow wall/fluid)
            args=(mdot_h, mdot_c, T_h_in, T_c_in, hot_stream, cold_stream),
            rtol=1e-7, atol=1e-9,
        )

        Th = sol.y[0:N, :]
        Tc = sol.y[N:2 * N, :]
        Tw = sol.y[2 * N:3 * N, :]

        # hot exits at node N-1; cold (counterflow) exits at node 0
        T_h_out = Th[N - 1, :]
        T_c_out = Tc[0, :]

        cp_h = self._props(hot_stream)[0]
        cp_c = self._props(cold_stream)[0]
        C_h = mdot_h * cp_h
        C_c = mdot_c * cp_c
        C_min = min(C_h, C_c)

        Q_W = C_h * (T_h_in - T_h_out)            # duty from hot side [W]
        Q_max = C_min * (T_h_in - T_c_in)
        eps = np.where(Q_max > 1e-9, Q_W / np.maximum(Q_max, 1e-9), 0.0)

        return {
            "t": sol.t,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "Q_kW": Q_W / 1000.0,
            "effectiveness": np.clip(eps, 0.0, 1.0),
            "T_h_profile": Th[:, -1],
            "T_c_profile": Tc[:, -1],
            "T_wall_profile": Tw[:, -1],
            "dP_h_Pa": self.pressure_drop(mdot_h, hot_stream),
            "dP_c_Pa": self.pressure_drop(mdot_c, cold_stream),
            "Re_h": self.reynolds(mdot_h, hot_stream),
            "Re_c": self.reynolds(mdot_c, cold_stream),
            "h_h": self.htc(hot_stream),
            "h_c": self.htc(cold_stream),
            "UA": self.UA(mdot_h, mdot_c, hot_stream, cold_stream),
        }
