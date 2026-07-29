"""
EC190 -- LNG Regasification Terminal -- F2a Physics-Lumped Vaporizer Thermal Model

Physics-lumped (0D) first-principles model of an LNG send-out train:

    LNG tank (-162 C, ~1.15 bar)
        -> HP cryogenic pump  (raise to ~80 bar pipeline pressure)
        -> Vaporizer (ORV seawater / SCV combustion)  (latent + sensible heat)
        -> Send-out gas (~5 C, pipeline)

Send-out energy balance (per unit mass of LNG regasified):
    q_regas = h_fg + cp_liquid*(T_boil - T_storage) + cp_gas*(T_sendout - T_boil)   [J/kg]
    Q_process = m_dot * q_regas                                                      [W]
This is the heat the process stream must absorb (Mokhatab et al. 2014, ch. 9).

Lumped vaporizer thermal transient (ODE, integrated with scipy.solve_ivp):
    A single lumped metal node (vaporizer panels / tube wall) couples the heat
    source to the cryogenic process stream:

        m_metal*cp_metal * dT_metal/dt = Q_source_in - Q_process_out

        Q_source_in  = UA       * (T_heat_source - T_metal)   (seawater/flue -> metal)
        Q_process_out = UA_proc * (T_metal - T_LNG_mean)      (metal -> cold stream)

    The process actually absorbs min(Q_process_out, Q_demand) where Q_demand is the
    heat needed to fully vaporize+superheat the requested send-out flow. The metal
    relaxes to a steady temperature between the warm source and the cold stream.
    This is the classic lumped-capacitance heat-exchanger transient (Incropera,
    Fundamentals of Heat & Mass Transfer; same form as TES/HX lumped models).

Cryogenic pump work (mechanical, ~incompressible liquid):
    W_pump = m_dot * (P_discharge - P_tank) / (rho_LNG * eta_pump)   [W]

Cold-energy recovery potential (exergy of the cryogen vs ambient):
    The physical cold exergy released when warming the stream from T_storage to
    ambient is bounded by the Carnot/exergy integral. A practical estimate
    (Pospíšil et al. 2019):
        Ex_cold = m_dot * [ (h_amb - h_LNG) - T_amb*(s_amb - s_LNG) ]
    Approximated here with constant cp's:
        Ex_cold ~ m_dot * SUM cp_i*( (T_hi - T_lo) - T_amb*ln(T_hi/T_lo) ) + latent term
    This is recoverable shaft/power potential, NOT free energy.

References:
    Mokhatab, S., Mak, J.Y., Valappil, J.V., Wood, D.A. (2014).
        Handbook of Liquefied Natural Gas, Elsevier. (send-out, vaporization)
    Pospisil, J. et al. (2019). Energy demand of liquefaction and regasification
        of natural gas and the potential of LNG for operative thermal energy
        storage. Renewable & Sustainable Energy Reviews, 99, 1-15. (cold energy)
    Younglove, B.A. & Ely, J.F. (1987). Thermophysical properties of methane.
        J. Phys. Chem. Ref. Data 16(4). (h_fg, cp, T_boil)
    Incropera, F.P. (2007). Fundamentals of Heat and Mass Transfer. (lumped HX)
"""

import numpy as np
from scipy.integrate import solve_ivp


class LNGRegasF2a:
    """Physics-lumped LNG regasification vaporizer with thermal transient ODE."""

    def __init__(self, params: dict):
        v = params["vaporizer"]
        p = params["pump"]
        g = params["lng"]

        # Vaporizer thermal
        self.UA = v["UA"]["value"]                  # W/K  source -> metal
        self.UA_proc = v["UA_process"]["value"]     # W/K  metal -> process
        self.m_metal = v["m_metal"]["value"]        # kg
        self.cp_metal = v["cp_metal"]["value"]      # J/(kg.K)
        self.T_heat_source = v["T_heat_source"]["value"]  # K
        self.eta_comb = v["eta_combustion"]["value"]      # - SCV fuel fraction

        # Pump
        self.P_disc = p["P_discharge"]["value"]     # Pa
        self.P_tank = p["P_tank"]["value"]          # Pa
        self.eta_pump = p["eta_pump"]["value"]      # -

        # LNG / methane properties
        self.rho_LNG = g["rho_LNG"]["value"]        # kg/m3
        self.h_fg = g["h_fg"]["value"]              # J/kg
        self.cp_liq = g["cp_liquid"]["value"]       # J/(kg.K)
        self.cp_gas = g["cp_gas"]["value"]          # J/(kg.K)
        self.T_storage = g["T_storage"]["value"]    # K
        self.T_boil = g["T_boil"]["value"]          # K
        self.T_sendout = g["T_sendout"]["value"]    # K
        self.LHV = g["LHV"]["value"]                # J/kg

    # ------------------------------------------------------------------
    # Algebraic energy balances
    # ------------------------------------------------------------------
    def regas_specific_heat(self, T_sendout=None):
        """Heat to regasify 1 kg LNG: latent + sensible (liquid + gas) [J/kg]."""
        Ts = self.T_sendout if T_sendout is None else T_sendout
        q_liq = self.cp_liq * (self.T_boil - self.T_storage)   # warm liquid to boil
        q_lat = self.h_fg                                       # vaporize
        q_gas = self.cp_gas * (Ts - self.T_boil)               # superheat gas
        return q_liq + q_lat + q_gas

    def process_heat_demand_W(self, sendout_rate_ton_per_h, T_sendout=None):
        """Heat the process stream needs to absorb to meet send-out [W]."""
        m_dot = self._mdot_kg_s(sendout_rate_ton_per_h)
        return m_dot * self.regas_specific_heat(T_sendout)

    def pump_work_W(self, sendout_rate_ton_per_h):
        """Cryogenic HP pump shaft power (incompressible liquid) [W]."""
        m_dot = self._mdot_kg_s(sendout_rate_ton_per_h)
        return m_dot * (self.P_disc - self.P_tank) / (self.rho_LNG * self.eta_pump)

    def combustion_fuel_W(self, sendout_rate_ton_per_h):
        """SCV fuel-gas heat input (self-consumed LNG) [W]. Zero for ORV."""
        m_dot = self._mdot_kg_s(sendout_rate_ton_per_h)
        return self.eta_comb * m_dot * self.LHV

    def cold_exergy_W(self, sendout_rate_ton_per_h, T_ambient_K=288.15):
        """
        Recoverable cold-energy (exergy) potential of the regasified stream [W].
        Carnot-weighted using constant-cp segments + latent term at T_boil
        (Pospisil et al. 2019). Bounded below by 0.
        """
        m_dot = self._mdot_kg_s(sendout_rate_ton_per_h)
        Ta = float(T_ambient_K)
        # Sensible liquid segment: T_storage -> T_boil
        ex_liq = self.cp_liq * ((self.T_boil - self.T_storage)
                                - Ta * np.log(self.T_boil / self.T_storage))
        # Latent phase change at T_boil: exergy = h_fg*(Ta/T_boil - 1) released as cold
        ex_lat = self.h_fg * (Ta / self.T_boil - 1.0)
        # Sensible gas segment: T_boil -> ambient
        ex_gas = self.cp_gas * ((Ta - self.T_boil)
                                - Ta * np.log(Ta / self.T_boil))
        ex_specific = max(ex_liq + ex_lat + ex_gas, 0.0)
        return m_dot * ex_specific

    # ------------------------------------------------------------------
    # Lumped thermal ODE
    # ------------------------------------------------------------------
    def _process_stream_mean_T(self):
        """Mean cold-stream temperature seen by the metal wall [K]."""
        # Latent plateau dominates; use a weighted mean biased toward T_boil.
        return 0.5 * (self.T_storage + self.T_sendout)

    def _q_source_in(self, T_metal, T_heat_source):
        return self.UA * (T_heat_source - T_metal)

    def _q_process_out(self, T_metal):
        T_stream = self._process_stream_mean_T()
        return self.UA_proc * (T_metal - T_stream)

    def _rhs(self, t, y, T_heat_source_fn, sendout_fn, T_sendout):
        T_metal = y[0]
        Tsrc = T_heat_source_fn(t)
        q_in = self._q_source_in(T_metal, Tsrc)
        q_out_capacity = self._q_process_out(T_metal)
        # Heat actually drawn is bounded by what the requested flow demands.
        m_rate = sendout_fn(t)
        q_demand = self._mdot_kg_s(m_rate) * self.regas_specific_heat(T_sendout)
        q_out = max(min(q_out_capacity, q_demand), 0.0) if q_out_capacity > 0 else 0.0
        dTdt = (q_in - q_out) / (self.m_metal * self.cp_metal)
        return [dTdt]

    def simulate(self, sendout_rate_ton_per_h, T_metal0=None,
                 T_heat_source_K=None, T_ambient_K=288.15,
                 dt=10.0, duration_s=3600.0, T_sendout=None):
        """
        Integrate the lumped vaporizer thermal transient with solve_ivp.

        sendout_rate_ton_per_h : float OR callable(t)->ton/h (demand profile)
        T_metal0               : initial vaporizer metal temperature [K]
        T_heat_source_K        : float OR callable(t)->K  (seawater/flue temp)
        Returns dict of time-series arrays + scalar energy summary.
        """
        Tsrc_val = self.T_heat_source if T_heat_source_K is None else T_heat_source_K
        if callable(T_heat_source_K):
            T_src_fn = T_heat_source_K
            Tsrc_nom = T_src_fn(0.0)
        else:
            T_src_fn = lambda t: float(Tsrc_val)
            Tsrc_nom = float(Tsrc_val)

        if callable(sendout_rate_ton_per_h):
            send_fn = sendout_rate_ton_per_h
            send_nom = send_fn(0.0)
        else:
            send_fn = lambda t: float(sendout_rate_ton_per_h)
            send_nom = float(sendout_rate_ton_per_h)

        Ts = self.T_sendout if T_sendout is None else T_sendout

        if T_metal0 is None:
            # cold start: metal somewhere between source and cold stream
            T_metal0 = 0.5 * (Tsrc_nom + self._process_stream_mean_T())

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T_metal0],
            t_eval=t_eval, args=(T_src_fn, send_fn, Ts),
            method="RK45", rtol=1e-6, atol=1e-3, max_step=dt,
        )

        T_metal = sol.y[0]
        t = sol.t

        # Post-process time series
        q_in = self._q_source_in(T_metal, np.array([T_src_fn(ti) for ti in t]))
        q_cap = self._q_process_out(T_metal)
        m_rate = np.array([send_fn(ti) for ti in t])
        m_dot = self._mdot_kg_s(m_rate)
        q_demand = m_dot * self.regas_specific_heat(Ts)
        q_process = np.clip(np.minimum(q_cap, q_demand), 0.0, None)

        # Send-out gas actually produced (tracks delivered heat / demand)
        sendout_kg_s = np.where(q_demand > 0,
                                m_dot * q_process / np.maximum(q_demand, 1e-9),
                                0.0)

        pump_W = self.pump_work_W(m_rate)
        cold_ex_W = np.array([self.cold_exergy_W(mr, T_ambient_K) for mr in m_rate])

        # Energy accounting over the run (trapezoid)
        E_source = np.trapz(q_in, t)
        E_process = np.trapz(q_process, t)
        E_stored = self.m_metal * self.cp_metal * (T_metal[-1] - T_metal[0])

        return {
            "t": t,
            "T_metal": T_metal,
            "Q_source_W": q_in,
            "Q_process_W": q_process,
            "Q_demand_W": q_demand,
            "sendout_kg_s": sendout_kg_s,
            "pump_W": pump_W,
            "cold_exergy_W": cold_ex_W,
            "energy_balance": {
                "E_source_J": float(E_source),
                "E_process_J": float(E_process),
                "E_stored_J": float(E_stored),
                "residual_J": float(E_source - E_process - E_stored),
            },
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mdot_kg_s(sendout_rate_ton_per_h):
        return np.asarray(sendout_rate_ton_per_h, dtype=float) * 1000.0 / 3600.0

    def steady_metal_T(self, sendout_rate_ton_per_h, T_heat_source_K=None):
        """Analytic steady-state metal temperature when not flow-limited."""
        Tsrc = self.T_heat_source if T_heat_source_K is None else float(T_heat_source_K)
        T_stream = self._process_stream_mean_T()
        # UA*(Tsrc - Tm) = UA_proc*(Tm - Tstream)
        return (self.UA * Tsrc + self.UA_proc * T_stream) / (self.UA + self.UA_proc)
