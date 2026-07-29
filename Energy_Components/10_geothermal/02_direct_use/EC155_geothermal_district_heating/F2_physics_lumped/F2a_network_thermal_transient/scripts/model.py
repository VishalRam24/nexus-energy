"""
EC155 -- Geothermal District Heating -- F2a Lumped Network Thermal Transient

Physics-lumped (0D) dynamic model of a direct-use geothermal district-heating
plant.  A geothermal production well delivers hot brine (T_geo_source) which
passes through a plate heat exchanger and transfers heat to the closed
district-heating (DH) network loop.  The cooled brine optionally cascades to
lower-temperature users (greenhouses, aquaculture) before reinjection.  The DH
network is modelled as two lumped, perfectly-mixed water capacitances -- the
supply header (T_s) and the return header (T_r) -- circulated at constant mass
flow m_dot_net.  Consumers extract Q_load between supply and return; an optional
peak boiler tops up the supply temperature on cold days.

State vector  x = [T_s, T_r]  (degC), integrated with scipy.integrate.solve_ivp.

Governing lumped energy ODEs (first law on each well-mixed control volume):

    C_s dT_s/dt =  Q_geo_to_net            (heat exchanger into supply)
                 + Q_boiler                (peak boiler into supply)
                 - m_dot_net cp (T_s - T_r)   (advection: cooled return enters,
                                               hot supply leaves to consumers)
                 - UA_s (T_s - T_ground)    (distribution heat loss)

    C_r dT_r/dt =  m_dot_net cp (T_s - T_r)   (advection: hot supply leaves
                                               consumers as the return after
                                               giving up Q_load)
                 - Q_load                   (heat extracted by consumers)
                 - UA_r (T_r - T_ground)    (distribution heat loss)

with C_s = rho cp V_supply,  C_r = rho cp V_return  (J/K).

Adding the two equations gives the network energy balance:
    C_s dT_s/dt + C_r dT_r/dt =
        Q_geo_to_net + Q_boiler - Q_load - UA_s(T_s-Tg) - UA_r(T_r-Tg)
i.e. stored = injected (geo + boiler) - delivered (load) - distribution losses.
At steady state this reduces to the F1 algebraic heat balance, so F2a is the
dynamic upgrade of EC155 F1a.

Heat-exchanger coupling (epsilon-NTU style, lumped):
    Q_geo_avail = m_dot_geo cp (T_geo_source - T_r)          (max recoverable
                  against the cold network return acting as the cold stream)
    Q_geo_to_net = eta_hx * Q_geo_avail, clamped so the brine cannot be cooled
                   below T_reinject_min and so it never *heats* the supply above
                   the brine source temperature.
    T_reinject = T_geo_source - Q_geo_to_net/(eta_hx m_dot_geo cp)
    Cascade users recover Q_cascade = f_cascade * m_dot_geo cp
                   (T_reinject - T_reinject_min)  before final reinjection.

cp of liquid water (4186 J/(kg.K) near 60-70 C) and density (985 kg/m3 at 65 C)
are hardcoded from IAPWS-IF97 / CRC Handbook per the F2 build spec.

References:
    Lund, J.W. & Toth, A.N. (2021). Direct Utilization of Geothermal Energy 2020
        Worldwide Review. Geothermics, 90, 101915.
    Lund, J.W., Freeston, D.H. & Boyd, T.L. (2011). Direct utilization of
        geothermal energy 2010 worldwide review. Geothermics, 40(3), 159-180.
    Frederiksen, S. & Werner, S. (2013). District Heating and Cooling.
        Studentlitteratur, Lund (network thermal dynamics, HX effectiveness).
    Wagner, W. & Pruss, A. (2002). IAPWS-IF97 formulation (water properties).
"""

import numpy as np
from scipy.integrate import solve_ivp


class GeothermalDH_F2a:
    """Geothermal district heating -- lumped network thermal-transient model."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Geothermal source / heat exchanger
        self.T_geo_source = u["T_geo_source"]["value"]            # degC
        self.T_reinject_min = u["T_geo_reinject_min"]["value"]    # degC
        self.m_dot_geo = u["m_dot_geo"]["value"]                  # kg/s
        self.eta_hx = u["eta_hx"]["value"]                        # -
        self.f_cascade = u["f_cascade"]["value"]                  # -

        # Network set points (used for boiler control + defaults)
        self.T_supply_set = u["T_supply_setpoint"]["value"]       # degC
        self.T_return_nom = u["T_return_nominal"]["value"]        # degC

        # Network hydraulics + thermal capacitance
        self.m_dot_net = u["m_dot_net"]["value"]                  # kg/s
        self.V_supply = u["V_supply"]["value"]                    # m3
        self.V_return = u["V_return"]["value"]                    # m3
        self.rho = u["rho_water"]["value"]                        # kg/m3
        self.cp = u["cp_water"]["value"]                          # J/(kg.K)

        # Distribution losses
        self.UA_s = u["UA_supply"]["value"]                       # W/K
        self.UA_r = u["UA_return"]["value"]                       # W/K
        self.T_ground = u["T_ground"]["value"]                    # degC

        # Boiler
        self.Q_boiler_max = u["Q_boiler_max"]["value"] * 1e3      # kW -> W
        self.eta_boiler = u["eta_boiler"]["value"]                # -

        # Derived lumped heat capacities (J/K)
        self.C_s = self.rho * self.cp * self.V_supply
        self.C_r = self.rho * self.cp * self.V_return

    # ------------------------------------------------------------------ #
    #  Component sub-models (all powers in W internally)
    # ------------------------------------------------------------------ #
    def hx_heat_to_network(self, T_r, T_geo_source=None):
        """
        Heat delivered by the geothermal HX into the network supply (W).

        The cold network return (T_r) is the cold stream; the brine is the hot
        stream.  Q = eta_hx * m_dot_geo * cp * (T_geo_source - T_r), clamped so
        (i) it is non-negative and (ii) the brine is not cooled below the
        minimum reinjection temperature.
        """
        if T_geo_source is None:
            T_geo_source = self.T_geo_source
        dT = T_geo_source - T_r
        if dT <= 0.0:
            return 0.0
        Q = self.eta_hx * self.m_dot_geo * self.cp * dT
        # cap: brine cannot leave the HX colder than T_reinject_min
        Q_cap = self.eta_hx * self.m_dot_geo * self.cp * (T_geo_source - self.T_reinject_min)
        return min(Q, max(Q_cap, 0.0))

    def reinjection_temperature(self, Q_geo_to_net, T_geo_source=None):
        """Brine temperature leaving the HX before cascade use (degC)."""
        if T_geo_source is None:
            T_geo_source = self.T_geo_source
        denom = self.eta_hx * self.m_dot_geo * self.cp
        return T_geo_source - Q_geo_to_net / denom

    def cascade_heat(self, T_reinject):
        """
        Cascade (lower-grade direct-use) heat recovered before final
        reinjection (W).  Recovers a fraction of the residual enthalpy above
        the minimum reinjection temperature.
        """
        dT = max(T_reinject - self.T_reinject_min, 0.0)
        return self.f_cascade * self.m_dot_geo * self.cp * dT

    def boiler_heat(self, T_s, Q_geo_to_net, Q_load):
        """
        Peak-boiler heat into supply (W).  Proportional controller: only fires
        when geothermal alone cannot hold the supply set point against the load,
        capped at Q_boiler_max.
        """
        deficit = Q_load - Q_geo_to_net
        if deficit <= 0.0:
            return 0.0
        return min(deficit, self.Q_boiler_max)

    def distribution_loss(self, T_s, T_r):
        """Total network distribution heat loss to ground (W)."""
        return self.UA_s * (T_s - self.T_ground) + self.UA_r * (T_r - self.T_ground)

    # ------------------------------------------------------------------ #
    #  ODE right-hand side
    # ------------------------------------------------------------------ #
    def _rhs(self, t, x, Q_load_func, T_geo_func, boiler_on):
        T_s, T_r = x
        Q_load = float(Q_load_func(t))            # W
        T_geo = float(T_geo_func(t))              # degC

        Q_geo = self.hx_heat_to_network(T_r, T_geo)
        Q_boil = self.boiler_heat(T_s, Q_geo, Q_load) if boiler_on else 0.0

        m_cp = self.m_dot_net * self.cp           # W/K
        advect = m_cp * (T_s - T_r)               # W

        dTs = (Q_geo + Q_boil - advect - self.UA_s * (T_s - self.T_ground)) / self.C_s
        dTr = (advect - Q_load - self.UA_r * (T_r - self.T_ground)) / self.C_r
        return [dTs, dTr]

    # ------------------------------------------------------------------ #
    #  Public simulate
    # ------------------------------------------------------------------ #
    def simulate(self, Q_load_kW=None, T_s0=None, T_r0=None, T_geo_source=None,
                 dt=60.0, duration_s=86400.0, boiler_on=True):
        """
        Integrate the lumped network thermal transient with scipy.solve_ivp.

        Parameters
        ----------
        Q_load_kW    : float | callable(t)->kW | None
                       District heat demand.  Default = Q_load_design.
        T_s0, T_r0   : float (degC) initial supply / return temperatures.
        T_geo_source : float | callable(t)->degC | None  wellhead temperature.
        dt           : output sampling interval (s).
        duration_s   : simulation horizon (s).
        boiler_on    : enable peak boiler top-up.

        Returns dict of numpy arrays (SI/engineering units, powers in kW).
        """
        if T_s0 is None:
            T_s0 = self.T_supply_set
        if T_r0 is None:
            T_r0 = self.T_return_nom
        if T_geo_source is None:
            T_geo_source = self.T_geo_source

        # Build load function (W)
        if callable(Q_load_kW):
            Q_load_func = lambda t: max(float(Q_load_kW(t)) * 1e3, 0.0)
        else:
            Q_const = (Q_load_kW if Q_load_kW is not None else
                       self._default_load_kW()) * 1e3
            Q_load_func = lambda t, q=Q_const: q

        if callable(T_geo_source):
            T_geo_func = T_geo_source
        else:
            T_geo_func = lambda t, T=T_geo_source: T

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T_s0, T_r0],
            t_eval=t_eval, args=(Q_load_func, T_geo_func, boiler_on),
            method="RK45", rtol=1e-7, atol=1e-7, max_step=dt,
        )

        T_s = sol.y[0]
        T_r = sol.y[1]
        t = sol.t

        # Post-process power flows (kW)
        Q_load = np.array([Q_load_func(ti) for ti in t]) / 1e3
        T_geo_arr = np.array([T_geo_func(ti) for ti in t])
        Q_geo = np.array([self.hx_heat_to_network(tr, tg)
                          for tr, tg in zip(T_r, T_geo_arr)]) / 1e3
        Q_boiler = np.array([
            (self.boiler_heat(ts, self.hx_heat_to_network(tr, tg),
                              Q_load_func(ti)) if boiler_on else 0.0)
            for ts, tr, tg, ti in zip(T_s, T_r, T_geo_arr, t)
        ]) / 1e3
        T_reinject = np.array([
            self.reinjection_temperature(self.hx_heat_to_network(tr, tg), tg)
            for tr, tg in zip(T_r, T_geo_arr)
        ])
        Q_cascade = np.array([self.cascade_heat(tri) for tri in T_reinject]) / 1e3
        Q_loss = np.array([self.distribution_loss(ts, tr)
                           for ts, tr in zip(T_s, T_r)]) / 1e3
        Q_advect = self.m_dot_net * self.cp * (T_s - T_r) / 1e3   # heat to consumers loop

        return {
            "t": t,
            "T_supply": T_s,
            "T_return": T_r,
            "T_reinject": T_reinject,
            "T_geo_source": T_geo_arr,
            "Q_geo_kW": Q_geo,
            "Q_boiler_kW": Q_boiler,
            "Q_load_kW": Q_load,
            "Q_cascade_kW": Q_cascade,
            "Q_loss_kW": Q_loss,
            "Q_delivered_kW": Q_advect,
            "success": sol.success,
        }

    def _default_load_kW(self):
        """Steady load that the geothermal supply can meet at nominal return."""
        Q_geo_nom = self.hx_heat_to_network(self.T_return_nom) / 1e3
        Q_loss_nom = self.distribution_loss(self.T_supply_set, self.T_return_nom) / 1e3
        return max(Q_geo_nom - Q_loss_nom, 0.0)

    # ------------------------------------------------------------------ #
    #  Steady-state helper (algebraic, for validation against F1)
    # ------------------------------------------------------------------ #
    def steady_state(self, Q_load_kW, T_geo_source=None, boiler_on=True,
                     duration_s=None):
        """Run long enough to settle and return the final-state snapshot."""
        if duration_s is None:
            # ~6 thermal time constants of the larger capacitance
            tau = max(self.C_s, self.C_r) / (self.m_dot_net * self.cp)
            duration_s = max(6.0 * tau, 6.0 * 3600.0)
        r = self.simulate(Q_load_kW=Q_load_kW, T_geo_source=T_geo_source,
                          dt=duration_s / 200.0, duration_s=duration_s,
                          boiler_on=boiler_on)
        return {k: (v[-1] if isinstance(v, np.ndarray) else v)
                for k, v in r.items()}
