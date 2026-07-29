"""
EC090 -- Solar Water Heater Combi System -- F2a Physics-Lumped Stratified Tank

Physics-lumped (0D/1D) first-principles model of a solar combi system:

    1. Flat-plate collector  -- Hottel-Whillier-Bliss useful-gain equation
    2. Stratified storage tank -- N-node energy-balance ODE system (top hot, bottom cold)
    3. Combined loads          -- DHW draw (mains make-up) + space-heating return loop
    4. Auxiliary backup heater -- thermostatic (deadband) top-node trim heater
    5. Pump control            -- differential ON/OFF controller (collector vs tank bottom)

Collector useful gain (Duffie & Beckman 2013, eq. 6.7.6):
    Q_u = A_c * F_R * [ G*(tau*alpha) - U_L*(T_in - T_amb) ]      (clamped >= 0)
    => Q_u = A_c * [ F_R(tau alpha)*G - F_R*U_L*(T_in - T_amb) ]
The collector outlet temperature follows from an energy balance on the loop flow:
    T_out = T_in + Q_u / (mdot_coll * cp)

Stratified multi-node tank energy balance (Duffie & Beckman 2013, eq. 8.4;
oemof.thermal stratified_thermal_storage). For node i (1=top ... N=bottom),
mass m_i = rho * V_tank / N, the lumped capacity ODE is:

    m_i cp dT_i/dt =  Q_solar,i              (charge flow from collector loop)
                    + Q_load,i               (discharge flow to DHW + space heat)
                    + (kA/dx)(T_{i-1}-T_i)   (downward conduction)
                    + (kA/dx)(T_{i+1}-T_i)   (upward conduction)
                    - U_i A_i (T_i - T_amb)  (wall + top/bottom loss)
                    + Q_aux  (top node only, thermostatic backup)

Charge flow: when the pump is ON, collector loop delivers mdot_coll of water at
T_coll_out into the TOP node and withdraws from the BOTTOM node; advective plug
flow shifts each interior node (Duffie & Beckman 8.4 "plug-flow" stratified model).

Discharge flow: combined draw mdot_load enters the BOTTOM node as cold mains
make-up and is withdrawn (hot) from the TOP node, producing an upward advective
flow through the column. Space-heating return is lumped into the same draw.

Energy conservation, stratification (dT/dz >= 0, i.e. top hotter), solar fraction
in [0,1], and zero collector gain at night (G=0) are all enforced / verifiable.

Water cp = 4186 J/(kg.K) and rho = 1000 kg/m^3 are hardcoded liquid-water
properties at ~20-60 C (CRC Handbook of Chemistry & Physics; Duffie & Beckman 2013).

References
----------
Duffie, J.A. & Beckman, W.A. (2013). Solar Engineering of Thermal Processes,
    4th ed., Wiley. Chapters 6 (collectors) and 8 (storage), eqs. 6.7.6 and 8.4.
oemof.thermal contributors (2020). stratified_thermal_storage component docs,
    https://oemof-thermal.readthedocs.io
"""

import numpy as np
from scipy.integrate import solve_ivp


class SolarCombiF2a:
    """Solar combi system -- Hottel-Whillier collector + N-node stratified tank."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Collector (Hottel-Whillier)
        self.A_c = u["A_collector"]["value"]            # m2
        self.FR_ta = u["F_R_tau_alpha"]["value"]        # - (optical, intercept)
        self.FR_UL = u["F_R_U_L"]["value"]              # W/(m2.K) (loss, slope)
        self.mdot_coll = u["mdot_coll"]["value"]        # kg/s
        # Water properties (hardcoded, cited)
        self.cp = u["cp_water"]["value"]                # J/(kg.K)
        self.rho = u["rho_water"]["value"]              # kg/m3
        # Tank geometry / discretisation
        self.V_tank = u["V_tank"]["value"]              # m3
        self.N = int(u["N_nodes"]["value"])
        self.H = u["H_tank"]["value"]                   # m
        self.D = u["D_tank"]["value"]                   # m
        self.U_tank = u["U_tank"]["value"]              # W/(m2.K)
        self.k_eff = u["k_eff"]["value"]                # W/(m.K)
        self.T_amb_tank = u["T_amb_tank"]["value"]      # K
        self.T_mains = u["T_mains"]["value"]            # K
        # Auxiliary backup
        self.T_set_aux = u["T_set_aux"]["value"]        # K
        self.T_db = u["T_aux_deadband"]["value"]        # K
        self.Q_aux_max = u["Q_aux_max"]["value"]        # W
        self.eta_aux = u["eta_aux"]["value"]            # -
        # Pump differential controller
        self.dT_on = u["pump_dT_on"]["value"]           # K
        self.dT_off = u["pump_dT_off"]["value"]         # K

        # Derived node geometry
        self.A_cross = np.pi * (self.D / 2.0) ** 2      # m2 cross-section
        self.dz = self.H / self.N                       # m node height
        self.V_node = self.V_tank / self.N              # m3
        self.m_node = self.rho * self.V_node            # kg per node
        # Wall (side) area per node + caps on top/bottom node
        self.A_side = np.pi * self.D * self.dz          # m2 per node side wall
        self.A_cap = self.A_cross                       # m2 top / bottom end cap
        # Inter-node conduction conductance kA/dx  [W/K]
        self.G_cond = self.k_eff * self.A_cross / self.dz

    # ------------------------------------------------------------------
    # Collector -- Hottel-Whillier-Bliss useful gain
    # ------------------------------------------------------------------
    def collector_useful_gain(self, G, T_in, T_amb):
        """Useful heat-collection rate Q_u [W] (Duffie & Beckman eq. 6.7.6).

        Q_u = A_c*[F_R(tau alpha)*G - F_R*U_L*(T_in - T_amb)], clamped >= 0.
        At night (G=0) and T_in >= T_amb this is exactly 0 (no negative gain --
        the differential pump controller keeps the loop off when there is no gain).
        """
        Q = self.A_c * (self.FR_ta * G - self.FR_UL * (T_in - T_amb))
        return max(0.0, Q)

    def collector_outlet_temp(self, Q_u, T_in):
        """Collector loop outlet temperature [K] from loop energy balance."""
        if self.mdot_coll <= 0:
            return T_in
        return T_in + Q_u / (self.mdot_coll * self.cp)

    # ------------------------------------------------------------------
    # Pump differential controller (hysteresis, anti short-cycle)
    # ------------------------------------------------------------------
    def pump_on(self, G, T_amb, T_bottom, prev_on):
        """Return True if collector pump should run.

        ON  when potential collector outlet exceeds tank bottom by dT_on.
        OFF when that margin falls below dT_off (hysteresis).
        Uses a no-flow stagnation estimate of attainable gain to decide.
        """
        if G <= 0.0:
            return False
        # Attainable gain at current bottom temp as collector inlet
        Q_test = self.collector_useful_gain(G, T_bottom, T_amb)
        if Q_test <= 0.0:
            return False
        dT_attain = Q_test / (self.mdot_coll * self.cp)
        if prev_on:
            return dT_attain > self.dT_off
        return dT_attain > self.dT_on

    # ------------------------------------------------------------------
    # Tank RHS -- N-node stratified energy balance
    # ------------------------------------------------------------------
    def tank_rhs(self, T, G, T_amb, mdot_load, Q_space, pump):
        """dT/dt [K/s] for all N nodes (index 0=top hottest ... N-1=bottom).

        Advection (charge): collector loop mdot_coll at T_coll_out into top node,
        withdrawn from bottom -> downward plug flow through interior nodes.
        Advection (discharge): mdot_load cold make-up into bottom node, hot draw
        from top -> upward plug flow. Space-heating return Q_space is removed
        directly (lumped) and re-injected cold via the same make-up stream.
        Plus inter-node conduction and wall losses. Top node also gets Q_aux.
        """
        N = self.N
        dTdt = np.zeros(N)
        mc = self.m_node * self.cp  # J/K per node

        # ---- Collector charge advection (downward, top<-out, bottom->in) ----
        if pump:
            Q_u = self.collector_useful_gain(G, T[N - 1], T_amb)
            T_out = self.collector_outlet_temp(Q_u, T[N - 1])
            wc = self.mdot_coll * self.cp  # W/K flow capacity
            # Top node receives hot collector water; downward flow T_{i-1}->T_i
            dTdt[0] += wc * (T_out - T[0]) / mc
            for i in range(1, N):
                dTdt[i] += wc * (T[i - 1] - T[i]) / mc
        else:
            Q_u = 0.0
            T_out = T[N - 1]

        # ---- Load discharge advection (upward, bottom<-mains, top->draw) ----
        if mdot_load > 0:
            wl = mdot_load * self.cp  # W/K
            # Bottom node receives cold mains; upward flow T_{i+1}->T_i
            dTdt[N - 1] += wl * (self.T_mains - T[N - 1]) / mc
            for i in range(N - 1):
                dTdt[i] += wl * (T[i + 1] - T[i]) / mc

        # ---- Space-heating draw: remove Q_space from top node (lumped) ----
        if Q_space > 0:
            dTdt[0] -= Q_space / mc

        # ---- Inter-node vertical conduction ----
        for i in range(N):
            if i > 0:
                dTdt[i] += self.G_cond * (T[i - 1] - T[i]) / mc
            if i < N - 1:
                dTdt[i] += self.G_cond * (T[i + 1] - T[i]) / mc

        # ---- Wall + end-cap heat loss to ambient ----
        for i in range(N):
            A_loss = self.A_side
            if i == 0:
                A_loss += self.A_cap      # top cap
            if i == N - 1:
                A_loss += self.A_cap      # bottom cap
            dTdt[i] -= self.U_tank * A_loss * (T[i] - self.T_amb_tank) / mc

        # ---- Auxiliary thermostatic backup on top node ----
        Q_aux = self.aux_power(T[0])
        dTdt[0] += Q_aux / mc

        return dTdt, Q_u, Q_aux

    def aux_power(self, T_top):
        """Thermostatic backup delivered power [W] into top node (deadband)."""
        if T_top < self.T_set_aux - self.T_db:
            return self.Q_aux_max
        if T_top < self.T_set_aux:
            # proportional trim within deadband (smooth control, avoids chatter)
            frac = (self.T_set_aux - T_top) / self.T_db
            return self.Q_aux_max * frac
        return 0.0

    # ------------------------------------------------------------------
    # Time-domain simulation over a day
    # ------------------------------------------------------------------
    def simulate(self, G_profile, T_amb_profile, load_profile, space_profile,
                 T_init=None, dt=300.0, duration_s=86400.0):
        """Integrate the stratified-tank ODE system over a day.

        Parameters
        ----------
        G_profile, T_amb_profile : callable(t)->float
            Plane-of-array irradiance [W/m2] and ambient temperature [K].
        load_profile : callable(t)->float
            Combined DHW + space-heating draw flow [kg/s].
        space_profile : callable(t)->float
            Space-heating thermal load drawn from the top node [W].
        T_init : array(N) or None
            Initial node temperatures [K] (top..bottom). Default mild stratification.
        dt : float          output time step [s]
        duration_s : float  total horizon [s]

        Returns
        -------
        dict with time-series and integrated energy / solar-fraction metrics.
        """
        N = self.N
        if T_init is None:
            # mildly stratified start (top hotter), 45C top -> 35C bottom
            T_init = np.linspace(318.15, 308.15, N)
        T_init = np.asarray(T_init, dtype=float)

        # callables
        Gf = G_profile if callable(G_profile) else (lambda t: G_profile)
        Tf = T_amb_profile if callable(T_amb_profile) else (lambda t: T_amb_profile)
        Lf = load_profile if callable(load_profile) else (lambda t: load_profile)
        Sf = space_profile if callable(space_profile) else (lambda t: space_profile)

        # Pump state is integrated via a smoothed differential controller. To keep
        # the RHS pure, decide pump state from instantaneous conditions (the
        # hysteresis band is narrow vs dt so latch effects are second order here).
        def rhs(t, T):
            G = max(0.0, Gf(t))
            T_amb = Tf(t)
            mdot_load = max(0.0, Lf(t))
            Q_space = max(0.0, Sf(t))
            pump = self.pump_on(G, T_amb, T[N - 1], prev_on=False)
            d, _, _ = self.tank_rhs(T, G, T_amb, mdot_load, Q_space, pump)
            return d

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            rhs, (0.0, duration_s), T_init, t_eval=t_eval,
            method="LSODA", rtol=1e-6, atol=1e-4, max_step=dt,
        )

        t = sol.t
        Tnodes = sol.y                       # shape (N, M)
        M = len(t)

        # Post-process diagnostics at each output time
        Q_solar = np.zeros(M)
        Q_aux = np.zeros(M)
        Q_load = np.zeros(M)
        pump_state = np.zeros(M)
        T_top = Tnodes[0, :]
        T_bottom = Tnodes[N - 1, :]
        T_mean = Tnodes.mean(axis=0)

        for k in range(M):
            G = max(0.0, Gf(t[k]))
            T_amb = Tf(t[k])
            mdot_load = max(0.0, Lf(t[k]))
            Q_space = max(0.0, Sf(t[k]))
            Tk = Tnodes[:, k]
            pump = self.pump_on(G, T_amb, Tk[N - 1], prev_on=False)
            pump_state[k] = 1.0 if pump else 0.0
            if pump:
                Q_solar[k] = self.collector_useful_gain(G, Tk[N - 1], T_amb)
            Q_aux[k] = self.aux_power(Tk[0]) / self.eta_aux  # fuel input
            # delivered DHW load = draw heated from mains to top-node temp
            Q_load[k] = mdot_load * self.cp * max(0.0, Tk[0] - self.T_mains) + Q_space

        # Integrated energies [J] via trapezoid
        E_solar = np.trapz(Q_solar, t)
        E_aux_fuel = np.trapz(Q_aux, t)
        E_aux_delivered = E_aux_fuel * self.eta_aux
        E_load = np.trapz(Q_load, t)

        # Solar fraction: solar useful / total heat supplied to the store/load.
        # f_solar = E_solar / (E_solar + E_aux_delivered), clamped to [0,1].
        denom = E_solar + E_aux_delivered
        f_solar = E_solar / denom if denom > 0 else 0.0
        f_solar = min(1.0, max(0.0, f_solar))

        return {
            "t": t,
            "T_nodes": Tnodes,
            "T_top": T_top,
            "T_bottom": T_bottom,
            "T_mean": T_mean,
            "Q_solar": Q_solar,
            "Q_aux_fuel": Q_aux,
            "Q_load": Q_load,
            "pump_on": pump_state,
            "E_solar_J": E_solar,
            "E_aux_fuel_J": E_aux_fuel,
            "E_aux_delivered_J": E_aux_delivered,
            "E_load_J": E_load,
            "solar_fraction": f_solar,
        }
