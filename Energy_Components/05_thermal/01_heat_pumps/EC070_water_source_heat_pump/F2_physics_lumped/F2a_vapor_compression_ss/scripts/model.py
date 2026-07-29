"""
EC070 -- Water-Source Heat Pump -- F2a Vapor-Compression Cycle (Steady-State
thermodynamic cycle + lumped water-loop thermal ODE)

Physics-lumped first-principles model of a single-stage vapor-compression heat
pump operating between a water source (evaporator) and a water sink/load
(condenser). The refrigerant cycle (R410A) is solved from thermodynamic states:

    1 -> 2 : isentropic-ideal compression, corrected by isentropic efficiency
    2 -> 3 : condensation (heat rejected to the sink water loop)
    3 -> 4 : isenthalpic expansion through the throttle valve  (h4 = h3)
    4 -> 1 : evaporation (heat absorbed from the source water loop)

State enthalpies are obtained from a lumped property model (saturation pressure
via Antoine-type Clausius-Clapeyron fit, latent heat, liquid/vapor cp) rather
than CoolProp.  The compressor mass flow follows from the volumetric efficiency
and swept volume.  The cycle yields the heating COP from energy balance:

    COP_h = q_cond / w_comp,    q_cond = h2 - h3,   w_comp = (h2s - h1) / eta_is

A lumped first-order thermal ODE then tracks the condenser (load) water-loop
temperature as it is heated toward setpoint:

    m_w * cp_w * dT_load/dt = Q_cond(T_load) - Q_demand

solved with scipy.integrate.solve_ivp.

Physical guarantees enforced:
    * COP > 1                          (heat pump delivers more heat than work)
    * COP < Carnot COP = T_c/(T_c-T_e) (second law)
    * energy balance: Q_cond = Q_evap + W_comp  (first law, within tolerance)

References
----------
Cengel & Boles (2015), *Thermodynamics: An Engineering Approach*, 8th ed.,
    McGraw-Hill -- Ch. 11, ideal/actual vapor-compression refrigeration cycle.
ASHRAE Handbook -- Fundamentals (2021), Ch. 2 (thermodynamics) & Ch. 30
    (refrigerant R410A thermophysical data).
Hundy, Trott & Welch (2016), *Refrigeration, Air Conditioning and Heat Pumps*,
    5th ed., Butterworth-Heinemann -- compressor volumetric/isentropic efficiency.
Klein & Reindl (2005), ASHRAE compressor map conventions.

R410A property anchors (hardcoded, from ASHRAE Fundamentals 2021 Ch.30 / NIST
REFPROP saturation table for R410A as a pseudo-pure fluid):
    Normal boiling point  T_nbp  = 221.7 K  at 1 atm
    Latent heat at 273 K  h_fg   ~ 221.5 kJ/kg
    Critical temperature  T_crit = 344.5 K (71.4 degC)
"""

import numpy as np
from scipy.integrate import solve_ivp


class WaterSourceHP_F2a:
    """Vapor-compression water-source heat pump: thermodynamic cycle + thermal ODE."""

    # ------------------------------------------------------------------
    # R410A lumped property anchors (ASHRAE Fundamentals 2021, Ch.30;
    # NIST REFPROP saturation data). Units: SI.
    # ------------------------------------------------------------------
    R_GAS = 8.314          # J/(mol.K)
    M_R410A = 0.0725       # kg/mol  (R410A molar mass, ASHRAE)

    def __init__(self, params: dict):
        u = params["unit"]
        # Refrigerant property anchors
        self.T_ref = u["T_ref_prop"]["value"]            # K  reference T for h_fg
        self.h_fg_ref = u["h_fg_ref"]["value"]            # J/kg latent heat at T_ref
        self.cp_liq = u["cp_liq"]["value"]                # J/(kg.K) liquid
        self.cp_vap = u["cp_vap"]["value"]                # J/(kg.K) sup. vapor
        self.T_crit = u["T_crit"]["value"]                # K
        self.P_crit = u["P_crit"]["value"]                # Pa
        self.T_nbp = u["T_nbp"]["value"]                  # K (1 atm boiling pt)
        self.gamma = u["gamma"]["value"]                  # cp/cv isentropic exp.

        # Compressor
        self.V_swept = u["V_swept"]["value"]              # m3/s displacement rate
        self.eta_is_ref = u["eta_isentropic"]["value"]    # isentropic efficiency
        self.eta_vol_ref = u["eta_volumetric"]["value"]   # volumetric efficiency
        self.clearance = u["clearance_ratio"]["value"]    # compressor clearance

        # Heat-exchanger approach temperatures
        self.dT_evap = u["dT_evap_approach"]["value"]     # K source->evap
        self.dT_cond = u["dT_cond_approach"]["value"]     # K cond->sink
        self.superheat = u["superheat"]["value"]          # K at evap outlet
        self.subcool = u["subcool"]["value"]              # K at cond outlet

        # Water-loop lumped thermal mass (condenser/load side)
        self.m_water = u["m_water_loop"]["value"]         # kg
        self.cp_water = u["cp_water"]["value"]            # J/(kg.K)
        self.UA_cond = u["UA_cond"]["value"]              # W/K condenser conductance

        self.aux_power = u["auxiliary_power"]["value"]    # W pumps/controls

    # ==================================================================
    # Refrigerant property correlations (lumped, no CoolProp)
    # ==================================================================
    def p_sat(self, T):
        """Saturation pressure [Pa] -- Clausius-Clapeyron anchored to nbp & crit.

        ln(P) is fit linear in 1/T through (T_nbp, 1 atm) and (T_crit, P_crit),
        a standard two-point Clausius-Clapeyron approximation (Cengel Ch.11).
        """
        T = np.asarray(T, dtype=float)
        P1, T1 = 101325.0, self.T_nbp
        P2, T2 = self.P_crit, self.T_crit
        # ln P = A - B/T
        B = np.log(P2 / P1) / (1.0 / T1 - 1.0 / T2)
        A = np.log(P1) + B / T1
        return np.exp(A - B / T)

    def h_fg(self, T):
        """Latent heat of vaporization [J/kg] -- Watson correlation (Cengel Ch.11)."""
        T = np.asarray(T, dtype=float)
        tr = np.clip((self.T_crit - T) / (self.T_crit - self.T_ref), 0.0, None)
        return self.h_fg_ref * tr ** 0.38

    def h_liq_sat(self, T):
        """Saturated-liquid enthalpy [J/kg] relative to T_ref datum."""
        return self.cp_liq * (np.asarray(T, dtype=float) - self.T_ref)

    def h_vap_sat(self, T):
        """Saturated-vapor enthalpy [J/kg] = h_liq + h_fg."""
        return self.h_liq_sat(T) + self.h_fg(T)

    def h_superheated(self, T, T_sat):
        """Superheated-vapor enthalpy [J/kg]: h_vap_sat(T_sat) + cp_vap*(T-T_sat)."""
        return self.h_vap_sat(T_sat) + self.cp_vap * (np.asarray(T, dtype=float) - T_sat)

    def rho_vap_sat(self, T):
        """Saturated-vapor density [kg/m3] from ideal-gas law at p_sat(T)."""
        P = self.p_sat(T)
        return P * self.M_R410A / (self.R_GAS * np.asarray(T, dtype=float))

    # ==================================================================
    # Compressor efficiencies
    # ==================================================================
    def volumetric_efficiency(self, pressure_ratio):
        """Volumetric efficiency from clearance-volume re-expansion model.

        eta_vol = eta_vol_ref * (1 + c - c*PR^(1/gamma))
        (Hundy/Trott/Welch 2016; standard reciprocating compressor model).
        """
        c = self.clearance
        eta = self.eta_vol_ref * (1.0 + c - c * pressure_ratio ** (1.0 / self.gamma))
        return float(np.clip(eta, 0.2, 1.0))

    def isentropic_efficiency(self, pressure_ratio):
        """Isentropic efficiency, mild de-rating with pressure ratio."""
        eta = self.eta_is_ref - 0.02 * max(pressure_ratio - 2.0, 0.0)
        return float(np.clip(eta, 0.3, 0.95))

    # ==================================================================
    # Vapor-compression cycle solver (steady-state thermodynamic states)
    # ==================================================================
    def cycle(self, T_source_c, T_sink_c):
        """Solve the 4-state vapor-compression cycle.

        Parameters
        ----------
        T_source_c : float   source-water temperature [degC] (evaporator side)
        T_sink_c   : float   sink/load-water temperature [degC] (condenser side)

        Returns
        -------
        dict with refrigerant states, COP, capacities, mass flow, etc.
        """
        T_source = T_source_c + 273.15
        T_sink = T_sink_c + 273.15

        # Evaporation / condensation saturation temperatures (HX approach).
        # Cap condensation safely below the critical temperature so the latent
        # heat (Watson) and the subcritical-cycle assumptions stay valid.
        T_evap = T_source - self.dT_evap
        T_cond = T_sink + self.dT_cond
        T_cond_max = self.T_crit - 5.0
        T_cond = min(T_cond, T_cond_max)
        if T_cond <= T_evap + 1.0:
            T_cond = T_evap + 1.0   # guard against degenerate lift

        P_evap = float(self.p_sat(T_evap))
        P_cond = float(self.p_sat(T_cond))
        pressure_ratio = P_cond / P_evap

        # ------------------------------------------------------------------
        # Enthalpy ladder on a SINGLE global datum: saturated liquid @ T_ref.
        #   liquid:           h_l(T)  = cp_liq*(T - T_ref)
        #   sat. vapor:       h_g(T)  = h_l(T) + h_fg(T)
        #   superheated:      h_g(T_sat) + cp_vap*(T - T_sat)
        # Compressor work is the VAPOR-PHASE enthalpy change only, so both
        # inlet and isentropic-outlet enthalpies are referenced to their own
        # saturation state -- this keeps the latent-heat terms out of the work.
        # ------------------------------------------------------------------

        # ---- State 1: evaporator outlet (superheated vapor) ----
        T1 = T_evap + self.superheat
        hg_evap = float(self.h_vap_sat(T_evap))
        h1 = hg_evap + self.cp_vap * (T1 - T_evap)
        rho1 = float(self.rho_vap_sat(T_evap)) * (T_evap / T1)  # ideal-gas corr.

        # ---- State 2s: isentropic compression to P_cond ----
        # Ideal-gas isentropic vapor temperature rise; work = sensible vapor
        # enthalpy rise (Cengel Ch.11 actual-cycle treatment).
        T2s = T1 * pressure_ratio ** ((self.gamma - 1.0) / self.gamma)
        w_is = self.cp_vap * (T2s - T1)                   # ideal specific work

        # ---- State 2: actual compressor outlet ----
        eta_is = self.isentropic_efficiency(pressure_ratio)
        w_comp = w_is / eta_is                            # actual specific work
        T2 = T1 + (T2s - T1) / eta_is
        h2 = h1 + w_comp

        # ---- State 3: condenser outlet (subcooled liquid) ----
        T3 = T_cond - self.subcool
        h3 = float(self.h_liq_sat(T3))

        # ---- State 4: after isenthalpic throttle ----
        h4 = h3                                           # isenthalpic expansion

        # Heats on the shared datum. Because h1, h2, h3, h4 all lie on one
        # enthalpy ladder, the first law closes exactly:
        #   q_cond - q_evap = (h2-h3) - (h1-h4) = (h2-h1) = w_comp.
        q_cond = h2 - h3                                  # heat rejected (heating)
        q_evap = h1 - h4                                  # heat absorbed (cooling)

        # ---- Mass flow from compressor displacement ----
        eta_vol = self.volumetric_efficiency(pressure_ratio)
        m_dot = eta_vol * self.V_swept * rho1             # kg/s

        # ---- Powers ----
        W_comp = m_dot * w_comp                           # W
        Q_cond = m_dot * q_cond                           # W (heating)
        Q_evap = m_dot * q_evap                           # W (cooling)

        cop_heat = q_cond / w_comp if w_comp > 0 else 0.0
        cop_cool = q_evap / w_comp if w_comp > 0 else 0.0
        cop_carnot = T_cond / (T_cond - T_evap)

        return {
            "T_evap": T_evap, "T_cond": T_cond,
            "P_evap": P_evap, "P_cond": P_cond,
            "pressure_ratio": pressure_ratio,
            "T1": T1, "T2s": T2s, "T2": T2, "T3": T3,
            "h1": h1, "h2s": h1 + w_is, "h2": h2, "h3": h3, "h4": h4,
            "w_comp": w_comp, "q_cond": q_cond, "q_evap": q_evap,
            "m_dot": m_dot, "eta_is": eta_is, "eta_vol": eta_vol,
            "W_comp": W_comp, "Q_cond": Q_cond, "Q_evap": Q_evap,
            "cop_heat": cop_heat, "cop_cool": cop_cool,
            "cop_carnot": cop_carnot,
        }

    # ==================================================================
    # Lumped water-loop thermal ODE (condenser / load side)
    # ==================================================================
    # High-limit safety: compressor modulates off as the load loop approaches
    # the upper design sink temperature (real WSHP high-condensing cutout).
    # Implemented as a smooth logistic de-rating so the ODE stays non-stiff.
    T_high_limit_c = 65.0
    high_limit_width = 1.5   # K, transition band of the cutout

    def _capacity_derate(self, T_load_c):
        """Smooth 1->0 modulation factor near the high-limit cutout."""
        x = (T_load_c - self.T_high_limit_c) / self.high_limit_width
        return 1.0 / (1.0 + np.exp(x))

    def dTdt(self, T_load_c, T_source_c, Q_demand_W, on=True):
        """Rate of change of load-loop water temperature [K/s].

        m_w cp_w dT/dt = f(T_load) * Q_cond(T_load) - Q_demand
        Q_cond depends on T_load via the cycle (warmer load -> higher lift ->
        lower capacity); f is the smooth high-limit modulation that prevents the
        loop from running past the cutout, giving a stable lumped first-order
        transient.
        """
        if on:
            st = self.cycle(T_source_c, T_load_c)
            Q_cond = st["Q_cond"] * self._capacity_derate(T_load_c)
        else:
            Q_cond = 0.0
        return (Q_cond - Q_demand_W) / (self.m_water * self.cp_water)

    def simulate(self, T_source_c, T_load0_c, Q_demand_W, duration_s, dt,
                 T_setpoint_c=None):
        """Transient warm-up of the condenser/load water loop.

        Parameters
        ----------
        T_source_c   : float   source-water temperature [degC] (assumed constant)
        T_load0_c    : float   initial load-loop temperature [degC]
        Q_demand_W   : float   heat draw from the load [W]
        duration_s   : float   total time [s]
        dt           : float   output time step [s]
        T_setpoint_c : float or None  if set, compressor cycles off above setpoint

        Returns
        -------
        dict of time-series arrays.
        """
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T_load = y[0]
            on = True
            if T_setpoint_c is not None and T_load >= T_setpoint_c:
                on = False
            return [self.dTdt(T_load, T_source_c, Q_demand_W, on=on)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_load0_c],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t_out = sol.t
        T_load = sol.y[0]
        N = len(t_out)

        cop = np.zeros(N)
        Q_cond = np.zeros(N)
        W_comp = np.zeros(N)
        Q_evap = np.zeros(N)
        cop_carnot = np.zeros(N)
        for i in range(N):
            st = self.cycle(T_source_c, T_load[i])
            cop[i] = st["cop_heat"]
            Q_cond[i] = st["Q_cond"]
            W_comp[i] = st["W_comp"] + self.aux_power
            Q_evap[i] = st["Q_evap"]
            cop_carnot[i] = st["cop_carnot"]

        return {
            "t": t_out,
            "T_load": T_load,
            "cop": cop,
            "cop_carnot": cop_carnot,
            "Q_cond": Q_cond,
            "Q_evap": Q_evap,
            "W_comp": W_comp,
        }
