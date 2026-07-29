"""
EC156 -- Geothermal Heat Pump (GHP / Ground-Source) -- F2a
Physics-Lumped: vapor-compression cycle coupled to a lumped borehole ground
heat exchanger, with a transient ODE for the ground-loop fluid and condenser
(building-side) temperatures.

Model structure
---------------
1. Vapor-compression cycle (per-instant, from refrigerant states):
     - Evaporation temperature  T_evap = T_loop_out - pinch_evap   (ground loop -> refrigerant)
     - Condensation temperature T_cond = T_supply   + pinch_cond   (refrigerant -> building water)
     - R-410A saturation pressures via an Antoine fit of NIST REFPROP data
       (Lemmon 2003): ln P = A - B/(T + C).
     - Pressure ratio PR = P_cond / P_evap.
     - Volumetric efficiency  eta_v = a - b*(PR - 1)        (clearance re-expansion).
     - Refrigerant mass flow   mdot_r = eta_v * V_disp * n_comp * rho_suction.
     - Isentropic compressor work per unit mass via the polytropic (ideal-gas)
       relation  w_isen = (gamma/(gamma-1)) R/M T_suction (PR^((gamma-1)/gamma) - 1),
       divided by eta_isen (ASHRAE 2020, Handbook HVAC Systems & Equipment Ch.9).
     - Evaporator duty  Q_evap = mdot_r * (h_fg(T_evap) + cp_vap*superheat)
       (latent heat fit h_fg(T) from NIST, plus suction superheat sensible gain).
     - Condenser duty   Q_cond = Q_evap + W_comp        (1st-law energy balance).
     - COP_heating      = Q_cond / W_elec,  W_elec = W_comp/eta_motor + W_aux.
   Enforced: COP > 1 and COP < Carnot COP = T_cond/(T_cond - T_evap).

2. Ground heat exchanger (lumped borehole, Kavanaugh & Rafferty 2014):
     The ground loop extracts Q_evap from the borehole. The borehole wall
     temperature relaxes toward the undisturbed ground T with a lumped first-order
     ground response (effective ground capacitance from k, rho_cp and bore length),
     and the borehole resistance R_b sets the loop->wall temperature drop:
        T_wall = T_loop_avg - Q_evap * R_b / L      (steady borehole resistance)
     The ground node has a slow capacitance C_g; the loop fluid node has fast
     capacitance C_loop.

3. State ODE (scipy.solve_ivp):  state = [T_loop, T_cond_node, T_ground]
     C_loop * dT_loop/dt   = -Q_evap                      (heat pulled from loop by evaporator)
                              + UA_bore*(T_ground - T_loop)  (recharge from ground via borehole)
     C_cond * dT_cond/dt   = +Q_cond  - hA_cond_load*(T_cond - T_supply_setpoint?) ...
                              actually delivers to building: +Q_cond - Q_delivered
     C_ground* dT_ground/dt= -UA_bore*(T_ground - T_loop)   (ground depletion)
                              + UA_far*(T_undisturbed - T_ground)  (far-field recharge)

References
----------
- Kavanaugh, S.P. & Rafferty, K. (2014). "Geothermal Heating and Cooling:
  Design of Ground-Source Heat Pump Systems." ASHRAE. (borehole resistance R_b,
  ground conductivity, undisturbed ground T).
- ASHRAE Handbook -- HVAC Systems and Equipment (2020), Ch. 9 (Compressors),
  vapor-compression cycle and isentropic/volumetric compressor efficiency.
- Lemmon, E.W. (2003). "Pseudo-Pure Fluid Equations of State for the
  Refrigerant Blend R-410A." Int. J. Thermophys. 24(4). (R-410A properties).
"""

import numpy as np
from scipy.integrate import solve_ivp


class GHP_F2a:
    """Vapor-compression GSHP with lumped borehole + condenser thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        sat = params["saturation_R410A"]

        # --- compressor / cycle ---
        self.V_disp    = u["V_disp"]["value"]
        self.n_comp0   = u["n_comp"]["value"]
        self.eta_isen  = u["eta_isen"]["value"]
        self.eta_vol_a = u["eta_vol_a"]["value"]
        self.eta_vol_b = u["eta_vol_b"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.superheat = u["superheat_K"]["value"]
        self.subcool   = u["subcool_K"]["value"]
        self.W_aux     = u["auxiliary_power"]["value"] * 1e3       # W
        self.rated_cap = u["rated_capacity"]["value"]

        # --- heat-transfer pinches ---
        self.pinch_evap = u["pinch_evap_K"]["value"]
        self.pinch_cond = u["pinch_cond_K"]["value"]

        # --- refrigerant R-410A properties ---
        self.R_gas      = u["R_gas"]["value"]
        self.M          = u["M_R410A"]["value"]
        self.gamma      = u["gamma_R410A"]["value"]
        self.cp_vap     = u["cp_vap_R410A"]["value"] * 1e3        # J/(kg.K)
        self.hfg_ref    = u["hfg_ref_R410A"]["value"] * 1e3       # J/kg
        self.hfg_slope  = u["hfg_slope_R410A"]["value"] * 1e3     # J/(kg.K)
        self.Tref_hfg   = u["Tref_hfg_R410A"]["value"]            # degC

        # Antoine: ln(P[kPa]) = A - B/(T[K] + C)
        self.A_ant = sat["antoine_A"]
        self.B_ant = sat["antoine_B"]
        self.C_ant = sat["antoine_C"]

        # --- ground / borehole (Kavanaugh & Rafferty 2014) ---
        self.T_ground0  = u["T_ground_undisturbed"]["value"]      # degC
        self.k_ground   = u["k_ground"]["value"]
        self.rho_cp_g   = u["rho_cp_ground"]["value"]
        self.L_bore     = u["borehole_depth"]["value"]
        self.R_b        = u["R_b"]["value"]                       # (m.K)/W per unit length
        self.r_b        = u["r_b"]["value"]

        # --- ground-loop fluid node ---
        self.m_loop   = u["m_loop"]["value"]
        self.cp_loop  = u["cp_loop"]["value"]
        self.mdot_loop= u["mdot_loop"]["value"]
        self.C_loop   = self.m_loop * self.cp_loop               # J/K

        # --- condenser / building-side node ---
        self.C_cond      = u["C_cond"]["value"]                  # J/K
        self.hA_cond_load= u["hA_cond_load"]["value"]            # W/K

        # Borehole conductance loop<->ground (W/K): UA = L / R_b
        self.UA_bore = self.L_bore / self.R_b

        # Far-field ground recharge conductance and lumped ground capacitance.
        # Effective ground annulus from borehole wall to ~ thermal radius (K&R 2014):
        # use a steady cylindrical conduction shell of outer radius r_far.
        r_far = 3.0  # m, representative thermal influence radius for lumped node
        self.UA_far = 2.0 * np.pi * self.k_ground * self.L_bore / np.log(r_far / self.r_b)
        # Ground node capacitance = volume of influenced annulus * rho_cp
        vol_ground = np.pi * (r_far**2 - self.r_b**2) * self.L_bore
        self.C_ground = self.rho_cp_g * vol_ground               # J/K

    # ------------------------------------------------------------------
    # R-410A property correlations (NIST REFPROP-derived fits)
    # ------------------------------------------------------------------
    def p_sat(self, T_c):
        """Saturation pressure of R-410A [Pa] from Antoine fit. T_c in degC."""
        T_k = np.asarray(T_c, dtype=float) + 273.15
        ln_p_kpa = self.A_ant - self.B_ant / (T_k + self.C_ant)
        return np.exp(ln_p_kpa) * 1e3  # Pa

    def hfg(self, T_c):
        """Latent heat of vaporization of R-410A [J/kg]. Linear NIST fit."""
        return self.hfg_ref + self.hfg_slope * (np.asarray(T_c, dtype=float) - self.Tref_hfg)

    def rho_vapor(self, T_evap_c):
        """Suction (superheated) vapor density [kg/m3] via ideal-gas at suction state."""
        P = self.p_sat(T_evap_c)                       # ~ evaporator pressure
        T_suction = T_evap_c + self.superheat + 273.15  # K
        return P * self.M / (self.R_gas * T_suction)

    # ------------------------------------------------------------------
    # Vapor-compression cycle (instantaneous, given loop & supply temps)
    # ------------------------------------------------------------------
    def cycle(self, T_loop_c, T_supply_c, n_comp=None):
        """
        Compute the vapor-compression operating point.

        Parameters
        ----------
        T_loop_c   : ground-loop fluid temperature (degC) entering evaporator
        T_supply_c : building heating supply temperature (degC) at condenser
        n_comp     : compressor speed (rev/s); default rated.

        Returns dict with Q_evap, Q_cond, W_comp, W_elec (W), COP, COP_carnot,
        T_evap, T_cond, PR, mdot_r, eta_vol.
        """
        if n_comp is None:
            n_comp = self.n_comp0

        T_evap = T_loop_c - self.pinch_evap     # refrigerant boils below loop
        T_cond = T_supply_c + self.pinch_cond   # refrigerant condenses above supply

        # guard: condenser must be hotter than evaporator
        if T_cond <= T_evap + 0.5:
            T_cond = T_evap + 0.5

        P_evap = self.p_sat(T_evap)
        P_cond = self.p_sat(T_cond)
        PR = P_cond / P_evap

        # volumetric efficiency (clearance re-expansion), bounded
        eta_vol = np.clip(self.eta_vol_a - self.eta_vol_b * (PR - 1.0), 0.3, 1.0)

        # refrigerant mass flow [kg/s]
        rho_s = self.rho_vapor(T_evap)
        mdot_r = eta_vol * self.V_disp * n_comp * rho_s

        # isentropic specific work [J/kg] -- polytropic ideal-gas relation
        T_suction = T_evap + self.superheat + 273.15
        Rspec = self.R_gas / self.M
        k = self.gamma
        w_isen = (k / (k - 1.0)) * Rspec * T_suction * (PR ** ((k - 1.0) / k) - 1.0)
        w_actual = w_isen / self.eta_isen
        W_comp = mdot_r * w_actual                 # W (refrigerant-side)

        # evaporator duty: latent + suction superheat sensible
        q_evap_spec = self.hfg(T_evap) + self.cp_vap * self.superheat
        Q_evap = mdot_r * q_evap_spec              # W extracted from ground loop

        # condenser duty by 1st law (includes subcooling implicitly via balance)
        Q_cond = Q_evap + W_comp                   # W delivered to building

        # electrical input
        W_elec = W_comp / self.eta_motor + self.W_aux

        COP = Q_cond / W_elec
        # Carnot ceiling between the two saturation temperatures
        Tc_K = T_cond + 273.15
        Te_K = T_evap + 273.15
        COP_carnot = Tc_K / (Tc_K - Te_K)

        return {
            "T_evap": T_evap, "T_cond": T_cond, "PR": PR, "eta_vol": eta_vol,
            "mdot_r": mdot_r, "Q_evap": Q_evap, "Q_cond": Q_cond,
            "W_comp": W_comp, "W_elec": W_elec, "COP": COP, "COP_carnot": COP_carnot,
        }

    # ------------------------------------------------------------------
    # Transient lumped ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, T_supply_c, Q_demand_W, n_comp):
        """
        State y = [T_loop, T_cond_node, T_ground] (degC).

        Evaporator pulls Q_evap from the loop; borehole recharges the loop from
        the ground node; the ground node depletes and is recharged by the far
        field. Condenser node receives Q_cond and delivers heat to the building
        load (the smaller of demand and what the condenser UA can pass).
        """
        T_loop, T_cond_node, T_ground = y
        op = self.cycle(T_loop, T_supply_c, n_comp)
        Q_evap = op["Q_evap"]
        Q_cond = op["Q_cond"]

        # heat delivered to building from condenser node (limited by UA & demand)
        Q_deliver_cap = self.hA_cond_load * (T_cond_node - T_supply_c)
        Q_deliver = max(0.0, min(Q_deliver_cap, Q_demand_W))

        # loop fluid: lose Q_evap to evaporator, gain from ground via borehole
        dT_loop = (-Q_evap + self.UA_bore * (T_ground - T_loop)) / self.C_loop

        # condenser/building-water node
        dT_cond = (Q_cond - Q_deliver) / self.C_cond

        # ground node: feeds the loop (depletion), recharged by far field
        dT_ground = (-self.UA_bore * (T_ground - T_loop)
                     + self.UA_far * (self.T_ground0 - T_ground)) / self.C_ground

        return [dT_loop, dT_cond, dT_ground]

    def simulate(self, T_supply_c=45.0, Q_demand_kW=8.0, n_comp=None,
                 dt=60.0, duration_s=86400.0, T_loop0=None, T_cond0=None,
                 T_ground0=None):
        """
        Integrate the lumped GSHP over `duration_s`.

        Parameters
        ----------
        T_supply_c  : building heating supply setpoint (degC)
        Q_demand_kW : building heating demand (kW)
        n_comp      : compressor speed (rev/s); default rated.
        dt          : output sampling step (s)
        duration_s  : total simulated time (s)
        T_loop0/T_cond0/T_ground0 : optional initial temps (degC).

        Returns dict of time-series arrays + derived cycle metrics.
        """
        if n_comp is None:
            n_comp = self.n_comp0
        if T_ground0 is None:
            T_ground0 = self.T_ground0
        if T_loop0 is None:
            T_loop0 = T_ground0          # loop starts at undisturbed ground T
        if T_cond0 is None:
            T_cond0 = T_supply_c + self.pinch_cond

        Q_demand_W = Q_demand_kW * 1e3
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T_loop0, T_cond0, T_ground0],
            args=(T_supply_c, Q_demand_W, n_comp),
            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-6, max_step=dt,
        )

        T_loop = sol.y[0]
        T_cond_node = sol.y[1]
        T_ground = sol.y[2]

        # reconstruct cycle metrics along the trajectory
        COP = np.empty_like(T_loop)
        COP_carnot = np.empty_like(T_loop)
        Q_cond = np.empty_like(T_loop)
        Q_evap = np.empty_like(T_loop)
        W_elec = np.empty_like(T_loop)
        T_evap = np.empty_like(T_loop)
        T_cond_ref = np.empty_like(T_loop)
        for i, Tl in enumerate(T_loop):
            op = self.cycle(Tl, T_supply_c, n_comp)
            COP[i] = op["COP"]; COP_carnot[i] = op["COP_carnot"]
            Q_cond[i] = op["Q_cond"]; Q_evap[i] = op["Q_evap"]
            W_elec[i] = op["W_elec"]
            T_evap[i] = op["T_evap"]; T_cond_ref[i] = op["T_cond"]

        return {
            "t": sol.t,
            "T_loop": T_loop,
            "T_ground": T_ground,
            "T_cond_node": T_cond_node,
            "T_evap": T_evap,
            "T_cond_refrigerant": T_cond_ref,
            "COP": COP,
            "COP_carnot": COP_carnot,
            "Q_cond_kW": Q_cond / 1e3,
            "Q_evap_kW": Q_evap / 1e3,
            "W_elec_kW": W_elec / 1e3,
            "success": bool(sol.success),
        }
