"""
EC124 -- Liquid Air Energy Storage (LAES / CES) -- F2a Cryo-Tank Thermo Model

Physics-lumped (0D) first-principles model of a stand-alone LAES plant with three
operating modes coupled through a single lumped cryogenic tank state.

State integrated with scipy.integrate.solve_ivp:
    y = [m_liq, Q_cold_stored]
      m_liq        : liquid-air mass in tank                       [kg]
      Q_cold_stored: high-grade "cold" inventory available for     [kWh]
                     cold recycle into the next charge cycle

Three modes (chosen by the caller, may be sequenced):

1. CHARGE  -- air liquefaction (Claude/Linde cycle).
   Specific liquefaction work is reduced by the cold-recycle inventory:
       w_liq_eff = w_liq_floor + (w_liq_nom - w_liq_floor)*(1 - x_cold*eps_cr)
   where x_cold in [0,1] is the fraction of a full cold charge currently stored.
   Reversible (isothermal) lower bound w_liq_floor ~0.2 kWh/kg (Guizzi 2015).
       dm_liq/dt   = + m_dot_in
       dQ_cold/dt  = + m_dot_in * e_cold_per_kg   (cold produced & stored)
   Electrical draw:
       P_in = m_dot_in * w_liq_eff * 3600 / eta_liq          [kW]

2. STORE / IDLE -- cryogenic tank with parasitic heat ingress (boil-off).
   Lumped tank energy balance, Newton heat leak:
       Q_leak = UA*(T_amb - T_storage)                       [W]
   Boil-off mass rate:
       m_dot_bo = Q_leak / h_vap                              [kg/s]
       dm_liq/dt = - m_dot_bo
   Stored cold decays with the lost mass (cold leaves with vapour):
       dQ_cold/dt = -(m_dot_bo/m_liq)*Q_cold

3. DISCHARGE -- cryo-pump -> evaporate (cold recycle + waste heat) -> turbine.
   Available specific expansion work scales with the liquid-air physical exergy
   and is boosted by external/waste heat (hot recycle) superheating the air:
       w_exp = e_x_liq * (1 + beta*cp_g*dT_hot/(e_x_liq*3600))
   Net electrical output:
       P_out = m_dot_out * w_exp * 3600 * eta_pump*eta_exp*eta_gen   [kW]
       dm_liq/dt = - m_dot_out
   On discharge the cold is released (not stored): dQ_cold/dt = -(m_dot_out/m_liq)*Q_cold

Round-trip efficiency:
    eta_RT = E_out / E_in
Energy & mass conservation are enforced by the ODE structure; cold recycle and
hot (waste-heat) recycle both raise RTE, consistent with Highview / Morgan 2015.

Hardcoded cryogenic air properties (cited in parameters.json):
    h_vap ~205 kJ/kg, cp_liq ~1.68 kJ/kgK, cp_gas ~1.005 kJ/kgK,
    liquid-air physical exergy ~0.205 kWh/kg (739 kJ/kg) vs ambient,
    reversible liquefaction floor ~0.20-0.21 kWh/kg.

References:
    Morgan, R., Nelmes, S., Gibson, E., Brett, G. (2015).
        "Liquid air energy storage -- Analysis and first results from a pilot
         scale demonstration plant." Applied Energy 137, 845-853.
    Guizzi, G.L., Manno, M., Tolomei, L.M., Vitali, R.M. (2015).
        "Thermodynamic analysis of a liquid air energy storage system."
         Energy 93(1), 1382-1394.
    Sciacovelli, A., Vecchi, A., Ding, Y. (2017). Applied Energy 190, 84-98.
    Highview Power -- CRYOBattery technical literature.
"""

import numpy as np
from scipy.integrate import solve_ivp


class LAES_F2a:
    """Liquid Air Energy Storage -- lumped cryo-tank + thermodynamic cycle ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.w_liq_nom = u["specific_liquefaction_kwh_per_kg"]["value"]   # kWh/kg
        self.w_liq_floor = u["w_liq_floor_kwh_per_kg"]["value"]           # kWh/kg
        self.eta_liq = u["eta_liquefier"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_exp = u["eta_expander"]["value"]
        self.eta_gen = u["eta_generator"]["value"]

        self.e_x_liq = u["specific_exergy_liquid_kwh_per_kg"]["value"]    # kWh/kg
        self.m_tank_max = u["tank_capacity_kg"]["value"]                  # kg
        self.T_storage = u["T_storage"]["value"]                         # K
        self.rho_liq = u["rho_liquid_air"]["value"]                      # kg/m3

        self.cp_liq = u["cp_liquid_air_kJ_kgK"]["value"]                 # kJ/kgK
        self.cp_gas = u["cp_gas_air_kJ_kgK"]["value"]                    # kJ/kgK
        self.h_vap = u["h_vap_air_kJ_kg"]["value"]                       # kJ/kg

        self.UA = u["UA_tank_W_per_K"]["value"]                          # W/K
        self.T_amb_ref = u["T_amb_ref"]["value"]                        # K
        self.eps_cr_ref = u["cold_recycle_eff_ref"]["value"]            # -
        self.T_superheat_max = u["T_superheat_max_K"]["value"]          # K

        # Cold produced per kg liquefied [kWh/kg]: the high-grade cold that the
        # liquefaction process deposits in the cold-store. Tied to exergy of the
        # cold stream; use the liquid physical exergy as the reference quantum.
        self.e_cold_per_kg = self.e_x_liq

    # ------------------------------------------------------------------
    # Property / static helpers
    # ------------------------------------------------------------------

    def cold_fraction(self, Q_cold):
        """Fraction x_cold in [0,1] of a full cold charge currently stored."""
        Q_full = self.m_tank_max * self.e_cold_per_kg
        return np.clip(Q_cold / Q_full, 0.0, 1.0)

    def liquefaction_work(self, Q_cold=0.0, eps_cr=None):
        """
        Effective specific liquefaction work [kWh/kg].

        w_liq_eff = w_liq_floor + (w_liq_nom - w_liq_floor)*(1 - x_cold*eps_cr)

        More stored cold (x_cold) and better recycle effectiveness (eps_cr)
        push the work toward the reversible floor -> lower charge energy.
        """
        if eps_cr is None:
            eps_cr = self.eps_cr_ref
        x = self.cold_fraction(Q_cold)
        w = self.w_liq_floor + (self.w_liq_nom - self.w_liq_floor) * (1.0 - x * eps_cr)
        return float(np.clip(w, self.w_liq_floor, self.w_liq_nom))

    def expansion_work(self, hot_recycle_dT_K=0.0):
        """
        Specific net expansion (discharge) work delivered to the shaft [kWh/kg]
        before the pump/expander/generator efficiencies.

        Base = liquid physical exergy e_x_liq. Waste-heat (hot recycle)
        superheat dT adds reheat enthalpy converted via a Carnot-like factor.
        """
        dT = max(0.0, float(hot_recycle_dT_K))
        T_in = min(self.T_amb_ref + dT, self.T_superheat_max)
        dT = T_in - self.T_amb_ref
        # Extra shaft work from superheat: cp_g*dT [kJ/kg] * Carnot factor
        # (1 - T_amb/T_in), converted kJ->kWh.
        carnot = 0.0 if T_in <= self.T_amb_ref else (1.0 - self.T_amb_ref / T_in)
        w_extra_kwh = self.cp_gas * dT * carnot / 3600.0
        return self.e_x_liq + w_extra_kwh

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def charge_power_kw(self, m_dot_in, Q_cold=0.0, eps_cr=None):
        """Electrical input power [kW] to liquefy m_dot_in [kg/s]."""
        w = self.liquefaction_work(Q_cold, eps_cr)
        return float(np.asarray(m_dot_in)) * w * 3600.0 / self.eta_liq

    def discharge_power_kw(self, m_dot_out, hot_recycle_dT_K=0.0):
        """Net electrical output power [kW] from expanding m_dot_out [kg/s]."""
        w = self.expansion_work(hot_recycle_dT_K)
        return (float(np.asarray(m_dot_out)) * w * 3600.0
                * self.eta_pump * self.eta_exp * self.eta_gen)

    def boil_off_rate_kgs(self, T_amb_K=None):
        """Boil-off mass rate [kg/s] from parasitic heat leak."""
        T_amb = self.T_amb_ref if T_amb_K is None else float(T_amb_K)
        Q_leak_W = max(0.0, self.UA * (T_amb - self.T_storage))
        return Q_leak_W / (self.h_vap * 1000.0)  # h_vap kJ/kg -> J/kg

    def boil_off_per_day(self, T_amb_K=None, m_liq=None):
        """Boil-off rate as fraction of inventory per day."""
        m = self.m_tank_max if m_liq is None else float(m_liq)
        if m <= 0:
            return 0.0
        return self.boil_off_rate_kgs(T_amb_K) * 86400.0 / m

    # ------------------------------------------------------------------
    # ODE right-hand sides
    # ------------------------------------------------------------------

    def _rhs(self, t, y, mode, m_dot, T_amb_K, eps_cr, hot_dT):
        m_liq, Q_cold = y
        m_liq = max(m_liq, 0.0)

        if mode == "charge":
            cap_left = max(self.m_tank_max - m_liq, 0.0)
            md = m_dot if cap_left > 1e-6 else 0.0
            dm = md
            # Cold store is held at its recycled level during cyclic charge
            # (cold produced is balanced by cold consumed in liquefaction);
            # the static Q_cold0 sets the reduced-work operating point.
            dQ = 0.0
            return [dm, dQ]

        if mode == "discharge":
            md = m_dot if m_liq > 1e-6 else 0.0
            dm = -md
            # cold leaves proportionally with mass drawn
            dQ = -(md / m_liq) * Q_cold if m_liq > 1e-9 else 0.0
            return [dm, dQ]

        # store / idle: boil-off only
        m_dot_bo = self.boil_off_rate_kgs(T_amb_K)
        m_dot_bo = m_dot_bo if m_liq > 1e-6 else 0.0
        dm = -m_dot_bo
        dQ = -(m_dot_bo / m_liq) * Q_cold if m_liq > 1e-9 else 0.0
        return [dm, dQ]

    # ------------------------------------------------------------------
    # Single-mode time integration
    # ------------------------------------------------------------------

    def simulate(self, mode, duration_s, m_dot=0.0, m_liq0=0.0, Q_cold0=0.0,
                 T_amb_K=None, eps_cr=None, hot_recycle_dT_K=0.0, n_out=200):
        """
        Integrate the lumped tank ODE for one operating mode.

        Args:
            mode      : 'charge' | 'discharge' | 'store'
            duration_s: simulation horizon [s]
            m_dot     : liquid-air mass flow for charge/discharge [kg/s]
            m_liq0    : initial liquid mass [kg]
            Q_cold0   : initial stored cold [kWh]
            T_amb_K   : ambient temperature [K] (boil-off / hot recycle ref)
            eps_cr    : cold-recycle effectiveness [-]
            hot_recycle_dT_K : waste-heat superheat above ambient [K]
            n_out     : output sample count

        Returns dict of time-series arrays + scalar energies.
        """
        if T_amb_K is None:
            T_amb_K = self.T_amb_ref
        if eps_cr is None:
            eps_cr = self.eps_cr_ref

        t_eval = np.linspace(0.0, duration_s, n_out)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [m_liq0, Q_cold0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-3,
            args=(mode, m_dot, T_amb_K, eps_cr, hot_recycle_dT_K),
        )
        m_liq = np.clip(sol.y[0], 0.0, self.m_tank_max)
        Q_cold = np.clip(sol.y[1], 0.0, None)
        soc = m_liq / self.m_tank_max

        # Power & cumulative energy
        n = len(sol.t)
        P = np.zeros(n)
        if mode == "charge":
            for i in range(n):
                P[i] = self.charge_power_kw(m_dot if m_liq[i] < self.m_tank_max else 0.0,
                                            Q_cold[i], eps_cr)
        elif mode == "discharge":
            for i in range(n):
                P[i] = self.discharge_power_kw(m_dot if m_liq[i] > 0 else 0.0,
                                               hot_recycle_dT_K)

        dt = np.gradient(sol.t)
        E_kwh = np.cumsum(P * dt) / 3600.0  # kW * s / 3600 -> kWh

        return {
            "t": sol.t,
            "m_liq": m_liq,
            "soc": soc,
            "Q_cold": Q_cold,
            "power_kW": P,
            "energy_kWh": E_kwh,
            "mode": mode,
            "boil_off_per_day": self.boil_off_per_day(T_amb_K,
                                                      m_liq[-1] if m_liq[-1] > 0 else None),
        }

    # ------------------------------------------------------------------
    # Full round-trip cycle (charge -> store -> discharge)
    # ------------------------------------------------------------------

    def round_trip(self, charge_mass_kg=None, store_hours=0.0, T_amb_K=None,
                   eps_cr=None, hot_recycle_dT_K=0.0, m_dot=100.0):
        """
        Run a complete charge -> store -> discharge cycle and report RTE.

        Returns dict with E_in_kWh, E_out_kWh, eta_RT, boil_off_loss_kg, etc.
        """
        if T_amb_K is None:
            T_amb_K = self.T_amb_ref
        if eps_cr is None:
            eps_cr = self.eps_cr_ref
        if charge_mass_kg is None:
            charge_mass_kg = self.m_tank_max

        # --- Charge (steady cyclic state) ---
        # In cyclic operation the cold released during the previous discharge's
        # evaporation step is stored and recycled into this charge. Pre-load the
        # cold store with eps_cr * (full cold inventory) so the liquefier runs at
        # its reduced-work point from the start (Highview / Sciacovelli 2017).
        Q_cold_recycled = eps_cr * charge_mass_kg * self.e_cold_per_kg
        t_charge = charge_mass_kg / m_dot
        rc = self.simulate("charge", t_charge, m_dot=m_dot, m_liq0=0.0,
                           Q_cold0=Q_cold_recycled, T_amb_K=T_amb_K, eps_cr=eps_cr)
        E_in = rc["energy_kWh"][-1]
        m_after_charge = rc["m_liq"][-1]
        Q_after_charge = rc["Q_cold"][-1]

        # --- Store (boil-off) ---
        loss_kg = 0.0
        m_after_store = m_after_charge
        Q_after_store = Q_after_charge
        if store_hours > 0:
            rs = self.simulate("store", store_hours * 3600.0, m_liq0=m_after_charge,
                              Q_cold0=Q_after_charge, T_amb_K=T_amb_K)
            m_after_store = rs["m_liq"][-1]
            Q_after_store = rs["Q_cold"][-1]
            loss_kg = m_after_charge - m_after_store

        # --- Discharge ---
        t_disch = m_after_store / m_dot
        rd = self.simulate("discharge", t_disch, m_dot=m_dot, m_liq0=m_after_store,
                          Q_cold0=Q_after_store, T_amb_K=T_amb_K,
                          hot_recycle_dT_K=hot_recycle_dT_K)
        E_out = rd["energy_kWh"][-1]

        eta_RT = E_out / E_in if E_in > 0 else 0.0
        return {
            "E_in_kWh": E_in,
            "E_out_kWh": E_out,
            "eta_RT": eta_RT,
            "charge_mass_kg": m_after_charge,
            "boil_off_loss_kg": loss_kg,
            "w_liq_eff_kwh_per_kg": self.liquefaction_work(Q_after_charge, eps_cr),
            "w_exp_kwh_per_kg": self.expansion_work(hot_recycle_dT_K),
            "eps_cr": eps_cr,
            "hot_recycle_dT_K": hot_recycle_dT_K,
        }
