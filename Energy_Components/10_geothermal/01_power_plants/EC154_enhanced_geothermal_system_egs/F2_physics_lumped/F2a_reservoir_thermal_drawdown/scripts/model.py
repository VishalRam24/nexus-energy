"""
EC154 -- Enhanced Geothermal System (EGS) -- F2a Physics-Lumped Model
Reservoir thermal drawdown ODE + lumped rock->fluid heat exchange + binary/ORC cycle.

============================================================================
PHYSICS
============================================================================
An EGS doublet circulates water from an injection well, through an engineered
(hydraulically stimulated) fracture network in hot dry rock, to a production
well, and through a surface binary/ORC power plant. The cold reinjected water
progressively MINES the heat out of the rock, so the rock temperature -- and
hence the produced fluid temperature and net power -- DECLINE over years
("thermal drawdown" / "thermal breakthrough"). This is the defining lifetime
limit of an EGS reservoir (Tester et al. 2006, "Heat Mining").

State variable (lumped 0-D):
    T_rock(t)  = bulk temperature of the stimulated rock volume [K]

(1) Lumped rock energy balance ODE (first principles, energy conservation):

      (rho_r * cp_r * V_res) * dT_rock/dt  =  - Q_extract(t)

    where the heat removed by the circulating fluid is set by a lumped
    convective rock->fluid exchanger over the fracture surface area A_f
    (effective NTU formulation, Gringarten et al. 1975; Armstead & Tester 1987):

      NTU      = h * A_f / (m_dot * cp_w)            (number of transfer units)
      eps      = 1 - exp(-NTU)                        (exchanger effectiveness)
      T_prod   = T_inj + eps * (T_rock - T_inj)       (produced fluid temp)
      Q_extract= m_dot * cp_w * (T_prod - T_inj)
               = m_dot * cp_w * eps * (T_rock - T_inj)

    Substituting, the ODE is linear and relaxes T_rock -> T_inj with a
    physical reservoir time constant:

      dT_rock/dt = -(m_dot*cp_w*eps)/(rho_r*cp_r*V_res) * (T_rock - T_inj)
      tau_res    =  (rho_r*cp_r*V_res) / (m_dot*cp_w*eps)     [s]

    => T_rock(t) = T_inj + (T_rock0 - T_inj)*exp(-t/tau_res), the classic
       exponential thermal-drawdown signature observed in EGS reservoirs.
    Integrated numerically with scipy.integrate.solve_ivp (no analytic shortcut
    is taken in the solver; the RHS is evaluated directly).

(2) Binary / ORC surface power cycle at each instant (DiPippo 2015, Ch.8/16):

      eta_carnot = 1 - T_cond / T_prod                (K, Carnot upper bound)
      eta_cycle  = eta_util * eta_carnot              (2nd-law utilization)
      Q_in       = m_dot * cp_w * (T_prod - T_inj)     (heat to the cycle, kW)
      P_gross    = eta_cycle * Q_in
      P_pump     = pump_frac * P_gross                 (circulation parasitic)
      P_net      = P_gross - P_pump = P_gross*(1 - pump_frac)

    Enforced physical bounds:
      - eta_cycle < eta_carnot strictly (eta_util < 1)
      - T_prod between T_inj and T_rock (0 < eps < 1)
      - reservoir cools monotonically:  dT_rock/dt <= 0  for T_rock > T_inj
      - energy conservation: integral of Q_extract = rock internal energy lost

============================================================================
REFERENCES
============================================================================
  Tester, J.W. et al. (2006). The Future of Geothermal Energy: Impact of
      Enhanced Geothermal Systems (EGS) on the United States in the 21st
      Century. MIT / US-DOE. (rock properties, drawdown, parasitics)
  DiPippo, R. (2015). Geothermal Power Plants, 4th ed., Ch.8 (binary cycles),
      Ch.16 (EGS). (cycle utilization efficiency, Carnot bound)
  Gringarten, A.C., Witherspoon, P.A., Ohnishi, Y. (1975). Theory of heat
      extraction from fractured hot dry rock. J. Geophys. Res., 80(8),
      1120-1124. (fracture heat-exchange model)
  Armstead, H.C.H. & Tester, J.W. (1987). Heat Mining. E. & F.N. Spon.
"""

import numpy as np
from scipy.integrate import solve_ivp

SEC_PER_YEAR = 365.25 * 24.0 * 3600.0


class EGS_F2a:
    """Physics-lumped EGS: reservoir thermal-drawdown ODE + binary/ORC cycle."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Operating point
        self.T_geo_init = u["T_geo_init_degC"]["value"]      # degC
        self.T_inject   = u["T_inject_degC"]["value"]        # degC
        self.m_dot      = u["m_dot_kg_s"]["value"]           # kg/s

        # Rock properties (cited: Tester 2006)
        self.rho_rock = u["rho_rock"]["value"]               # kg/m3
        self.cp_rock  = u["cp_rock_J_kgK"]["value"]          # J/(kg.K)
        self.k_rock   = u["k_rock_W_mK"]["value"]            # W/(m.K)

        # Water properties (cited: IAPWS approx)
        self.rho_water = u["rho_water"]["value"]             # kg/m3
        self.cp_water  = u["cp_water_J_kgK"]["value"]        # J/(kg.K)

        # Reservoir geometry / exchange
        self.V_res = u["V_reservoir_m3"]["value"]            # m3
        self.A_f   = u["A_fracture_m2"]["value"]             # m2
        self.h_rf  = u["h_rock_fluid_W_m2K"]["value"]        # W/(m2.K)

        # Surface power cycle
        self.eta_util  = u["eta_utilization"]["value"]       # -
        self.pump_frac = u["pump_parasitic_frac"]["value"]   # -
        self.T_amb     = u["T_ambient_degC"]["value"]        # degC
        self.P_rated   = u["P_rated_kW"]["value"]            # kW

    # ------------------------------------------------------------------
    # Lumped fracture heat-exchanger effectiveness (NTU formulation)
    # ------------------------------------------------------------------
    def effectiveness(self, m_dot=None):
        """
        Exchanger effectiveness eps = 1 - exp(-NTU), NTU = h*A_f/(m_dot*cp_w).
        Gringarten (1975) / Armstead & Tester (1987) lumped fracture exchange.
        Lower flow -> longer residence -> higher eps (fluid approaches T_rock).
        """
        m = self.m_dot if m_dot is None else float(m_dot)
        m = max(m, 1e-6)
        NTU = self.h_rf * self.A_f / (m * self.cp_water)
        return float(1.0 - np.exp(-NTU))

    def reservoir_time_constant_yr(self, m_dot=None):
        """Thermal-drawdown time constant tau_res [years]."""
        m = self.m_dot if m_dot is None else float(m_dot)
        eps = self.effectiveness(m)
        C_rock = self.rho_rock * self.cp_rock * self.V_res    # J/K (rock thermal mass)
        UA_eff = max(m, 1e-6) * self.cp_water * eps           # W/K (effective)
        tau_s = C_rock / UA_eff
        return float(tau_s / SEC_PER_YEAR)

    # ------------------------------------------------------------------
    # Produced fluid temperature given current rock temperature
    # ------------------------------------------------------------------
    def produced_temperature(self, T_rock_degC, m_dot=None):
        """T_prod = T_inj + eps*(T_rock - T_inj).  Bounded in [T_inj, T_rock]."""
        eps = self.effectiveness(m_dot)
        T_prod = self.T_inject + eps * (float(T_rock_degC) - self.T_inject)
        return float(T_prod)

    # ------------------------------------------------------------------
    # Surface binary/ORC cycle at one instant
    # ------------------------------------------------------------------
    def cycle_power(self, T_prod_degC, m_dot=None):
        """
        Binary/ORC instantaneous performance (DiPippo 2015).
        Returns dict: Q_in_kW, eta_carnot, eta_cycle, P_gross_kW,
        P_pump_kW, P_net_kW.
        """
        m = self.m_dot if m_dot is None else float(m_dot)
        T_prod = float(T_prod_degC)
        T_cond_K = self.T_amb + 273.15        # condenser ~ ambient cold side
        T_prod_K = T_prod + 273.15

        # Heat available to the cycle (cooled from T_prod down to T_inject)
        Q_in = m * self.cp_water * max(0.0, T_prod - self.T_inject) / 1000.0  # kW

        # Carnot bound and 2nd-law utilization
        eta_carnot = max(0.0, 1.0 - T_cond_K / T_prod_K)
        eta_cycle = self.eta_util * eta_carnot          # strictly < eta_carnot

        P_gross = eta_cycle * Q_in
        P_gross = min(P_gross, self.P_rated)            # nameplate cap
        P_pump = self.pump_frac * P_gross
        P_net = P_gross - P_pump
        return {
            "Q_in_kW": float(Q_in),
            "eta_carnot": float(eta_carnot),
            "eta_cycle": float(eta_cycle),
            "P_gross_kW": float(P_gross),
            "P_pump_kW": float(P_pump),
            "P_net_kW": float(max(0.0, P_net)),
        }

    # ------------------------------------------------------------------
    # Reservoir thermal-drawdown ODE RHS:  dT_rock/dt  [K/s]
    # ------------------------------------------------------------------
    def _rhs(self, t, y, m_dot, eps, C_rock):
        T_rock = y[0]
        # Heat extracted by circulating fluid (W)
        Q_extract = m_dot * self.cp_water * eps * (T_rock - self.T_inject)
        Q_extract = max(0.0, Q_extract)          # cannot reverse-heat the rock
        return [-Q_extract / C_rock]

    # ------------------------------------------------------------------
    # Integrate over a multi-year horizon with solve_ivp
    # ------------------------------------------------------------------
    def simulate(self, years=30.0, n_points=200, m_dot=None,
                 T_geo_init_degC=None):
        """
        Integrate the reservoir thermal-drawdown ODE over `years` and evaluate
        the binary/ORC cycle at each sample point.

        Returns dict of arrays (length n_points):
            t_years, T_rock_degC, T_prod_degC, Q_in_kW, eta_carnot,
            eta_cycle, P_gross_kW, P_pump_kW, P_net_kW
        plus scalars: tau_res_yr, effectiveness, energy_balance_err.
        """
        m = self.m_dot if m_dot is None else float(m_dot)
        T0 = self.T_geo_init if T_geo_init_degC is None else float(T_geo_init_degC)
        eps = self.effectiveness(m)
        C_rock = self.rho_rock * self.cp_rock * self.V_res    # J/K

        t_end = years * SEC_PER_YEAR
        t_eval = np.linspace(0.0, t_end, int(n_points))

        sol = solve_ivp(
            self._rhs, (0.0, t_end), [T0],
            t_eval=t_eval, args=(m, eps, C_rock),
            method="RK45", rtol=1e-8, atol=1e-6, max_step=t_end / 50.0,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        T_rock = sol.y[0]
        t_years = sol.t / SEC_PER_YEAR

        # Per-step cycle evaluation
        T_prod = np.empty_like(T_rock)
        Q_in = np.empty_like(T_rock)
        eta_carnot = np.empty_like(T_rock)
        eta_cycle = np.empty_like(T_rock)
        P_gross = np.empty_like(T_rock)
        P_pump = np.empty_like(T_rock)
        P_net = np.empty_like(T_rock)
        for i, Tr in enumerate(T_rock):
            Tp = self.produced_temperature(Tr, m)
            cyc = self.cycle_power(Tp, m)
            T_prod[i] = Tp
            Q_in[i] = cyc["Q_in_kW"]
            eta_carnot[i] = cyc["eta_carnot"]
            eta_cycle[i] = cyc["eta_cycle"]
            P_gross[i] = cyc["P_gross_kW"]
            P_pump[i] = cyc["P_pump_kW"]
            P_net[i] = cyc["P_net_kW"]

        # Energy conservation check:
        #   rock internal energy lost  ==  integral of Q_extract dt
        dU_rock = C_rock * (T_rock[0] - T_rock[-1])                # J
        Q_extract_W = m * self.cp_water * eps * (T_rock - self.T_inject)
        E_extracted = np.trapezoid(Q_extract_W, sol.t)            # J
        denom = max(abs(dU_rock), 1.0)
        energy_balance_err = abs(dU_rock - E_extracted) / denom

        return {
            "t_years": t_years,
            "T_rock_degC": T_rock,
            "T_prod_degC": T_prod,
            "Q_in_kW": Q_in,
            "eta_carnot": eta_carnot,
            "eta_cycle": eta_cycle,
            "P_gross_kW": P_gross,
            "P_pump_kW": P_pump,
            "P_net_kW": P_net,
            "tau_res_yr": self.reservoir_time_constant_yr(m),
            "effectiveness": eps,
            "energy_balance_err": float(energy_balance_err),
        }
