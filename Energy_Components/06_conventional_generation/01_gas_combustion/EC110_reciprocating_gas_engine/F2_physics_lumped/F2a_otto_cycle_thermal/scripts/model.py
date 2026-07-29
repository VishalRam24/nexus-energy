"""
EC110 -- Reciprocating Gas Engine -- F2a Otto/Miller Cycle Physics-Lumped Model

Physics-lumped first-principles model of a lean-burn natural-gas spark-ignition
reciprocating engine. The thermodynamic cycle is modelled as an air-standard
Otto cycle with a Miller-cycle effective-expansion correction; mechanical losses
are captured with a friction-mean-effective-pressure (FMEP) correlation; and the
engine-block temperature is evolved with a lumped first-order thermal ODE solved
by scipy.integrate.solve_ivp.

Thermodynamic chain (per Heywood 1988, Ch.5 fuel-air cycles, Ch.2 mep):
    1. Ideal Otto thermal efficiency:
           eta_otto = 1 - 1/r_eff^(gamma-1)
       with r_eff = compression_ratio * f_miller (Miller early-IVC reduces the
       *effective* compression ratio used in the cycle work term).
    2. Trapped charge mass per cylinder per cycle from displacement, volumetric
       efficiency and intake charge density (Heywood Sec.6.2).
    3. Fuel energy release from air-fuel ratio (lambda * stoichiometric AFR) and
       combustion efficiency; heat-release Q_in = m_fuel * LHV * eta_comb.
    4. Indicated work  W_i = eta_otto * Q_in              (gross indicated)
       Indicated efficiency eta_i = eta_otto * eta_comb.
    5. Brake work  W_b = W_i - W_fric, where friction work follows FMEP:
           W_fric = FMEP * V_displaced
           FMEP   = a + b*N + c*BMEP_load   (Chen-Flynn form, Heywood Sec.13)
       so brake efficiency eta_b = W_b / Q_in_chem < eta_i (always).
    6. Engine-block thermal ODE (lumped capacitance):
           m_blk*cp_blk * dT/dt = Q_to_block - hA*(T - T_coolant)
       with Q_to_block a fixed fraction of fuel power (Heywood energy balance).

Physical guarantees enforced / testable:
    * eta_otto < Carnot bound between charge temperature and adiabatic flame temp.
    * eta_brake < eta_indicated < eta_otto (loss ordering).
    * Energy conservation: P_brake + P_friction + P_coolant + P_exhaust = P_fuel.
    * Part-load: brake efficiency falls as PLR -> 0 (friction fraction grows).

References:
    Heywood, J.B. (1988). Internal Combustion Engine Fundamentals. McGraw-Hill.
        - Sec.2.4  mean effective pressure (imep, bmep, fmep)
        - Ch.5     ideal air/fuel-air cycle efficiency, eta_otto = 1-r^(1-gamma)
        - Sec.4.9  combustion efficiency
        - Sec.6.2  volumetric efficiency
        - Sec.13   engine friction / FMEP correlations
    Miller, R.H. (1947). Supercharging and internally cooling air charge.
        Trans. ASME 69, 453-464.  (Miller cycle effective expansion)
    US EPA (2017). Catalog of CHP Technologies, Sec.2 Reciprocating Engines.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ReciprocatingGasEngineF2a:
    """Lean-burn gas engine -- Otto/Miller cycle + lumped engine-block thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_brake_rated = u["P_brake_rated_kw"]["value"] * 1e3       # W
        self.N_rated = u["N_rated_rpm"]["value"]                        # rpm
        self.n_cyl = u["n_cyl"]["value"]
        self.bore = u["bore_m"]["value"]                                # m
        self.stroke = u["stroke_m"]["value"]                            # m
        self.cr = u["compression_ratio"]["value"]
        self.gamma = u["gamma"]["value"]
        self.afr_stoich = u["afr_stoich"]["value"]
        self.lambda_ref = u["lambda_excess_air"]["value"]
        self.eta_comb = u["combustion_efficiency"]["value"]
        self.f_miller = u["f_miller"]["value"]
        self.fmep_a = u["fmep_a_kpa"]["value"] * 1e3                    # Pa
        self.fmep_b = u["fmep_b_kpa_per_rps"]["value"] * 1e3           # Pa per rpm
        self.fmep_c = u["fmep_c_kpa_per_bar"]["value"]                  # dimensionless factor on load mep
        self.LHV = u["LHV_gas_mjkg"]["value"] * 1e6                     # J/kg
        self.rho_gas = u["rho_gas_kgm3"]["value"]                       # kg/m3
        self.rho_air = u["rho_air_kgm3"]["value"]                       # kg/m3
        self.cp_gas = u["cp_gas_jkgk"]["value"]                         # J/kgK
        self.vol_eff = u["vol_efficiency"]["value"]
        self.T_intake = u["T_intake_K"]["value"]                        # K
        self.m_block = u["m_block_kg"]["value"]                         # kg
        self.cp_block = u["cp_block_jkgk"]["value"]                     # J/kgK
        self.hA_cool = u["hA_cool_wk"]["value"]                         # W/K
        self.T_coolant = u["T_coolant_K"]["value"]                      # K
        self.frac_block = u["frac_heat_to_block"]["value"]

        # Geometry (Heywood Sec.2.1)
        self.V_disp_cyl = (np.pi / 4.0) * self.bore**2 * self.stroke    # m3 per cylinder
        self.V_disp_total = self.V_disp_cyl * self.n_cyl               # m3 total displacement

    # ------------------------------------------------------------------
    # Thermodynamic cycle
    # ------------------------------------------------------------------
    def otto_efficiency(self, miller=True):
        """Air-standard (Otto) ideal cycle thermal efficiency [-].

        eta_otto = 1 - 1 / r_eff^(gamma-1)     (Heywood 1988, Eq.5.26)
        Miller cycle uses a reduced effective compression ratio.
        """
        r_eff = self.cr * (self.f_miller if miller else 1.0)
        r_eff = max(r_eff, 1.0001)
        return 1.0 - 1.0 / (r_eff ** (self.gamma - 1.0))

    def carnot_bound(self, T_charge=None, lam=None):
        """Carnot efficiency between charge temperature and adiabatic flame temp.

        Upper thermodynamic bound; eta_otto must lie below this.
        Adiabatic flame temperature estimated from energy balance on the
        trapped charge: T_flame = T_charge + Q_fuel/(m_charge*cp).
        """
        if T_charge is None:
            T_charge = self.T_intake
        if lam is None:
            lam = self.lambda_ref
        m_air, m_fuel = self._charge_masses(lam)
        m_charge = m_air + m_fuel
        Q = m_fuel * self.LHV * self.eta_comb
        T_flame = T_charge + Q / (m_charge * self.cp_gas)
        return 1.0 - T_charge / T_flame

    def _charge_masses(self, lam):
        """Trapped air & fuel mass per cylinder per cycle [kg].

        Air mass from displacement * volumetric efficiency * charge density.
        Fuel from air-fuel ratio: AFR = lambda * AFR_stoich.
        """
        m_air = self.rho_air * self.V_disp_cyl * self.vol_eff
        afr = lam * self.afr_stoich
        m_fuel = m_air / afr
        return m_air, m_fuel

    # ------------------------------------------------------------------
    # Friction (FMEP) -- Heywood Sec.13 / Chen-Flynn
    # ------------------------------------------------------------------
    def fmep(self, speed_rpm, bmep_load_pa):
        """Friction mean effective pressure [Pa].

        FMEP = a + b*N + c*BMEP_load   (Chen-Flynn correlation form).
        Always positive; grows with speed and load.
        """
        return self.fmep_a + self.fmep_b * speed_rpm + self.fmep_c * max(bmep_load_pa, 0.0)

    # ------------------------------------------------------------------
    # Steady-state performance at an operating point
    # ------------------------------------------------------------------
    def operating_point(self, part_load_ratio, speed_rpm=None, lam=None):
        """Full thermodynamic + mechanical balance at one operating point.

        Returns a dict of powers [W], efficiencies [-] and mean effective
        pressures [Pa]. 4-stroke: one power stroke every 2 revolutions, so
        cycles/s per cylinder = (N/60)/2.
        """
        if speed_rpm is None:
            speed_rpm = self.N_rated
        if lam is None:
            lam = self.lambda_ref
        plr = float(np.clip(part_load_ratio, 0.0, 1.0))

        # Fuelling scales with load; 4-stroke firing frequency
        cycles_per_s = (speed_rpm / 60.0) / 2.0
        m_air, m_fuel_full = self._charge_masses(lam)
        m_fuel = m_fuel_full * plr                                     # throttled fuelling
        Q_fuel_cycle = m_fuel * self.LHV                              # J per cyl per cycle (chemical)
        P_fuel = Q_fuel_cycle * cycles_per_s * self.n_cyl            # W chemical fuel power

        # Indicated work: Otto efficiency on the released heat
        eta_otto = self.otto_efficiency(miller=True)
        eta_ind = eta_otto * self.eta_comb                           # indicated < otto
        P_ind = eta_ind * P_fuel                                     # W indicated

        # Mean effective pressures: imep = W_ind / V_disp_total per cycle
        W_ind_cycle_total = P_ind / (cycles_per_s * self.n_cyl)      # J per cyl per cycle
        imep = W_ind_cycle_total / self.V_disp_cyl                    # Pa

        # Friction: FMEP grows with speed and indicated load
        fmep = self.fmep(speed_rpm, imep)
        bmep = imep - fmep                                            # net brake mep
        bmep = max(bmep, 0.0)

        # Brake power from BMEP
        P_brake = bmep * self.V_disp_cyl * cycles_per_s * self.n_cyl  # W
        P_fric = max(P_ind - P_brake, 0.0)

        eta_brake = P_brake / P_fuel if P_fuel > 0 else 0.0

        # Energy split: coolant (block) + exhaust = remainder
        P_to_block = self.frac_block * P_fuel
        P_exhaust = max(P_fuel - P_brake - P_fric - P_to_block, 0.0)

        return {
            "part_load_ratio": plr,
            "speed_rpm": speed_rpm,
            "lambda": lam,
            "P_fuel_w": P_fuel,
            "P_indicated_w": P_ind,
            "P_brake_w": P_brake,
            "P_friction_w": P_fric,
            "P_block_w": P_to_block,
            "P_exhaust_w": P_exhaust,
            "eta_otto": eta_otto,
            "eta_indicated": eta_ind,
            "eta_brake": eta_brake,
            "imep_pa": imep,
            "bmep_pa": bmep,
            "fmep_pa": fmep,
            "fuel_mass_flow_kgs": m_fuel * cycles_per_s * self.n_cyl,
        }

    # ------------------------------------------------------------------
    # Lumped engine-block thermal ODE
    # ------------------------------------------------------------------
    def _block_rhs(self, t, y, load_fn, speed_rpm, lam):
        """RHS of m*cp dT/dt = Q_to_block - hA*(T - T_coolant)."""
        T = y[0]
        plr = load_fn(t) if callable(load_fn) else load_fn
        op = self.operating_point(plr, speed_rpm, lam)
        Q_in = op["P_block_w"]
        Q_out = self.hA_cool * (T - self.T_coolant)
        dTdt = (Q_in - Q_out) / (self.m_block * self.cp_block)
        return [dTdt]

    def simulate(self, part_load_ratio, T0_K=298.15, speed_rpm=None,
                 lam=None, dt=1.0, duration_s=600.0):
        """Integrate the engine-block thermal transient with solve_ivp.

        part_load_ratio may be a scalar or a callable t -> PLR (load schedule).
        Returns time series of block temperature and per-step performance.
        """
        if speed_rpm is None:
            speed_rpm = self.N_rated
        if lam is None:
            lam = self.lambda_ref

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._block_rhs,
            (0.0, duration_s),
            [T0_K],
            t_eval=t_eval,
            args=(part_load_ratio, speed_rpm, lam),
            method="RK45",
            rtol=1e-6,
            atol=1e-3,
            max_step=dt,
        )
        T = sol.y[0]
        t = sol.t

        # Per-step steady performance (cycle is fast vs thermal transient)
        P_brake = np.zeros_like(t)
        P_fuel = np.zeros_like(t)
        eta_brake = np.zeros_like(t)
        eta_ind = np.zeros_like(t)
        for i, ti in enumerate(t):
            plr_i = part_load_ratio(ti) if callable(part_load_ratio) else part_load_ratio
            op = self.operating_point(plr_i, speed_rpm, lam)
            P_brake[i] = op["P_brake_w"]
            P_fuel[i] = op["P_fuel_w"]
            eta_brake[i] = op["eta_brake"]
            eta_ind[i] = op["eta_indicated"]

        return {
            "t": t,
            "temperature": T,
            "P_brake_w": P_brake,
            "P_fuel_w": P_fuel,
            "eta_brake": eta_brake,
            "eta_indicated": eta_ind,
            "eta_otto": self.otto_efficiency(miller=True),
        }
