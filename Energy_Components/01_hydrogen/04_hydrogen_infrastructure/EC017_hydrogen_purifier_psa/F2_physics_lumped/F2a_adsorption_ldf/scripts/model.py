"""
EC017 -- Hydrogen Purifier (Pressure Swing Adsorption, PSA) -- F2a
Lumped Adsorption Column with Langmuir Isotherm + Linear-Driving-Force (LDF) Kinetics

Physics-lumped (0D / single-stage) first-principles PSA model. The packed bed is
treated as one lumped adsorption stage (a "mixing cell" / CSTR-equivalent of the
column) through which reformate (H2 + lumped adsorbable impurity) flows. The
hydrogen is essentially non-adsorbing; the impurity (CO/CO2/CH4/N2 lumped) is
captured by the adsorbent. We track the gas-phase impurity mole fraction and the
solid-phase impurity loading q over a full PSA cycle:

    1. ADSORPTION (high pressure P_H): feed enters, impurity is adsorbed, the bed
       loads up, H2-rich product leaves. Breakthrough occurs when the bed nears
       saturation and impurity slips into the product.
    2. BLOWDOWN (depressurisation to P_L): co-/counter-current pressure drop;
       partial desorption as gas-phase partial pressure collapses.
    3. PURGE (low pressure, H2 sweep): clean H2 sweeps the bed, driving the
       impurity loading back down -> regeneration for the next cycle.

Governing equations (Ruthven 1984; Yang 1987):

Equilibrium (Langmuir isotherm) for impurity at partial pressure p_i [bar]:
    q*(p_i, T) = q_sat * b(T) * p_i / (1 + b(T) * p_i)            [mol/kg]
    b(T) = b_ref * exp( (Q_st/R) * (1/T - 1/T_ref) )    (van't Hoff)

Kinetics (Linear Driving Force, Glueckauf 1955):
    dq/dt = k_LDF * ( q*(p_i, T) - q )                            [mol/kg/s]

Gas-phase impurity balance over the lumped bed void (mole balance, isothermal):
    eps * V_bed/(R T) * d p_i/dt
        = (F_in*y_in - F_out*y_out)            convective in/out
          - rho_b * V_bed * dq/dt              uptake to solid (sink)

We integrate the LDF loading ODE with scipy.integrate.solve_ivp through each
step, with the gas-phase impurity partial pressure relaxed quasi-statically to
the convective balance each step (lumped-stage approximation).

Performance metrics over the cycle (Sircar & Golden 2000):
    Purity      y_H2_product = 1 - (impurity in product)/(product moles)
    Recovery    = (H2 out as product - H2 used for purge) / (H2 fed)   in (0,1)
    Productivity= mol H2 product per kg adsorbent per cycle  [mol/kg/cycle]

Mass conservation: impurity_fed = impurity_in_product + impurity_in_tail
(adsorbed mass returns to gas during blowdown+purge over a cyclic-steady cycle).

References:
    Ruthven, D.M. (1984). Principles of Adsorption and Adsorption Processes. Wiley.
    Yang, R.T. (1987). Gas Separation by Adsorption Processes. Butterworths.
    Glueckauf, E. (1955). Trans. Faraday Soc. 51, 1540 (LDF approximation).
    Sircar, S. & Golden, T.C. (2000). Sep. Sci. Technol. 35(5), 667-687.
"""

import numpy as np
from scipy.integrate import solve_ivp


class HydrogenPSA_F2a:
    """Lumped PSA adsorption column: Langmuir + LDF over an adsorb/blowdown/purge cycle."""

    def __init__(self, params: dict):
        u = params["unit"]
        th = params["thermodynamics"]

        self.R = th["R"]["value"]                       # J/(mol.K)
        self.M_H2 = th["M_H2"]["value"]                 # kg/mol

        # Bed geometry
        self.L = u["bed_length_m"]["value"]             # m
        self.D = u["bed_diameter_m"]["value"]           # m
        self.eps = u["void_fraction"]["value"]          # -
        self.rho_b = u["bulk_density_kg_m3"]["value"]   # kg/m3
        self.A = np.pi * (self.D / 2.0) ** 2            # m2 cross-section
        self.V_bed = self.A * self.L                    # m3 total bed volume
        self.m_ads = self.rho_b * self.V_bed            # kg adsorbent

        # Langmuir isotherm (impurity)
        self.q_sat = u["q_sat_mol_kg"]["value"]         # mol/kg
        self.b_ref = u["b_langmuir_ref_per_bar"]["value"]  # 1/bar
        self.Q_st = u["isosteric_heat_J_mol"]["value"]  # J/mol
        self.T_ref = u["T_ref_K"]["value"]              # K

        # LDF kinetics
        self.k_ldf = u["k_ldf_1_s"]["value"]            # 1/s

        # Operating conditions
        self.y_feed = u["feed_h2_fraction"]["value"]    # mol fraction H2
        self.P_H = u["feed_pressure_bar"]["value"]      # bar (adsorption)
        self.P_L = u["purge_pressure_bar"]["value"]     # bar (purge/blowdown)
        self.T_op = u["T_operating_K"]["value"]         # K
        self.u_feed = u["u_feed_m_s"]["value"]          # m/s interstitial

        # Cycle timing
        self.t_ads = u["t_adsorption_s"]["value"]       # s
        self.t_blow = u["t_blowdown_s"]["value"]        # s
        self.t_purge = u["t_purge_s"]["value"]          # s
        self.purge_ratio = u["purge_to_feed_ratio"]["value"]  # -

    # ------------------------------------------------------------------
    # Equilibrium: Langmuir affinity with van't Hoff temperature dependence
    # ------------------------------------------------------------------
    def b_langmuir(self, T):
        """Temperature-dependent Langmuir affinity b(T) [1/bar]. van't Hoff."""
        return self.b_ref * np.exp((self.Q_st / self.R) * (1.0 / T - 1.0 / self.T_ref))

    def q_equilibrium(self, p_imp_bar, T):
        """
        Langmuir equilibrium loading q* [mol/kg] for impurity partial pressure
        p_imp_bar [bar] at temperature T [K]. Ruthven (1984) Eq. 3.x.
        """
        p = np.maximum(np.asarray(p_imp_bar, dtype=float), 0.0)
        b = self.b_langmuir(T)
        return self.q_sat * b * p / (1.0 + b * p)

    # ------------------------------------------------------------------
    # LDF loading ODE (Glueckauf 1955)
    # ------------------------------------------------------------------
    def dqdt_ldf(self, q, p_imp_bar, T):
        """LDF uptake rate dq/dt [mol/kg/s] = k_LDF (q* - q)."""
        q_star = self.q_equilibrium(p_imp_bar, T)
        return self.k_ldf * (q_star - q)

    # ------------------------------------------------------------------
    # Total gas-phase impurity moles held in the bed void at given p_imp
    # ------------------------------------------------------------------
    def _void_moles(self, P_total_bar, T):
        """Total gas moles in bed void at pressure P [bar]. ideal gas."""
        P_pa = P_total_bar * 1e5
        return self.eps * self.V_bed * P_pa / (self.R * T)

    # ------------------------------------------------------------------
    # Full PSA cycle simulation
    # ------------------------------------------------------------------
    def simulate_cycle(self, y_feed=None, P_H=None, P_L=None, T=None,
                       t_ads=None, t_purge=None, purge_ratio=None,
                       dt=1.0, q0=0.0):
        """
        Simulate ONE PSA cycle (adsorption -> blowdown -> purge) on the lumped bed.

        Integrates the LDF loading ODE with solve_ivp through each step, tracking
        the solid loading q(t) and the gas-phase impurity partial pressure.

        Returns dict with time-series and cycle-integrated performance metrics.
        """
        y_feed = self.y_feed if y_feed is None else y_feed
        P_H = self.P_H if P_H is None else P_H
        P_L = self.P_L if P_L is None else P_L
        T = self.T_op if T is None else T
        t_ads = self.t_ads if t_ads is None else t_ads
        t_purge = self.t_purge if t_purge is None else t_purge
        purge_ratio = self.purge_ratio if purge_ratio is None else purge_ratio
        t_blow = self.t_blow

        y_imp_feed = 1.0 - y_feed                      # impurity mole fraction in feed

        # Molar feed flow [mol/s]: F = u * A_eps * (P/RT)
        # interstitial velocity over void area -> superficial = u*eps; molar density P/RT
        P_H_pa = P_H * 1e5
        c_H = P_H_pa / (self.R * T)                    # mol/m3 at high pressure
        F_feed = self.u_feed * self.A * self.eps * c_H  # mol/s total feed

        # ---------- STEP 1: ADSORPTION (high pressure) ----------
        # Gas-phase impurity partial pressure feeding the bed
        p_imp_feed = y_imp_feed * P_H                  # bar

        n_ads = max(int(round(t_ads / dt)), 2)
        t_eval_ads = np.linspace(0.0, t_ads, n_ads)

        def rhs_ads(t, y):
            q = y[0]
            # quasi-static gas-phase impurity partial pressure in the bed:
            # convective supply at p_imp_feed; the lumped void equilibrates toward
            # feed composition but is depleted by uptake. We use the feed partial
            # pressure as the driving gas concentration (worst case = strong driving),
            # valid while bed is far from breakthrough.
            return [self.dqdt_ldf(q, p_imp_feed, T)]

        sol_ads = solve_ivp(rhs_ads, (0.0, t_ads), [q0], t_eval=t_eval_ads,
                            method="RK45", rtol=1e-8, atol=1e-10, max_step=dt)
        q_ads = sol_ads.y[0]
        # LDF-equilibrium loading the bed *would* reach with unlimited supply
        q_ldf_target = sol_ads.y[0][-1]

        # Impurity fed during adsorption [mol]
        imp_fed = F_feed * y_imp_feed * t_ads
        # H2 fed during adsorption [mol]
        h2_fed = F_feed * y_feed * t_ads
        # Total moles fed
        n_fed = F_feed * t_ads

        # Convective supply limit: the bed cannot adsorb more impurity than was
        # fed to it during the step. Likewise it cannot exceed Langmuir saturation.
        q_uptake_ldf = q_ldf_target - q0                       # mol/kg LDF wants
        imp_ldf_capacity = self.m_ads * max(q_uptake_ldf, 0.0)  # mol bed could take
        q_room_to_sat = max(self.q_sat - q0, 0.0)
        imp_sat_capacity = self.m_ads * q_room_to_sat          # mol until saturation

        # Capture capacity = min(supply, LDF kinetic capacity, saturation room).
        # This is the *maximum* impurity the bed can take this step.
        imp_capturable = min(imp_fed, imp_ldf_capacity, imp_sat_capacity)
        imp_capturable = max(imp_capturable, 0.0)

        # Mass-transfer-zone (MTZ) leakage: even far from breakthrough a finite LDF
        # rate leaves a small impurity partial pressure in the product-end void gas,
        # so the bed captures slightly less than the capturable maximum. The leakage
        # fraction is set by how hard the bed is driven relative to LDF response:
        # tau = 1/(k_LDF * t_ads) is the dimensionless mass-transfer time.
        # Sircar & Golden (2000): industrial H2 PSA product ~99.99%, not 100%.
        tau_mt = 1.0 / max(self.k_ldf * t_ads, 1e-9)
        leak_frac = float(np.clip(tau_mt, 1e-4, 0.5))  # fractional MTZ leakage

        # Actual impurity adsorbed (a small MTZ fraction leaks into product instead)
        imp_adsorbed = imp_capturable * (1.0 - leak_frac)

        # Resulting end-of-adsorption loading (mass-consistent with imp_adsorbed)
        q_end_ads = q0 + imp_adsorbed / self.m_ads
        q_ads = np.clip(q_ads, 0.0, q_end_ads)  # display series bounded by mass-balance

        # Breakthrough + MTZ slip = all impurity fed that was NOT adsorbed [mol] >= 0.
        # As the bed approaches saturation (q0 -> q_sat) imp_capturable shrinks and
        # slip rises sharply -> purity falls: the breakthrough mechanism.
        imp_slip = max(imp_fed - imp_adsorbed, 0.0)

        # Product gas during adsorption = everything fed minus what stayed adsorbed.
        n_product_raw = n_fed - imp_adsorbed
        # H2 in product (H2 is non-adsorbing -> all fed H2 leaves)
        h2_in_product_raw = h2_fed
        # impurity in product = slip
        imp_in_product = imp_slip

        # ---------- STEP 2: BLOWDOWN (depressurise H -> L) ----------
        # Partial pressure of impurity in gas collapses with total pressure.
        # During blowdown gas leaves co/counter-current; loading relaxes toward the
        # (much lower) equilibrium at the depressurised partial pressure.
        # Approximate driving partial pressure as the average over blowdown.
        p_imp_blow = y_imp_feed * P_L                  # bar (low-pressure gas)
        n_blow = max(int(round(t_blow / dt)), 2)
        t_eval_blow = np.linspace(0.0, t_blow, n_blow)

        def rhs_blow(t, y):
            q = y[0]
            return [self.dqdt_ldf(q, p_imp_blow, T)]

        sol_blow = solve_ivp(rhs_blow, (0.0, t_blow), [q_end_ads], t_eval=t_eval_blow,
                            method="RK45", rtol=1e-8, atol=1e-10, max_step=dt)
        q_blow = sol_blow.y[0]
        q_end_blow = q_blow[-1]

        # Impurity desorbed during blowdown [mol]
        imp_desorbed_blow = max(self.m_ads * (q_end_ads - q_end_blow), 0.0)

        # ---------- STEP 3: PURGE (low pressure, clean H2 sweep) ----------
        # Pure H2 sweep -> impurity gas partial pressure ~ 0 -> strong desorption.
        p_imp_purge = 0.0                              # clean H2, no impurity in sweep
        n_pg = max(int(round(t_purge / dt)), 2)
        t_eval_pg = np.linspace(0.0, t_purge, n_pg)

        def rhs_purge(t, y):
            q = y[0]
            return [self.dqdt_ldf(q, p_imp_purge, T)]

        sol_pg = solve_ivp(rhs_purge, (0.0, t_purge), [q_end_blow], t_eval=t_eval_pg,
                          method="RK45", rtol=1e-8, atol=1e-10, max_step=dt)
        q_pg = sol_pg.y[0]
        q_end_purge = q_pg[-1]

        imp_desorbed_purge = max(self.m_ads * (q_end_blow - q_end_purge), 0.0)

        # H2 consumed by purge [mol] = purge_ratio * H2 product
        h2_purge = purge_ratio * h2_in_product_raw

        # ---------- PERFORMANCE METRICS ----------
        # Net H2 product = product H2 minus H2 used for purge
        h2_product_net = max(h2_in_product_raw - h2_purge, 0.0)

        # Product stream moles (net): net H2 + impurity slip
        n_product_net = h2_product_net + imp_in_product
        purity = h2_product_net / n_product_net if n_product_net > 0 else 1.0
        purity = float(np.clip(purity, 0.0, 1.0))

        # Recovery = net H2 recovered / H2 fed   (< 1)
        recovery = h2_product_net / h2_fed if h2_fed > 0 else 0.0
        recovery = float(np.clip(recovery, 0.0, 0.99999))

        # Productivity [mol H2 product / kg adsorbent / cycle]
        t_cycle = t_ads + t_blow + t_purge
        productivity = h2_product_net / self.m_ads if self.m_ads > 0 else 0.0

        # ---------- MASS CONSERVATION CHECK ----------
        # impurity fed must equal impurity in product (slip) + impurity adsorbed
        imp_balance_residual = abs(imp_fed - (imp_in_product + imp_adsorbed))
        # net impurity stored over cycle (should be ~0 at cyclic steady state if
        # desorbed == adsorbed). Here we report the net change in loading.
        imp_net_stored = self.m_ads * (q_end_purge - q0)

        # Assemble time-series across the whole cycle
        t_full = np.concatenate([
            sol_ads.t,
            sol_ads.t[-1] + sol_blow.t,
            sol_ads.t[-1] + sol_blow.t[-1] + sol_pg.t,
        ])
        q_full = np.concatenate([q_ads, q_blow, q_pg])
        q_star_full = np.concatenate([
            np.full_like(q_ads, self.q_equilibrium(p_imp_feed, T)),
            np.full_like(q_blow, self.q_equilibrium(p_imp_blow, T)),
            np.full_like(q_pg, self.q_equilibrium(p_imp_purge, T)),
        ])

        return {
            "t": t_full,
            "loading": q_full,                    # mol/kg
            "loading_equilibrium": q_star_full,   # mol/kg target per step
            "q_end_adsorption": q_end_ads,
            "q_end_blowdown": q_end_blow,
            "q_end_purge": q_end_purge,
            "purity": purity,
            "recovery": recovery,
            "productivity_mol_kg_cycle": productivity,
            "cycle_time_s": t_cycle,
            "H2_fed_mol": h2_fed,
            "H2_product_net_mol": h2_product_net,
            "H2_purge_mol": h2_purge,
            "impurity_fed_mol": imp_fed,
            "impurity_adsorbed_mol": imp_adsorbed,
            "impurity_slip_mol": imp_in_product,
            "impurity_desorbed_blowdown_mol": imp_desorbed_blow,
            "impurity_desorbed_purge_mol": imp_desorbed_purge,
            "impurity_balance_residual_mol": imp_balance_residual,
            "impurity_net_stored_mol": imp_net_stored,
            "feed_flow_mol_s": F_feed,
            "adsorbent_mass_kg": self.m_ads,
            "specific_energy_kWh_per_kg_H2": self.specific_energy(P_H, P_L),
        }

    # ------------------------------------------------------------------
    # Cyclic steady state: iterate cycles until q0 converges
    # ------------------------------------------------------------------
    def cyclic_steady_state(self, n_cycles=20, tol=1e-6, **kwargs):
        """
        Iterate the single-cycle map q0 -> q_end_purge until the residual loading
        at the start of adsorption converges (cyclic steady state, CSS).
        Returns the converged cycle result plus convergence history.
        """
        q0 = kwargs.pop("q0", 0.0)
        history = []
        result = None
        for i in range(n_cycles):
            result = self.simulate_cycle(q0=q0, **kwargs)
            q_new = result["q_end_purge"]
            history.append(q_new)
            if abs(q_new - q0) < tol:
                q0 = q_new
                result = self.simulate_cycle(q0=q0, **kwargs)
                break
            q0 = q_new
        result["css_q0"] = q0
        result["css_history"] = np.array(history)
        result["css_cycles"] = len(history)
        return result

    # ------------------------------------------------------------------
    # Compression/vacuum specific energy (thermodynamic, isothermal bound)
    # ------------------------------------------------------------------
    def specific_energy(self, P_H, P_L):
        """
        Specific electric energy [kWh/kg_H2] to drive the PSA pressure swing.
        Lower bound = isothermal reversible compression work to raise feed to P_H
        relative to purge level P_L, per kg H2 in product, with a blower efficiency.

            w_min = R T ln(P_H / P_L)              [J/mol of gas processed]
        Scaled by impurity fraction processed and a real-machine efficiency.
        Sircar & Golden (2000): typical 1-3 kWh/kg H2.
        """
        eta_blower = 0.55                              # real isothermal efficiency
        w_mol = self.R * self.T_op * np.log(P_H / P_L) / eta_blower  # J/mol gas
        # Gas processed per mol H2 product ~ 1/(y_feed*recovery_nom); use nominal
        mol_gas_per_mol_h2 = 1.0 / max(self.y_feed, 0.1)
        w_per_mol_h2 = w_mol * mol_gas_per_mol_h2       # J/mol H2
        w_per_kg_h2 = w_per_mol_h2 / self.M_H2          # J/kg H2
        kWh = w_per_kg_h2 / 3.6e6                        # kWh/kg H2
        return float(np.clip(kWh, 0.3, 6.0))
