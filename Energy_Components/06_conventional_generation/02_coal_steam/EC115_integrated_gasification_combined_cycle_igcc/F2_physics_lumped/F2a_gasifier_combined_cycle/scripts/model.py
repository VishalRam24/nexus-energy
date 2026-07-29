"""
EC115 -- Integrated Gasification Combined Cycle (IGCC) -- F2a Physics-Lumped

Physics-lumped (0D) first-principles model of a coal IGCC plant:

    Coal  --gasifier-->  Syngas (CO+H2)  --gas turbine (Brayton, topping)-->
        --HRSG--> steam (Rankine, bottoming)  --> net electricity

The plant is decomposed into three energy stages, each a lumped energy balance,
plus ONE lumped transient ODE for the gas-turbine combustor/casing metal
temperature integrated with scipy.integrate.solve_ivp.

------------------------------------------------------------------------------
STAGE 1 -- Gasifier (cold-gas efficiency)
------------------------------------------------------------------------------
    Q_coal   = m_coal * LHV_coal                          [MW_th]  coal chemical power in
    Q_syngas = CGE * Q_coal                               [MW_th]  chemical power in clean syngas
    CGE (cold-gas efficiency) = chemical energy in cold syngas / coal LHV input.
    Higman & van der Burgt (2008): entrained-flow CGE 0.78-0.83, ALWAYS < 1
    (the balance leaves as sensible heat, slag, unconverted carbon, char).

------------------------------------------------------------------------------
STAGE 2 -- Combined cycle (Brayton topping + HRSG + Rankine bottoming)
------------------------------------------------------------------------------
Brayton (gas turbine) burns the syngas:
    W_GT      = eta_B * Q_syngas                          [MW_e]
    Q_exhaust = (1 - eta_B) * Q_syngas                    [MW_th] rejected to HRSG
HRSG passes a fraction eps_HRSG of the exhaust into the steam cycle:
    Q_HRSG    = eps_HRSG * Q_exhaust
Rankine (steam) bottoming:
    W_ST      = eta_R * Q_HRSG                            [MW_e]

Combined-cycle efficiency (Cengel & Boles 2015, eq. for topping/bottoming):
    eta_CC = eta_B + eta_R * eps_HRSG * (1 - eta_B)
           = 1 - (1 - eta_B)(1 - eta_R*eps_HRSG)          (with eps_HRSG=1)
This is ALWAYS greater than either single cycle alone (proven in tests).

------------------------------------------------------------------------------
STAGE 3 -- Net plant
------------------------------------------------------------------------------
    W_gross = W_GT + W_ST
    W_net   = W_gross * (1 - aux_fraction)                ASU + BOP parasitics
    eta_net = W_net / Q_coal      (~0.38-0.45, LHV basis)
            = CGE * eta_CC * (1 - aux_fraction)

Carnot bound (Cengel & Boles 2015):
    eta_Carnot = 1 - T_ambient / T_firing
Every stage efficiency and the net efficiency must satisfy eta < eta_Carnot.

------------------------------------------------------------------------------
LUMPED TRANSIENT ODE -- gas-turbine combustor metal temperature
------------------------------------------------------------------------------
A single first-order energy balance on the combustor/first-stage casing metal:

    m_metal * cp_metal * dT_metal/dt = hA * (T_gas - T_metal)

    T_gas = T_compressor_exit + (T_firing_design - T_compressor_exit) * load
    load  = Q_syngas / Q_syngas_design     (fuel-driven firing fraction)

The metal lags the gas temperature with time-constant tau = m*cp/hA. This is the
classic lumped-capacitance (Biot << 1) thermal model (Cengel & Boles 2015,
transient lumped systems) and governs GT thermal-stress / start-up dynamics.

------------------------------------------------------------------------------
References
------------------------------------------------------------------------------
    Higman, C. & van der Burgt, M. (2008). Gasification, 2nd ed., Elsevier.
        (cold-gas efficiency, syngas LHV ~10-12 MJ/Nm3, coal LHV/cp)
    Cengel, Y. & Boles, M. (2015). Thermodynamics: An Engineering Approach,
        8th ed., McGraw-Hill. (Brayton, Rankine, combined-cycle topping/
        bottoming efficiency, Carnot bound, lumped-capacitance transients)
    Cormos, C.-C. (2012). Int. J. Hydrogen Energy 37(4), 3083-3095.
    IEA GHG R&D Programme (2003). Improvements in gasification combined cycle.
    Booras, G. & Holt, N. (2004). EPRI Gasification Technologies Conference.
"""

import numpy as np
from scipy.integrate import solve_ivp


class IGCC_F2a:
    """IGCC -- physics-lumped gasifier + combined cycle with thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated        = u["rated_power_mw"]["value"]            # MW_e

        # Fuel / gasifier
        self.LHV_coal       = u["LHV_coal"]["value"]                 # MJ/kg
        self.cp_coal        = u["cp_coal"]["value"]                  # kJ/(kg.K)
        self.syngas_lhv     = u["syngas_lhv"]["value"]              # MJ/Nm3
        self.cp_syngas      = u["cp_syngas"]["value"]              # kJ/(kg.K)
        self.syngas_M       = u["syngas_mw_kg_per_kmol"]["value"]   # kg/kmol
        self.cge            = u["cold_gas_efficiency"]["value"]     # -

        # Combined cycle
        self.eta_B          = u["eta_brayton_design"]["value"]      # -
        self.eta_R          = u["eta_rankine_design"]["value"]      # -
        self.eps_hrsg       = u["hrsg_effectiveness"]["value"]      # -

        # Temperatures
        self.T_fire         = u["T_firing_design_K"]["value"]       # K
        self.T_comp         = u["T_compressor_exit_K"]["value"]     # K
        self.T_amb          = u["T_ambient_K"]["value"]             # K

        # Combustor metal thermal mass
        self.m_metal        = u["m_metal"]["value"]                # kg
        self.cp_metal       = u["cp_metal"]["value"]               # J/(kg.K)
        self.hA             = u["hA_combustor"]["value"]           # W/K

        # Parasitics / emissions
        self.aux_fraction   = u["aux_fraction"]["value"]           # -
        self.CO2_per_coal   = u["CO2_per_kg_coal"]["value"]        # kg/kg

        # Design-point fuel power (firing fraction reference): the coal rate that
        # delivers rated net power at design net efficiency.
        eta_net_design = self.net_efficiency()
        self.Q_coal_design = self.P_rated / max(eta_net_design, 1e-6)   # MW_th coal
        self.Q_syngas_design = self.cge * self.Q_coal_design           # MW_th syngas

    # ==================================================================
    # Bounds / thermodynamic references
    # ==================================================================
    def carnot_efficiency(self, T_hot=None, T_cold=None):
        """Carnot upper bound 1 - T_cold/T_hot (Cengel & Boles 2015)."""
        T_hot = self.T_fire if T_hot is None else T_hot
        T_cold = self.T_amb if T_cold is None else T_cold
        return 1.0 - T_cold / T_hot

    # ==================================================================
    # STAGE 1 -- Gasifier
    # ==================================================================
    def coal_power_mw(self, m_coal_kgs):
        """Coal chemical power in [MW_th] = m_coal[kg/s] * LHV[MJ/kg]."""
        return np.asarray(m_coal_kgs, float) * self.LHV_coal

    def syngas_power_mw(self, m_coal_kgs):
        """Clean-syngas chemical power [MW_th] = CGE * coal power."""
        return self.cge * self.coal_power_mw(m_coal_kgs)

    def syngas_rate_nm3s(self, m_coal_kgs):
        """Syngas volumetric flow [Nm3/s] = syngas_MW / syngas_LHV(MJ/Nm3)."""
        return self.syngas_power_mw(m_coal_kgs) / self.syngas_lhv

    # ==================================================================
    # STAGE 2 -- Combined cycle
    # ==================================================================
    def combined_cycle_efficiency(self):
        """
        Topping (Brayton) + bottoming (Rankine via HRSG) combined-cycle
        efficiency (Cengel & Boles 2015):
            eta_CC = eta_B + eta_R * eps_HRSG * (1 - eta_B)
        Strictly greater than eta_B alone whenever eta_R*eps_HRSG > 0.
        """
        return self.eta_B + self.eta_R * self.eps_hrsg * (1.0 - self.eta_B)

    def brayton_work_mw(self, m_coal_kgs):
        """Gas-turbine (topping) electrical work [MW_e]."""
        return self.eta_B * self.syngas_power_mw(m_coal_kgs)

    def rankine_work_mw(self, m_coal_kgs):
        """Steam (bottoming) electrical work [MW_e] off recovered HRSG heat."""
        Q_syn = self.syngas_power_mw(m_coal_kgs)
        Q_exhaust = (1.0 - self.eta_B) * Q_syn
        Q_hrsg = self.eps_hrsg * Q_exhaust
        return self.eta_R * Q_hrsg

    def gross_power_mw(self, m_coal_kgs):
        """Gross combined-cycle electrical power [MW_e]."""
        return self.brayton_work_mw(m_coal_kgs) + self.rankine_work_mw(m_coal_kgs)

    # ==================================================================
    # STAGE 3 -- Net plant
    # ==================================================================
    def net_power_mw(self, m_coal_kgs):
        """Net electrical power after auxiliary (ASU/BOP) loads [MW_e]."""
        return self.gross_power_mw(m_coal_kgs) * (1.0 - self.aux_fraction)

    def net_efficiency(self, m_coal_kgs=None):
        """
        Net plant LHV efficiency (dimensionless):
            eta_net = CGE * eta_CC * (1 - aux_fraction)
        Load-independent in this lumped form (constant component efficiencies);
        accepts m_coal_kgs for API symmetry. Returns ~0.38-0.45.
        """
        return self.cge * self.combined_cycle_efficiency() * (1.0 - self.aux_fraction)

    def co2_rate_kgs(self, m_coal_kgs):
        """CO2 emission rate [kg/s] (no CCS)."""
        return np.asarray(m_coal_kgs, float) * self.CO2_per_coal

    def co2_intensity_g_per_kwh(self, m_coal_kgs):
        """CO2 intensity [g/kWh_e] = CO2[g/s] / P_net[kW] * 3600."""
        P_net_kw = self.net_power_mw(m_coal_kgs) * 1e3
        co2_gs = self.co2_rate_kgs(m_coal_kgs) * 1e3
        P_safe = np.where(np.asarray(P_net_kw) > 1e-9, P_net_kw, 1e-9)
        return co2_gs / P_safe * 3600.0

    # ==================================================================
    # LUMPED TRANSIENT ODE -- combustor metal temperature
    # ==================================================================
    def _gas_temperature(self, m_coal_kgs):
        """
        Combustor gas temperature as function of fuel firing fraction:
            T_gas = T_comp + (T_fire - T_comp) * load,  load = Q_syn/Q_syn_design.
        """
        load = self.syngas_power_mw(m_coal_kgs) / max(self.Q_syngas_design, 1e-9)
        load = np.clip(load, 0.0, 1.5)
        return self.T_comp + (self.T_fire - self.T_comp) * load

    def time_constant(self):
        """Lumped thermal time constant tau = m*cp/hA [s]."""
        return self.m_metal * self.cp_metal / self.hA

    def _dTdt(self, t, T, coal_fn):
        m_coal = coal_fn(t)
        T_gas = self._gas_temperature(m_coal)
        return [self.hA * (T_gas - T[0]) / (self.m_metal * self.cp_metal)]

    def simulate(self, m_coal_kgs, T_metal_0=None, dt=2.0, duration_s=600.0):
        """
        Integrate the lumped combustor-metal temperature ODE with solve_ivp.

        m_coal_kgs : float OR callable(t)->kg/s  (coal feed schedule)
        T_metal_0  : initial metal temperature [K] (default = T_compressor_exit)
        Returns dict of time-series arrays plus steady plant performance.
        """
        if callable(m_coal_kgs):
            coal_fn = m_coal_kgs
        else:
            val = float(m_coal_kgs)
            coal_fn = lambda t: val

        if T_metal_0 is None:
            T_metal_0 = self.T_comp

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._dTdt, (0.0, duration_s), [T_metal_0],
            t_eval=t_eval, args=(coal_fn,), method="RK45",
            rtol=1e-7, atol=1e-6, max_step=dt,
        )

        t = sol.t
        T_metal = sol.y[0]
        m_coal = np.array([coal_fn(ti) for ti in t])
        T_gas = self._gas_temperature(m_coal)

        return {
            "t": t,
            "T_metal": T_metal,
            "T_gas": T_gas,
            "m_coal_kgs": m_coal,
            "coal_power_mw": self.coal_power_mw(m_coal),
            "syngas_power_mw": self.syngas_power_mw(m_coal),
            "syngas_rate_nm3s": self.syngas_rate_nm3s(m_coal),
            "brayton_mw": self.brayton_work_mw(m_coal),
            "rankine_mw": self.rankine_work_mw(m_coal),
            "gross_power_mw": self.gross_power_mw(m_coal),
            "net_power_mw": self.net_power_mw(m_coal),
            "net_efficiency": np.full_like(t, self.net_efficiency()),
            "combined_cycle_efficiency": self.combined_cycle_efficiency(),
            "carnot_efficiency": self.carnot_efficiency(),
            "co2_intensity_g_per_kwh": self.co2_intensity_g_per_kwh(m_coal),
            "tau_s": self.time_constant(),
        }
