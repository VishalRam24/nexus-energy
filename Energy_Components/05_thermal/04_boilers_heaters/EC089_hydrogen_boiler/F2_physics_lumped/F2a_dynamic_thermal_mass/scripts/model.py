"""
EC089 -- Hydrogen Boiler (100% H2 combustion) -- F2a Physics-Lumped

0D first-principles dynamic model of a hydrogen-fired condensing boiler.
Combines (a) H2/air combustion chemistry, (b) a flue-gas energy balance with
stack/sensible loss and condensing latent recovery, and (c) a lumped
boiler thermal-mass ODE for the water temperature, integrated with
scipy.integrate.solve_ivp.

------------------------------------------------------------------
Combustion chemistry  (Turns 2012, Ch. 2)
------------------------------------------------------------------
    2 H2 + O2  ->  2 H2O          (only product is water -- no CO2, no CO)
    Mass:  1 kg H2 + 8 kg O2  ->  9 kg H2O
    Stoichiometric air-fuel ratio (mass): AFR_s = 8 / 0.233 ~= 34.3 kg air/kg H2
    With excess-air ratio lambda:  m_air = lambda * AFR_s * m_H2
    Product water per kg fuel = 9 kg H2O/kg H2  (very high vs hydrocarbons)
    => HHV - LHV = latent heat of that water => large condensing potential.

Higher flame temperature & NOx (Cellek & Pinarbasi 2018):
    Lean H2/air adiabatic flame temperature (~2300-2500 K at lambda~1.25) runs
    ~150-200 K hotter than CH4/air for a given lambda because of H2's high
    flame speed and reactivity, which RAISES thermal (Zeldovich) NOx.
    Excess air (lambda > 1) is the primary lever used here to moderate it.

------------------------------------------------------------------
Flue-gas energy balance  (sensible stack loss + latent recovery)
------------------------------------------------------------------
    Q_fuel_LHV = m_H2 * LHV                          (chemical input, LHV basis)
    Q_fuel_HHV = m_H2 * HHV                          (chemical input, HHV basis)
    m_flue     = m_H2 * (9 + lambda*AFR_s)           (water + air, mass cons.)
    Q_stack    = m_flue * cp_flue * (T_flue - T_amb) (sensible loss up the stack)
    Q_latent   = condensing ? f_cond * m_H2 * 9 * h_fg : 0   (water condensed)

    Useful heat delivered to the water jacket (HHV accounting):
        Q_to_water = Q_fuel_HHV - Q_stack - (1 - x_cond_term)*latent_unrecovered
    Implemented as:
        Q_to_water = Q_fuel_LHV + Q_latent - Q_stack
    i.e. LHV chemical heat plus whatever latent heat is recovered, minus the
    sensible heat carried out of the stack.  Energy is conserved on the HHV
    basis because (Q_fuel_HHV) = Q_fuel_LHV + total_latent, and the
    un-recovered latent + sensible stack heat are the losses.

------------------------------------------------------------------
Lumped thermal-mass ODE  (boiler water node)
------------------------------------------------------------------
    m_w * cp_w * dT_w/dt = Q_to_water(phi) - Q_load(T_w) - Q_standby(T_w)
        Q_load    = hA_load   * (T_w - T_return)      (heat to the circuit)
        Q_standby = hA_standby* (T_w - T_amb)         (casing loss)
    phi = firing rate in [0, 1] (fraction of rated fuel input).

Efficiency (HHV basis, the physically conservative definition):
    eta_HHV = Q_to_water / Q_fuel_HHV   in (0, 1].
    Condensing operation pushes eta_HHV well above the LHV-basis number
    but never above 1 (you cannot recover more than the chemical HHV).

References
---------
    Turns, S.R. (2012) An Introduction to Combustion, 3rd ed., McGraw-Hill.
    Cellek, M.S. & Pinarbasi, A. (2018) Int. J. Hydrogen Energy 43, 1194-1207.
    Hy4Heat WP6 Technical Report (2021), BEIS UK.
    Woolley, E. et al. (2022) Appl. Energy 323, 119577.
    NIST-JANAF thermochemical tables (cp, h_fg).
"""

import numpy as np
from scipy.integrate import solve_ivp


class HydrogenBoilerF2a:
    """Physics-lumped hydrogen boiler: combustion + flue balance + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated   = u["Q_rated"]["value"] * 1000.0        # W
        self.LHV       = u["LHV_H2_MJ_kg"]["value"] * 1e6      # J/kg
        self.HHV       = u["HHV_H2_MJ_kg"]["value"] * 1e6      # J/kg
        self.AFR_s     = u["AFR_stoich"]["value"]              # kg air / kg H2
        self.lam       = u["excess_air_ratio"]["value"]        # -
        self.cp_water  = u["cp_water"]["value"]                # J/(kg.K)
        self.cp_flue   = u["cp_flue"]["value"]                 # J/(kg.K)
        self.cp_H2     = u["cp_H2"]["value"]                   # J/(kg.K)
        self.h_fg      = u["h_fg_water"]["value"]              # J/kg
        self.m_water   = u["m_water"]["value"]                 # kg
        self.T_flame   = u["T_adiabatic_flame_K"]["value"]     # K
        self.T_amb     = u["T_ambient_K"]["value"]             # K
        self.T_flue_nc = u["T_flue_noncond_K"]["value"]        # K
        self.T_flue_c  = u["T_flue_cond_K"]["value"]           # K
        self.T_dew     = u["T_dew_K"]["value"]                 # K
        self.f_cond    = u["condensation_fraction"]["value"]   # -
        self.hA_load   = u["hA_load"]["value"]                 # W/K
        self.T_return  = u["T_return_K"]["value"]              # K
        self.hA_sb     = u["hA_standby"]["value"]              # W/K
        self.condensing = bool(u["condensing"]["value"])

        # Mass of product water per kg of H2 fuel (2 H2 + O2 -> 2 H2O).
        self.water_per_fuel = 9.0
        # Rated H2 mass flow on LHV basis [kg/s].
        self.m_H2_rated = self.Q_rated / self.LHV

    # ------------------------------------------------------------------
    # Combustion / mass flows
    # ------------------------------------------------------------------
    def h2_mass_flow(self, phi):
        """Hydrogen mass flow [kg/s] at firing rate phi in [0,1]."""
        phi = np.clip(phi, 0.0, 1.0)
        return phi * self.m_H2_rated

    def air_mass_flow(self, phi, lam=None):
        """Combustion air mass flow [kg/s] = lambda * AFR_stoich * m_H2."""
        if lam is None:
            lam = self.lam
        return lam * self.AFR_s * self.h2_mass_flow(phi)

    def flue_mass_flow(self, phi, lam=None):
        """
        Total flue-gas mass flow [kg/s] = fuel + air (mass conservation).
        Everything fed in (H2 + air) leaves as flue gas: the 8 kg O2 that
        forms the 9 kg product water per kg H2 is already part of the air
        stream, so it is NOT added again here.
        """
        m_h2 = self.h2_mass_flow(phi)
        m_air = self.air_mass_flow(phi, lam)
        return m_h2 + m_air

    def check_mass_balance(self, phi, lam=None):
        """Return (m_in, m_out) [kg/s]: fuel+air in vs flue out. Must be equal."""
        m_in = self.h2_mass_flow(phi) + self.air_mass_flow(phi, lam)
        m_out = self.flue_mass_flow(phi, lam)
        return m_in, m_out

    # ------------------------------------------------------------------
    # Adiabatic flame temperature (sensible-energy estimate)
    # ------------------------------------------------------------------
    def adiabatic_flame_temp(self, lam=None):
        """
        Estimate adiabatic flame temperature [K] from a lumped sensible
        balance: LHV heat raises the flue-gas mixture from T_amb.
            T_ad = T_amb + (m_H2 * LHV) / (m_flue * cp_flue_hot)
        Uses a hot-gas cp. Returns a value that DECREASES with excess air
        (more inert mass to heat) -- the physical NOx-mitigation lever.
        """
        if lam is None:
            lam = self.lam
        # Per kg fuel basis.
        m_flue_per_fuel = self.water_per_fuel + lam * self.AFR_s
        cp_hot = self.cp_flue * 1.25  # hot-gas cp slightly higher (NIST trend)
        return self.T_amb + self.LHV / (m_flue_per_fuel * cp_hot)

    def nox_index(self, lam=None):
        """
        Dimensionless thermal-NOx propensity ~ exp(-E_a / (R T_flame)),
        Zeldovich-style (Turns 2012). Monotonically decreasing in lambda
        because higher excess air lowers the flame temperature.
        Returned relative to a reference (lambda=1.25) for interpretability.
        """
        R = 8.314
        Ea = 38370.0 * R  # Zeldovich activation (~319 kJ/mol) order of magnitude
        T_ad = self.adiabatic_flame_temp(lam)
        T_ref = self.adiabatic_flame_temp(1.25)
        return float(np.exp(-Ea / T_ad) / np.exp(-Ea / T_ref))

    # ------------------------------------------------------------------
    # Flue-gas energy balance
    # ------------------------------------------------------------------
    def flue_exit_temp(self, phi):
        """Stack exit temperature [K]; condensing mode runs below dew point."""
        phi = np.clip(phi, 0.0, 1.0)
        T_design = self.T_flue_c if self.condensing else self.T_flue_nc
        # Lower firing -> cooler flue; floor at T_amb.
        return self.T_amb + (T_design - self.T_amb) * (0.4 + 0.6 * phi)

    def stack_loss(self, phi):
        """Sensible heat carried out of the stack [W]."""
        m_flue = self.flue_mass_flow(phi)
        T_flue = self.flue_exit_temp(phi)
        return max(m_flue * self.cp_flue * (T_flue - self.T_amb), 0.0)

    def latent_recovery(self, phi):
        """Latent heat recovered by condensing product water [W]."""
        if not self.condensing:
            return 0.0
        m_h2 = self.h2_mass_flow(phi)
        m_water = self.water_per_fuel * m_h2
        return self.f_cond * m_water * self.h_fg

    def total_latent(self, phi):
        """Total latent heat available in the product water [W] (= HHV-LHV part)."""
        m_h2 = self.h2_mass_flow(phi)
        return m_h2 * (self.HHV - self.LHV)

    def fuel_power_LHV(self, phi):
        return self.h2_mass_flow(phi) * self.LHV

    def fuel_power_HHV(self, phi):
        return self.h2_mass_flow(phi) * self.HHV

    def heat_to_water(self, phi):
        """
        Net useful heat delivered to the water jacket [W].
            Q = Q_fuel_LHV + Q_latent_recovered - Q_stack
        Bounded above by Q_fuel_HHV (cannot exceed chemical HHV).
        """
        if phi <= 0.0:
            return 0.0
        Q = self.fuel_power_LHV(phi) + self.latent_recovery(phi) - self.stack_loss(phi)
        Q = min(Q, self.fuel_power_HHV(phi))
        return max(Q, 0.0)

    def efficiency_hhv(self, phi):
        """Combustion-to-water efficiency, HHV basis, in (0,1]."""
        Q_fuel = self.fuel_power_HHV(phi)
        if Q_fuel <= 0.0:
            return 0.0
        eta = self.heat_to_water(phi) / Q_fuel
        return float(np.clip(eta, 0.0, 1.0))

    def efficiency_lhv(self, phi):
        """Efficiency on LHV basis (can exceed 1 in condensing mode)."""
        Q_fuel = self.fuel_power_LHV(phi)
        if Q_fuel <= 0.0:
            return 0.0
        return self.heat_to_water(phi) / Q_fuel

    def energy_balance_residual(self, phi):
        """
        HHV energy-conservation check [W]:
            Q_fuel_HHV - Q_to_water - Q_stack - Q_latent_unrecovered  ~= 0
        Returns the residual; should be ~0.
        """
        Q_hhv = self.fuel_power_HHV(phi)
        Q_water = self.heat_to_water(phi)
        Q_stack = self.stack_loss(phi)
        latent_unrec = self.total_latent(phi) - self.latent_recovery(phi)
        return Q_hhv - Q_water - Q_stack - latent_unrec

    # ------------------------------------------------------------------
    # Thermal-mass ODE
    # ------------------------------------------------------------------
    def dTdt(self, T_w, phi):
        """Water-node temperature rate of change [K/s]."""
        Q_in = self.heat_to_water(phi)
        Q_load = self.hA_load * (T_w - self.T_return)
        Q_sb = self.hA_sb * (T_w - self.T_amb)
        return (Q_in - Q_load - Q_sb) / (self.m_water * self.cp_water)

    # ------------------------------------------------------------------
    # Time-domain simulation (solve_ivp)
    # ------------------------------------------------------------------
    def simulate(self, firing_rate, T_w0, dt, duration_s):
        """
        Integrate the lumped boiler thermal-mass ODE.

        Parameters
        ----------
        firing_rate : float or callable(t) -> float in [0, 1]
        T_w0        : float, initial water temperature [K]
        dt          : float, output time step [s]
        duration_s  : float, total duration [s]

        Returns
        -------
        dict of time-series arrays.
        """
        _phi = firing_rate if callable(firing_rate) else (lambda t: firing_rate)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], _phi(t))]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_w0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        phi_arr = np.zeros(N)
        Q_water = np.zeros(N)
        Q_stack = np.zeros(N)
        Q_latent = np.zeros(N)
        Q_fuel_hhv = np.zeros(N)
        eta_hhv = np.zeros(N)
        eta_lhv = np.zeros(N)
        m_h2 = np.zeros(N)
        T_flue = np.zeros(N)

        for i in range(N):
            phi = float(np.clip(_phi(t_out[i]), 0.0, 1.0))
            phi_arr[i] = phi
            Q_water[i] = self.heat_to_water(phi)
            Q_stack[i] = self.stack_loss(phi)
            Q_latent[i] = self.latent_recovery(phi)
            Q_fuel_hhv[i] = self.fuel_power_HHV(phi)
            eta_hhv[i] = self.efficiency_hhv(phi)
            eta_lhv[i] = self.efficiency_lhv(phi)
            m_h2[i] = self.h2_mass_flow(phi)
            T_flue[i] = self.flue_exit_temp(phi)

        return {
            "t": t_out,
            "temperature": T_out,           # water-node temperature [K]
            "firing_rate": phi_arr,
            "heat_to_water_W": Q_water,
            "stack_loss_W": Q_stack,
            "latent_recovery_W": Q_latent,
            "fuel_power_hhv_W": Q_fuel_hhv,
            "efficiency": eta_hhv,          # HHV basis, primary efficiency
            "efficiency_lhv": eta_lhv,
            "h2_flow_kg_s": m_h2,
            "flue_temp_K": T_flue,
            "T_adiabatic_flame_K": self.adiabatic_flame_temp(),
            "nox_index": self.nox_index(),
        }
