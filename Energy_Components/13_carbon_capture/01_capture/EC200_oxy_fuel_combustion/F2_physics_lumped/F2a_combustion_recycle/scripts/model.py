"""
EC200 -- Oxy-Fuel Combustion Capture -- F2a Combustion + Flue-Gas Recycle Model

Physics-lumped (0D) first-principles model of an oxy-fuel furnace.

Fuel is burned in nearly-pure O2 from an Air Separation Unit (ASU). A fraction
of the cooled, dried flue gas (mostly CO2) is recycled into the burner to
moderate the flame/furnace temperature (the wet/dry recycle ratio is the main
control). The product gas is CO2 + H2O; after water knockout (condensation) the
dry stream is high-purity CO2 (>0.9), ready for compression.

Combustion stoichiometry (per kg fuel, daf ultimate analysis):
    C + O2 -> CO2
    2 H + 1/2 O2 -> H2O      (H is atomic-H mass fraction)
    S + O2 -> SO2
The oxidant is O2 (from ASU) + recycled flue gas (RFG). Recycle adds CO2/H2O
thermal mass that absorbs heat, lowering the adiabatic flame temperature.

Lumped furnace energy balance (ODE, transient to steady state):
    m_g * cp_g * dT/dt = Q_comb - Q_wall - H_sensible_out
where
    Q_comb     = LHV * mdot_fuel               (chemical heat release, W)
    Q_wall     = (h A)_wall * (T - T_wall)     (heat to steam/walls)
    H_out      = mdot_gas_out * cp_g * (T - T_in)   (enthalpy leaving)
The recycle ratio sets mdot of recycled gas added at T_in and hence the
gas thermal mass and the steady flame temperature.

CO2 purity after water knockout:
    x_CO2_dry = n_CO2 / (n_CO2 + n_O2_excess + n_SO2 + n_inert)
H2O is removed by condensation -> dry-basis purity.

References:
    Buhre, B.J.P. et al. (2005). Oxy-fuel combustion technology for coal-fired
        power generation. Prog. Energy Combust. Sci. 31(4), 283-307.
    Toftegaard, M.B. et al. (2010). Oxy-fuel combustion of solid fuels.
        Prog. Energy Combust. Sci. 36(5), 581-625.
    Turns, S.R. (2012). An Introduction to Combustion, 3rd ed., McGraw-Hill
        (adiabatic flame temperature, sensible enthalpy).
"""

import numpy as np
from scipy.integrate import solve_ivp


class OxyFuelF2a:
    """Oxy-fuel furnace -- lumped combustion + recycle + energy-balance ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Fuel ultimate analysis (mass fractions, daf)
        self.w_C = u["fuel_C"]["value"]
        self.w_H = u["fuel_H"]["value"]
        self.w_S = u["fuel_S"]["value"]
        self.w_O = u["fuel_O"]["value"]
        self.w_N = u["fuel_N"]["value"]
        self.LHV = u["fuel_LHV"]["value"] * 1e6        # J/kg
        # ASU / oxidant
        self.O2_purity = u["o2_purity"]["value"]       # mol O2 / mol oxidant (rest Ar+N2)
        self.excess_O2 = u["excess_o2_ratio"]["value"] # lambda - 1
        # Furnace thermal
        self.cp_gas = u["cp_gas"]["value"]             # J/(kg.K) mean flue-gas cp
        # Effective high-T cp near flame (CO2/H2O cp rises strongly with T,
        # plus endothermic dissociation) -- used for adiabatic flame temp.
        self.cp_flame = u.get("cp_gas_flame", {"value": 1700.0})["value"]
        self.hA_wall = u["hA_wall_per_kgfuel"]["value"]  # W/K per (kg/s fuel)
        self.T_wall = u["T_wall"]["value"]             # K (steam-side wall)
        self.T_in = u["T_inlet"]["value"]              # K (preheated oxidant + recycle)
        self.m_furnace = u["m_furnace_gas_per_kgfuel"]["value"]  # kg gas holdup per kg/s fuel
        # Molar masses (g/mol)
        self.MW_C = 12.011
        self.MW_H = 1.008
        self.MW_S = 32.06
        self.MW_O2 = 31.998
        self.MW_N = 14.007
        self.MW_CO2 = 44.01
        self.MW_H2O = 18.015
        self.MW_SO2 = 64.06
        self.MW_Ar = 39.948

    # ------------------------------------------------------------------
    # Stoichiometry (per kg fuel) -> molar flows of products & oxidant
    # ------------------------------------------------------------------
    def stoichiometry(self):
        """
        Return molar amounts (mol per kg fuel) of products and required O2.
        C + O2 -> CO2 ; 4H + O2 -> 2 H2O ; S + O2 -> SO2.
        """
        n_C = self.w_C / self.MW_C * 1000.0      # mol/kg fuel  (w in kg/kg, MW g/mol)
        n_H = self.w_H / self.MW_H * 1000.0      # mol H atoms
        n_S = self.w_S / self.MW_S * 1000.0
        n_O_fuel = self.w_O / self.MW_O2 * 1000.0 * 2.0  # mol O atoms in fuel
        n_N = self.w_N / self.MW_N * 1000.0      # mol N atoms (-> N2)

        n_CO2 = n_C
        n_H2O = 0.5 * n_H
        n_SO2 = n_S
        # O2 needed (mol): C->1, H atoms->0.25, S->1, minus fuel O (atoms/2)
        n_O2_stoich = n_C + 0.25 * n_H + n_S - 0.5 * n_O_fuel
        n_N2_fuel = 0.5 * n_N
        return {
            "n_CO2": n_CO2, "n_H2O": n_H2O, "n_SO2": n_SO2,
            "n_O2_stoich": n_O2_stoich, "n_N2_fuel": n_N2_fuel,
        }

    def o2_supplied(self):
        """O2 supplied per kg fuel (mol/kg), including excess."""
        s = self.stoichiometry()
        return s["n_O2_stoich"] * (1.0 + self.excess_O2)

    def oxidant_inerts(self):
        """
        Inert moles (Ar + small N2) per kg fuel carried in with the ASU O2
        stream. ASU O2 purity < 1 -> the balance is mostly Ar (+ trace N2).
        """
        n_O2 = self.o2_supplied()
        # total oxidant moles such that O2/total = purity
        n_oxidant = n_O2 / self.O2_purity
        return n_oxidant - n_O2     # mol inert (Ar/N2) per kg fuel

    # ------------------------------------------------------------------
    # Flue-gas composition (wet) per kg fuel, and dry CO2 purity
    # ------------------------------------------------------------------
    def flue_gas_moles(self):
        """Wet flue-gas molar composition (mol/kg fuel)."""
        s = self.stoichiometry()
        n_O2_excess = self.o2_supplied() - s["n_O2_stoich"]
        n_inert = self.oxidant_inerts() + s["n_N2_fuel"]
        return {
            "CO2": s["n_CO2"],
            "H2O": s["n_H2O"],
            "SO2": s["n_SO2"],
            "O2": n_O2_excess,
            "inert": n_inert,
        }

    def co2_purity_dry(self):
        """
        Dry-basis CO2 mole fraction after water knockout (condensation).
        x_CO2_dry = n_CO2 / (n_CO2 + n_SO2 + n_O2_excess + n_inert).
        Oxy-fuel gives >0.9 typically (Buhre 2005).
        """
        fg = self.flue_gas_moles()
        n_dry = fg["CO2"] + fg["SO2"] + fg["O2"] + fg["inert"]
        return fg["CO2"] / n_dry

    def co2_purity_wet(self):
        """Wet-basis CO2 mole fraction (before water knockout)."""
        fg = self.flue_gas_moles()
        n_wet = sum(fg.values())
        return fg["CO2"] / n_wet

    # ------------------------------------------------------------------
    # Gas mass flow per kg fuel (wet product) and with recycle
    # ------------------------------------------------------------------
    def product_gas_mass(self):
        """Wet product-gas mass per kg fuel (kg/kg)."""
        fg = self.flue_gas_moles()
        mass = (fg["CO2"] * self.MW_CO2 + fg["H2O"] * self.MW_H2O
                + fg["SO2"] * self.MW_SO2 + fg["O2"] * self.MW_O2
                + fg["inert"] * self.MW_Ar) / 1000.0   # g->kg
        return mass

    # ------------------------------------------------------------------
    # Lumped furnace energy-balance ODE
    # ------------------------------------------------------------------
    def _gas_flows(self, mdot_fuel, recycle_ratio):
        """
        Mass flows (kg/s) of through-gas and recycled gas for the furnace
        energy balance. recycle_ratio R = m_recycle / m_product (0..~0.8).
        """
        m_prod = self.product_gas_mass() * mdot_fuel    # kg/s wet product
        m_recycle = recycle_ratio * m_prod
        # Total gas mass passing the flame = fresh O2-stream products + recycle
        m_through = m_prod + m_recycle
        return m_prod, m_recycle, m_through

    def dTdt(self, T, mdot_fuel, recycle_ratio):
        """Furnace gas temperature rate of change [K/s]."""
        Q_comb = self.LHV * mdot_fuel                    # W
        m_prod, m_recycle, m_through = self._gas_flows(mdot_fuel, recycle_ratio)
        # Heat to steam/walls (radiative+convective, lumped linear)
        Q_wall = self.hA_wall * mdot_fuel * (T - self.T_wall)
        # Sensible enthalpy carried out by product gas leaving at T from T_in
        H_out = m_prod * self.cp_gas * (T - self.T_in)
        # Recycle returns at T_in: it has to be heated from T_in -> T (heat sink)
        H_recycle = m_recycle * self.cp_gas * (T - self.T_in)
        m_hold = self.m_furnace * mdot_fuel              # kg gas holdup
        denom = max(m_hold * self.cp_gas, 1e-6)
        return (Q_comb - Q_wall - H_out - H_recycle) / denom

    def adiabatic_flame_temp(self, mdot_fuel, recycle_ratio):
        """
        Approximate adiabatic flame temperature (no wall loss): all heat goes
        into raising the through-gas from T_in. T_ad = T_in + Q_comb / (m_th*cp).
        Recycle increases m_through -> lowers T_ad (Turns 2012).
        """
        Q_comb = self.LHV * mdot_fuel
        _, _, m_through = self._gas_flows(mdot_fuel, recycle_ratio)
        # Use the high-T effective cp (CO2/H2O cp grows with T + dissociation),
        # which keeps the adiabatic flame temperature physically bounded.
        return self.T_in + Q_comb / (m_through * self.cp_flame)

    def simulate(self, mdot_fuel, recycle_ratio, T0=None, dt=0.5, duration_s=120.0):
        """
        Transient furnace energy balance to steady state.

        Parameters
        ----------
        mdot_fuel : float        fuel mass flow (kg/s)
        recycle_ratio : float    m_recycle / m_product (0..0.8)
        T0 : float               initial gas temperature (K), default T_wall+200
        dt : float               output time step (s)
        duration_s : float       simulation length (s)

        Returns
        -------
        dict: t, temperature, T_steady, T_adiabatic, co2_purity_dry,
              co2_purity_wet, o2_demand_kgs, co2_produced_kgs,
              product_gas_kgs, recycle_kgs
        """
        if T0 is None:
            T0 = self.T_wall + 200.0
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], mdot_fuel, recycle_ratio)]

        sol = solve_ivp(rhs, (0.0, duration_s), [T0], t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-6, max_step=dt)

        T_out = sol.y[0]
        # Derived quantities
        o2_kgs = self.o2_supplied() * self.MW_O2 / 1000.0 * mdot_fuel
        co2_kgs = self.stoichiometry()["n_CO2"] * self.MW_CO2 / 1000.0 * mdot_fuel
        m_prod, m_recycle, _ = self._gas_flows(mdot_fuel, recycle_ratio)

        return {
            "t": sol.t,
            "temperature": T_out,
            "T_steady": float(T_out[-1]),
            "T_adiabatic": self.adiabatic_flame_temp(mdot_fuel, recycle_ratio),
            "co2_purity_dry": self.co2_purity_dry(),
            "co2_purity_wet": self.co2_purity_wet(),
            "o2_demand_kgs": o2_kgs,
            "co2_produced_kgs": co2_kgs,
            "product_gas_kgs": m_prod,
            "recycle_kgs": m_recycle,
        }
