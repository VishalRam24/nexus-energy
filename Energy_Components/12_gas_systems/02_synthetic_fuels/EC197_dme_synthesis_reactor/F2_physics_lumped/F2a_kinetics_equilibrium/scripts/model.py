"""
EC197 -- DME Synthesis Reactor -- F2a Kinetics + Equilibrium (Physics-Lumped)

Direct (single-step) DME synthesis from syngas over a bifunctional catalyst
(CuO-ZnO-Al2O3 methanol-synthesis component + gamma-Al2O3 / HZSM-5 acidic
dehydration component). Three coupled reactions are tracked:

    (R1) Methanol synthesis:      CO  + 2 H2  <-> CH3OH            dH = -90.6 kJ/mol
    (R2) Water-gas shift (WGS):   CO2 +   H2  <-> CO + H2O         dH = +41.2 kJ/mol
    (R3) Methanol dehydration:  2 CH3OH      <-> CH3OCH3 + H2O     dH = -23.4 kJ/mol

Overall direct DME (R1*2 + R3, with H2O consumed by reverse WGS):
    2 CO + 4 H2 -> CH3OCH3 + H2O          dH ~ -204.8 kJ/mol DME (strongly exothermic)

This F2 model is the physics-lumped upgrade of the F1a equilibrium-fit model:
it integrates LHHW reaction-rate ODEs along a plug-flow / batch-equivalent
reactor coordinate together with a lumped (0-D) reactor energy balance, using
scipy.integrate.solve_ivp. State integrated vs. residence time tau (or W/F):

    dn_i/dtau = sum_j  nu_ij * r_j * rho_bulk          (mol balance, per reaction)
    (m*cp) dT/dtau-equivalent handled as an adiabatic/cooled lumped energy
    balance:  Cp_mix * dT/dtau = -sum_j (dH_j) * r_j * rho_b - U a (T - T_cool)

Kinetics
--------
Methanol synthesis (R1) + WGS (R2): Graaf et al. (1988) LHHW rate expressions
    (the Graaf/Vanden Bussche-Froment family), Arrhenius rate + thermodynamic
    equilibrium driving force (1 - Q/Keq).
Methanol dehydration (R3): Bercic & Levec (1992, 1993) LHHW rate over gamma-Al2O3
    r3 = k3 * Ks^2 * (C_M^2 - C_W C_D / Keq3) / (1 + 2 sqrt(Ks C_M) + Ks C_W)^4

Equilibrium constants (van 't Hoff / fitted):
    Keq1 (MeOH synth):  Graaf et al. (1986)
    Keq2 (WGS):         Graaf et al. (1986) / Moe (1962)
    Keq3 (dehydration): Bercic & Levec (1992)

References
----------
  Ng, K.L., Chadwick, D., Toseland, B.A. (1999). Kinetics and modelling of
      dimethyl ether synthesis from synthesis gas. Chem. Eng. Sci. 54, 3587-3592.
  Bercic, G., Levec, J. (1992). Intrinsic and global reaction rate of methanol
      dehydration over gamma-Al2O3 pellets. Ind. Eng. Chem. Res. 31, 1035-1040.
  Bercic, G., Levec, J. (1993). Catalytic dehydration of methanol to dimethyl
      ether. Kinetic investigation and reactor simulation. IEC Res. 32, 2478-2484.
  Graaf, G.H., Stamhuis, E.J., Beenackers, A.A.C.M. (1988). Kinetics of low-
      pressure methanol synthesis. Chem. Eng. Sci. 43, 3185-3195.
  Graaf, G.H., Sijtsema, P.J.J.M., Stamhuis, E.J., Joosten, G.E.H. (1986).
      Chemical equilibria in methanol synthesis. Chem. Eng. Sci. 41, 2883-2890.
"""

import numpy as np
from scipy.integrate import solve_ivp

R = 8.314          # J/(mol.K)

# Species index map for the molar state vector
#   0: CO   1: H2   2: CO2   3: H2O   4: CH3OH (MeOH)   5: CH3OCH3 (DME)
SPECIES = ["CO", "H2", "CO2", "H2O", "CH3OH", "DME"]
IDX = {s: i for i, s in enumerate(SPECIES)}

# Standard reaction enthalpies [J/mol] (per mol of key product), ~constant over band
DH_R1 = -90.6e3    # CO + 2H2 -> CH3OH
DH_R2 = +41.2e3    # CO2 + H2 -> CO + H2O   (reverse WGS as written; +41.2 endothermic)
DH_R3 = -23.4e3    # 2 CH3OH -> DME + H2O

# Mean molar heat capacities of the gas mixture components [J/(mol.K)]
# (approx values around 500 K, Perry's / NIST). Used for lumped energy balance.
CP = {
    "CO": 29.6, "H2": 29.0, "CO2": 41.3, "H2O": 36.0,
    "CH3OH": 52.0, "DME": 75.0,
}


class DMEReactorF2a:
    """
    Direct DME synthesis reactor -- physics-lumped kinetics + equilibrium model.

    Integrates the three coupled reaction rates along a residence-time
    coordinate together with a lumped reactor energy balance.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        # Methanol synthesis (R1) Arrhenius
        self.k1_0   = u["k1_0"]["value"]        # pre-exp, rate units
        self.Ea1    = u["Ea1"]["value"]         # J/mol
        # WGS (R2) Arrhenius
        self.k2_0   = u["k2_0"]["value"]
        self.Ea2    = u["Ea2"]["value"]
        # Methanol dehydration (R3) Bercic-Levec
        self.k3_0   = u["k3_0"]["value"]
        self.Ea3    = u["Ea3"]["value"]
        self.Ks_0   = u["Ks_0"]["value"]        # methanol adsorption pre-exp
        self.dHs    = u["dH_ads"]["value"]      # adsorption enthalpy (J/mol, negative)
        # Adsorption lumped term for R1/R2 LHHW denominator
        self.K_ads  = u["K_ads"]["value"]
        # Reactor
        self.rho_b  = u["rho_bulk"]["value"]    # kg cat / m3
        self.Ua     = u["Ua"]["value"]          # W/(m3.K) volumetric cooling
        self.T_cool = u["T_coolant"]["value"]   # K
        self.P_ref  = u["P_ref"]["value"]       # bar
        # Feed defaults
        self.n_CO_in = u["n_CO_in"]["value"]
        self.H2_CO   = u["H2_CO_ratio"]["value"]
        self.CO2_frac = u.get("CO2_frac", {"value": 0.03})["value"]

    # ------------------------------------------------------------------
    # Equilibrium constants (van 't Hoff fits, dimensionless / bar-based)
    # ------------------------------------------------------------------
    def Keq1(self, T):
        """MeOH synthesis equilibrium (Graaf 1986). Kp in bar^-2."""
        # log10 Kp = 5139/T - 12.621   (Graaf et al. 1986)
        return 10.0 ** (5139.0 / T - 12.621)

    def Keq2(self, T):
        """WGS equilibrium (as written CO2+H2<->CO+H2O), dimensionless."""
        # Moe (1962): WGS  CO + H2O <-> CO2 + H2 :  Kp = exp(4577.8/T - 4.33)
        # We write R2 as reverse (CO2+H2->CO+H2O), so Keq2 = 1/Kp_wgs
        Kp_wgs = np.exp(4577.8 / T - 4.33)
        return 1.0 / Kp_wgs

    def Keq3(self, T):
        """
        Methanol dehydration equilibrium 2 CH3OH <-> DME + H2O (dimensionless).
        Diep & Wainwright (1987) correlation (also used by Aguayo, Ereña):
            ln Kp = 2835.2/T + 1.675 ln T - 2.39e-4 T - 0.21e-6 T^2 - 13.360
        Gives Keq3 ~ 16 (200C), ~10.6 (250C), ~7.5 (300C): mildly exothermic.
        """
        lnK = (2835.2 / T + 1.675 * np.log(T)
               - 2.39e-4 * T - 0.21e-6 * T ** 2 - 13.360)
        return np.exp(lnK)

    # ------------------------------------------------------------------
    # Concentrations / partial pressures from molar flows
    # ------------------------------------------------------------------
    def _partial_pressures(self, n, P_bar, T):
        """Partial pressures [bar] for each species from molar flow vector n."""
        n = np.maximum(n, 0.0)
        ntot = np.sum(n)
        if ntot <= 0:
            return np.zeros_like(n)
        y = n / ntot
        return y * P_bar

    def _concentrations(self, n, P_bar, T):
        """Molar concentrations [mol/m3] via ideal gas: C_i = p_i/(RT)."""
        p = self._partial_pressures(n, P_bar, T)      # bar
        p_pa = p * 1e5
        return p_pa / (R * T)

    # ------------------------------------------------------------------
    # Reaction rates [mol/(kg_cat.s)]
    # ------------------------------------------------------------------
    def rates(self, n, T, P_bar):
        """
        Return (r1, r2, r3) reaction rates [mol/(kg_cat.s)].
        r1: MeOH synthesis CO+2H2->MeOH (Graaf-type LHHW with eq. driving force)
        r2: WGS as written CO2+H2->CO+H2O
        r3: MeOH dehydration 2MeOH->DME+H2O (Bercic-Levec LHHW)
        """
        p = self._partial_pressures(n, P_bar, T)
        p_CO, p_H2, p_CO2, p_H2O, p_M, p_D = (p[IDX[s]] for s in SPECIES)
        eps = 1e-12

        # --- R1: methanol synthesis (Graaf family, simplified LHHW) ---
        k1 = self.k1_0 * np.exp(-self.Ea1 / (R * T))
        Keq1 = self.Keq1(T)
        # driving force: p_CO p_H2^2 - p_M/Keq1   (Kp in bar^-2)
        drive1 = p_CO * p_H2 ** 2 - p_M / max(Keq1, eps)
        denom1 = (1.0 + self.K_ads * (p_CO + p_CO2) + np.sqrt(max(p_H2, 0.0))) ** 3
        r1 = k1 * drive1 / max(denom1, eps)

        # --- R2: water-gas shift (reverse, CO2+H2->CO+H2O) ---
        k2 = self.k2_0 * np.exp(-self.Ea2 / (R * T))
        Keq2 = self.Keq2(T)
        drive2 = p_CO2 * p_H2 - (p_CO * p_H2O) / max(Keq2, eps)
        r2 = k2 * drive2 / max(denom1, eps)

        # --- R3: methanol dehydration (Bercic-Levec 1992) ---
        k3 = self.k3_0 * np.exp(-self.Ea3 / (R * T))
        Ks = self.Ks_0 * np.exp(-self.dHs / (R * T))   # methanol adsorption const
        C = self._concentrations(n, P_bar, T)
        C_M = max(C[IDX["CH3OH"]], 0.0)
        C_W = max(C[IDX["H2O"]], 0.0)
        C_D = max(C[IDX["DME"]], 0.0)
        Keq3 = self.Keq3(T)
        drive3 = C_M ** 2 - (C_W * C_D) / max(Keq3, eps)
        denom3 = (1.0 + 2.0 * np.sqrt(max(Ks * C_M, 0.0)) + Ks * C_W) ** 4
        r3 = k3 * Ks ** 2 * drive3 / max(denom3, eps)

        return r1, r2, r3

    # ------------------------------------------------------------------
    # ODE system: molar balances + lumped energy balance vs residence time
    # ------------------------------------------------------------------
    def _rhs(self, tau, y, P_bar, adiabatic):
        n = y[:6]
        T = y[6]
        r1, r2, r3 = self.rates(n, T, P_bar)
        rb = self.rho_b

        dn = np.zeros(6)
        # R1: CO + 2H2 -> CH3OH
        dn[IDX["CO"]]    += -1.0 * r1
        dn[IDX["H2"]]    += -2.0 * r1
        dn[IDX["CH3OH"]] += +1.0 * r1
        # R2: CO2 + H2 -> CO + H2O
        dn[IDX["CO2"]]   += -1.0 * r2
        dn[IDX["H2"]]    += -1.0 * r2
        dn[IDX["CO"]]    += +1.0 * r2
        dn[IDX["H2O"]]   += +1.0 * r2
        # R3: 2 CH3OH -> DME + H2O
        dn[IDX["CH3OH"]] += -2.0 * r3
        dn[IDX["DME"]]   += +1.0 * r3
        dn[IDX["H2O"]]   += +1.0 * r3
        dn *= rb   # mol/(kg.s) * kg/m3 = mol/(m3.s) per unit residence time

        # Lumped energy balance: Cp_mix dT/dtau = -sum dH_j r_j rho_b - Ua(T-Tcool)
        Cp_mix = sum(max(n[IDX[s]], 0.0) * CP[s] for s in SPECIES)  # J/(K) per (mol/s) basis
        Cp_mix = max(Cp_mix, 1e-6)
        Q_rxn = -(DH_R1 * r1 + DH_R2 * r2 + DH_R3 * r3) * rb        # W/m3
        Q_cool = 0.0 if adiabatic else self.Ua * (T - self.T_cool)
        dT = (Q_rxn - Q_cool) / Cp_mix
        return np.concatenate([dn, [dT]])

    # ------------------------------------------------------------------
    # Feed builder
    # ------------------------------------------------------------------
    def build_feed(self, n_CO_in=None, H2_CO=None, CO2_frac=None):
        """Construct initial molar flow vector [mol/s]."""
        n_CO = self.n_CO_in if n_CO_in is None else n_CO_in
        r = self.H2_CO if H2_CO is None else H2_CO
        f = self.CO2_frac if CO2_frac is None else CO2_frac
        n = np.zeros(6)
        n[IDX["CO"]]  = n_CO
        n[IDX["H2"]]  = n_CO * r
        n[IDX["CO2"]] = n_CO * f
        return n

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, T_in_K, P_bar, tau_max=2.0, n_eval=120,
                 n_CO_in=None, H2_CO=None, CO2_frac=None, adiabatic=False):
        """
        Integrate the reactor along residence-time coordinate tau in [0, tau_max].

        Parameters
        ----------
        T_in_K : float     inlet/initial temperature [K]
        P_bar  : float     operating pressure [bar]
        tau_max: float     max residence-time coordinate (proportional to W/F) [s-equiv]
        n_eval : int       number of output points
        adiabatic : bool   if True, no coolant removal (T rises with conversion)

        Returns
        -------
        dict of arrays: tau, T, n (per species), CO_conversion,
            methanol_conversion(of produced MeOH -> DME), DME_yield,
            DME_selectivity, heat_release_kW, plus equilibrium constants.
        """
        n0 = self.build_feed(n_CO_in, H2_CO, CO2_frac)
        y0 = np.concatenate([n0, [T_in_K]])
        tau_eval = np.linspace(0.0, tau_max, n_eval)

        sol = solve_ivp(
            self._rhs, (0.0, tau_max), y0, t_eval=tau_eval,
            args=(P_bar, adiabatic), method="LSODA",
            rtol=1e-7, atol=1e-10,
        )

        n = sol.y[:6, :]
        T = sol.y[6, :]
        n = np.maximum(n, 0.0)

        n_CO0 = n0[IDX["CO"]]
        n_CO2_0 = n0[IDX["CO2"]]
        carbon_in = n_CO0 + n_CO2_0

        n_CO_t  = n[IDX["CO"], :]
        n_CO2_t = n[IDX["CO2"], :]
        n_M_t   = n[IDX["CH3OH"], :]
        n_D_t   = n[IDX["DME"], :]

        # CO conversion (relative to inlet CO)
        X_CO = np.clip((n_CO0 - n_CO_t) / max(n_CO0, 1e-12), -1.0, 1.0)
        # Carbon-to-DME yield: 2 C atoms per DME, relative to total carbon fed
        dme_yield = np.clip(2.0 * n_D_t / max(carbon_in, 1e-12), 0.0, 1.0)
        # Methanol conversion to DME: of all MeOH ever produced (MeOH + 2*DME),
        # fraction that went to DME
        meoh_made = n_M_t + 2.0 * n_D_t
        meoh_conv = np.where(meoh_made > 1e-12, (2.0 * n_D_t) / np.maximum(meoh_made, 1e-12), 0.0)
        # DME selectivity among carbon products (MeOH + DME)
        carbon_in_products = n_M_t + 2.0 * n_D_t
        dme_sel = np.where(carbon_in_products > 1e-12,
                           (2.0 * n_D_t) / np.maximum(carbon_in_products, 1e-12), 0.0)

        # Heat release rate along reactor [kW] (per reaction, total)
        Q = np.zeros_like(T)
        for i in range(len(T)):
            r1, r2, r3 = self.rates(n[:, i], T[i], P_bar)
            Q[i] = -(DH_R1 * r1 + DH_R2 * r2 + DH_R3 * r3) * self.rho_b / 1000.0  # kW/m3

        return {
            "tau": sol.t,
            "T": T,
            "n_CO": n_CO_t,
            "n_H2": n[IDX["H2"], :],
            "n_CO2": n_CO2_t,
            "n_H2O": n[IDX["H2O"], :],
            "n_CH3OH": n_M_t,
            "n_DME": n_D_t,
            "n_total": np.sum(n, axis=0),
            "CO_conversion": X_CO,
            "methanol_conversion": meoh_conv,
            "DME_yield": dme_yield,
            "DME_selectivity": dme_sel,
            "heat_release_kW": Q,
            "Keq1": self.Keq1(T),
            "Keq2": self.Keq2(T),
            "Keq3": self.Keq3(T),
            "n0": n0,
        }
