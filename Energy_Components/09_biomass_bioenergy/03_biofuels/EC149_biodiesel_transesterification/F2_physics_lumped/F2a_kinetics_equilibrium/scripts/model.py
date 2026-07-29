"""
EC149 -- Biodiesel Transesterification -- F2a Physics-Lumped Kinetics + Reactor Energy Balance

Physics-lumped (0D, well-mixed batch reactor) first-principles model of
base-catalyzed transesterification of triglyceride with methanol to FAME
(fatty-acid methyl ester, "biodiesel") and glycerol.

The reaction proceeds stepwise through di- and mono-glyceride intermediates,
each step reversible and second-order, releasing one FAME molecule per step:

    TG + MeOH  <-->  DG + FAME        (k1 forward, k2 reverse)
    DG + MeOH  <-->  MG + FAME        (k3 forward, k4 reverse)
    MG + MeOH  <-->  GL + FAME        (k5 forward, k6 reverse)

Six elementary second-order rate constants k1..k6 [L/(mol.min)] follow an
Arrhenius temperature law:

    k_i(T) = A_i * exp(-Ea_i / (R T))

calibrated so each equals its reported value at T_ref (50 degC). Species mole
balances (per litre of mixture) give the coupled ODE system:

    d[TG]/dt   = -k1[TG][A] + k2[DG][E]
    d[DG]/dt   =  k1[TG][A] - k2[DG][E] - k3[DG][A] + k4[MG][E]
    d[MG]/dt   =  k3[DG][A] - k4[MG][E] - k5[MG][A] + k6[GL][E]
    d[GL]/dt   =  k5[MG][A] - k6[GL][E]
    d[E]/dt    =  k1[TG][A] - k2[DG][E] + k3[DG][A] - k4[MG][E] + k5[MG][A] - k6[GL][E]   (E = FAME / ester)
    d[A]/dt    = -d[E]/dt                                                                  (A = methanol; consumed 1:1 with FAME formed)

coupled to a lumped reactor energy balance ODE (well-mixed jacketed batch):

    rho V cp dT/dt = (-dH_rxn) * V * R_FAME  -  UA (T - T_jacket)

where R_FAME = d[E]/dt * V is the molar FAME production rate and dH_rxn is the
mildly exothermic net heat of reaction. The chemical ODEs and the thermal ODE
are integrated together with scipy.integrate.solve_ivp so temperature feeds back
into the Arrhenius constants (self-accelerating then equilibrium-limited).

Conserved quantities:
    glyceride backbone:  [TG] + [DG] + [MG] + [GL] = const
    acyl (ester) balance: 3[TG] + 2[DG] + [MG] + [E] = const = 3[TG]_0  (3 FAME max per TG)
    methanol+ester:      [A] + [E] = const

References:
    Noureddini, H. & Zhu, D. (1997). Kinetics of transesterification of soybean
        oil. J. Am. Oil Chem. Soc. 74(11), 1457-1463.   (3-step reversible scheme)
    Freedman, B., Butterfield, R.O. & Pryde, E.H. (1986). Transesterification
        kinetics of soybean oil. J. Am. Oil Chem. Soc. 63(10), 1375-1380.
    Freedman, B., Pryde, E.H. & Mounts, T.L. (1984). Variables affecting the
        yields of fatty esters from transesterified vegetable oils. JAOCS 61:1638.
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314  # J/(mol.K)


class TransesterificationF2a:
    """Base-catalyzed transesterification -- lumped kinetics + reactor energy balance."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Arrhenius pre-exponentials [L/(mol.min)] and activation energies [J/mol]
        self.A = [u[f"A{i}"]["value"] for i in range(1, 7)]
        self.Ea = [u[f"Ea{i}"]["value"] for i in range(1, 7)]
        self.T_ref = float(u["T_ref"]["value"])
        # Reported rate constants at T_ref (used to recalibrate A so k(T_ref)=k_ref)
        self.k_ref = [u[f"k{i}_ref"]["value"] for i in range(1, 7)]
        # Recompute consistent pre-exponentials: A_i = k_ref_i * exp(Ea_i/(R T_ref))
        self.A = [self.k_ref[i] * np.exp(self.Ea[i] / (R_GAS * self.T_ref))
                  for i in range(6)]

        self.MW = {
            "TG": u["MW_TG"]["value"], "DG": u["MW_DG"]["value"],
            "MG": u["MW_MG"]["value"], "FAME": u["MW_FAME"]["value"],
            "GL": u["MW_GL"]["value"], "MeOH": u["MW_MeOH"]["value"],
        }
        self.density_oil = u["density_oil"]["value"]
        self.density_MeOH = u["density_MeOH"]["value"]

        # Thermal
        self.dH_rxn = u["dH_rxn"]["value"]     # J/mol_FAME (negative = exothermic)
        self.rho_mix = u["rho_mix"]["value"]    # kg/L
        self.cp_mix = u["cp_mix"]["value"]      # J/(kg.K)
        self.V = u["V_reactor"]["value"]        # L
        self.UA = u["UA_jacket"]["value"]       # W/K
        self.T_jacket = u["T_jacket"]["value"]  # K

    # ------------------------------------------------------------------
    # Arrhenius rate constants
    # ------------------------------------------------------------------
    def rate_constants(self, T):
        """Return [k1..k6] in L/(mol.min) at temperature T [K]."""
        return [self.A[i] * np.exp(-self.Ea[i] / (R_GAS * T)) for i in range(6)]

    # ------------------------------------------------------------------
    # Initial concentrations from methanol:oil molar ratio
    # ------------------------------------------------------------------
    def initial_concentrations(self, TG0, methanol_ratio):
        """
        Build initial concentration vector [TG, DG, MG, GL, E, A] in mol/L.

        TG0            : initial triglyceride concentration [mol/L]
        methanol_ratio : MeOH:oil molar ratio (e.g. 6.0)
        """
        A0 = methanol_ratio * TG0  # methanol
        return np.array([TG0, 0.0, 0.0, 0.0, 0.0, A0], dtype=float)

    # ------------------------------------------------------------------
    # Combined chemical + thermal ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, catalyst_factor, isothermal):
        TG, DG, MG, GL, E, A = y[0], y[1], y[2], y[3], y[4], y[5]
        T = y[6]
        # clamp concentrations against tiny negative excursions
        TG = max(TG, 0.0); DG = max(DG, 0.0); MG = max(MG, 0.0)
        GL = max(GL, 0.0); E = max(E, 0.0); A = max(A, 0.0)

        k = self.rate_constants(T)
        # catalyst_factor scales all forward+reverse rates (NaOH loading proxy)
        k = [catalyst_factor * ki for ki in k]
        k1, k2, k3, k4, k5, k6 = k

        r1 = k1 * TG * A   # TG + MeOH -> DG + FAME
        r2 = k2 * DG * E   # DG + FAME -> TG + MeOH
        r3 = k3 * DG * A   # DG + MeOH -> MG + FAME
        r4 = k4 * MG * E   # MG + FAME -> DG + MeOH
        r5 = k5 * MG * A   # MG + MeOH -> GL + FAME
        r6 = k6 * GL * E   # GL + FAME -> MG + MeOH

        dTG = -r1 + r2
        dDG = r1 - r2 - r3 + r4
        dMG = r3 - r4 - r5 + r6
        dGL = r5 - r6
        dE = r1 - r2 + r3 - r4 + r5 - r6   # net FAME formation
        dA = -dE                            # methanol consumed 1:1 with FAME

        if isothermal:
            dT = 0.0
        else:
            # FAME molar production rate [mol/min] in reactor volume V [L]
            R_FAME = dE * self.V
            # energy balance: rho V cp dT/dt = (-dH) V dE  -  UA (T - Tj)
            # rates are per minute -> convert UA (W/K) to J/(min.K)
            heat_gen = (-self.dH_rxn) * R_FAME              # J/min
            heat_rem = self.UA * 60.0 * (T - self.T_jacket)  # J/min
            mass = self.rho_mix * self.V                     # kg
            dT = (heat_gen - heat_rem) / (mass * self.cp_mix)  # K/min

        return [dTG, dDG, dMG, dGL, dE, dA, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, TG0=1.0, methanol_ratio=6.0, T0=333.15,
                 catalyst_factor=1.0, duration_min=90.0, n_points=120,
                 isothermal=False):
        """
        Integrate the coupled kinetics + reactor energy balance.

        Parameters
        ----------
        TG0            : initial triglyceride concentration [mol/L]
        methanol_ratio : MeOH:oil molar ratio [-]
        T0             : initial reactor temperature [K]
        catalyst_factor: multiplicative scaling of all rate constants [-]
        duration_min   : total reaction time [min]
        n_points       : number of output samples
        isothermal     : if True hold T = T0 (energy balance disabled)

        Returns
        -------
        dict of time-series arrays: t, TG, DG, MG, FAME, glycerol, methanol,
             temperature, conversion, FAME_yield  (and scalar finals).
        """
        y0c = self.initial_concentrations(TG0, methanol_ratio)
        y0 = np.concatenate([y0c, [T0]])

        t_eval = np.linspace(0.0, duration_min, int(n_points))

        sol = solve_ivp(
            self._rhs, (0.0, duration_min), y0,
            t_eval=t_eval, method="LSODA",
            args=(catalyst_factor, isothermal),
            rtol=1e-8, atol=1e-11,
        )

        TG, DG, MG, GL, E, A = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], sol.y[5]
        T = sol.y[6]

        # TG conversion
        conversion = np.clip((TG0 - TG) / TG0, 0.0, 1.0) if TG0 > 0 else np.zeros_like(TG)
        # FAME yield: fraction of theoretical max (3 mol FAME per mol TG fed)
        FAME_max = 3.0 * TG0
        FAME_yield = np.clip(E / FAME_max, 0.0, 1.0) if FAME_max > 0 else np.zeros_like(E)

        return {
            "t": sol.t,
            "TG": TG, "DG": DG, "MG": MG, "FAME": E, "glycerol": GL,
            "methanol": A,
            "temperature": T,
            "conversion": conversion,
            "FAME_yield": FAME_yield,
            "TG0": TG0,
            "FAME_final": float(E[-1]),
            "conversion_final": float(conversion[-1]),
            "FAME_yield_final": float(FAME_yield[-1]),
            "T_final": float(T[-1]),
        }

    # ------------------------------------------------------------------
    # Conservation diagnostics
    # ------------------------------------------------------------------
    def mass_balance_residuals(self, result):
        """
        Return max relative drift of the three conserved quantities over the run:
            glyceride backbone, ester balance, methanol+ester.
        """
        TG, DG, MG, GL = result["TG"], result["DG"], result["MG"], result["glycerol"]
        E, A = result["FAME"], result["methanol"]

        backbone = TG + DG + MG + GL                  # glycerol backbones conserved
        # acyl-group (ester bond) balance: TG=3, DG=2, MG=1, GL=0 acyls on
        # backbone, plus 1 acyl per free FAME -> total acyl = 3*TG0 = const
        ester_bal = 3.0 * TG + 2.0 * DG + MG + E      # constant (=3*TG0)
        meoh_bal = A + E                              # methanol+ester conserved

        def drift(x):
            x0 = x[0]
            if abs(x0) < 1e-12:
                return float(np.max(np.abs(x - x0)))
            return float(np.max(np.abs(x - x0)) / abs(x0))

        return {
            "backbone": drift(backbone),
            "ester_balance": drift(ester_bal),
            "methanol_balance": drift(meoh_bal),
        }
