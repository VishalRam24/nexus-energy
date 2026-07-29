"""
EC199 -- Pre-Combustion Capture (WGS + Separation) -- F2a Physics-Lumped

Two coupled first-principles ODE sub-models integrated with scipy.solve_ivp:

  (1) WATER-GAS-SHIFT REACTOR  (lumped plug-flow / residence-time ODE)
        CO + H2O  <->  CO2 + H2      DH = -41.1 kJ/mol  (exothermic)
      Reversible power-law kinetics approaching the equilibrium constant
      Keq(T) (Moe 1962 correlation). The molar extent xi of the shift is
      integrated over the gas residence time tau_WGS:
          d(xi)/d(t) = r_fwd - r_rev
          r = k(T) * [ p_CO * p_H2O  -  (p_CO2 * p_H2)/Keq(T) ]
      where partial pressures are evaluated from the instantaneous species
      composition at total pressure P. xi is bounded by the equilibrium
      (driving force -> 0) so conversion never exceeds the thermodynamic
      WGS limit. This concentrates carbon as CO2 and produces an H2-rich gas.

  (2) PHYSICAL-SOLVENT ABSORBER  (Henry's-law mass-transfer ODE)
        CO2(gas)  ->  CO2(dissolved in Selexol/Rectisol)
      Lumped two-film mass transfer driven by departure from Henry
      equilibrium (Kohl & Nielsen 1997). Per unit gas residence time tau_abs:
          d(n_CO2_gas)/dt = -KLa * (C_CO2_gas_equiv  -  C_star(x_liq))
      where C_star = p_CO2 / H(T)  is the gas-phase concentration that would
      be in equilibrium with the loaded liquid. High CO2 PARTIAL PRESSURE
      (IGCC syngas at 20-40 bar) gives a large driving force, so physical
      solvents capture CO2 cheaply -- the pre-combustion advantage. H2 is
      nearly insoluble (selective), so the H2 fuel is retained.

Energy penalty (IPCC 2005 SRCCS Ch.3; Kunze & Spliethoff 2012):
      E_total[GJ/tCO2] = E_pump (solvent circulation + flash regen)
                       + E_compression (CO2 to pipeline pressure)
  Physical absorption needs little thermal regeneration energy, so the
  penalty is dominated by pumping + compression -> much lower than amine
  post-combustion capture.

Conservation: carbon (CO + CO2) and hydrogen atoms are conserved across the
WGS reactor; total CO2 in product = CO2(captured) + CO2(slip) across absorber.

References
----------
  Moe, J.M. (1962). Chem. Eng. Prog. 58(3), 33-36.   [WGS Keq(T)]
  Smith, R.J.B., Loganathan, M., Shantha, M.S. (2010).
      Int. J. Chem. React. Eng. 8, R4. / Chem. Eng. Res. Des. 88. [LT-shift kinetics]
  Kohl, A.L. & Nielsen, R.B. (1997). Gas Purification, 5th ed.,
      Gulf Publishing, Ch.14.   [Selexol/Rectisol physical absorption, Henry's law]
  IPCC (2005). Special Report on CO2 Capture and Storage, Ch.3.
  Kunze, C. & Spliethoff, H. (2012). Applied Energy 94, 109-116.
"""

import numpy as np
from scipy.integrate import solve_ivp


class PreCombustionCaptureF2a:
    """Pre-combustion CO2 capture -- physics-lumped WGS reactor + absorber ODEs."""

    R = 8.314          # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        # WGS kinetics
        self.k0_WGS = u["k0_WGS"]["value"]            # mol/(m3.s.bar^2)
        self.Ea_WGS = u["Ea_WGS"]["value"]            # J/mol
        self.DH_WGS = u["DH_WGS"]["value"]            # J/mol (exothermic, <0)
        self.tau_WGS = u["tau_WGS_s"]["value"]        # s
        self.T_WGS_nom = u["T_WGS_K"]["value"]        # K
        # Absorber / Henry
        self.H_CO2_ref = u["henry_CO2_molperm3bar"]["value"]   # mol/(m3.bar) @ T_abs_ref
        self.H_H2_ref = u["henry_H2_molperm3bar"]["value"]     # mol/(m3.bar)
        self.dH_sol = u["dH_sol_CO2"]["value"]        # J/mol
        self.KLa = u["KLa_absorber"]["value"]         # 1/s
        self.tau_abs = u["tau_abs_s"]["value"]        # s
        self.L_over_G = u["L_over_G"]["value"]        # mol solvent / mol gas
        self.T_abs_nom = u["T_abs_K"]["value"]        # K
        # Energy
        self.E_pump = u["E_pump_GJ_tCO2"]["value"]
        self.E_comp = u["E_compression_GJ_tCO2"]["value"]
        # Molar masses (kg/mol)
        self.MW_CO2 = u["MW_CO2"]["value"] / 1000.0
        self.MW_H2 = u["MW_H2"]["value"] / 1000.0
        self.MW_CO = u["MW_CO"]["value"] / 1000.0
        self.MW_H2O = u["MW_H2O"]["value"] / 1000.0

    # =====================================================================
    # WATER-GAS-SHIFT thermodynamics + kinetics
    # =====================================================================
    def keq_wgs(self, T):
        """
        WGS equilibrium constant Keq(T) (dimensionless, pressure-independent).
        Moe (1962) correlation: Keq = exp(4577.8/T - 4.33).
        Favoured (Keq>>1) at low temperature -> LT shift maximises conversion.
        """
        return np.exp(4577.8 / T - 4.33)

    def wgs_rate(self, p_CO, p_H2O, p_CO2, p_H2, T):
        """
        Reversible WGS volumetric reaction rate [mol/(m3.s)].
        r = k(T) * (p_CO*p_H2O - p_CO2*p_H2/Keq).  Driving force -> 0 at equilibrium.
        """
        k = self.k0_WGS * np.exp(-self.Ea_WGS / (self.R * T))
        Keq = self.keq_wgs(T)
        return k * (p_CO * p_H2O - (p_CO2 * p_H2) / Keq)

    def simulate_wgs(self, n_in, T_WGS=None, P_bar=30.0, steam_co_ratio=3.0):
        """
        Integrate the WGS reactor over the gas residence time.

        Parameters
        ----------
        n_in : dict   inlet molar flows {CO, H2, CO2, H2O, inert} [mol/s]
        T_WGS : float reactor temperature [K]   (default nominal)
        P_bar : float total pressure [bar]
        steam_co_ratio : float  used only if H2O not supplied in n_in

        Returns
        -------
        dict: outlet flows, conversion X_CO, extent, time-series, heat released.
        """
        if T_WGS is None:
            T_WGS = self.T_WGS_nom

        n_CO0 = float(n_in.get("CO", 0.0))
        n_H20 = float(n_in.get("H2", 0.0))
        n_CO20 = float(n_in.get("CO2", 0.0))
        n_inert = float(n_in.get("inert", 0.0))
        # Steam: explicit, else from S/C ratio relative to CO
        n_H2O0 = float(n_in.get("H2O", steam_co_ratio * n_CO0))

        n_tot0 = n_CO0 + n_H20 + n_CO20 + n_H2O0 + n_inert
        if n_tot0 <= 0:
            raise ValueError("WGS inlet total flow must be > 0")

        # Total moles are conserved by WGS (1 mol -> 1 mol). State variable:
        # xi = molar shift flow of CO converted [mol/s], bounded [0, n_CO0].
        def partials(xi):
            n_CO = n_CO0 - xi
            n_H2O = n_H2O0 - xi
            n_CO2 = n_CO20 + xi
            n_H2 = n_H20 + xi
            n_tot = n_tot0  # total unchanged
            p_CO = P_bar * max(n_CO, 0.0) / n_tot
            p_H2O = P_bar * max(n_H2O, 0.0) / n_tot
            p_CO2 = P_bar * max(n_CO2, 0.0) / n_tot
            p_H2 = P_bar * max(n_H2, 0.0) / n_tot
            return p_CO, p_H2O, p_CO2, p_H2

        # d(xi)/dt is proportional to volumetric rate; absorb reactor-volume /
        # molar-flow grouping into the calibrated tau so xi-extent tracks
        # conversion over the residence time. Scale by inlet CO to keep the
        # ODE well-conditioned across plant sizes.
        scale = max(n_CO0, 1e-9)

        def rhs(t, y):
            xi = np.clip(y[0], 0.0, min(n_CO0, n_H2O0))
            p_CO, p_H2O, p_CO2, p_H2 = partials(xi)
            r = self.wgs_rate(p_CO, p_H2O, p_CO2, p_H2, T_WGS)
            return [r * scale]

        t_span = (0.0, self.tau_WGS)
        t_eval = np.linspace(0.0, self.tau_WGS, 50)
        sol = solve_ivp(rhs, t_span, [0.0], t_eval=t_eval,
                        method="LSODA", rtol=1e-8, atol=1e-10)

        xi_f = float(np.clip(sol.y[0][-1], 0.0, min(n_CO0, n_H2O0)))
        X_CO = xi_f / n_CO0 if n_CO0 > 0 else 0.0

        n_out = {
            "CO": n_CO0 - xi_f,
            "H2": n_H20 + xi_f,
            "CO2": n_CO20 + xi_f,
            "H2O": n_H2O0 - xi_f,
            "inert": n_inert,
        }
        # Exothermic heat released [kW] = extent * (-DH)
        Q_kW = xi_f * (-self.DH_WGS) / 1000.0

        return {
            "n_in": {"CO": n_CO0, "H2": n_H20, "CO2": n_CO20,
                     "H2O": n_H2O0, "inert": n_inert},
            "n_out": n_out,
            "extent_mol_s": xi_f,
            "X_CO": X_CO,
            "Keq": float(self.keq_wgs(T_WGS)),
            "X_eq": self._equilibrium_conversion(n_CO0, n_H20, n_CO20, n_H2O0, T_WGS),
            "Q_WGS_kW": Q_kW,
            "t": sol.t,
            "xi_t": np.clip(sol.y[0], 0.0, min(n_CO0, n_H2O0)),
            "T_WGS_K": T_WGS,
            "P_bar": P_bar,
        }

    def _equilibrium_conversion(self, n_CO0, n_H20, n_CO20, n_H2O0, T):
        """Analytic WGS equilibrium conversion (mole-ratio form, P cancels)."""
        Keq = self.keq_wgs(T)
        # (n_CO2+x)(n_H2+x) = Keq (n_CO-x)(n_H2O-x)  -> quadratic in x
        a = 1.0 - Keq
        b = (n_CO20 + n_H20) + Keq * (n_CO0 + n_H2O0)
        c = n_CO20 * n_H20 - Keq * n_CO0 * n_H2O0
        if abs(a) < 1e-12:
            x = -c / b if abs(b) > 1e-12 else 0.0
        else:
            disc = b * b - 4 * a * c
            disc = max(disc, 0.0)
            x = (-b + np.sqrt(disc)) / (2 * a)
            x2 = (-b - np.sqrt(disc)) / (2 * a)
            # pick physical root in [0, min(n_CO0, n_H2O0)]
            hi = min(n_CO0, n_H2O0)
            cand = [r for r in (x, x2) if -1e-9 <= r <= hi + 1e-9]
            x = max(cand) if cand else 0.0
        return float(np.clip(x, 0.0, min(n_CO0, n_H2O0)) / max(n_CO0, 1e-12))

    # =====================================================================
    # PHYSICAL-SOLVENT ABSORBER (Henry's law mass-transfer ODE)
    # =====================================================================
    def henry_CO2(self, T):
        """Henry solubility coefficient for CO2 in Selexol [mol/(m3.bar)].
        van't Hoff temperature correction: H(T) = H_ref*exp(-dH_sol/R*(1/T-1/Tref)).
        Solubility rises (H larger) as T drops -> chilled solvent absorbs more."""
        return self.H_CO2_ref * np.exp(-self.dH_sol / self.R *
                                       (1.0 / T - 1.0 / self.T_abs_nom))

    def simulate_absorber(self, n_gas_in, T_abs=None, P_bar=30.0):
        """
        Integrate CO2 absorption into physical solvent over contact time.

        Lumped per-mole-of-gas basis: track moles of CO2 remaining in the gas
        phase and moles absorbed into liquid, driven by Henry departure.

        Parameters
        ----------
        n_gas_in : dict   gas molar flows entering absorber {CO2, H2, CO, H2O, inert}
        T_abs : float     absorber temperature [K]
        P_bar : float     pressure [bar]

        Returns
        -------
        dict with captured CO2, slip, capture fraction, H2 retained, time series.
        """
        if T_abs is None:
            T_abs = self.T_abs_nom

        n_CO2_g0 = float(n_gas_in.get("CO2", 0.0))
        n_tot_gas = sum(float(v) for v in n_gas_in.values())
        if n_tot_gas <= 0:
            raise ValueError("Absorber inlet gas flow must be > 0")

        H = self.henry_CO2(T_abs)                       # mol/(m3.bar)
        # Liquid solvent molar flow from L/G ratio; capacity to dissolve CO2.
        n_solvent = self.L_over_G * n_tot_gas           # mol/s solvent

        # State: y[0] = CO2 absorbed into liquid [mol/s]. Gas CO2 = n_CO2_g0 - y0.
        # Effective gas-phase "concentration" ~ partial pressure; equilibrium
        # gas conc in balance with liquid loading via Henry: when liquid is at
        # equilibrium with current gas, driving force -> 0.
        # Liquid loading C_liq [mol CO2 / mol solvent]; equilibrium partial
        # pressure over liquid p* = C_liq * n_solvent / (H * V_liq_equiv).
        # We lump V terms into KLa (1/s) acting on the partial-pressure gap.
        eps = 1e-12

        def rhs(t, y):
            n_abs = np.clip(y[0], 0.0, n_CO2_g0)
            n_CO2_g = n_CO2_g0 - n_abs
            # gas CO2 partial pressure [bar]
            p_CO2 = P_bar * n_CO2_g / n_tot_gas
            # liquid loading -> equilibrium back-pressure via Henry.
            # C_liq (mol/m3-equiv) ~ n_abs / (n_solvent/H_scale); use Henry so
            # that p_eq = (n_abs / n_solvent) * (P_bar_ref) / Henry_capacity.
            # Capacity (mol CO2 dissolvable per mol solvent per bar) ~ H/ C_ref.
            cap = H / 1000.0  # mol CO2 / (mol solvent . bar)  (lumped from H)
            p_eq = (n_abs / n_solvent) / max(cap, eps)
            driving = max(p_CO2 - p_eq, 0.0)            # bar, only absorb (>=0)
            # mass-transfer flux [mol/s] = KLa * driving * (gas holdup proxy)
            flux = self.KLa * driving * (n_tot_gas / P_bar)
            return [flux]

        t_span = (0.0, self.tau_abs)
        t_eval = np.linspace(0.0, self.tau_abs, 50)
        sol = solve_ivp(rhs, t_span, [0.0], t_eval=t_eval,
                        method="LSODA", rtol=1e-8, atol=1e-12)

        n_abs_f = float(np.clip(sol.y[0][-1], 0.0, n_CO2_g0))
        cap_frac = n_abs_f / n_CO2_g0 if n_CO2_g0 > 0 else 0.0
        n_CO2_slip = n_CO2_g0 - n_abs_f

        # H2 co-absorption (selectivity loss) via its much smaller Henry coeff.
        n_H2_g = float(n_gas_in.get("H2", 0.0))
        sel = self.H_H2_ref / max(self.H_CO2_ref, eps)
        n_H2_lost = n_H2_g * cap_frac * sel             # tiny
        n_H2_retained = n_H2_g - n_H2_lost

        return {
            "n_CO2_in": n_CO2_g0,
            "n_CO2_captured_mol_s": n_abs_f,
            "n_CO2_slip_mol_s": n_CO2_slip,
            "capture_fraction": cap_frac,
            "n_H2_retained_mol_s": n_H2_retained,
            "n_H2_lost_mol_s": n_H2_lost,
            "H_CO2": H,
            "p_CO2_in_bar": P_bar * n_CO2_g0 / n_tot_gas,
            "t": sol.t,
            "n_abs_t": np.clip(sol.y[0], 0.0, n_CO2_g0),
            "T_abs_K": T_abs,
        }

    # =====================================================================
    # ENERGY PENALTY
    # =====================================================================
    def energy_penalty_GJ_tCO2(self):
        """Specific energy penalty [GJ/tCO2] = pumping/regen + compression."""
        return self.E_pump + self.E_comp

    # =====================================================================
    # FULL COUPLED SIMULATION (WGS -> absorber)
    # =====================================================================
    def simulate(self, syngas_flow_mol_s, co_fraction, h2_fraction,
                 T_WGS_K=None, T_abs_K=None, P_bar=30.0,
                 steam_co_ratio=3.0, co2_fraction=0.0, inert_fraction=0.0):
        """
        Full pre-combustion train: WGS reactor then physical-solvent absorber.

        Parameters
        ----------
        syngas_flow_mol_s : float  total dry-ish syngas molar flow [mol/s]
        co_fraction, h2_fraction, co2_fraction, inert_fraction : float
            inlet mole fractions of the syngas (remainder -> inert)
        T_WGS_K, T_abs_K : float   reactor / absorber temperatures [K]
        P_bar : float              operating pressure [bar]
        steam_co_ratio : float     steam-to-CO molar ratio fed to WGS

        Returns
        -------
        dict: capture_rate, h2_rich_fuel, energy_penalty, conservation residuals,
              and the two sub-model result dicts.
        """
        n = float(syngas_flow_mol_s)
        n_CO = n * co_fraction
        n_H2 = n * h2_fraction
        n_CO2 = n * co2_fraction
        n_inert = n * inert_fraction

        wgs_in = {"CO": n_CO, "H2": n_H2, "CO2": n_CO2,
                  "H2O": steam_co_ratio * n_CO, "inert": n_inert}
        wgs = self.simulate_wgs(wgs_in, T_WGS=T_WGS_K, P_bar=P_bar,
                                steam_co_ratio=steam_co_ratio)

        # Gas entering absorber = WGS outlet (drop most steam by condensation;
        # keep small residual so it doesn't dominate partial pressures).
        n_out = wgs["n_out"]
        abs_in = {
            "CO2": n_out["CO2"],
            "H2": n_out["H2"],
            "CO": n_out["CO"],
            "H2O": 0.05 * n_out["H2O"],   # most steam knocked out before absorber
            "inert": n_out["inert"],
        }
        ab = self.simulate_absorber(abs_in, T_abs=T_abs_K, P_bar=P_bar)

        # --- overall metrics ---
        # Carbon basis: total carbon in (CO + CO2) at inlet.
        C_in = n_CO + n_CO2
        co2_captured = ab["n_CO2_captured_mol_s"]
        capture_rate = co2_captured / C_in if C_in > 0 else 0.0

        co2_captured_kg_s = co2_captured * self.MW_CO2
        co2_captured_tps = co2_captured_kg_s / 1000.0          # t/s
        E_spec = self.energy_penalty_GJ_tCO2()                 # GJ/tCO2
        # GJ/tCO2 * tCO2/s = GJ/s = GW -> *1e3 = MW
        power_penalty_MW = E_spec * co2_captured_tps * 1e3

        # H2-rich fuel leaving (retained H2 + unreacted CO + slip)
        h2_fuel = ab["n_H2_retained_mol_s"]
        h2_purity = h2_fuel / max(
            h2_fuel + ab["n_CO2_slip_mol_s"] + abs_in["CO"] + abs_in["inert"], 1e-12)

        # --- conservation checks ---
        # Carbon: CO_out + CO2_out(slip) + CO2_captured == C_in
        C_out = n_out["CO"] + ab["n_CO2_slip_mol_s"] + co2_captured
        carbon_residual = abs(C_out - C_in) / max(C_in, 1e-12)
        # Total moles across WGS conserved:
        tot_in = sum(wgs["n_in"].values())
        tot_out = sum(n_out.values())
        mole_residual_wgs = abs(tot_out - tot_in) / max(tot_in, 1e-12)
        # Hydrogen atom balance across WGS: 2*H2 + 2*H2O conserved
        H_in = 2 * wgs["n_in"]["H2"] + 2 * wgs["n_in"]["H2O"]
        H_out = 2 * n_out["H2"] + 2 * n_out["H2O"]
        h_atom_residual = abs(H_out - H_in) / max(H_in, 1e-12)

        return {
            "capture_rate": capture_rate,
            "wgs_conversion": wgs["X_CO"],
            "wgs_equilibrium_conversion": wgs["X_eq"],
            "co2_captured_kg_s": co2_captured_kg_s,
            "co2_captured_mol_s": co2_captured,
            "h2_rich_fuel_mol_s": h2_fuel,
            "h2_purity": h2_purity,
            "energy_penalty_GJ_tCO2": E_spec,
            "power_penalty_MW": power_penalty_MW,
            "wgs_heat_kW": wgs["Q_WGS_kW"],
            "p_CO2_absorber_in_bar": ab["p_CO2_in_bar"],
            "carbon_residual": carbon_residual,
            "mole_residual_wgs": mole_residual_wgs,
            "h_atom_residual": h_atom_residual,
            "wgs": wgs,
            "absorber": ab,
        }
