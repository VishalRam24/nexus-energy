"""
EC015 -- Chemical H2 Storage (LOHC / Ammonia) -- F2a Lumped Kinetics Reactor

Physics-lumped 0D batch reactor for the ENDOTHERMIC H2-release step:
    LOHC:    H18-DBT  ->  H0-DBT + 9 H2      DH = +65 kJ/mol_H2  (dehydrogenation)
    Ammonia: 2 NH3    ->  N2 + 3 H2          DH = +46 kJ/mol_H2  (cracking)

Two coupled ODEs integrated with scipy.integrate.solve_ivp:

  (1) Conversion ODE (first-order Arrhenius kinetics, Fogler 2016 Ch.3-4):
        k(T)  = A * exp(-E_act / (R T))            [1/s]   (Arrhenius)
        dX/dt = k(T) * (1 - X)                      first order in unreacted carrier
      X in [0,1] is fraction of hydrogenated carrier that has released its H2.

  (2) Lumped energy balance (Fogler 2016 Ch.12, energy balance on a batch reactor):
        m_r cp_r dT/dt = Q_heat - Q_rxn
        Q_heat = hA (T_set - T)                     external heater (jacket)
        Q_rxn  = (dn_H2/dt) * DH_per_mol_H2         endothermic sink (>0 removes heat)
      dn_H2/dt = n_carrier * nu_H2 * dX/dt          molar H2 release rate

Tracked outputs: conversion X(t), H2 release rate (mol/s and kg/s),
reactor temperature T(t), cumulative H2 released, and the thermal energy
penalty per kg H2 (DH / M_H2 / eta_heat-style accounting).

Conservation:
  * Mass: total carrier moles conserved; H2 released = n_carrier*nu_H2*X (<= max).
  * Energy: integral(Q_heat) = sensible heat stored + integral(Q_rxn);
            reaction enthalpy charged to released H2 equals n_H2 * DH.

References:
    Fogler, H.S. (2016). Elements of Chemical Reaction Engineering, 5th ed.,
        Prentice Hall -- Arrhenius rate law (Ch.3) and batch energy balance (Ch.12).
    Preuster, P., Papp, C., Wasserscheid, P. (2017). Acc. Chem. Res. 50(1), 74-85.
    Duerr, S. et al. (2021). Int. J. Hydrogen Energy 46, 32583 (DBT dehydrogenation).
    Di Carlo, A. et al. (2014). Int. J. Hydrogen Energy 39, 808 (NH3 cracking kinetics).
    Lamb, K.E. et al. (2019). Int. J. Hydrogen Energy 44(7), 3580-3593.
"""

import numpy as np
from scipy.integrate import solve_ivp

M_H2 = 0.002016  # kg/mol


class ChemicalH2StorageF2a:
    """Lumped batch reactor: Arrhenius conversion ODE + energy balance."""

    R = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        self._p = params
        lohc = params["lohc_dbt"]
        nh3 = params["ammonia"]
        rxr = params["reactor"]
        thermo = params["thermodynamics"]

        # --- LOHC (DBT dehydrogenation) ---
        self.lohc = {
            "A": lohc["A_pre_exp"]["value"],
            "Ea": lohc["E_act"]["value"],
            "dH": lohc["delta_H_dehydrogenation"]["value"],   # J/mol_H2 (>0)
            "nu_H2": lohc["H2_mol_per_mol_carrier"]["value"], # mol H2 / mol carrier
            "T_set": lohc["T_reactor_K"]["value"],
            "M_carrier": lohc["molar_mass_carrier_kg_per_mol"]["value"],
        }
        # --- Ammonia cracking ---
        self.nh3 = {
            "A": nh3["A_pre_exp"]["value"],
            "Ea": nh3["E_act"]["value"],
            "dH": nh3["delta_H_cracking"]["value"],            # J/mol_H2 (>0)
            "nu_H2": nh3["H2_mol_per_mol_carrier"]["value"],   # mol H2 / mol NH3
            "T_set": nh3["T_reactor_K"]["value"],
            "M_carrier": nh3["molar_mass_nh3_kg_per_mol"]["value"],
        }

        self.m_reactor = rxr["m_reactor_kg"]["value"]
        self.cp_reactor = rxr["cp_reactor"]["value"]
        self.hA = rxr["hA_heater"]["value"]
        self.n_carrier0 = rxr["n_carrier_mol"]["value"]

        self.LHV_H2 = thermo["LHV_H2"]["value"]

    # ------------------------------------------------------------------
    def _mode(self, mode):
        if mode == "lohc":
            return self.lohc
        elif mode == "ammonia":
            return self.nh3
        raise ValueError(f"unknown mode '{mode}' (use 'lohc' or 'ammonia')")

    # ------------------------------------------------------------------
    # Arrhenius rate constant
    # ------------------------------------------------------------------
    def rate_constant(self, T, mode):
        """Arrhenius rate constant k(T) [1/s]  (Fogler 2016, Ch.3)."""
        p = self._mode(mode)
        return p["A"] * np.exp(-p["Ea"] / (self.R * T))

    # ------------------------------------------------------------------
    # Reaction rate (conversion derivative), first order
    # ------------------------------------------------------------------
    def conversion_rate(self, X, T, mode):
        """dX/dt = k(T) (1 - X), clipped at X=1 [1/s]."""
        X = min(max(X, 0.0), 1.0)
        return self.rate_constant(T, mode) * (1.0 - X)

    # ------------------------------------------------------------------
    # H2 molar release rate
    # ------------------------------------------------------------------
    def h2_release_rate(self, X, T, mode, n_carrier=None):
        """dn_H2/dt [mol/s] = n_carrier * nu_H2 * dX/dt."""
        p = self._mode(mode)
        if n_carrier is None:
            n_carrier = self.n_carrier0
        return n_carrier * p["nu_H2"] * self.conversion_rate(X, T, mode)

    # ------------------------------------------------------------------
    # Specific thermal energy demand (penalty)
    # ------------------------------------------------------------------
    def specific_energy(self, mode):
        """Reaction thermal demand per kg H2 released [MJ/kg_H2]."""
        p = self._mode(mode)
        return (p["dH"] / M_H2) / 1e6  # J/mol_H2 -> J/kg -> MJ/kg

    def energy_penalty_fraction(self, mode):
        """Reaction enthalpy as a fraction of H2 LHV (energy penalty) [-]."""
        p = self._mode(mode)
        q_per_kg = p["dH"] / M_H2          # J/kg_H2
        return q_per_kg / self.LHV_H2

    # ------------------------------------------------------------------
    # Coupled ODE RHS:  y = [X, T]
    # ------------------------------------------------------------------
    def _rhs(self, t, y, mode, n_carrier, T_set, hA):
        X, T = y
        p = self._mode(mode)
        dXdt = self.conversion_rate(X, T, mode)
        # molar H2 release rate
        dnH2 = n_carrier * p["nu_H2"] * dXdt           # mol/s
        Q_rxn = dnH2 * p["dH"]                          # W absorbed by endotherm
        Q_heat = hA * (T_set - T)                       # W from heater
        dTdt = (Q_heat - Q_rxn) / (self.m_reactor * self.cp_reactor)
        return [dXdt, dTdt]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, mode="lohc", T0=None, T_set=None, n_carrier=None,
                 dt=10.0, duration_s=3600.0, X0=0.0, hA=None):
        """
        Simulate batch dehydrogenation / cracking reactor.

        Parameters
        ----------
        mode : 'lohc' or 'ammonia'
        T0 : float
            Initial reactor temperature [K] (default = setpoint).
        T_set : float
            Heater setpoint temperature [K] (default = literature reactor T).
        n_carrier : float
            Initial moles of hydrogenated carrier [mol].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation time [s].
        X0 : float
            Initial conversion [-].
        hA : float
            Heater conductance [W/K] (default from params).

        Returns
        -------
        dict with arrays: t, conversion, h2_rate_mol_s, h2_rate_kg_s,
            h2_released_kg, temperature, q_heat_W, q_rxn_W, and scalars
            specific_energy_MJ_per_kg, energy_penalty_frac, h2_total_kg.
        """
        p = self._mode(mode)
        if T_set is None:
            T_set = p["T_set"]
        if T0 is None:
            T0 = T_set
        if n_carrier is None:
            n_carrier = self.n_carrier0
        if hA is None:
            hA = self.hA

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [X0, T0],
            t_eval=t_eval, args=(mode, n_carrier, T_set, hA),
            method="LSODA", rtol=1e-8, atol=1e-10, max_step=dt,
        )

        t_out = sol.t
        X_out = np.clip(sol.y[0], 0.0, 1.0)
        T_out = sol.y[1]
        N = len(t_out)

        rate_mol = np.zeros(N)
        q_rxn = np.zeros(N)
        q_heat = np.zeros(N)
        for i in range(N):
            dXdt = self.conversion_rate(X_out[i], T_out[i], mode)
            rate_mol[i] = n_carrier * p["nu_H2"] * dXdt
            q_rxn[i] = rate_mol[i] * p["dH"]
            q_heat[i] = hA * (T_set - T_out[i])

        rate_kg = rate_mol * M_H2
        n_H2_max = n_carrier * p["nu_H2"]
        h2_released_kg = X_out * n_H2_max * M_H2

        return {
            "t": t_out,
            "conversion": X_out,
            "h2_rate_mol_s": rate_mol,
            "h2_rate_kg_s": rate_kg,
            "h2_released_kg": h2_released_kg,
            "temperature": T_out,
            "q_heat_W": q_heat,
            "q_rxn_W": q_rxn,
            "specific_energy_MJ_per_kg": self.specific_energy(mode),
            "energy_penalty_frac": self.energy_penalty_fraction(mode),
            "h2_total_kg": n_H2_max * M_H2,
            "n_H2_max_mol": n_H2_max,
            "mode": mode,
        }
