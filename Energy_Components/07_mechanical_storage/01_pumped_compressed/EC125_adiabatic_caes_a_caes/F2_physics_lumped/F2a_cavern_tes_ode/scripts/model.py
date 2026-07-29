"""
EC125 — Adiabatic Compressed Air Energy Storage (A-CAES) — F2a Physics-Lumped

Coupled cavern + thermal-energy-store (TES) energy-balance ODE model, integrated
with scipy.integrate.solve_ivp. This is the first-principles (0D lumped) upgrade of
the EC125 F1 semi-empirical round-trip / thermal models.

Key A-CAES physics (vs diabatic CAES):
---------------------------------------
* CHARGE  : air is compressed (multi-stage). The compression heat is NOT rejected to
            atmosphere — it is captured into a sensible-heat TES (concrete/rock bed).
            The cooled air is stored in the cavern.
* DISCHARGE: cavern air is RE-HEATED by the stored TES heat (NO FUEL COMBUSTION —
            the defining A-CAES advantage), then expanded through the turbine train.
* Because the compression heat is recycled, A-CAES achieves RTE ~0.65-0.75, strictly
  above diabatic CAES (Huntorf 0.42 / McIntosh 0.54, which burn natural gas on
  expansion).

State vector  y = [m_air, T_cav, U_tes]
---------------------------------------
  m_air  : air mass in cavern                              [kg]
  T_cav  : bulk cavern-air temperature                     [K]
  U_tes  : TES internal (sensible) energy above ambient    [J]
           ->  T_tes = T_tes_ambient + U_tes / Cm_tes

Governing ODEs (lumped energy/mass balance)
-------------------------------------------
Mass balance (cavern):
    dm_air/dt = +m_dot              (charge)
              = -m_dot              (discharge)

Cavern-air energy balance (first law, open system, ideal gas, cv basis):
    d(m_air·cv·T_cav)/dt = cp·T_in·m_dot_in            (enthalpy of incoming air)
                         - cp·T_cav·m_dot_out          (enthalpy of leaving air)
                         - UA_cav·(T_cav - T_rock)     (wall heat loss to rock)
  expanded with the product rule for dT_cav/dt.
  On CHARGE the inlet air has already been cooled by the TES-charging heat exchanger,
  so T_in ≈ T_rock (near-isothermal storage). On DISCHARGE air leaves the cavern.

TES energy balance (sensible store, Newton coupling to ambient):
    dU_tes/dt = +Q_dot_store        (charge: compression heat captured, ×eta_tes)
              = -Q_dot_release       (discharge: heat delivered to re-heat expander air, /eta_tes)
              - UA_tes·(T_tes - T_amb)            (standby insulation loss)

References
----------
  ADELE project, RWE Power / GE / DLR / Zunft S. (2009-2013), "Adiabatic CAES for
      electricity supply", 350 MWh demonstrator concept.
  Wolf, D. & Budt, M. (2014). LTA-CAES — a low-temperature approach to Adiabatic
      Compressed Air Energy Storage. Applied Energy, 125, 158-164.
  Budt, M., Wolf, D., Span, R., Yan, J. (2016). A review on compressed air energy
      storage: Basic principles, past milestones and recent developments.
      Applied Energy, 170, 250-268.
  Barbour, E., Mignard, D., Ding, Y., Li, Y. (2015). Adiabatic Compressed Air Energy
      Storage with packed bed thermal energy storage. Applied Energy, and
      Renewable & Sustainable Energy Reviews, 45, 598-614.
  Air properties: Lemmon et al. (2000) J.Phys.Chem.Ref.Data 29:331; Cengel & Boles,
      Thermodynamics 8th ed., Table A-2.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ACAES_F2a:
    """Adiabatic CAES — coupled cavern + TES energy-balance ODE model."""

    def __init__(self, params: dict):
        u = params["unit"]

        # --- Air properties (hardcoded, cited) ---
        self.R_air   = u["R_air"]["value"]      # J/(kg·K)
        self.cp_air  = u["cp_air"]["value"]     # J/(kg·K)
        self.cv_air  = u["cv_air"]["value"]     # J/(kg·K)
        self.gamma   = u["gamma_air"]["value"]  # -

        # --- TES (sensible-heat store) ---
        self.cp_tes        = u["cp_tes"]["value"]         # J/(kg·K)
        self.m_tes         = u["m_tes"]["value"]          # kg
        self.Cm_tes        = self.m_tes * self.cp_tes     # J/K
        self.T_tes_design  = u["T_tes_design"]["value"]   # K
        self.T_tes_ambient = u["T_tes_ambient"]["value"]  # K
        self.UA_tes        = u["UA_tes"]["value"]         # W/K
        self.eta_tes       = u["eta_tes"]["value"]        # -
        self.tau_tes       = self.Cm_tes / self.UA_tes    # s

        # --- Cavern ---
        self.V          = u["cavern_volume"]["value"]       # m3
        self.p_max      = u["p_max"]["value"]               # Pa
        self.p_min      = u["p_min"]["value"]               # Pa
        self.T_rock     = u["T_rock"]["value"]              # K
        self.T_cav_nom  = u["T_cavern_nominal"]["value"]    # K
        self.UA_cav     = u["UA_cavern_rock"]["value"]      # W/K
        self.Cm_cav     = u["cavern_thermal_mass"]["value"] # J/K (informational)
        self.tau_cav    = self.Cm_cav / self.UA_cav         # s

        # --- Machinery ---
        self.eta_comp  = u["eta_compressor"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.eta_exp   = u["eta_expander"]["value"]
        self.eta_gen   = u["eta_generator"]["value"]
        self.w_comp_ref = u["specific_work"]["value"]                  # kJ/kg
        self.w_exp_ref  = u["specific_expansion_adiabatic"]["value"]   # kJ/kg
        self.T_ref_comp = u["T_ref_comp"]["value"]                     # K
        self.k_comp_T   = u["k_comp_T"]["value"]                       # 1/K

        # --- Diabatic reference (for RTE > diabatic constraint) ---
        self.rte_diabatic_ref = u["rte_diabatic_reference"]["value"]

        # --- Derived cavern mass bounds (at nominal T) ---
        self.m_max = self.p_max * self.V / (self.R_air * self.T_cav_nom)
        self.m_min = self.p_min * self.V / (self.R_air * self.T_cav_nom)
        self.m_usable = self.m_max - self.m_min

    # ------------------------------------------------------------------
    # Algebraic helpers
    # ------------------------------------------------------------------
    def cavern_pressure(self, m_air, T_cav):
        """Ideal-gas cavern pressure [Pa] from air mass and temperature."""
        return m_air * self.R_air * T_cav / self.V

    def soc_from_mass(self, m_air):
        """State of charge in [0,1] from air mass."""
        return float(np.clip((m_air - self.m_min) / self.m_usable, 0.0, 1.0))

    def mass_from_soc(self, soc):
        """Air mass [kg] from SOC."""
        s = float(np.clip(soc, 0.0, 1.0))
        return self.m_min + s * self.m_usable

    def tes_temperature(self, U_tes):
        """TES temperature [K] from stored sensible energy above ambient."""
        return self.T_tes_ambient + np.asarray(U_tes, dtype=float) / self.Cm_tes

    def tes_energy_from_temperature(self, T_tes):
        """TES sensible energy above ambient [J] for a given temperature."""
        return (np.asarray(T_tes, dtype=float) - self.T_tes_ambient) * self.Cm_tes

    def specific_compression_work(self, T_amb_K=None):
        """Compressor specific work [kJ/kg], corrected for intake temperature."""
        if T_amb_K is None:
            return self.w_comp_ref
        T = np.asarray(T_amb_K, dtype=float)
        return self.w_comp_ref * (1.0 + self.k_comp_T * (T - self.T_ref_comp))

    def tes_heat_available_fraction(self, T_tes_K):
        """
        Fraction (0..1) of design re-heat available given current TES temperature.
        Linear in the sensible-energy ratio above ambient, capped at 1.
        f = (T_tes - T_ambient) / (T_tes_design - T_ambient)
        """
        T = np.asarray(T_tes_K, dtype=float)
        num = T - self.T_tes_ambient
        den = self.T_tes_design - self.T_tes_ambient
        return np.clip(num / den, 0.0, 1.0)

    def expansion_work_effective(self, T_tes_K=None):
        """Effective expander specific work [kJ/kg] given TES re-heat temperature."""
        if T_tes_K is None:
            return self.w_exp_ref
        return self.w_exp_ref * self.tes_heat_available_fraction(T_tes_K)

    def fuel_power(self, p_elec_out=0.0):
        """Fuel thermal input [kW] — ALWAYS ZERO for A-CAES (no combustion)."""
        return np.zeros_like(np.asarray(p_elec_out, dtype=float))

    # ------------------------------------------------------------------
    # Round-trip efficiency (design / instantaneous)
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, T_amb_K=None, T_tes_K=None):
        """
        A-CAES RTE = E_out / E_in.
        E_in  = w_comp(T_amb) / (eta_comp·eta_motor)
        E_out = w_exp_eff(T_tes) · eta_exp · eta_gen
        Physical band 0.65-0.75 at/near design; must exceed diabatic CAES.
        """
        w_c = self.specific_compression_work(T_amb_K)
        w_e = self.expansion_work_effective(T_tes_K)
        E_out = w_e * self.eta_exp * self.eta_gen / 3600.0       # kWh/kg
        E_in  = w_c / (self.eta_comp * self.eta_motor) / 3600.0  # kWh/kg
        return float(E_out / E_in)

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, mode, m_dot, T_amb):
        """
        y = [m_air, T_cav, U_tes]
        mode: 'charge' | 'discharge' | 'idle'
        m_dot: air mass-flow magnitude [kg/s]
        T_amb: ambient/intake temperature [K]
        """
        m_air, T_cav, U_tes = y
        m_air = max(m_air, 1.0e-6)
        T_tes = self.T_tes_ambient + U_tes / self.Cm_tes

        # default: no flow
        dm = 0.0
        dU_tes = 0.0
        # cavern wall loss to rock (always active)
        Q_wall = self.UA_cav * (T_cav - self.T_rock)        # W (positive = cavern loses heat)
        # cavern enthalpy flux term placeholder
        enthalpy_flux = 0.0  # W into cavern air

        if mode == "charge" and m_dot > 0.0:
            # mass in; air cooled by TES-charging HX -> enters near rock temperature
            dm = +m_dot
            T_in = self.T_rock
            enthalpy_flux = self.cp_air * T_in * m_dot       # W carried in by inflow
            # SENSIBLE compression heat captured by the inter/after-coolers into the
            # TES: per kg ~ cp_air * (T_tes_design - T_rock) (the air is cooled from
            # the hot compressor-train outlet down to near storage temperature, and
            # that heat is what charges the regenerator). times TES effectiveness.
            dT_comp = max(self.T_tes_design - self.T_rock, 0.0)
            Q_store = m_dot * self.cp_air * dT_comp * self.eta_tes   # W into TES
            # Once the TES reaches its design temperature it is "full": further
            # compression heat is rejected by the HX rather than over-heating the
            # store (smooth ramp-down over the top 5 K to keep the ODE continuous).
            fill = np.clip((self.T_tes_design - T_tes) / 5.0, 0.0, 1.0)
            dU_tes = +Q_store * fill

        elif mode == "discharge" and m_dot > 0.0:
            # mass out; leaving air carries its enthalpy out of the cavern
            dm = -m_dot
            enthalpy_flux = -self.cp_air * T_cav * m_dot     # W (leaving)
            # SENSIBLE heat drawn from the TES to re-heat the expander air (no fuel).
            # Per kg this is cp_air * (T_tes - T_cav): the regenerator pre-heats the
            # cavern air toward the TES temperature before it enters the expander.
            # (This is the heat the store gives up — distinct from the expander
            #  shaft work w_exp, which also draws on the air's pressure exergy.)
            dT_reheat = max(T_tes - T_cav, 0.0)
            Q_release = m_dot * self.cp_air * dT_reheat / self.eta_tes   # W from TES
            dU_tes = -Q_release

        # Cavern-air energy balance:
        #   d(m·cv·T)/dt = enthalpy_flux - Q_wall
        #   m·cv·dT/dt + cv·T·dm/dt = enthalpy_flux - Q_wall
        denom = m_air * self.cv_air
        dT_cav = (enthalpy_flux - Q_wall - self.cv_air * T_cav * dm) / denom

        # TES standby insulation loss to ambient (always active)
        dU_tes -= self.UA_tes * (T_tes - self.T_tes_ambient)

        return [dm, dT_cav, dU_tes]

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------
    def simulate(self, mode="charge", m_dot=100.0, duration_s=3600.0, dt=60.0,
                 soc0=0.5, T_cav0=None, T_tes0=None, T_amb=288.15):
        """
        Integrate the coupled cavern+TES ODEs over `duration_s` with scipy.solve_ivp.

        Parameters
        ----------
        mode       : 'charge' | 'discharge' | 'idle'
        m_dot      : air mass-flow magnitude [kg/s]
        duration_s : simulation horizon [s]
        dt         : output sampling interval [s]
        soc0       : initial state of charge [0,1]
        T_cav0     : initial cavern temperature [K] (default nominal)
        T_tes0     : initial TES temperature [K]   (default design)
        T_amb      : ambient/intake temperature [K]

        Returns dict of time-series arrays + scalar diagnostics.
        """
        if T_cav0 is None:
            T_cav0 = self.T_cav_nom
        if T_tes0 is None:
            T_tes0 = self.T_tes_design

        m0 = self.mass_from_soc(soc0)
        U0 = self.tes_energy_from_temperature(T_tes0)
        y0 = [m0, float(T_cav0), float(U0)]

        n = max(int(round(duration_s / dt)) + 1, 2)
        t_eval = np.linspace(0.0, duration_s, n)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45",
            args=(mode, float(m_dot), float(T_amb)),
            rtol=1e-7, atol=1e-3, max_step=dt,
        )

        m_air = sol.y[0]
        T_cav = sol.y[1]
        U_tes = sol.y[2]
        T_tes = self.tes_temperature(U_tes)
        soc   = np.clip((m_air - self.m_min) / self.m_usable, 0.0, 1.0)
        p_cav = self.cavern_pressure(m_air, T_cav)

        # Electrical power flows (kW). Sign convention: charge=input, discharge=output.
        if mode == "charge":
            w_c = self.specific_compression_work(T_amb)          # kJ/kg
            p_elec = m_dot * w_c / (self.eta_comp * self.eta_motor)   # kW (input)
        elif mode == "discharge":
            w_e = self.expansion_work_effective(T_tes)           # kJ/kg array
            p_elec = m_dot * w_e * self.eta_exp * self.eta_gen   # kW (output)
        else:
            p_elec = np.zeros_like(sol.t)

        p_elec = np.atleast_1d(np.asarray(p_elec, dtype=float)) * np.ones_like(sol.t)
        fuel = self.fuel_power(p_elec)   # always zero

        # Energy integrals over horizon (kWh)
        E_elec = np.trapz(p_elec, sol.t) / 3600.0
        # TES energy change (kWh)
        dU_tes_kwh = (U_tes[-1] - U_tes[0]) / 3.6e6

        return {
            "t": sol.t,
            "m_air": m_air,
            "soc": soc,
            "T_cav": T_cav,
            "T_tes": T_tes,
            "U_tes": U_tes,
            "pressure": p_cav,
            "power_elec_kw": p_elec,
            "fuel_power_kw": fuel,
            "mode": mode,
            "E_elec_kwh": float(E_elec),
            "dU_tes_kwh": float(dU_tes_kwh),
            "rte_design": self.round_trip_efficiency(T_amb, T_tes[-1]),
            "solver_success": bool(sol.success),
        }

    # ------------------------------------------------------------------
    # Full round-trip (charge -> idle? -> discharge) RTE via ODE energy
    # ------------------------------------------------------------------
    def round_trip_simulation(self, m_dot=100.0, T_amb=288.15,
                              charge_s=None, discharge_s=None, dt=60.0):
        """
        Simulate a full charge phase followed by a full discharge phase and
        compute the round-trip efficiency from the integrated electrical energies.

        Represents a cyclic-steady-state operating round-trip about the design
        point: the TES sits at its design charge temperature (reached after the
        first few cycles, when per-cycle compression heat in ≈ re-heat out), and
        the SAME air mass that was stored on charge is released on discharge
        (equal mass throughput). The integrated RTE recovers the per-kg design
        RTE less the cavern/TES standby losses incurred over the cycle.

        Returns (rte, charge_result, discharge_result).
        """
        # default: store ~20% of the usable cavern mass per cycle (representative
        # daily-cycling depth), keeping the large TES close to its design state.
        if charge_s is None:
            charge_s = 0.20 * self.m_usable / max(m_dot, 1e-9)
        # TES seeded at design (cyclic-steady-state); compression heat tops it up.
        ch = self.simulate("charge", m_dot, charge_s, dt,
                           soc0=0.0, T_tes0=self.T_tes_design, T_amb=T_amb)
        soc_after   = ch["soc"][-1]
        T_tes_after = min(ch["T_tes"][-1], self.T_tes_design)  # cap at design (excess heat dumped)
        T_cav_after = ch["T_cav"][-1]

        if discharge_s is None:
            m_charged = ch["m_air"][-1] - ch["m_air"][0]
            discharge_s = max(m_charged / max(m_dot, 1e-9), dt)

        dis = self.simulate("discharge", m_dot, discharge_s, dt,
                            soc0=soc_after, T_cav0=T_cav_after,
                            T_tes0=T_tes_after, T_amb=T_amb)
        E_in = ch["E_elec_kwh"]
        E_out = dis["E_elec_kwh"]
        rte = E_out / E_in if E_in > 0 else 0.0
        return float(rte), ch, dis
