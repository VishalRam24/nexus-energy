"""
EC053 -- Thermophotovoltaic (TPV) -- F2a Spectral Radiative Physics-Lumped Model

Physics-lumped (0D) first-principles TPV model. A hot emitter (Planck blackbody
modulated by a selective emissivity) radiates across a gap to a low-bandgap PV
cell (GaSb, E_g ~ 0.72 eV; InGaAs alternative ~0.74 eV). The model:

  1. Spectral emission. Planck spectral radiance integrated over wavelength,
     split into above-bandgap (lambda < lambda_g = hc/E_g, convertible) and
     sub-bandgap (lambda > lambda_g, lost unless recycled) bands.

        M_lambda(lambda,T) = (2*pi*h*c^2 / lambda^5) / (exp(hc/(lambda*kB*T)) - 1)   [W/m^2/m]

     Integrating Planck over all lambda recovers Stefan-Boltzmann:
        integral_0^inf M_lambda dlambda = sigma * T^4
     so total in-band power scales ~ T^4 (and steeper near the band edge), as
     required physically.

  2. Photocurrent. Each above-bandgap photon (energy >= E_g) can promote one
     carrier. Photon flux above the gap -> photocurrent via EQE:
        Jph = q * EQE * Phi_photon(>E_g)              [A/m^2]
     with Phi_photon = integral_{0}^{lambda_g} (M_lambda / E_photon) dlambda.

  3. Single-diode I-V (low-bandgap cell, dark-current dominated):
        J(V) = Jph - J0(T_cell)*(exp(q(V+J*Rs)/(n*kB*T_cell)) - 1) - (V+J*Rs)/Rsh
     solved implicitly per voltage point; MPP found by sweeping V.
     J0 follows ~ exp(-Eg/(kB*T)) Arrhenius scaling about a reference.

  4. Spectral utilization & recycling efficiency. Sub-bandgap photons are
     reflected by a back-surface reflector / front-side filter and returned to
     the emitter (photon recycling). The radiative spectral efficiency is
        eta_spec = P_inband_useful / P_radiated_net
     where P_radiated_net accounts for the recycled sub-bandgap fraction.
     This makes 0 < eta < 1 with sub-bandgap loss explicitly accounted.

  5. Lumped cell thermal ODE (cell cooling) integrated with scipy.solve_ivp:
        m*cp dT_cell/dt = Q_absorbed - P_elec - Q_cool
        Q_cool = hA*(T_cell - T_coolant)
     Q_absorbed is the radiative heat deposited in the cell (in-band thermalization
     above E_g plus absorbed sub-bandgap that is not recycled).

Physical guarantees enforced by construction / tested:
  * P_elec increases monotonically and ~T_emitter^4-or-steeper with emitter T.
  * P_elec -> 0 as T_emitter -> 0 (no above-bandgap photons; "P=0 below useful T").
  * 0 < eta_spec < 1 and 0 < eta_system < eta_carnot-like ceiling.
  * Sub-bandgap loss is separated out and (partially) recycled.

References:
  Coutts, T.J. (1999). A review of progress in thermophotovoltaic generation
      of electricity. Renewable and Sustainable Energy Reviews, 3, 77-184.
  Bauer, T. (2011). Thermophotovoltaics: Basic Principles and Critical Aspects
      of System Design. Springer.
  LaPotin, A. et al. (2022). Thermophotovoltaic efficiency of 40%.
      Nature, 604, 287-291.
  Wurfel, P. (2009). Physics of Solar Cells, Wiley-VCH (single-diode, detailed
      balance).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Physical constants (CODATA)
H = 6.62607015e-34      # J s   Planck
C = 2.99792458e8        # m/s   speed of light
KB = 1.380649e-23       # J/K   Boltzmann
Q = 1.602176634e-19     # C     elementary charge
SIGMA = 5.670374419e-8  # W/m^2/K^4  Stefan-Boltzmann
EV = 1.602176634e-19    # J per eV


class TPV_F2a:
    """Thermophotovoltaic cell -- spectral radiative + single-diode + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.E_g_eV = u["E_g_eV"]["value"]
        self.E_g = self.E_g_eV * EV                  # J
        self.lambda_g = H * C / self.E_g             # band-edge wavelength [m]
        self.A_cell = u["A_cell"]["value"]           # m^2
        self.T_emitter_ref = u["T_emitter_ref"]["value"]
        self.emissivity = u["emissivity"]["value"]
        self.F_view = u["F_view"]["value"]
        self.EQE = u["EQE"]["value"]
        self.r_sub = u["sub_bandgap_reflectance"]["value"]
        self.J0_ref = u["J0_ref"]["value"]           # A/m^2
        self.n = u["n_ideality"]["value"]
        self.Rs = u["Rs"]["value"]                   # ohm.m^2
        self.Rsh = u["Rsh"]["value"]                 # ohm.m^2
        self.T_cell_ref = u["T_cell_ref"]["value"]
        self.m_cell = u["m_cell"]["value"]
        self.cp_cell = u["cp_cell"]["value"]
        self.hA_cool = u["hA_cool"]["value"]
        self.T_coolant = u["T_coolant"]["value"]

    # ------------------------------------------------------------------
    # Planck spectral radiance / exitance
    # ------------------------------------------------------------------
    @staticmethod
    def planck_exitance(lam, T):
        """Spectral radiant exitance M_lambda [W/m^2/m] (hemispherical, into 2pi)."""
        lam = np.asarray(lam, dtype=float)
        if T <= 0:
            return np.zeros_like(lam)
        x = H * C / (lam * KB * T)
        # guard overflow
        x = np.clip(x, 0, 700.0)
        return (2.0 * np.pi * H * C ** 2) / (lam ** 5) / np.expm1(x)

    def _lambda_grid(self, T, n=400):
        """Wavelength grid spanning the thermal spectrum for emitter temperature T."""
        if T <= 0:
            return np.linspace(1e-7, 1e-5, n)
        # Wien peak lambda_max = b / T ; span ~0.1x .. ~25x peak
        lam_peak = 2.897771955e-3 / T
        return np.linspace(max(lam_peak * 0.05, 1e-8), lam_peak * 30.0, n)

    # ------------------------------------------------------------------
    # Spectral integrals: in-band power and photon flux above the gap
    # ------------------------------------------------------------------
    def spectral_power(self, T_emitter):
        """
        Integrate the (selective) emitter spectrum.

        Returns dict with area-specific (per m^2 of emitter surface) quantities:
            M_total        total radiated exitance [W/m^2]  (~ eps*sigma*T^4)
            M_inband       above-bandgap exitance [W/m^2]
            M_subband      sub-bandgap exitance [W/m^2]
            phi_inband     above-bandgap photon flux [1/m^2/s]
        """
        T = float(T_emitter)
        if T <= 0:
            return {"M_total": 0.0, "M_inband": 0.0, "M_subband": 0.0,
                    "phi_inband": 0.0}

        lam = self._lambda_grid(T)
        M = self.emissivity * self.planck_exitance(lam, T)   # W/m^2/m

        # masks
        in_band = lam <= self.lambda_g
        # photon energy E = hc/lambda
        E_photon = H * C / lam

        M_total = np.trapz(M, lam)
        M_inband = np.trapz(np.where(in_band, M, 0.0), lam)
        M_subband = M_total - M_inband
        # photon flux above the gap: M/E integrated over in-band region
        integrand_phi = np.where(in_band, M / E_photon, 0.0)
        phi_inband = np.trapz(integrand_phi, lam)

        return {"M_total": M_total, "M_inband": M_inband,
                "M_subband": max(M_subband, 0.0), "phi_inband": max(phi_inband, 0.0)}

    # ------------------------------------------------------------------
    # Photocurrent
    # ------------------------------------------------------------------
    def photocurrent_density(self, T_emitter):
        """Area-specific photocurrent density Jph [A/m^2] (per cell area)."""
        sp = self.spectral_power(T_emitter)
        # photon flux reaching the cell scaled by view factor
        phi = sp["phi_inband"] * self.F_view
        Jph = Q * self.EQE * phi
        return Jph

    # ------------------------------------------------------------------
    # Diode saturation current (Arrhenius in cell temperature)
    # ------------------------------------------------------------------
    def J0(self, T_cell):
        """Reverse saturation current density [A/m^2], ~ exp(-Eg/(n kB T))."""
        # detailed-balance-like dark current scaling about the reference T
        exponent = -(self.E_g / (self.n * KB)) * (1.0 / T_cell - 1.0 / self.T_cell_ref)
        exponent = np.clip(exponent, -300.0, 300.0)
        return self.J0_ref * np.exp(exponent)

    # ------------------------------------------------------------------
    # Single-diode I-V (implicit, area-specific)
    # ------------------------------------------------------------------
    def current_density(self, V, T_emitter, T_cell, Jph=None):
        """Solve single-diode equation for current density J(V) [A/m^2]."""
        if Jph is None:
            Jph = self.photocurrent_density(T_emitter)
        J0 = self.J0(T_cell)
        Vt = self.n * KB * T_cell / Q

        def f(J):
            Vd = V + J * self.Rs
            return Jph - J0 * np.expm1(Vd / Vt) - Vd / self.Rsh - J

        # bracket: J in [-|Jph|-margin, Jph]
        lo = -abs(Jph) - 10.0
        hi = Jph + 1e-9
        # ensure sign change; f(lo) should be >0, f(hi) <0 typically
        flo, fhi = f(lo), f(hi)
        if flo == 0:
            return lo
        if fhi == 0:
            return hi
        if np.sign(flo) == np.sign(fhi):
            # widen
            lo = -abs(Jph) - 1e6
            flo = f(lo)
            if np.sign(flo) == np.sign(fhi):
                return 0.0
        return brentq(f, lo, hi, maxiter=200, xtol=1e-12)

    def open_circuit_voltage(self, T_emitter, T_cell, Jph=None):
        """Voc where J=0."""
        if Jph is None:
            Jph = self.photocurrent_density(T_emitter)
        if Jph <= 0:
            return 0.0
        J0 = self.J0(T_cell)
        Vt = self.n * KB * T_cell / Q
        # ignore Rs (J=0) ; include Rsh implicitly small effect -> approximate
        Voc = Vt * np.log(Jph / J0 + 1.0)
        return max(Voc, 0.0)

    # ------------------------------------------------------------------
    # Maximum power point
    # ------------------------------------------------------------------
    def mpp(self, T_emitter, T_cell, n_pts=120):
        """
        Sweep V in [0, Voc] and find the maximum power point.

        Returns dict: Vmp, Jmp, Pmp_density [W/m^2 cell], P_W [W], Voc, Jsc, FF.
        """
        Jph = self.photocurrent_density(T_emitter)
        Voc = self.open_circuit_voltage(T_emitter, T_cell, Jph=Jph)
        if Jph <= 0 or Voc <= 0:
            return {"Vmp": 0.0, "Jmp": 0.0, "Pmp_density": 0.0, "P_W": 0.0,
                    "Voc": 0.0, "Jsc": 0.0, "FF": 0.0}
        Vsweep = np.linspace(0.0, Voc, n_pts)
        Pden = np.empty_like(Vsweep)
        Jvals = np.empty_like(Vsweep)
        for i, V in enumerate(Vsweep):
            J = self.current_density(V, T_emitter, T_cell, Jph=Jph)
            Jvals[i] = J
            Pden[i] = V * J
        imax = int(np.argmax(Pden))
        Vmp, Jmp, Pmp = Vsweep[imax], Jvals[imax], Pden[imax]
        Jsc = self.current_density(0.0, T_emitter, T_cell, Jph=Jph)
        FF = Pmp / (Voc * Jsc) if (Voc * Jsc) > 0 else 0.0
        return {"Vmp": Vmp, "Jmp": Jmp, "Pmp_density": Pmp,
                "P_W": Pmp * self.A_cell, "Voc": Voc, "Jsc": Jsc, "FF": FF}

    # ------------------------------------------------------------------
    # Efficiencies and radiative balance (per cell area)
    # ------------------------------------------------------------------
    def radiative_balance(self, T_emitter, T_cell):
        """
        Area-specific (per cell) radiative powers in W using view factor.

        Returns dict:
            P_inband_W      above-bandgap radiative power reaching cell
            P_subband_W     sub-bandgap radiative power reaching cell
            P_subband_lost  sub-bandgap NOT recycled (absorbed -> heat)
            P_rad_net       net radiated power leaving emitter to cell
                            after sub-bandgap recycling
        """
        sp = self.spectral_power(T_emitter)
        scale = self.A_cell * self.F_view
        P_inband = sp["M_inband"] * scale
        P_subband = sp["M_subband"] * scale
        P_subband_lost = P_subband * (1.0 - self.r_sub)  # absorbed -> heat
        # recycled portion returns to emitter, so net radiated drops
        P_rad_net = P_inband + P_subband_lost
        return {"P_inband_W": P_inband, "P_subband_W": P_subband,
                "P_subband_lost_W": P_subband_lost, "P_rad_net_W": P_rad_net}

    def efficiencies(self, T_emitter, T_cell):
        """Spectral and system efficiencies plus MPP power."""
        rb = self.radiative_balance(T_emitter, T_cell)
        mpp = self.mpp(T_emitter, T_cell)
        P_elec = mpp["P_W"]
        # spectral utilization: in-band fraction of net radiated
        eta_spec = rb["P_inband_W"] / rb["P_rad_net_W"] if rb["P_rad_net_W"] > 0 else 0.0
        # system efficiency: electrical out / net radiative heat supplied
        eta_sys = P_elec / rb["P_rad_net_W"] if rb["P_rad_net_W"] > 0 else 0.0
        return {"eta_spectral": eta_spec, "eta_system": eta_sys,
                "P_elec_W": P_elec, **mpp, **rb}

    # ------------------------------------------------------------------
    # Lumped cell thermal ODE (cell cooling) via solve_ivp
    # ------------------------------------------------------------------
    def _dTdt(self, t, T_cell, T_emit_fn):
        Tc = float(T_cell[0])
        Tc = max(Tc, 1.0)
        T_emit = T_emit_fn(t)
        rb = self.radiative_balance(T_emit, Tc)
        mpp = self.mpp(T_emit, Tc)
        P_elec = mpp["P_W"]
        # heat deposited in cell = in-band thermalization + non-recycled sub-gap
        # absorbed in-band power minus electrical extracted = heat
        Q_absorbed = rb["P_inband_W"] + rb["P_subband_lost_W"]
        Q_cool = self.hA_cool * (Tc - self.T_coolant)
        dT = (Q_absorbed - P_elec - Q_cool) / (self.m_cell * self.cp_cell)
        return [dT]

    def simulate(self, T_emitter, T_cell0=300.0, dt=0.1, duration_s=60.0):
        """
        Integrate the lumped cell thermal ODE while tracking electrical output.

        T_emitter : float OR callable f(t)->K (time-varying emitter).
        Returns dict of time-series arrays.
        """
        if callable(T_emitter):
            T_emit_fn = T_emitter
        else:
            Tval = float(T_emitter)
            T_emit_fn = lambda t: Tval

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._dTdt, (0.0, duration_s), [T_cell0],
            t_eval=t_eval, args=(T_emit_fn,),
            method="RK45", rtol=1e-6, atol=1e-6, max_step=dt,
        )
        T_arr = sol.y[0]
        t_arr = sol.t

        # post-process electrical outputs along the trajectory
        P_elec = np.empty_like(t_arr)
        Vmp = np.empty_like(t_arr)
        eta_sys = np.empty_like(t_arr)
        eta_spec = np.empty_like(t_arr)
        T_emit_arr = np.empty_like(t_arr)
        for i, (ti, Tc) in enumerate(zip(t_arr, T_arr)):
            Te = T_emit_fn(ti)
            T_emit_arr[i] = Te
            eff = self.efficiencies(Te, max(Tc, 1.0))
            P_elec[i] = eff["P_elec_W"]
            Vmp[i] = eff["Vmp"]
            eta_sys[i] = eff["eta_system"]
            eta_spec[i] = eff["eta_spectral"]

        return {
            "t": t_arr,
            "T_cell": T_arr,
            "T_emitter": T_emit_arr,
            "P_elec_W": P_elec,
            "Vmp": Vmp,
            "eta_system": eta_sys,
            "eta_spectral": eta_spec,
        }
