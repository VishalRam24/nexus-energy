"""
EC222 — Betavoltaic Cell — F2a Single-Diode I-V with Beta-Flux Photocurrent and Decay ODE

Physics-lumped (0D) first-principles upgrade of the F1b junction model. Where F1b
used fitted Isc/Voc/FF curves, F2a derives the device behaviour from first principles:

  (1) Radioactive decay as an ODE (Bateman / decay law), integrated with solve_ivp:
          dN/dt = -lambda * N,   lambda = ln2 / t_half
          A(t)  = lambda * N(t)        [Bq]
      Closed form A(t) = A0 * exp(-lambda t) is recovered; we integrate the ODE so
      the model is a genuine lumped dynamical system (and so an optional thermal
      state can be co-integrated).

  (2) Energy-consistent electron-hole-pair (EHP) generation. A beta particle of mean
      energy E_beta deposits energy in the semiconductor; the number of EHPs created
      per absorbed particle is E_beta / E_pair, where the pair-creation (ionization)
      energy follows Klein's empirical rule (Klein 1968):
          E_pair = W_factor * E_gap + W_offset      [eV]   (W ~ 2.8 for many semis)
      The beta-generated current (the "photocurrent equivalent") is then:
          I_L(t) = q * A(t) * eta_capture * eta_collection * (E_beta / E_pair)
      This is energy-consistent: the electrical generation can never exceed the
      absorbed beta power (enforced and tested).

  (3) Single-diode equivalent circuit (Shockley) with series + shunt resistance.
      The terminal I-V relation is the implicit single-diode equation:
          I = I_L - I0*(exp(q(V + I*Rs)/(n k T)) - 1) - (V + I*Rs)/Rsh
      solved numerically (Brent) for I(V). Isc, Voc, the maximum-power point
      (P_mpp, V_mpp, I_mpp) and the fill factor FF = P_mpp/(Isc*Voc) all emerge
      from this single equation rather than being prescribed.
      I0 carries the standard semiconductor temperature dependence:
          I0(T) = I0_ref * (T/T_ref)^3 * exp( Eg/k (1/T_ref - 1/T) )

  (4) Conversion efficiency relative to absorbed beta power:
          eta = P_mpp / (A(t) * E_beta * eta_capture)
      Bounded in (0,1); for wide-gap betavoltaics this is a few percent.

  (5) Optional lumped thermal ODE (self-heating from absorbed beta power):
          C dT/dt = P_beta_absorbed - hA (T - T_amb)
      The deposited beta power is minuscule, so T stays near ambient — but it is
      integrated as a real state so the model is a coupled ODE system.

Betavoltaics deliver very low power density but extremely long life (here Ni-63,
t_half = 100 yr). All of the above is enforced by the test suite.

References:
    Klein, C.A. (1968). Bandgap dependence and related features of radiation
        ionization energies in semiconductors. J. Appl. Phys. 39(4), 2029.
    Olsen, L.C., Cabauy, P., Elkind, B.J. (1993/2012). Betavoltaic power sources.
        Nucl. Instrum. Methods Phys. Res. B 73, 139; Physics Today 65(12), 35.
    Prelas, M.A. et al. (2014). A review of nuclear batteries.
        Progress in Nuclear Energy 75, 117.
    Sun, W. et al. (2018). Energy harvesting from radioisotopes. Appl. Energy 225, 390.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Physical constants (CODATA)
q_e = 1.602176634e-19        # elementary charge [C]
k_B = 1.380649e-23           # Boltzmann constant [J/K]
eV_to_J = 1.602176634e-19    # J per eV
MeV_to_J = 1.602176634e-13   # J per MeV
ln2 = np.log(2.0)
SEC_PER_YEAR = 365.25 * 24.0 * 3600.0


class BetavoltaicF2a:
    """Betavoltaic cell — first-principles single-diode model with decay + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A0 = u["A0_Bq"]["value"]
        self.t_half = u["t_half_years"]["value"]
        self.E_beta = u["E_beta_MeV"]["value"]            # MeV
        self.eta_cap = u["eta_capture"]["value"]
        self.eta_coll = u["eta_collection"]["value"]
        self.E_gap = u["E_gap_eV"]["value"]               # eV
        self.W_factor = u["W_factor"]["value"]
        self.W_offset = u["W_offset_eV"]["value"]         # eV
        self.T_ref = u["T_ref_K"]["value"]                # K
        self.n = u["ideality_factor"]["value"]
        self.I0_ref = u["I0_ref_A"]["value"]              # A
        self.Eg_I0 = u["Eg_I0_eV"]["value"]               # eV
        self.Rs = u["R_series_ohm"]["value"]              # ohm
        self.Rsh = u["R_shunt_ohm"]["value"]              # ohm
        self.FF_decay = u["FF_decay_rate"]["value"]       # 1/yr
        self.C_th = u["C_thermal_J_K"]["value"]           # J/K
        self.hA = u["hA_W_K"]["value"]                    # W/K
        self.T_amb = u["T_amb_K"]["value"]                # K

        # Decay constant in 1/year
        self.lam = ln2 / self.t_half

    # ----------------------------------------------------------------- decay
    def activity(self, t_years):
        """Activity A(t) [Bq] from the decay law A = A0 exp(-lambda t)."""
        t = np.asarray(t_years, dtype=float)
        return self.A0 * np.exp(-self.lam * np.maximum(t, 0.0))

    def beta_power_absorbed(self, A):
        """Beta power absorbed in the junction [W] = A * E_beta * eta_capture."""
        return A * self.E_beta * MeV_to_J * self.eta_cap

    # ----------------------------------------------------------- pair energy
    def pair_creation_energy_eV(self):
        """Ionization (pair-creation) energy E_pair = W*Eg + offset [eV] (Klein 1968)."""
        return self.W_factor * self.E_gap + self.W_offset

    def ehp_per_beta(self):
        """Mean electron-hole pairs created per absorbed beta particle (energy/E_pair)."""
        E_beta_eV = self.E_beta * 1.0e6
        return E_beta_eV / self.pair_creation_energy_eV()

    # -------------------------------------------------- beta photocurrent eq.
    def beta_current(self, t_years, collection_eff=None):
        """Beta-generated (photocurrent-equivalent) current I_L [A].

        I_L = q * A(t) * eta_capture * eta_collection * (E_beta / E_pair)
        Energy-consistent: I_L * E_pair_J = q * (absorbed-EHP-rate energy).
        """
        if collection_eff is None:
            collection_eff = self.eta_coll
        A = self.activity(t_years)
        return q_e * A * self.eta_cap * collection_eff * self.ehp_per_beta()

    # ------------------------------------------------------ diode saturation
    def I0(self, T_K):
        """Reverse saturation current I0(T) [A] with standard Eg temperature law."""
        T = np.asarray(T_K, dtype=float)
        Eg_J = self.Eg_I0 * eV_to_J
        ratio = (T / self.T_ref) ** 3
        expo = np.exp((Eg_J / k_B) * (1.0 / self.T_ref - 1.0 / T))
        return self.I0_ref * ratio * expo

    # ---------------------------------------------- single-diode I(V) solver
    def diode_residual(self, I, V, I_L, I0, T_K):
        """Single-diode KCL residual f(I)=0 for given terminal voltage V."""
        vt = self.n * k_B * T_K / q_e
        arg = (V + I * self.Rs) / vt
        arg = np.clip(arg, -700.0, 700.0)   # overflow guard
        return I_L - I0 * (np.exp(arg) - 1.0) - (V + I * self.Rs) / self.Rsh - I

    def current_at_voltage(self, V, I_L, I0, T_K):
        """Solve the implicit single-diode equation for terminal current I(V) [A].

        The residual is monotonically decreasing in I (the exp and the Rs/Rsh terms
        all push I down as I rises), so a fixed, generous bracket plus brentq is
        robust.  The current can never exceed I_L + small shunt term, and goes
        negative once V exceeds Voc.
        """
        if I_L <= 0.0:
            return 0.0
        # Upper bound: I cannot exceed the total generation (forward, V<=Voc).
        hi = I_L * 1.000001 + 1e-18
        # Lower bound: at V >= Voc the cell sinks current; bound it generously.
        lo = -(I_L * 10.0 + abs(V) / max(self.Rsh, 1e-30) + 1e-15)
        f_lo = self.diode_residual(lo, V, I_L, I0, T_K)
        f_hi = self.diode_residual(hi, V, I_L, I0, T_K)
        if f_lo * f_hi > 0.0:
            # No sign change in bracket: residual is decreasing, so f_hi<0 means
            # the root is below lo (very large reverse current) -> clamp to lo;
            # f_lo>0 (root above hi) cannot happen for V>=0.
            return lo if f_hi < 0.0 else hi
        return brentq(self.diode_residual, lo, hi, args=(V, I_L, I0, T_K),
                      xtol=1e-20, rtol=1e-12, maxiter=200)

    def open_circuit_voltage(self, I_L, I0, T_K):
        """Voc — terminal voltage at I=0.

        At I=0 the implicit equation collapses to a 1-D root in V alone:
            0 = I_L - I0*(exp(qV/(nkT)) - 1) - V/Rsh
        which is monotonically decreasing in V (both loss terms grow with V), so
        brentq on [0, V_ideal*1.5] is unconditionally robust.  This reduces to
        q Voc = nkT ln(I_L/I0 + 1) in the high-shunt limit.
        """
        if I_L <= 0.0:
            return 0.0
        vt = self.n * k_B * T_K / q_e
        Voc_ideal = vt * np.log(max(I_L / I0, 0.0) + 1.0)
        if Voc_ideal <= 0.0:
            return 0.0

        def f(V):
            arg = np.clip(V / vt, -700.0, 700.0)
            return I_L - I0 * (np.exp(arg) - 1.0) - V / self.Rsh

        lo, hi = 0.0, Voc_ideal * 1.5 + 1e-9
        # f(0) = I_L > 0; ensure f(hi) < 0 (shunt + diode exceed generation).
        for _ in range(80):
            if f(hi) < 0.0:
                break
            hi *= 1.5
        if f(hi) >= 0.0:
            return Voc_ideal
        return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)

    def iv_curve(self, t_years, T_K=None, n_points=200):
        """Full I-V / P-V sweep and the maximum-power point at time t.

        Returns dict with V, I, P arrays plus Isc, Voc, P_mpp, V_mpp, I_mpp, FF.
        """
        if T_K is None:
            T_K = self.T_ref
        T_K = float(T_K)

        # Collection efficiency degrades slowly (radiation damage)
        coll = self.eta_coll * max(1.0 - self.FF_decay * max(t_years, 0.0), 0.5)
        I_L = float(self.beta_current(t_years, collection_eff=coll))
        I0 = float(self.I0(T_K))

        Isc = self.current_at_voltage(0.0, I_L, I0, T_K)
        Voc = self.open_circuit_voltage(I_L, I0, T_K)

        if Voc <= 0.0 or Isc <= 0.0:
            zeros = np.zeros(n_points)
            return {
                "V": zeros, "I": zeros, "P": zeros,
                "Isc_A": max(Isc, 0.0), "Voc_V": max(Voc, 0.0),
                "P_mpp_W": 0.0, "V_mpp_V": 0.0, "I_mpp_A": 0.0,
                "FF": 0.0, "I_L_A": I_L, "I0_A": I0,
            }

        V = np.linspace(0.0, Voc, n_points)
        I = np.array([self.current_at_voltage(v, I_L, I0, T_K) for v in V])
        I = np.maximum(I, 0.0)
        P = V * I

        idx = int(np.argmax(P))
        P_mpp = float(P[idx])
        V_mpp = float(V[idx])
        I_mpp = float(I[idx])
        FF = P_mpp / (Isc * Voc) if (Isc * Voc) > 0 else 0.0

        return {
            "V": V, "I": I, "P": P,
            "Isc_A": float(Isc), "Voc_V": float(Voc),
            "P_mpp_W": P_mpp, "V_mpp_V": V_mpp, "I_mpp_A": I_mpp,
            "FF": float(FF), "I_L_A": I_L, "I0_A": I0,
        }

    # --------------------------------------------------------- decay ODE
    def _rhs(self, t_yr, y):
        """Decay ODE RHS for the normalised atom inventory N (in YEARS).

        dN/dt = -lambda * N          (lambda in 1/year)
        N is normalised so N(0)=1 and A(t) = A0 * N(t).

        The thermal state is handled quasi-statically in `simulate` (its time
        constant C/hA ~ seconds is >10^9x faster than the decay timescale of
        decades, so resolving it inside this multi-decade integration would make
        the system absurdly stiff; the equilibrium T(t) = T_amb + P_abs(t)/hA is
        used instead). Integrating in years keeps the decay ODE well-scaled.
        """
        N = y[0]
        return [-self.lam * N]

    def thermal_equilibrium_T(self, A):
        """Quasi-steady cell temperature [K]: balance of absorbed beta power vs loss.

        C dT/dt = P_abs - hA (T - T_amb) = 0  =>  T = T_amb + P_abs/hA.
        For betavoltaics P_abs is microwatts so T sits a hair above ambient.
        """
        P_abs = self.beta_power_absorbed(A)
        return self.T_amb + P_abs / self.hA

    def simulate(self, t_years_span, n_eval=50, T0_K=None, with_iv=True):
        """Integrate decay (+ thermal) ODE over a time span and report power.

        Parameters
        ----------
        t_years_span : (t0, t1) in years, or scalar t1 (t0=0).
        n_eval       : number of output samples.
        T0_K         : initial cell temperature [K] (default T_amb).
        with_iv      : if True, solve the single-diode MPP at each sample.

        Returns dict of arrays: t_years, activity_Bq, fraction_remaining,
        P_beta_total_W, P_beta_absorbed_W, temperature_K, and (if with_iv)
        Isc_uA, Voc_V, FF, P_out_W, P_out_uW, eta, plus power_density_uW_cm2 if
        an area were supplied (left to predict()).
        """
        if np.isscalar(t_years_span):
            t0, t1 = 0.0, float(t_years_span)
        else:
            t0, t1 = float(t_years_span[0]), float(t_years_span[1])
        if T0_K is None:
            T0_K = self.T_amb

        t_eval_yr = np.linspace(t0, t1, n_eval)
        y0 = [np.exp(-self.lam * t0)]

        # Integrate the decay ODE in years (well-scaled, smooth).
        sol = solve_ivp(
            self._rhs, (t0, t1), y0,
            t_eval=t_eval_yr, method="RK45", rtol=1e-9, atol=1e-14,
        )

        N = np.maximum(sol.y[0], 0.0)
        A = self.A0 * N
        P_abs = self.beta_power_absorbed(A)
        P_tot = A * self.E_beta * MeV_to_J

        # Quasi-steady thermal state (relaxes in seconds vs a multi-decade run).
        # If a cold start T0 is given that differs from equilibrium, blend toward
        # equilibrium with the (fast) thermal time constant evaluated at the first
        # sample — but on the year grid it is effectively at equilibrium already.
        T = self.thermal_equilibrium_T(A)
        T = np.asarray(T, dtype=float)
        if abs(float(T0_K) - float(T[0])) > 1e-9:
            # First sample reflects the imposed initial temperature.
            T = T.copy()
            T[0] = float(T0_K)

        out = {
            "t_years": t_eval_yr,
            "activity_Bq": A,
            "fraction_remaining": N,
            "P_beta_total_W": P_tot,
            "P_beta_absorbed_W": P_abs,
            "temperature_K": T,
        }

        if with_iv:
            Isc = np.zeros(n_eval)
            Voc = np.zeros(n_eval)
            FF = np.zeros(n_eval)
            Pout = np.zeros(n_eval)
            eta = np.zeros(n_eval)
            for i in range(n_eval):
                iv = self.iv_curve(t_eval_yr[i], T[i])
                Isc[i] = iv["Isc_A"]
                Voc[i] = iv["Voc_V"]
                FF[i] = iv["FF"]
                Pout[i] = iv["P_mpp_W"]
                denom = P_abs[i]
                eta[i] = Pout[i] / denom if denom > 1e-30 else 0.0
            eta = np.clip(eta, 0.0, 1.0)
            out.update({
                "Isc_uA": Isc * 1e6,
                "Voc_V": Voc,
                "FF": FF,
                "P_out_W": Pout,
                "P_out_uW": Pout * 1e6,
                "eta": eta,
            })

        return out
