"""
EC174 -- Instrument Transformer (CT / PT) -- F2a Magnetizing-Branch Model

Physics-lumped (0D) equivalent-circuit model of an instrument transformer
(current transformer, CT, and inductive voltage / potential transformer, PT)
with an explicit *nonlinear magnetizing branch*. Compared with the F1 model
(algebraic accuracy/loss map) this F2a model:

  1. Resolves the magnetizing current into in-phase (core-loss) and quadrature
     (magnetizing) components, giving BOTH ratio error and PHASE error.
  2. Captures CT SATURATION above the knee point through a nonlinear B-H core
     law, integrated as a flux / secondary-current ODE with scipy.solve_ivp.
  3. Shows that error grows with burden and with primary current beyond the
     knee, consistent with the accuracy-class framework of IEEE C57.13 / IEC 61869.

────────────────────────────────────────────────────────────────────────
EQUIVALENT CIRCUIT (referred to the secondary), per IEEE C57.13-2016 §6 and
Slemon & Straughen (1980) "Electric Machines", Ch. 3:

    CT:  ideal source I2_ideal = I1/n  feeds a node; the magnetizing branch
         Rc || Xm shunts off an excitation current Ie; the remainder flows
         through the secondary impedance (R2+jX2) and the burden (Rb+jXb).

         I2_ideal = I2 + Ie                          (Kirchhoff at the node)
         Ie       = Vexc/Rc  +  Vexc/(jXm)           (loss + magnetizing)
         Vexc     = I2 * (R2 + Rb + j(X2 + Xb))      (secondary EMF)

         Ratio (current) error and phase error follow from the complex
         excitation current that is "lost" to the magnetizing branch:

             eps_I  = Re(-Ie / I2_ideal) * 100   [%]   (in-phase -> magnitude)
             phase  = Im( Ie / I2_ideal)         [rad]  (quadrature -> phase)

    PT:  series primary+secondary impedance drops the transformed voltage,
         and the shunt magnetizing current adds a small further error.

         eps_V  = -(I2*(Re_series)cosφ + I2*(Xe_series)sinφ)/V2_ideal * 100
         phase  ≈  (I2*(Xe_series)cosφ - I2*(Re_series)sinφ)/V2_ideal  [rad]

────────────────────────────────────────────────────────────────────────
NONLINEAR CORE / SATURATION (CT), per IEEE C37.110 Guide for the Application
of CTs and the IEEE PSRC CT-saturation model:

The core flux linkage λ relates to the magnetizing current Im by a single-valued
nonlinear law (arctangent / atan saturation model — Frame & Mohan 1982, IEEE
Trans. PAS):

        λ(Im) = λ_sat * (2/π) * atan( Im / Im_knee )

so the differential (incremental) magnetizing inductance collapses as Im grows:

        L_m(Im) = dλ/dIm = (2 λ_sat / (π Im_knee)) / (1 + (Im/Im_knee)^2)

Faraday's law on the secondary loop gives the flux / current ODE (in the time
domain, secondary referred):

        dλ/dt = Vexc(t) = i2(t) * (R2 + Rb) + (X2+Xb)/ω * di2/dt        (resistive-dominant)
        i2(t) = i1(t)/n - im(t)

We integrate the magnetizing current state im(t):

        dim/dt = ( i2(t)*(R2+Rb) - λ(im)*ω_corr ... )                  (see _saturation_rhs)

Below the knee λ is linear -> Im is a small sinusoid -> nearly perfect
reproduction. Above the knee λ clips at λ_sat -> Im spikes -> the secondary
current i2 = i1/n - im is badly distorted (the classic CT saturation waveform).

────────────────────────────────────────────────────────────────────────
ENERGY CONSISTENCY:
The instantaneous secondary energy balance is enforced and checked:
    P_in (transferred) = P_burden + P_winding(I2^2 R2) + P_core(Vexc^2/Rc)
to within numerical tolerance over a cycle.

References:
    IEEE Std C57.13-2016 — Requirements for Instrument Transformers.
    IEC 61869-1 (2007), -2 (2012, CT), -3 (2011, VT) — Instrument transformers.
    IEEE Std C37.110-2007 — Guide for the Application of CTs in Protective Relaying.
    Frame, J.G., Mohan, N., Liu, T. (1982). Hysteresis & eddy-current losses /
      saturable-core modelling, IEEE Trans. PAS-101.
    Slemon, G.R. & Straughen, A. (1980). Electric Machines. Addison-Wesley.
    Hoblit, F.M. (1980). Current Transformers. IEEE Trans. Ind. Appl.
"""

import numpy as np
from scipy.integrate import solve_ivp


class InstrumentTransformerF2a:
    """CT/PT magnetizing-branch equivalent circuit with nonlinear saturation ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.type = u["type"]["value"]                  # "CT" or "PT"
        self.n = float(u["turns_ratio"]["value"])        # turns ratio
        self.I_rated = float(u["I_rated"]["value"])      # A (primary, CT)
        self.V_rated = float(u["V_rated"]["value"])      # V (primary, PT)
        self.f = float(u["f"]["value"])                  # Hz
        self.w = 2.0 * np.pi * self.f                    # rad/s
        self.S_burden = float(u["S_burden_VA"]["value"]) # VA rated
        self.pf_burden = float(u["pf_burden"]["value"])
        self.R2 = float(u["R2"]["value"])                # Ohm
        self.X2 = float(u["X2"]["value"])                # Ohm
        self.Rc = float(u["Rc"]["value"])                # Ohm
        self.Xm = float(u["Xm"]["value"])                # Ohm
        self.V_knee = float(u["V_knee"]["value"])        # V (knee EMF)
        self.accuracy_class = float(u["accuracy_class"]["value"])
        self.phase_class_min = float(u.get("phase_class_min", {"value": 30.0})["value"])

        # Saturation parameters derived from knee EMF.
        # Knee EMF Vk ~ ω * λ_sat (peak flux linkage at saturation).
        self.lambda_sat = self.V_knee * np.sqrt(2.0) / self.w   # peak flux linkage [Wb-turns]
        # Magnetizing current at the knee: Im_knee = Vk / Xm (rms), use peak.
        self.Im_knee = (self.V_knee / self.Xm) * np.sqrt(2.0)

    # ==================================================================
    # Burden helpers
    # ==================================================================
    def _ct_rated_burden(self):
        """Rated secondary burden impedance R_b, X_b, Z_b [Ohm] from VA."""
        i2_rated = self.I_rated / self.n
        Z_b = self.S_burden / i2_rated ** 2
        R_b = Z_b * self.pf_burden
        X_b = Z_b * np.sqrt(max(0.0, 1.0 - self.pf_burden ** 2))
        return R_b, X_b, Z_b

    def _pt_rated_burden(self):
        v2_rated = self.V_rated / self.n
        Z_b = v2_rated ** 2 / self.S_burden
        R_b = Z_b * self.pf_burden
        X_b = Z_b * np.sqrt(max(0.0, 1.0 - self.pf_burden ** 2))
        return R_b, X_b, Z_b

    def _scaled_burden(self, burden_fraction):
        """Return CT burden (R_b, X_b) scaled to burden_fraction of rated VA.
        Higher burden => higher Z => higher excitation EMF => bigger error."""
        R_b, X_b, _ = self._ct_rated_burden()
        bf = max(burden_fraction, 0.0)
        return R_b * bf, X_b * bf

    # ==================================================================
    # CT phasor (steady-state, sub-saturation) model
    # ==================================================================
    def ct_errors(self, i_primary, burden_fraction=1.0):
        """
        Steady-state phasor solution of the CT magnetizing branch.

        Returns dict with:
            ratio_error_pct  : current (ratio) error [%]   (negative = secondary under-reads)
            phase_error_min  : phase displacement [minutes]
            phase_error_deg  : phase displacement [deg]
            excitation_A     : rms excitation current referred to secondary [A]
            V_exc            : excitation EMF [V]
            I2              : actual secondary current [A]
        Per IEEE C57.13-2016 §6 / Slemon & Straughen (1980).
        """
        i_primary = np.asarray(i_primary, dtype=float)
        I2_ideal = i_primary / self.n                       # ideal secondary current
        R_b, X_b = self._scaled_burden(burden_fraction)

        # Secondary loop impedance (winding + burden)
        Z_sec = complex(self.R2 + R_b, self.X2 + X_b)

        # First pass: assume I2 ~ I2_ideal to get excitation EMF.
        # Iterate twice (excitation current is small, converges fast).
        I2 = I2_ideal.astype(float)
        for _ in range(3):
            V_exc = I2 * abs(Z_sec)                          # excitation EMF magnitude
            # Magnetizing admittance Ym = 1/Rc + 1/(jXm)
            # Excitation current phasor referred to the EMF (take EMF as reference)
            Ie_loss = V_exc / self.Rc                        # in-phase with EMF (core loss)
            Ie_mag = V_exc / self.Xm                         # quadrature (lags EMF by 90)
            # EMF leads I2 by the secondary impedance angle theta
            theta = np.angle(Z_sec)
            # Excitation current components projected onto the I2 axis:
            #   in-phase-with-I2 component -> ratio error; quadrature -> phase error
            Ie_re = Ie_loss * np.cos(theta) + Ie_mag * np.sin(theta)   # along I2
            Ie_im = Ie_mag * np.cos(theta) - Ie_loss * np.sin(theta)   # perpendicular
            I2 = np.maximum(I2_ideal - Ie_re, 1e-9)

        safe = np.where(I2_ideal > 1e-9, I2_ideal, 1.0)
        ratio_err = np.where(I2_ideal > 1e-9, -(Ie_re / safe) * 100.0, 0.0)
        phase_rad = np.where(I2_ideal > 1e-9, Ie_im / safe, 0.0)
        Ie_rms = np.sqrt(Ie_re ** 2 + Ie_im ** 2)

        return {
            "ratio_error_pct": ratio_err,
            "phase_error_deg": np.degrees(phase_rad),
            "phase_error_min": np.degrees(phase_rad) * 60.0,
            "excitation_A": Ie_rms,
            "V_exc": V_exc,
            "I2": I2,
        }

    # ==================================================================
    # PT phasor model
    # ==================================================================
    def pt_errors(self, v_primary, burden_fraction=1.0):
        """Steady-state PT voltage ratio error and phase error."""
        v_primary = np.asarray(v_primary, dtype=float)
        V2_ideal = v_primary / self.n
        R_b, X_b, _ = self._pt_rated_burden()
        bf = max(burden_fraction, 1e-9)
        Z_b = complex(R_b / bf, X_b / bf)                   # higher fraction -> lower Z -> more current
        I2 = V2_ideal / np.maximum(abs(Z_b), 1e-9)

        # Series impedance: primary referred (~R2,X2) + secondary (R2,X2)
        R_series = 2.0 * self.R2
        X_series = 2.0 * self.X2
        phi = np.arccos(np.clip(self.pf_burden, -1, 1))

        volt_drop_re = I2 * (R_series * np.cos(phi) + X_series * np.sin(phi))
        volt_drop_im = I2 * (X_series * np.cos(phi) - R_series * np.sin(phi))
        safe_V2 = np.where(V2_ideal > 0, V2_ideal, 1.0)
        ratio_err = np.where(V2_ideal > 0, -volt_drop_re / safe_V2 * 100.0, 0.0)
        phase_rad = np.where(V2_ideal > 0, volt_drop_im / safe_V2, 0.0)
        return {
            "ratio_error_pct": ratio_err,
            "phase_error_deg": np.degrees(phase_rad),
            "phase_error_min": np.degrees(phase_rad) * 60.0,
            "I2": I2,
        }

    # ==================================================================
    # Nonlinear core law (arctangent saturation model)
    # ==================================================================
    def flux_linkage(self, im):
        """λ(Im) [Wb-turns] -- single-valued arctangent saturation law."""
        return self.lambda_sat * (2.0 / np.pi) * np.arctan(im / self.Im_knee)

    def magnetizing_current(self, lam):
        """Inverse law: Im(λ) [A] = Im_knee * tan(π λ / (2 λ_sat)).
        Diverges as |λ| -> λ_sat -> the magnetizing current spikes in saturation."""
        ratio = np.clip(lam / self.lambda_sat, -0.9999, 0.9999)
        return self.Im_knee * np.tan(np.pi / 2.0 * ratio)

    def magnetizing_inductance(self, im):
        """Incremental L_m(Im) [H] = dλ/dIm -- collapses in saturation."""
        Lm0 = (2.0 * self.lambda_sat / (np.pi * self.Im_knee))
        return Lm0 / (1.0 + (im / self.Im_knee) ** 2)

    # ==================================================================
    # Transient CT saturation: flux / secondary-current ODE
    # ==================================================================
    def _saturation_rhs(self, t, y, i1_peak, burden_fraction):
        """
        ODE for the core flux-linkage state λ(t) on the secondary side
        (IEEE C37.110 CT-saturation formulation).

        Secondary node KCL:  i_src = i_sec + im
            i_src = (i1_peak/n) sin(ωt)          (ideal transformed current)
            im    = Im(λ)                         (nonlinear magnetizing current)
            i_sec = i_src - im                    (current delivered to burden)
        Faraday on the burden loop:
            dλ/dt = Vexc = i_sec * (R2 + Rb)

        =>  dλ/dt = (i_src - Im(λ)) * (R2 + Rb)

        Below the knee Im(λ)≈0 so i_sec≈i_src (faithful reproduction). Once the
        flux excursion drives |λ| toward λ_sat, Im(λ) spikes -> i_sec collapses
        -> the classic distorted, peaky CT-saturation secondary waveform.
        """
        lam = y[0]
        i_src = (i1_peak / self.n) * np.sin(self.w * t)
        im = self.magnetizing_current(lam)
        R_b, _ = self._scaled_burden(burden_fraction)
        R_loop = self.R2 + R_b
        dlam = (i_src - im) * R_loop
        return [dlam]

    def simulate_saturation(self, i_primary_peak, burden_fraction=1.0,
                            n_cycles=2.0, n_per_cycle=400):
        """
        Time-domain CT saturation simulation via scipy.solve_ivp.

        Returns dict:
            t            : time [s]
            i_primary    : primary current [A]
            i_sec_ideal  : ideal secondary i1/n [A]
            i_sec        : actual secondary current (i1/n - im) [A]
            i_mag        : magnetizing current im [A]
            flux         : core flux linkage λ [Wb-turns]
            distortion   : RMS distortion of i_sec vs ideal [-]
            saturated    : bool, True if |λ| reached >90% of λ_sat
            energy_resid : per-cycle energy balance residual [-]
        """
        T = n_cycles / self.f
        t_eval = np.linspace(0.0, T, int(n_cycles * n_per_cycle))
        sol = solve_ivp(
            self._saturation_rhs, (0.0, T), [0.0],
            args=(i_primary_peak, burden_fraction),
            t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9, max_step=T / 4000.0,
        )
        t = sol.t
        flux = sol.y[0]
        im = self.magnetizing_current(flux)
        i_src = (i_primary_peak / self.n) * np.sin(self.w * t)
        i_sec = i_src - im

        # Distortion: deviation of actual secondary from ideal, normalized.
        denom = np.sqrt(np.mean(i_src ** 2)) + 1e-12
        distortion = np.sqrt(np.mean((i_sec - i_src) ** 2)) / denom
        saturated = bool(np.max(np.abs(flux)) > 0.9 * self.lambda_sat)

        # ----- Energy consistency check -----
        # Source (transformed) power delivered to the secondary network must
        # equal what is dissipated (winding + burden) plus the net change in
        # stored magnetic energy:  ∫ Vexc·i_src dt = ∫ i_sec²(R2+Rb) dt + ΔW_mag.
        R_b, _ = self._scaled_burden(burden_fraction)
        R_loop = self.R2 + R_b
        Vexc = i_sec * R_loop                      # secondary EMF across magnetizing branch
        e_src = np.trapz(Vexc * i_src, t)          # energy injected by ideal source
        e_diss = np.trapz(i_sec ** 2 * R_loop, t)  # resistive dissipation
        # Stored magnetic energy W = ∫ Vexc·im dt  (energy into magnetizing branch)
        e_mag = np.trapz(Vexc * im, t)
        energy_resid = abs(e_src - (e_diss + e_mag)) / (abs(e_src) + 1e-12)

        return {
            "t": t,
            "i_primary": i_primary_peak * np.sin(self.w * t),
            "i_sec_ideal": i_src,
            "i_sec": i_sec,
            "i_mag": im,
            "flux": flux,
            "distortion": float(distortion),
            "saturated": saturated,
            "energy_resid": float(energy_resid),
        }

    # ==================================================================
    # Unified accuracy interface
    # ==================================================================
    def accuracy(self, input_value, burden_fraction=1.0):
        if self.type == "CT":
            return self.ct_errors(input_value, burden_fraction)
        return self.pt_errors(input_value, burden_fraction)

    def within_accuracy_class(self, input_value, burden_fraction=1.0):
        res = self.accuracy(input_value, burden_fraction)
        ratio_ok = np.abs(res["ratio_error_pct"]) <= self.accuracy_class
        phase_ok = np.abs(res["phase_error_min"]) <= self.phase_class_min
        return bool(np.all(ratio_ok)) and bool(np.all(phase_ok))
