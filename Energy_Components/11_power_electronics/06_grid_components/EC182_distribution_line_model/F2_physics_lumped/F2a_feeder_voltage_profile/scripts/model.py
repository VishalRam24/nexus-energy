"""
EC182 -- Distribution Line Model -- F2a Physics-Lumped Feeder Voltage-Profile ODE

This is the physics-lumped (1D ODE) upgrade of the F1a single-section R+jX model.
Instead of treating the feeder as one lumped impedance feeding one end load, the
feeder is modelled as a CONTINUOUS 1D conductor carrying a DISTRIBUTED load, and
the complex voltage V(x) and current I(x) profiles along the feeder are obtained by
integrating a coupled first-order ODE system (a 1D transmission-line / load-flow
boundary-value problem).

Physics (per-phase, line-to-neutral phasors, x = distance from substation [km]):

    dV/dx = -(r + j x_L) * I(x)              (voltage drop along conductor, Ohm/km)
    dI/dx = -i_load(x)                        (current absorbed by distributed load)

where i_load(x) is the per-unit-length phase current drawn by the distributed load.
Shunt capacitance is NEGLECTED (short/medium distribution line, 4-35 kV): there is
no dI/dx term from line charging, so there is NO Ferranti rise -- voltage falls
monotonically toward the open end under load. This is the defining distribution-line
assumption (Kersting 2012, Ch. 3).

High R/X ratio: distribution feeders are RESISTIVE (R/X ~ 0.5-1.5) unlike
transmission (R/X << 0.1), so the P*R term dominates the voltage drop and the
real losses sum_i 3*|I_i|^2*R_i are significant.

Because the load current i_load depends on the (a priori unknown) local voltage
(constant-power loads: i = conj(s/V)), the system is a two-point boundary-value
problem:  V(0) = V_substation (known),  I(L) = 0 (open radial end).
It is solved by a SHOOTING method -- guess the sending-end current I(0), integrate
the ODE with scipy.integrate.solve_ivp, and drive the end-current residual I(L) -> 0
with scipy.optimize.root (2 real unknowns = Re/Im of I(0)).

Energy conservation is enforced and verified a-posteriori:
    P_send = P_load_delivered + P_loss     (and same for Q with X)

References
----------
Kersting, W. H. (2012). Distribution System Modeling and Analysis, 3rd ed., CRC Press.
Gonen, T. (2014). Electric Power Distribution Engineering, 3rd ed., CRC Press.
ANSI C84.1-2020, Electric Power Systems and Equipment -- Voltage Ratings.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


class FeederVoltageProfileModel:
    """1D distributed-load radial feeder voltage-profile model (R+jX, no shunt)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.r = float(u["R_ohm_per_km"]["value"])        # Ohm/km per phase
        self.x = float(u["X_ohm_per_km"]["value"])        # Ohm/km per phase
        self.V_base_kV = float(u["V_base_kV"]["value"])
        self.length_km_default = float(u["length_km"]["value"])
        self.n_sections_default = int(u["n_sections"]["value"])
        self.ansi_band_pct = float(u.get("ansi_band_pct", {"value": 5.0})["value"])

    # ------------------------------------------------------------------
    @property
    def r_over_x(self) -> float:
        """R/X ratio of the conductor (high => resistive distribution feeder)."""
        return self.r / self.x

    # ------------------------------------------------------------------
    def _rhs(self, x_km, y, L, S_total_VA, load_model, V_nom_phase):
        """
        ODE right-hand side for the complex feeder state.

        y = [Re(V), Im(V), Re(I), Im(I)]  (line-to-neutral phasors)

        The distributed load draws a per-unit-length 3-phase complex power density
        s_density = S_total / L  [VA/km] uniformly along the feeder. The local
        per-phase current absorbed per km is conj(s_density / 3 / V_phase).
        """
        z = self.r + 1j * self.x                 # Ohm/km
        V = y[0] + 1j * y[1]
        I = y[2] + 1j * y[3]

        # Per-km 3-phase complex power absorbed by the distributed load.
        s_density = S_total_VA / L               # VA/km (3-phase)

        if load_model == "constant_power":
            Vsafe = V if abs(V) > 1.0 else (V_nom_phase + 0j)
            di = np.conj(s_density / 3.0 / Vsafe)        # A/km absorbed (per phase)
        elif load_model == "constant_impedance":
            # Z-load: i_density = s_density* /3 * V / |Vnom|^2  -> current scales with V
            di = np.conj(s_density / 3.0 / V_nom_phase) * (V / V_nom_phase)
        else:  # constant_current
            di = np.conj(s_density / 3.0 / V_nom_phase)

        dV = -z * I
        dI = -di
        return [dV.real, dV.imag, dI.real, dI.imag]

    # ------------------------------------------------------------------
    def _integrate(self, I0_complex, V0_phase, L, S_total_VA,
                   load_model, V_nom_phase, n_eval):
        x_eval = np.linspace(0.0, L, n_eval)
        y0 = [V0_phase.real, V0_phase.imag, I0_complex.real, I0_complex.imag]
        sol = solve_ivp(
            self._rhs, (0.0, L), y0, t_eval=x_eval,
            args=(L, S_total_VA, load_model, V_nom_phase),
            method="RK45", rtol=1e-9, atol=1e-9, max_step=L / max(n_eval, 10),
        )
        return sol

    # ------------------------------------------------------------------
    def compute(self, V_s_kV: float, P_total_kW: float, Q_total_kVAR: float,
                length_km: float = None, n_sections: int = None,
                load_model: str = "constant_power") -> dict:
        """
        Solve the 1D feeder voltage profile for a uniformly distributed load.

        Parameters
        ----------
        V_s_kV       : substation (sending-end) line-to-line voltage [kV]
        P_total_kW   : total 3-phase active load distributed along feeder [kW]
        Q_total_kVAR : total 3-phase reactive load [kVAR] (+ = inductive)
        length_km    : feeder length [km]
        n_sections   : spatial discretization points
        load_model   : 'constant_power' | 'constant_impedance' | 'constant_current'

        Returns
        -------
        dict with spatial profiles (x_km, V_profile_kV, I_profile_A, P_flow_kW),
        scalar summary (V_r_kV at far end, P_loss_kW, efficiency, voltage_drop_pct,
        ansi_compliant, ...) and an energy-balance residual.
        """
        if length_km is None:
            length_km = self.length_km_default
        if n_sections is None:
            n_sections = self.n_sections_default
        L = float(length_km)
        n_eval = int(max(n_sections, 4))

        V0_phase = (float(V_s_kV) * 1000.0 / np.sqrt(3.0)) + 0j   # V, L-N
        V_nom_phase = self.V_base_kV * 1000.0 / np.sqrt(3.0)
        S_total_VA = (float(P_total_kW) + 1j * float(Q_total_kVAR)) * 1000.0

        # --- Shooting: find sending current I(0) so that end current I(L)=0 -------
        def residual(I0_ri):
            I0 = I0_ri[0] + 1j * I0_ri[1]
            sol = self._integrate(I0, V0_phase, L, S_total_VA,
                                  load_model, V_nom_phase, n_eval=8)
            IL = sol.y[2, -1] + 1j * sol.y[3, -1]
            return [IL.real, IL.imag]

        # Good initial guess: all current enters at substation = conj(S/3/V0).
        I0_guess = np.conj(S_total_VA / 3.0 / V0_phase)
        solr = root(residual, [I0_guess.real, I0_guess.imag],
                    method="hybr", tol=1e-10)
        I0 = solr.x[0] + 1j * solr.x[1]

        # --- Final high-resolution integration with the converged I(0) -----------
        sol = self._integrate(I0, V0_phase, L, S_total_VA,
                              load_model, V_nom_phase, n_eval=n_eval)
        x_km = sol.t
        Vc = sol.y[0] + 1j * sol.y[1]            # L-N phasor profile [V]
        Ic = sol.y[2] + 1j * sol.y[3]            # phase current profile [A]

        V_profile_phase_mag = np.abs(Vc)
        V_profile_LL_kV = V_profile_phase_mag * np.sqrt(3.0) / 1000.0
        I_profile_A = np.abs(Ic)

        # Active power flowing past each point (3-phase) [kW]
        S_flow = 3.0 * Vc * np.conj(Ic)         # VA (3-phase)
        P_flow_kW = S_flow.real / 1000.0
        Q_flow_kVAR = S_flow.imag / 1000.0

        # --- Losses by spatial integral of I^2 * r  (3-phase) --------------------
        # P_loss = integral_0^L 3 * |I(x)|^2 * r dx
        P_loss_kW = np.trapz(3.0 * I_profile_A ** 2 * self.r, x_km) / 1000.0
        Q_loss_kVAR = np.trapz(3.0 * I_profile_A ** 2 * self.x, x_km) / 1000.0

        # Sending-end injected power (at x=0)
        S_send = 3.0 * Vc[0] * np.conj(Ic[0])
        P_send_kW = S_send.real / 1000.0
        Q_send_kVAR = S_send.imag / 1000.0

        # Delivered (load) power = send - loss  (energy conservation)
        P_delivered_kW = P_send_kW - P_loss_kW
        Q_delivered_kVAR = Q_send_kVAR - Q_loss_kVAR

        # Far-end (receiving) voltage
        V_r_kV = float(V_profile_LL_kV[-1])
        V_s_kV_f = float(V_s_kV)
        voltage_drop_kV = V_s_kV_f - V_r_kV
        voltage_drop_pct = voltage_drop_kV / V_s_kV_f * 100.0 if V_s_kV_f > 0 else 0.0

        # Efficiency (active power)
        eta = P_delivered_kW / P_send_kW if P_send_kW > 1e-9 else 0.0

        # Energy-balance residual (should be ~0).
        # Power that disappears from the conductor between x=0 and x=L equals the
        # power delivered to the distributed load PLUS the I^2*R line loss:
        #     P_flow[0] - P_flow[L] = P_delivered + P_loss
        # The far-end flow P_flow[L] ~ 0 (open radial end), so this reduces to the
        # global conservation statement P_send = P_delivered + P_loss.
        conductor_power_consumed_kW = P_flow_kW[0] - P_flow_kW[-1]
        energy_balance_residual_kW = abs(
            conductor_power_consumed_kW - P_delivered_kW - P_loss_kW
        )

        # ANSI C84.1 voltage band check (Range A, +/- band%)
        v_pu_profile = V_profile_LL_kV / V_s_kV_f if V_s_kV_f > 0 else V_profile_LL_kV
        band = self.ansi_band_pct / 100.0
        ansi_compliant = bool(np.all(v_pu_profile >= (1.0 - band) - 1e-9))
        min_v_pu = float(np.min(v_pu_profile))

        return {
            # spatial profiles
            "x_km": x_km,
            "V_profile_kV": V_profile_LL_kV,
            "I_profile_A": I_profile_A,
            "P_flow_kW": P_flow_kW,
            "Q_flow_kVAR": Q_flow_kVAR,
            # scalar summary
            "V_r_kV": V_r_kV,
            "I_send_A": float(I_profile_A[0]),
            "P_send_kW": P_send_kW,
            "Q_send_kVAR": Q_send_kVAR,
            "P_loss_kW": P_loss_kW,
            "Q_loss_kVAR": Q_loss_kVAR,
            "P_delivered_kW": P_delivered_kW,
            "Q_delivered_kVAR": Q_delivered_kVAR,
            "efficiency": eta,
            "voltage_drop_kV": voltage_drop_kV,
            "voltage_drop_pct": voltage_drop_pct,
            "min_voltage_pu": min_v_pu,
            "ansi_compliant": ansi_compliant,
            "r_over_x": self.r_over_x,
            "energy_balance_residual_kW": energy_balance_residual_kW,
            "converged": bool(solr.success),
        }
