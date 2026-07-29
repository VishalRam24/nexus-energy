"""
EC170 -- Solid State Transformer (SST) -- F2a Cascaded Averaged Three-Stage Model

Physics-lumped, control-oriented model of a three-stage SST. An SST replaces a
line-frequency magnetic transformer with three cascaded power-electronic stages
that together provide voltage transformation, galvanic isolation and bidirectional
power-flow control (Huang 2016; She, Huang & Burgos 2013):

    MV AC  ──▶ [Stage 1: AC-DC rectifier]  ──▶  HV DC-link (C_hv)
                                                     │
           [Stage 2: isolated DAB DC-DC + medium-frequency transformer]
                                                     │
    LV DC-link (C_lv) ◀── [Stage 3: DC-AC inverter] ──▶ LV AC

Each stage is represented by its *averaged* (cycle-averaged, switching-ripple
removed) power model. The instantaneous switching is not resolved (that is F2b);
here every stage is a power-in/power-out block with an efficiency that depends on
load fraction:

    eta_stage(P) = eta_nom - k_load * |P/P_rated|        (load-dependent rolloff)

This captures the characteristic SST efficiency curve: high efficiency over a wide
load range with a gentle quadratic-ish rolloff toward full load (Krismer & Kolar
2012). The DAB stage additionally carries a constant core (Steinmetz) standing
loss when energised.

CASCADE (forward, MV->LV):
    eta_total = eta_rect * eta_dab * eta_inv          (product of stage efficiencies)
    P_out     = eta_total * P_in

Because 0 < eta_stage < 1 for every stage, the product is strictly in (0,1):
energy conservation holds stage-by-stage and overall, P_loss = P_in - P_out > 0.

BIDIRECTIONAL:
    Sign convention: P_command > 0 is forward (MV->LV); P_command < 0 is reverse
    (LV->MV, e.g. PV/battery export to the MV grid). In reverse the same stage
    efficiencies apply in the opposite order so the delivered power is again
    eta_total * |P_in|, and the source-side draw is |P_out|/eta_total.

VOLTAGE TRANSFORMATION:
    The averaged conversion ratio from MV AC to LV AC is set by the rectifier DC
    target, the DAB turns ratio n and the inverter modulation. For the nominal
    operating point V_lv_ac = V_hv_ac * (V_lv_dc / V_hv_dc) / n_effective; the
    lumped model tracks the two DC-link voltages explicitly.

LUMPED DC-LINK DYNAMICS (the ODE, integrated with scipy.solve_ivp):
    The two DC-link capacitors are energy buffers between stages. Charge balance:

        C_hv * dV_hv/dt = (P_rect_out - P_dab_in) / V_hv
        C_lv * dV_lv/dt = (P_dab_out - P_inv_in) / V_lv

    The DAB power follows the commanded power through a first-order actuation lag
    (inner phase-shift control loop, time constant tau_ctrl):

        dP_dab/dt = (P_command - P_dab) / tau_ctrl

    These three coupled ODEs are the lumped state; solve_ivp integrates them to
    show DC-link transients and settling after a power-flow command step, with
    cascaded energy conservation enforced at every instant.

References:
    Huang, A.Q. (2016). Medium-Voltage Solid-State Transformer: Technology for a
        Smarter and Resilient Grid. IEEE Industrial Electronics Magazine, 10(3), 29-42.
    She, X., Huang, A.Q., Burgos, R. (2013). Review of Solid-State Transformer
        Technologies and Their Application in Power Distribution Systems.
        IEEE J. Emerg. Sel. Topics Power Electron., 1(3), 186-198.
    Krismer, F. & Kolar, J.W. (2012). Efficiency-Optimized High-Current Dual Active
        Bridge Converter for Automotive Applications. IEEE Trans. Ind. Electron.,
        59(7), 2745-2760.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SST_F2a:
    """Three-stage averaged SST with lumped DC-link ODE dynamics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated"]["value"]
        self.V_hv_dc_nom = u["V_hv_dc_nom"]["value"]
        self.V_lv_dc_nom = u["V_lv_dc_nom"]["value"]
        self.n = u["turns_ratio"]["value"]

        # Stage nominal efficiencies + load rolloff coefficients
        self.eta_rect_nom = u["eta_rect_nom"]["value"]
        self.k_rect = u["k_rect_load"]["value"]
        self.eta_dab_nom = u["eta_dab_nom"]["value"]
        self.k_dab = u["k_dab_load"]["value"]
        self.P_core_dab = u["P_core_dab"]["value"]
        self.eta_inv_nom = u["eta_inv_nom"]["value"]
        self.k_inv = u["k_inv_load"]["value"]

        # DC-link energy buffers + control lag
        self.C_hv = u["C_hv"]["value"]
        self.C_lv = u["C_lv"]["value"]
        self.tau = u["tau_ctrl"]["value"]

    # ------------------------------------------------------------------
    # Per-stage averaged efficiency (load-dependent)
    # ------------------------------------------------------------------
    def _stage_eff(self, eta_nom, k_load, p_through):
        """Averaged stage efficiency at |load| fraction, clamped to (0,1)."""
        load = np.abs(np.asarray(p_through, dtype=float)) / self.P_rated
        eta = eta_nom - k_load * load
        # strictly inside (0,1)
        return np.clip(eta, 1e-3, 1.0 - 1e-6)

    def stage_efficiencies(self, p_through, power_factor=1.0):
        """Return (eta_rect, eta_dab, eta_inv) at the given through-power.

        The rectifier sees the (poor) input power factor: lower pf raises its
        apparent-current losses, reducing its averaged efficiency.
        """
        pf = np.clip(np.asarray(power_factor, dtype=float), 1e-3, 1.0)
        eta_rect = self._stage_eff(self.eta_rect_nom, self.k_rect / pf, p_through)
        eta_dab = self._stage_eff(self.eta_dab_nom, self.k_dab, p_through)
        eta_inv = self._stage_eff(self.eta_inv_nom, self.k_inv, p_through)
        return eta_rect, eta_dab, eta_inv

    def total_efficiency(self, p_through, power_factor=1.0):
        """Overall efficiency = PRODUCT of the three averaged stage efficiencies.

        Strictly in (0,1) by construction. The DAB core standing loss is folded
        in as an additional load-referenced term so that at very light load the
        efficiency correctly droops (fixed loss / small power)."""
        er, ed, ei = self.stage_efficiencies(p_through, power_factor)
        eta_switching = er * ed * ei

        p = np.abs(np.asarray(p_through, dtype=float))
        # core loss reduces delivered power: eta_core = (p - P_core)/p folded in
        # but only when energised; guard the p->0 limit.
        p_safe = np.where(p > 1.0, p, 1.0)
        eta_core = np.where(p > 1.0,
                            np.clip(1.0 - self.P_core_dab / p_safe, 1e-3, 1.0),
                            1.0)
        eta = eta_switching * eta_core
        return np.clip(eta, 1e-6, 1.0 - 1e-9)

    # ------------------------------------------------------------------
    # Static cascade (steady-state power balance), bidirectional
    # ------------------------------------------------------------------
    def cascade(self, p_command, power_factor=1.0):
        """Steady-state cascaded power balance.

        p_command : commanded through-power [W]. >0 forward (MV->LV),
                    <0 reverse (LV->MV).
        Returns dict with delivered power, source draw, per-stage powers,
        per-stage + total efficiency, and total loss (all energy-conserving).
        """
        p_cmd = np.asarray(p_command, dtype=float)
        mag = np.abs(p_cmd)
        eta = self.total_efficiency(mag, power_factor)
        er, ed, ei = self.stage_efficiencies(mag, power_factor)

        # Delivered (load side) and source draw
        p_delivered = eta * mag        # what reaches the far side
        p_source = mag                 # what the source supplies
        p_loss = p_source - p_delivered

        # Per-stage intermediate powers in the forward chain (rect->dab->inv)
        p1_out = mag * er
        p2_out = p1_out * ed
        p3_out = p2_out * ei

        direction = np.where(p_cmd >= 0, 1.0, -1.0)
        return {
            "direction": direction,                 # +1 forward, -1 reverse
            "p_source_w": p_source,
            "p_delivered_w": p_delivered * np.sign(np.where(p_cmd == 0, 1, p_cmd)),
            "p_delivered_mag_w": p_delivered,
            "p_loss_w": p_loss,
            "eta_rect": er,
            "eta_dab": ed,
            "eta_inv": ei,
            "eta_total": eta,
            "p_stage1_out_w": p1_out,
            "p_stage2_out_w": p2_out,
            "p_stage3_out_w": p3_out,
        }

    # ------------------------------------------------------------------
    # Voltage transformation
    # ------------------------------------------------------------------
    def voltage_transform(self, v_hv_ac_rms):
        """Averaged MV-AC -> LV-AC voltage transformation [V].

        Conversion is set by the DC-link ratio and the MF-transformer turns
        ratio. With the rectifier regulating V_hv_dc and the inverter producing
        a modulation-set LV AC from V_lv_dc:

            V_lv_ac = V_hv_ac * (V_lv_dc_nom / V_hv_dc_nom) / n
        """
        v = np.asarray(v_hv_ac_rms, dtype=float)
        ratio = (self.V_lv_dc_nom / self.V_hv_dc_nom) / self.n
        return v * ratio

    # ------------------------------------------------------------------
    # Lumped DC-link ODE dynamics (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def _rhs(self, t, y, p_cmd_func, power_factor):
        """State y = [V_hv_dc, V_lv_dc, P_dab]. Coupled DC-link + control ODEs."""
        v_hv, v_lv, p_dab = y
        v_hv = max(v_hv, 1.0)
        v_lv = max(v_lv, 1.0)

        p_cmd = float(p_cmd_func(t))
        mag = abs(p_cmd)
        er, ed, ei = self.stage_efficiencies(mag, power_factor)
        er = float(er); ed = float(ed); ei = float(ei)

        # Control loop drives DAB power toward command (first-order lag)
        dP_dab = (p_cmd - p_dab) / self.tau

        mag_dab = abs(p_dab)
        if p_cmd >= 0:
            # forward: rectifier feeds HV link, DAB draws from HV link,
            # delivers (eta_dab) to LV link, inverter draws from LV link.
            p_rect_out = mag * er                 # into HV link
            p_dab_in = mag_dab                     # out of HV link
            p_dab_out = mag_dab * ed               # into LV link
            p_inv_in = (mag_dab * ed) * ei         # out of LV link (to load)
        else:
            # reverse: inverter acts as rectifier feeding LV link, DAB pushes
            # HV-ward, rectifier (now inverter) delivers to MV grid.
            p_inv_in = -mag * ei                   # into LV link (from LV source)
            p_dab_out = -mag_dab                    # out of LV link
            p_dab_in = -mag_dab * ed                # into HV link
            p_rect_out = -mag * ed * er             # out of HV link (to MV grid)

        # Charge balance: C dV/dt = net_power / V
        dV_hv = (p_rect_out - p_dab_in) / (self.C_hv * v_hv)
        dV_lv = (p_dab_out - p_inv_in) / (self.C_lv * v_lv)
        return [dV_hv, dV_lv, dP_dab]

    def simulate(self, p_command, power_factor=1.0,
                 V_hv0=None, V_lv0=None, dt=0.001, duration_s=0.2):
        """Integrate the lumped DC-link + control ODE with scipy.solve_ivp.

        p_command : scalar [W] or callable t->[W] (power-flow command, signed).
        Returns time series of DC-link voltages, DAB power, instantaneous
        delivered power, efficiency, and loss.
        """
        if callable(p_command):
            p_cmd_func = p_command
            p0 = float(p_command(0.0))
        else:
            pc = float(p_command)
            p_cmd_func = lambda t: pc
            p0 = pc

        if V_hv0 is None:
            V_hv0 = self.V_hv_dc_nom
        if V_lv0 is None:
            V_lv0 = self.V_lv_dc_nom

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        y0 = [V_hv0, V_lv0, 0.0]   # start with zero transferred power

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(p_cmd_func, power_factor),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        v_hv = sol.y[0]
        v_lv = sol.y[1]
        p_dab = sol.y[2]

        # Reconstruct instantaneous delivered power / efficiency from |P_dab|
        cmd = np.array([p_cmd_func(tt) for tt in t])
        mag = np.abs(p_dab)
        eta = self.total_efficiency(mag, power_factor)
        p_delivered = eta * np.abs(cmd) * np.sign(np.where(cmd == 0, 1, cmd))
        p_loss = np.abs(cmd) - eta * np.abs(cmd)

        return {
            "t": t,
            "v_hv_dc": v_hv,
            "v_lv_dc": v_lv,
            "p_dab_w": p_dab,
            "p_command_w": cmd,
            "p_delivered_w": p_delivered,
            "efficiency": eta,
            "p_loss_w": p_loss,
            "success": sol.success,
        }
