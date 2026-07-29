"""
EC159 -- Buck-Boost Converter (Inverting) -- F2a State-Space Averaged Model

Physics-lumped 2-state ODE model of the inverting buck-boost converter in
continuous-conduction mode (CCM), obtained by state-space averaging the two
switch subintervals (switch ON / diode ON) over one switching period.

Topology (single MOSFET + single diode, inverting):
    During D*Ts  (switch ON) :  L diL/dt = Vin - iL*(Rds_on + R_L)
                                C dvC/dt = -vC/R
    During (1-D)*Ts (diode ON): L diL/dt = -vC - iL*R_L - Vf
                                C dvC/dt =  iL - vC/R

State-space averaging (Middlebrook & Cuk, 1976) -- weight each subinterval by
its duty fraction. States: iL = average inductor current [A] (magnitude),
vC = average output capacitor voltage [V] (magnitude; physical output is -vC):

    L diL/dt = d*Vin - (1-d)*vC - iL*(d*Rds_on + R_L) - (1-d)*Vf
    C dvC/dt = (1-d)*iL - vC/R

Ideal (lossless: R_L=Rds_on=Vf=0) steady state ->
    d*Vin = (1-d)*vC  =>  vC = d*Vin/(1-d)
    Output is inverting:  Vout = -vC = -d/(1-d) * Vin     (classic buck-boost gain)

Parasitic losses (R_L inductor DCR, Rds_on MOSFET conduction, Vf diode drop,
R_C capacitor ESR) reduce the realised gain and the conversion efficiency:
    eta = P_out / P_in = (vC^2 / R) / (d * Vin * iL)

Reference:
    Erickson, R.W. & Maksimovic, D. (2020). Fundamentals of Power Electronics,
        3rd ed., Springer. Ch. 7 (AC equivalent circuit / averaged modeling).
    Middlebrook, R.D. & Cuk, S. (1976). A general unified approach to modelling
        switching-converter power stages. IEEE PESC, 18-34.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BuckBoostConverterF2a:
    """Inverting buck-boost converter -- state-space averaged 2-state ODE (CCM)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Vin_nom = u["v_in_nominal"]["value"]      # V
        self.L = u["L"]["value"]                        # H
        self.C = u["C"]["value"]                        # F
        self.R_load_nom = u["R_load"]["value"]          # Ohm
        self.R_L = u["R_L"]["value"]                    # Ohm
        self.R_ds_on = u["R_ds_on"]["value"]            # Ohm
        self.V_f = u["V_f"]["value"]                    # V
        self.R_C = u["R_C"]["value"]                    # Ohm (ESR)

    # ------------------------------------------------------------------
    # Averaged state derivatives
    # ------------------------------------------------------------------
    def derivatives(self, iL, vC, d, Vin, R_load):
        """
        Averaged state derivatives [diL/dt, dvC/dt].

        iL : inductor current (magnitude) [A]
        vC : output capacitor voltage (magnitude) [V]
        d  : duty cycle in [0, 1]
        """
        # Inductor voltage balance (averaged), including parasitics
        v_L = (d * Vin
               - (1.0 - d) * vC
               - iL * (d * self.R_ds_on + self.R_L)
               - (1.0 - d) * self.V_f)
        # Capacitor charge balance (averaged)
        i_C = (1.0 - d) * iL - vC / R_load
        return v_L / self.L, i_C / self.C

    # ------------------------------------------------------------------
    # Ideal (lossless) steady-state conversion ratio
    # ------------------------------------------------------------------
    def ideal_gain(self, d):
        """Ideal inverting buck-boost gain Vout/Vin = -d/(1-d)."""
        d = np.clip(np.asarray(d, dtype=float), 0.0, 0.999)
        return -d / (1.0 - d)

    def duty_for_gain(self, vin, vout_target):
        """Duty cycle that yields |Vout|: d = |Vout| / (Vin + |Vout|)."""
        vin = float(vin)
        vo = abs(float(vout_target))
        denom = vin + vo
        d = vo / denom if denom > 0 else 0.5
        return float(np.clip(d, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Steady-state solution (lossy) -- solve averaged ODE fixed point
    # ------------------------------------------------------------------
    def steady_state(self, d, Vin=None, R_load=None):
        """
        Lossy steady-state (diL/dt = dvC/dt = 0).

        From charge balance:  iL = vC / ((1-d) R)
        Substitute into voltage balance and solve linear eq for vC.
        Returns dict: iL, vC, vout (= -vC), gain, efficiency.
        """
        Vin = self.Vin_nom if Vin is None else float(Vin)
        R = self.R_load_nom if R_load is None else float(R_load)
        d = float(np.clip(d, 0.01, 0.99))
        omd = 1.0 - d

        # iL = vC / (omd * R). Plug into:
        #   0 = d*Vin - omd*vC - iL*(d*Rds + R_L) - omd*Vf
        # => 0 = d*Vin - omd*Vf - vC*[ omd + (d*Rds + R_L)/(omd*R) ]
        coef = omd + (d * self.R_ds_on + self.R_L) / (omd * R)
        vC = (d * Vin - omd * self.V_f) / coef
        vC = max(vC, 0.0)
        iL = vC / (omd * R)

        p_out = vC * vC / R
        p_in = d * Vin * iL
        eta = p_out / p_in if p_in > 0 else 0.0
        return {
            "iL": iL,
            "vC": vC,
            "vout": -vC,
            "gain": -vC / Vin if Vin != 0 else 0.0,
            "efficiency": eta,
            "duty": d,
        }

    # ------------------------------------------------------------------
    # Time-domain simulation of the averaged model
    # ------------------------------------------------------------------
    def simulate(self, duty, Vin=None, R_load=None, dt=1.0e-6,
                 duration_s=2.0e-3, iL0=0.0, vC0=0.0):
        """
        Integrate the averaged state-space ODE.

        Parameters
        ----------
        duty : float or callable(t) -> float
            Duty cycle command in [0, 1].
        Vin : float or callable(t), optional
            Input voltage [V] (default nominal).
        R_load : float, optional
            Load resistance [Ohm] (default nominal).
        dt : float
            Output sampling step [s].
        duration_s : float
            Total simulated time [s].
        iL0, vC0 : float
            Initial inductor current / capacitor voltage.

        Returns
        -------
        dict with arrays: t, iL, vC, vout, vin, duty, p_in, p_out, efficiency.
        """
        Vin_nom = self.Vin_nom if Vin is None else Vin
        R = self.R_load_nom if R_load is None else float(R_load)

        d_fn = duty if callable(duty) else (lambda t: duty)
        v_fn = Vin_nom if callable(Vin_nom) else (lambda t: Vin_nom)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            iL, vC = y
            d = float(np.clip(d_fn(t), 0.0, 1.0))
            vin = float(v_fn(t))
            diL, dvC = self.derivatives(iL, vC, d, vin, R)
            return [diL, dvC]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [iL0, vC0],
            t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        t = sol.t
        iL = sol.y[0]
        vC = sol.y[1]
        N = len(t)

        duty_arr = np.array([float(np.clip(d_fn(tt), 0.0, 1.0)) for tt in t])
        vin_arr = np.array([float(v_fn(tt)) for tt in t])

        p_out = vC * vC / R
        p_in = duty_arr * vin_arr * iL
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(p_in > 1e-12, p_out / p_in, 0.0)
        eff = np.clip(eff, 0.0, 1.0)

        return {
            "t": t,
            "iL": iL,
            "vC": vC,
            "vout": -vC,
            "vin": vin_arr,
            "duty": duty_arr,
            "p_in": p_in,
            "p_out": p_out,
            "efficiency": eff,
        }
