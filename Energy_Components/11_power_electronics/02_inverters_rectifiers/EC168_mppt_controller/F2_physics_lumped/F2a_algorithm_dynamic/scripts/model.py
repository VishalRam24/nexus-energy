"""
EC168 -- MPPT Controller -- F2a Algorithm Dynamic

Physics-lumped model: Perturb & Observe (P&O) MPPT algorithm coupled
with buck converter averaged dynamics.

PV model (single-diode):
    I = I_ph - I_0*(exp((V + I*R_s)/(n*N_s*Vt)) - 1) - (V + I*R_s)/R_sh

P&O algorithm:
    Every T_mppt seconds: measure P_pv
    if dP > 0: V_ref += sign(dV)*dV_step
    else:      V_ref -= sign(dV)*dV_step

Buck converter (averaged model):
    L * dI_L/dt = D*V_pv - V_out - R_L*I_L
    C * dV_out/dt = I_L - V_out/R_load

Duty cycle D controlled by PI controller tracking V_ref.

References:
    Esram & Chapman (2007), IEEE Trans. Energy Conv., 22(2), 439-449
    Femia et al. (2005), IEEE Trans. Power Electron., 20(4), 963-973
"""

import numpy as np
from scipy.integrate import solve_ivp


class PVSingleDiode:
    """Single-diode PV model with series and shunt resistance."""

    k_B = 1.381e-23   # Boltzmann constant [J/K]
    q = 1.602e-19      # Electron charge [C]

    def __init__(self, params: dict):
        u = params["unit"]
        self.I_ph_stc = u["I_ph_stc"]["value"]
        self.I_0_ref = u["I_0"]["value"]
        self.n = u["n_diode"]["value"]
        self.N_s = u["N_s"]["value"]
        self.R_s = u["R_s"]["value"]
        self.R_sh = u["R_sh"]["value"]
        self.T_cell = u["T_cell"]["value"]
        self.T_ref = 298.15  # STC reference temperature [K]
        self.E_g = 1.12 * self.q  # Silicon bandgap energy [J]

    def thermal_voltage(self, T=None):
        """Thermal voltage Vt = kT/q [V]."""
        if T is None:
            T = self.T_cell
        return self.k_B * T / self.q

    def saturation_current(self, T=None):
        """Temperature-dependent reverse saturation current [A].
        I_0(T) = I_0_ref * (T/T_ref)^3 * exp((E_g/(n*k)) * (1/T_ref - 1/T))
        """
        if T is None:
            T = self.T_cell
        return self.I_0_ref * (T / self.T_ref)**3 * np.exp(
            (self.E_g / (self.n * self.k_B)) * (1.0 / self.T_ref - 1.0 / T)
        )

    def photocurrent(self, G, T=None):
        """Photocurrent scaled by irradiance [A].
        I_ph = I_ph_stc * (G/1000) * (1 + 0.0005*(T-298.15))
        """
        if T is None:
            T = self.T_cell
        return self.I_ph_stc * (G / 1000.0) * (1.0 + 0.0005 * (T - 298.15))

    def current(self, V, G, T=None):
        """PV output current [A] at terminal voltage V.

        Solves implicit equation iteratively (Newton).
        I = I_ph - I_0*(exp((V+I*Rs)/(n*Ns*Vt))-1) - (V+I*Rs)/R_sh
        """
        if T is None:
            T = self.T_cell
        Vt = self.thermal_voltage(T)
        I_ph = self.photocurrent(G, T)
        I_0 = self.saturation_current(T)
        a = self.n * self.N_s * Vt

        # Initial guess
        I = I_ph - V / self.R_sh

        for _ in range(50):
            V_d = V + I * self.R_s
            exp_term = np.exp(np.clip(V_d / a, -50, 50))
            f = I_ph - I_0 * (exp_term - 1.0) - V_d / self.R_sh - I
            df = -I_0 * (self.R_s / a) * exp_term - self.R_s / self.R_sh - 1.0
            dI = -f / df
            I = I + dI
            if abs(dI) < 1e-10:
                break

        return max(I, 0.0)

    def power(self, V, G, T=None):
        """PV output power [W]."""
        I = self.current(V, G, T)
        return V * I

    def mpp(self, G, T=None):
        """Find maximum power point by voltage sweep.
        Returns (V_mpp, I_mpp, P_mpp).
        """
        if G < 1.0:
            return 0.0, 0.0, 0.0
        V_arr = np.linspace(0.1, self.N_s * 0.65, 500)
        P_arr = np.array([self.power(v, G, T) for v in V_arr])
        idx = np.argmax(P_arr)
        V_m = V_arr[idx]
        I_m = self.current(V_m, G, T)
        return V_m, I_m, P_arr[idx]


class MPPTController_F2a:
    """MPPT Controller with P&O algorithm and buck converter dynamics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.pv = PVSingleDiode(params)

        # Buck converter
        self.L = u["L"]["value"]
        self.C = u["C"]["value"]
        self.R_L = u["R_L"]["value"]
        self.V_bat = u["V_bat"]["value"]
        self.R_load = u["R_load"]["value"]

        # P&O MPPT
        self.dV = u["dV_mppt"]["value"]
        self.T_mppt = u["T_mppt"]["value"]
        self.D_init = u["D_init"]["value"]
        self.D_min = u["D_min"]["value"]
        self.D_max = u["D_max"]["value"]

        # PI controller
        self.Kp = u["Kp_pi"]["value"]
        self.Ki = u["Ki_pi"]["value"]

    def simulate(self, irradiance, T_cell, dt, duration_s):
        """
        Simulate MPPT controller with buck converter.

        Parameters
        ----------
        irradiance : float or callable(t)
            Solar irradiance [W/m2].
        T_cell : float or callable(t)
            Cell temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation time [s].

        Returns
        -------
        dict with time series of V_pv, I_pv, P_pv, V_ref, D, I_L, V_out,
             P_out, tracking_efficiency.
        """
        _G = irradiance if callable(irradiance) else lambda t: irradiance
        _T = T_cell if callable(T_cell) else lambda t: T_cell

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        # P&O state
        V_ref = 30.0  # initial reference voltage (near MPP at STC)
        P_prev = 0.0
        V_prev = V_ref
        last_mppt_time = -self.T_mppt  # force first update at t=0

        # PI integrator state
        integral_error = 0.0
        D = self.D_init

        # State: [I_L, V_out]
        y0 = [self.V_bat / self.R_load, self.V_bat]

        # We'll do manual time-stepping for the coupled MPPT + ODE system
        N = len(t_eval)
        V_pv_arr = np.zeros(N)
        I_pv_arr = np.zeros(N)
        P_pv_arr = np.zeros(N)
        V_ref_arr = np.zeros(N)
        D_arr = np.zeros(N)
        I_L_arr = np.zeros(N)
        V_out_arr = np.zeros(N)
        P_out_arr = np.zeros(N)
        eta_track_arr = np.zeros(N)
        G_arr = np.zeros(N)

        I_L = y0[0]
        V_out = y0[1]

        for i in range(N):
            t = t_eval[i]
            G = _G(t)
            T = _T(t)
            G_arr[i] = G

            # PV operating point: V_pv determined by buck input
            # In averaged model: V_pv ~ V_out / D (buck converter relation)
            # But more accurately, PV voltage is set by the MPPT reference
            V_pv = V_ref
            I_pv = self.pv.current(V_pv, G, T)
            P_pv = V_pv * I_pv

            # P&O MPPT update
            if t - last_mppt_time >= self.T_mppt:
                dP = P_pv - P_prev
                dV_applied = V_ref - V_prev

                if dP > 0:
                    if dV_applied >= 0:
                        V_ref_new = V_ref + self.dV
                    else:
                        V_ref_new = V_ref - self.dV
                else:
                    if dV_applied >= 0:
                        V_ref_new = V_ref - self.dV
                    else:
                        V_ref_new = V_ref + self.dV

                # Clamp V_ref
                V_ref_new = max(V_ref_new, 1.0)
                V_ref_new = min(V_ref_new, self.pv.N_s * 0.65)

                V_prev = V_ref
                P_prev = P_pv
                V_ref = V_ref_new
                last_mppt_time = t

            # PI controller: D tracks V_ref
            # Error: we want V_pv = V_ref, and V_pv ~ V_out/D
            # So target D ~ V_out / V_ref (for buck: V_out = D * V_in)
            # PI on the voltage error
            error = V_ref - V_pv
            if i > 0:
                integral_error += error * dt
            D_target = self.V_bat / V_ref if V_ref > 0 else self.D_init
            D = D_target + self.Kp * error + self.Ki * integral_error
            D = np.clip(D, self.D_min, self.D_max)

            # Buck converter ODE (one step with RK4)
            def buck_rhs(I_L_loc, V_out_loc):
                dIL = (D * V_pv - V_out_loc - self.R_L * I_L_loc) / self.L
                dVout = (I_L_loc - V_out_loc / self.R_load) / self.C
                return dIL, dVout

            if i < N - 1:
                h = t_eval[i + 1] - t_eval[i] if i < N - 1 else dt
                # RK4
                k1_IL, k1_V = buck_rhs(I_L, V_out)
                k2_IL, k2_V = buck_rhs(I_L + 0.5*h*k1_IL, V_out + 0.5*h*k1_V)
                k3_IL, k3_V = buck_rhs(I_L + 0.5*h*k2_IL, V_out + 0.5*h*k2_V)
                k4_IL, k4_V = buck_rhs(I_L + h*k3_IL, V_out + h*k3_V)
                I_L += h/6 * (k1_IL + 2*k2_IL + 2*k3_IL + k4_IL)
                V_out += h/6 * (k1_V + 2*k2_V + 2*k3_V + k4_V)
                I_L = max(I_L, 0.0)
                V_out = max(V_out, 0.0)

            # Output power
            P_out = V_out * V_out / self.R_load

            # Tracking efficiency
            _, _, P_mpp = self.pv.mpp(G, T)
            eta_track = P_pv / P_mpp if P_mpp > 1.0 else 1.0

            # Store
            V_pv_arr[i] = V_pv
            I_pv_arr[i] = I_pv
            P_pv_arr[i] = P_pv
            V_ref_arr[i] = V_ref
            D_arr[i] = D
            I_L_arr[i] = I_L
            V_out_arr[i] = V_out
            P_out_arr[i] = P_out
            eta_track_arr[i] = min(eta_track, 1.0)

        return {
            "t": t_eval,
            "irradiance": G_arr,
            "V_pv": V_pv_arr,
            "I_pv": I_pv_arr,
            "P_pv": P_pv_arr,
            "V_ref": V_ref_arr,
            "duty_cycle": D_arr,
            "I_L": I_L_arr,
            "V_out": V_out_arr,
            "P_out": P_out_arr,
            "tracking_efficiency": eta_track_arr,
        }
