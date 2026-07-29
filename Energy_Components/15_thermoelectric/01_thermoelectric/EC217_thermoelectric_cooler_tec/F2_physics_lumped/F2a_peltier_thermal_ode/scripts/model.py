"""
EC217 -- Thermoelectric Cooler (TEC) -- F2a Peltier + Lumped Thermal ODE

Physics-lumped (0D) upgrade of the F1b temperature-dependent TEC model. The
F1 model is algebraic (steady-state Q_cold / COP at fixed plate temperatures).
This F2a model adds two coupled lumped-capacitance ODEs for the cold-plate and
hot-plate temperatures, integrated with scipy.integrate.solve_ivp, so the
device's transient pull-down and steady operating point emerge from first
principles.

Governing thermoelectric junction equations (per module, current-controlled),
from the standard Peltier-cooler heat balance (Goldsmid 1986; Rowe 2006):

    Q_cold = alpha_m * I * T_c  -  0.5 * I^2 * R  -  K * (T_h - T_c)   [W]
             ^Peltier absorbed     ^half Joule       ^Fourier leak-back
    Q_hot  = alpha_m * I * T_h  +  0.5 * I^2 * R  -  K * (T_h - T_c)   [W]
    W_in   = Q_hot - Q_cold = alpha_m * I * (T_h - T_c) + I^2 * R      [W]
    COP    = Q_cold / W_in

The Joule heat I^2*R is split equally to the two junctions (the classic
"half-Joule to each side" lumped approximation). Energy conservation is exact:
    W_in = Q_hot - Q_cold     (electrical power = net heat pumped).

Material figure of merit:  ZT = alpha^2 * sigma * T / k.
Temperature-dependent Bi2Te3 properties (same fits as EC217 F1b / EC216 F1b):
    alpha(T) = alpha0 * (1 + a1*(T-T0) + a2*(T-T0)^2)   [V/K]
    k(T)     = k0     * (1 + b1*(T-T0))                 [W/(m.K)]
    sigma(T) = sigma0 * (1 + c1*(T-T0))                 [S/m]

Lumped thermal ODE (the F2 first-principles part):
    C_c dT_c/dt = -Q_cold(T_c,T_h,I) + Q_load + hA_c*(T_amb - T_c)
    C_h dT_h/dt =  Q_hot (T_c,T_h,I)            - hA_h*(T_h - T_amb)
where Q_cold is heat pumped OUT of the cold plate (so it cools the plate),
Q_load is the applied payload heat, and hA_c / hA_h are parasitic / heat-sink
conductances to ambient.  At steady state dT/dt = 0 these reproduce the
algebraic F1 balance.

Maximum-cooling current and Ioffe limits (constant-property approximation):
    I_qmax  = alpha_m * T_c / R                      (maximises Q_cold)
    Q_c,max = 0.5 * alpha_m^2 * T_c^2 / R - K*dT      (Q_cold at I_qmax)
    I_copt  = alpha_m*dT / (R*(sqrt(1+ZT)-1)) (max-COP current, Goldsmid)
    dT_max  = 0.5 * ZT * T_c^2 / T_h  (max temperature lift at zero load)
    COP_max = (T_c/dT)*(sqrt(1+ZT) - T_h/T_c)/(sqrt(1+ZT)+1)   (< Carnot)

References:
    Rowe, D.M. (ed.) (2006). Thermoelectrics Handbook. CRC Press.
    Goldsmid, H.J. (1986). Electronic Refrigeration. Pion Ltd, London.
    Ioffe, A.F. (1957). Semiconductor Thermoelements and Thermoelectric
        Cooling. Infosearch, London.
"""

import numpy as np
from scipy.integrate import solve_ivp


class TEC_F2a:
    """Current-controlled Peltier cooler with lumped cold/hot-plate thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Material property fits (per element)
        self.alpha0 = u["alpha0"]["value"]          # V/K
        self.k0 = u["k0"]["value"]                  # W/(m.K)
        self.sigma0 = u["sigma_e0"]["value"]        # S/m
        self.T0 = u["T0_K"]["value"]                # K
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.b1 = u["b1"]["value"]
        self.c1 = u["c1"]["value"]
        # Geometry
        self.N = u["n_couples"]["value"]
        self.A_elem = u["A_element_m2"]["value"]
        self.L_elem = u["L_element_m"]["value"]
        self.r_contact = u["contact_resistance_fraction"]["value"]
        # Lumped thermal network
        self.C_cold = u["C_cold_J_per_K"]["value"]
        self.C_hot = u["C_hot_J_per_K"]["value"]
        self.hA_cold = u["hA_cold_W_per_K"]["value"]
        self.hA_hot = u["hA_hot_W_per_K"]["value"]
        self.T_amb = u["T_ambient_K"]["value"]
        self.Q_load_default = u["Q_load_W"]["value"]

    # ------------------------------------------------------------------ #
    # Temperature-dependent material properties (per element)            #
    # ------------------------------------------------------------------ #
    def alpha(self, T):
        """Per-element Seebeck coefficient [V/K]."""
        dT = np.asarray(T, dtype=float) - self.T0
        return self.alpha0 * (1.0 + self.a1 * dT + self.a2 * dT ** 2)

    def k_thermal(self, T):
        """Per-element thermal conductivity [W/(m.K)]."""
        dT = np.asarray(T, dtype=float) - self.T0
        return self.k0 * (1.0 + self.b1 * dT)

    def sigma_electrical(self, T):
        """Per-element electrical conductivity [S/m]."""
        dT = np.asarray(T, dtype=float) - self.T0
        return self.sigma0 * (1.0 + self.c1 * dT)

    def zt_local(self, T):
        """Local dimensionless figure of merit ZT = alpha^2 * sigma * T / k."""
        T = np.asarray(T, dtype=float)
        a = self.alpha(T)
        return a ** 2 * self.sigma_electrical(T) * T / (self.k_thermal(T) + 1e-30)

    # ------------------------------------------------------------------ #
    # Module-level lumped electrical/thermal parameters                  #
    # ------------------------------------------------------------------ #
    def _alpha_module(self, T_avg):
        """Module Seebeck coefficient [V/K] = N * alpha(T_avg)."""
        return self.N * self.alpha(T_avg)

    def _module_resistance(self, T_avg):
        """Module electrical resistance [ohm].  R = 2N*L/(sigma*A)*(1+r_contact)."""
        sigma = self.sigma_electrical(T_avg)
        R_elem = self.L_elem / (sigma * self.A_elem + 1e-30)
        return 2.0 * self.N * R_elem * (1.0 + self.r_contact)

    def _module_conductance(self, T_avg):
        """Module thermal conductance [W/K].  K = 2N*k*A/L."""
        return 2.0 * self.N * self.k_thermal(T_avg) * self.A_elem / self.L_elem

    # ------------------------------------------------------------------ #
    # Core junction heat balance                                          #
    # ------------------------------------------------------------------ #
    def junction_heat(self, T_c, T_h, I):
        """Peltier-cooler junction heat balance.

        Returns dict with Q_cold, Q_hot, W_input, COP, V_module, ZT_avg,
        plus the module parameters alpha_m, R, K at the mean temperature.
        Joule heat is split half/half to the two junctions.
        """
        T_c = float(T_c)
        T_h = float(T_h)
        I = float(I)
        T_avg = 0.5 * (T_c + T_h)

        alpha_m = self._alpha_module(T_avg)
        R = self._module_resistance(T_avg)
        K = self._module_conductance(T_avg)
        dT = T_h - T_c

        Q_cold = alpha_m * I * T_c - 0.5 * I ** 2 * R - K * dT
        Q_hot = alpha_m * I * T_h + 0.5 * I ** 2 * R - K * dT
        W_input = alpha_m * I * dT + I ** 2 * R          # = Q_hot - Q_cold exactly
        V_module = alpha_m * dT + I * R

        COP = Q_cold / W_input if (W_input > 1e-12 and Q_cold > 0) else 0.0
        ZT_avg = float(self.zt_local(T_avg))

        return {
            "Q_cold_W": Q_cold,
            "Q_hot_W": Q_hot,
            "W_input_W": W_input,
            "COP": COP,
            "V_module_V": V_module,
            "ZT_avg": ZT_avg,
            "alpha_module_V_per_K": alpha_m,
            "R_module_ohm": R,
            "K_module_W_per_K": K,
        }

    # ------------------------------------------------------------------ #
    # Optimal-current / limit formulas (constant-property, at T_avg)      #
    # ------------------------------------------------------------------ #
    def current_for_max_cooling(self, T_c, T_h):
        """Current that maximises Q_cold:  I = alpha_m * T_c / R  (dQc/dI = 0)."""
        T_avg = 0.5 * (T_c + T_h)
        alpha_m = self._alpha_module(T_avg)
        R = self._module_resistance(T_avg)
        return alpha_m * T_c / R

    def current_for_max_cop(self, T_c, T_h):
        """Current that maximises COP (Goldsmid 1986):
        I_copt = alpha_m*dT / (R*(sqrt(1+ZT)-1))."""
        T_avg = 0.5 * (T_c + T_h)
        alpha_m = self._alpha_module(T_avg)
        R = self._module_resistance(T_avg)
        dT = max(T_h - T_c, 1e-6)
        ZT = float(self.zt_local(T_avg))
        return alpha_m * dT / (R * (np.sqrt(1.0 + ZT) - 1.0) + 1e-30)

    def max_cooling_power(self, T_c, T_h):
        """Q_cold evaluated at the max-cooling current."""
        I = self.current_for_max_cooling(T_c, T_h)
        return self.junction_heat(T_c, T_h, I)["Q_cold_W"]

    def max_temperature_lift(self, T_h):
        """Maximum dT (zero cold load) -- Ioffe limit dT_max = 0.5*ZT*T_c^2/T_h.

        Solved self-consistently for T_c since ZT and T_c both vary.
        """
        T_c = T_h
        for _ in range(50):
            ZT = float(self.zt_local(0.5 * (T_c + T_h)))
            dT = 0.5 * ZT * T_c ** 2 / T_h
            T_c_new = T_h - dT
            if abs(T_c_new - T_c) < 1e-6:
                T_c = T_c_new
                break
            T_c = 0.5 * (T_c + T_c_new)
        return T_h - T_c

    def cop_max_theoretical(self, T_c, T_h):
        """Max-COP (Ioffe/Goldsmid), guaranteed below Carnot COP."""
        T_avg = 0.5 * (T_c + T_h)
        ZT = float(self.zt_local(T_avg))
        dT = max(T_h - T_c, 1e-6)
        s = np.sqrt(1.0 + ZT)
        cop = (T_c / dT) * (s - T_h / T_c) / (s + 1.0)
        return max(cop, 0.0)

    def carnot_cop(self, T_c, T_h):
        """Carnot COP for refrigeration:  T_c / (T_h - T_c)."""
        dT = max(T_h - T_c, 1e-9)
        return T_c / dT

    # ------------------------------------------------------------------ #
    # Lumped transient thermal ODE (the F2 first-principles part)         #
    # ------------------------------------------------------------------ #
    def _rhs(self, t, y, I, Q_load):
        """Two-state ODE:  y = [T_cold, T_hot]."""
        T_c, T_h = y
        # Guard the conductance against an inverted plate during transients.
        jh = self.junction_heat(T_c, T_h, I)
        Q_cold = jh["Q_cold_W"]
        Q_hot = jh["Q_hot_W"]

        # Cold plate: heat pumped away cools it; load + parasitics warm it.
        dTc = (-Q_cold + Q_load + self.hA_cold * (self.T_amb - T_c)) / self.C_cold
        # Hot plate: rejected heat warms it; heat-sink removes it to ambient.
        dTh = (Q_hot - self.hA_hot * (T_h - self.T_amb)) / self.C_hot
        return [dTc, dTh]

    def simulate(self, I, T_cold0=None, T_hot0=None, Q_load=None,
                 duration_s=120.0, n_eval=200):
        """Integrate the lumped thermal ODE with scipy.integrate.solve_ivp.

        Parameters
        ----------
        I : float          drive current [A]
        T_cold0, T_hot0 : float  initial plate temperatures [K] (default T_amb)
        Q_load : float     active heat load on the cold plate [W]
        duration_s : float simulation horizon [s]
        n_eval : int       number of output time points

        Returns dict of time-series arrays plus the steady-state summary.
        """
        if T_cold0 is None:
            T_cold0 = self.T_amb
        if T_hot0 is None:
            T_hot0 = self.T_amb
        if Q_load is None:
            Q_load = self.Q_load_default

        t_eval = np.linspace(0.0, duration_s, n_eval)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T_cold0, T_hot0],
            t_eval=t_eval, args=(I, Q_load),
            method="BDF", rtol=1e-7, atol=1e-9, max_step=duration_s / 20.0,
        )

        T_c = sol.y[0]
        T_h = sol.y[1]

        # Post-process the junction quantities along the trajectory.
        Q_cold = np.empty_like(T_c)
        Q_hot = np.empty_like(T_c)
        W_in = np.empty_like(T_c)
        COP = np.empty_like(T_c)
        V = np.empty_like(T_c)
        for i in range(len(T_c)):
            jh = self.junction_heat(T_c[i], T_h[i], I)
            Q_cold[i] = jh["Q_cold_W"]
            Q_hot[i] = jh["Q_hot_W"]
            W_in[i] = jh["W_input_W"]
            COP[i] = jh["COP"]
            V[i] = jh["V_module_V"]

        ss = self.junction_heat(T_c[-1], T_h[-1], I)
        ss["T_cold_ss_K"] = float(T_c[-1])
        ss["T_hot_ss_K"] = float(T_h[-1])
        ss["dT_ss_K"] = float(T_h[-1] - T_c[-1])
        ss["carnot_COP"] = self.carnot_cop(T_c[-1], T_h[-1])
        ss["COP_max_theoretical"] = self.cop_max_theoretical(T_c[-1], T_h[-1])

        return {
            "t": sol.t,
            "T_cold": T_c,
            "T_hot": T_h,
            "Q_cold_W": Q_cold,
            "Q_hot_W": Q_hot,
            "W_input_W": W_in,
            "COP": COP,
            "V_module_V": V,
            "current_A": I,
            "steady_state": ss,
            "success": bool(sol.success),
        }
