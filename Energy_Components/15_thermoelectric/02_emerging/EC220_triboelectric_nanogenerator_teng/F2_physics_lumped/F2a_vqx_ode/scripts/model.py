"""
EC220 -- Triboelectric Nanogenerator (TENG) -- F2a Physics-Lumped V-Q-x Model

Contact-separation mode TENG integrated with an external resistive load via the
Niu/Wang governing equation. This is the first-principles (lumped, 0D) upgrade of
the F1a/F1b capacitive equivalent-circuit models: instead of an algebraic
voltage-divider approximation, the transferred charge Q(t) is obtained by
integrating the coupled charge ODE over the time-dependent gap x(t).

Governing equation (Niu et al. 2013, "Theoretical study of contact-mode TENG"):

    V(t) = -Q(t) / C(x)  +  V_oc(x)                                          (1)

with the time-dependent device capacitance and open-circuit voltage

    C(x)    = eps_0 * A / (d0 + x)                                           (2)
    V_oc(x) = sigma * x / eps_0                                              (3)
    d0      = d / eps_r          (effective dielectric thickness, in vacuum) (4)

Here Q is the charge that has flowed through the external circuit, x(t) is the
contact-separation gap, sigma the triboelectric surface charge density, A the
electrode area, eps_0 the vacuum permittivity, and d0 the equivalent vacuum
thickness of the dielectric stack. C(x) varies with the gap x(t).

Coupling to an external resistive load R closes the circuit through Ohm's law,
V(t) = R * dQ/dt, giving the charge ODE that is integrated with scipy.solve_ivp:

    R * dQ/dt = -Q * (d0 + x(t)) / (eps_0 * A)  +  sigma * x(t) / eps_0      (5)

Motion profile (contact-separation, smooth so x in [0, x_max]):

    x(t) = x_max/2 * (1 - cos(omega * t)),   omega = 2*pi*f                  (6)

Outputs: charge Q(t), terminal voltage V(t) = R*dQ/dt, current I = dQ/dt,
instantaneous power P = I^2 * R, energy per cycle, average power; plus sweeps of
average power vs load resistance (with the optimum load) and vs frequency.

Conservation / limits enforced:
  * Charge conservation: with a periodic gap the cycle-averaged net charge
    transfer is zero (Q returns to its periodic orbit; integral of I over a
    cycle -> 0 at steady state).
  * The transferred charge is bounded by the saturation Q_sc = sigma*A*x_max/(d0+x_max).
  * Energy per cycle E = integral(P dt) > 0 and scales ~ sigma^2 and with x_max.
  * Power vs load has a single interior optimum (impedance match to the
    capacitive internal impedance ~ 1/(omega*C)).

References:
    Niu, S., Wang, S., Lin, L., Liu, Y., Zhou, Y.S., Hu, Y., Wang, Z.L. (2013).
        "Theoretical study of contact-mode triboelectric nanogenerators as an
        effective power source." Energy Environ. Sci. 6, 3576-3583.
    Niu, S. & Wang, Z.L. (2015). "Theoretical systems of triboelectric
        nanogenerators." Nano Energy 14, 161-192.
    Wang, Z.L. (2013). "Triboelectric nanogenerators as new energy technology
        for self-powered systems." ACS Nano 7(11), 9533-9557.
"""

import numpy as np
from scipy.integrate import solve_ivp

eps_0 = 8.854187817e-12   # F/m, vacuum permittivity


class TENG_F2a:
    """Contact-separation TENG -- V-Q-x governing ODE with external load."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.sigma = u["sigma"]["value"]                  # C/m^2
        self.A = u["electrode_area"]["value"]              # m^2
        self.x_max = u["gap_max"]["value"]                 # m
        self.eps_r = u["epsilon_r_dielectric"]["value"]    # -
        self.d = u["dielectric_thickness"]["value"]        # m
        # Effective dielectric thickness reduced to vacuum equivalent (Niu 2013, eq. 4)
        self.d0 = self.d / self.eps_r                       # m

    # ------------------------------------------------------------------ kinematics
    def gap(self, t, f):
        """Contact-separation gap x(t) in [0, x_max], smooth cosine profile."""
        omega = 2.0 * np.pi * f
        return 0.5 * self.x_max * (1.0 - np.cos(omega * t))

    def gap_velocity(self, t, f):
        """dx/dt of the cosine profile."""
        omega = 2.0 * np.pi * f
        return 0.5 * self.x_max * omega * np.sin(omega * t)

    # ------------------------------------------------------------------ statics
    def capacitance(self, x):
        """Device capacitance C(x) = eps_0*A/(d0+x)  [F]  (Niu 2013, eq. 2)."""
        x = np.asarray(x, dtype=float)
        return eps_0 * self.A / (self.d0 + x)

    def v_oc(self, x):
        """Open-circuit voltage V_oc(x) = sigma*x/eps_0  [V]  (Niu 2013, eq. 3)."""
        x = np.asarray(x, dtype=float)
        return self.sigma * x / eps_0

    def q_saturation(self):
        """Short-circuit (saturation) transferred charge Q_sc = sigma*A*x_max/(d0+x_max) [C].

        Upper bound on |Q| obtained by setting V=0 (short circuit) at x=x_max:
            0 = -Q/C(x_max) + V_oc(x_max)  =>  Q = C(x_max)*V_oc(x_max).
        """
        Cmax = eps_0 * self.A / (self.d0 + self.x_max)
        return Cmax * self.v_oc(self.x_max)

    # ------------------------------------------------------------------ ODE
    def _dQdt(self, t, Q, f, R):
        """RHS of the charge ODE (eq. 5): dQ/dt = V/R with V from the V-Q-x relation."""
        x = self.gap(t, f)
        # V = -Q/C(x) + V_oc(x);  C(x)=eps_0*A/(d0+x)  ->  1/C = (d0+x)/(eps_0*A)
        V = -Q * (self.d0 + x) / (eps_0 * self.A) + self.sigma * x / eps_0
        return V / R

    def simulate(self, frequency_hz=3.0, R_load_ohm=1e7, n_cycles=5,
                 points_per_cycle=400, Q0=0.0):
        """
        Integrate the TENG charge ODE over n_cycles of contact-separation motion.

        Returns dict with time series and cycle-averaged scalars. The last full
        cycle is used for the (periodic, steady-state) energy/power metrics.
        """
        f = float(frequency_hz)
        R = float(R_load_ohm)
        T_cycle = 1.0 / f
        t_end = n_cycles * T_cycle
        n_pts = int(n_cycles * points_per_cycle) + 1
        t_eval = np.linspace(0.0, t_end, n_pts)

        sol = solve_ivp(
            self._dQdt, (0.0, t_end), [Q0], t_eval=t_eval,
            args=(f, R), method="LSODA", rtol=1e-8, atol=1e-16,
            max_step=T_cycle / 50.0,
        )
        t = sol.t
        Q = sol.y[0]

        x = self.gap(t, f)
        V = -Q * (self.d0 + x) / (eps_0 * self.A) + self.sigma * x / eps_0
        I = V / R
        P = I * I * R   # = V^2/R, instantaneous dissipated power in load

        # Metrics over the LAST full cycle (steady periodic orbit)
        mask = t >= (t_end - T_cycle)
        t_c = t[mask]
        P_c = P[mask]
        I_c = I[mask]
        energy_per_cycle = np.trapz(P_c, t_c)          # J
        power_avg = energy_per_cycle / T_cycle          # W
        net_charge_per_cycle = np.trapz(I_c, t_c)       # C (should ~0 -> conservation)

        area_cm2 = self.A * 1e4
        power_density_mwcm2 = power_avg * 1e3 / area_cm2

        return {
            "t": t,
            "gap": x,
            "charge": Q,
            "voltage": V,
            "current": I,
            "power": P,
            "capacitance": self.capacitance(x),
            "v_oc": self.v_oc(x),
            "energy_per_cycle": energy_per_cycle,
            "power_avg": power_avg,
            "power_density_mwcm2": power_density_mwcm2,
            "net_charge_per_cycle": net_charge_per_cycle,
            "V_peak": float(np.max(np.abs(V))),
            "I_peak": float(np.max(np.abs(I))),
            "frequency_hz": f,
            "R_load_ohm": R,
        }

    # ------------------------------------------------------------------ sweeps
    def power_vs_load(self, frequency_hz=3.0, R_list=None, n_cycles=6):
        """Average power over a sweep of load resistances. Returns (R, P_avg, R_opt)."""
        if R_list is None:
            R_list = np.logspace(4, 10, 40)
        R_list = np.asarray(R_list, dtype=float)
        P = np.array([self.simulate(frequency_hz, R, n_cycles)["power_avg"]
                      for R in R_list])
        R_opt = float(R_list[int(np.argmax(P))])
        return R_list, P, R_opt

    def power_vs_frequency(self, f_list=None, R_load_ohm=1e7, n_cycles=6):
        """Average power over a sweep of frequencies. Returns (f, P_avg)."""
        if f_list is None:
            f_list = np.logspace(np.log10(0.5), np.log10(50.0), 25)
        f_list = np.asarray(f_list, dtype=float)
        P = np.array([self.simulate(fr, R_load_ohm, n_cycles)["power_avg"]
                      for fr in f_list])
        return f_list, P

    def optimal_load(self, frequency_hz=3.0):
        """Best resistive load (max average power) at a given frequency."""
        _, P, R_opt = self.power_vs_load(frequency_hz)
        return R_opt
