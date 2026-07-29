"""
EC126 -- Flywheel Energy Storage -- F2a Dynamic ODE Model

Physics-lumped model with rotational dynamics ODE.

State variable:
    omega - rotor angular speed [rad/s]

Governing equation:
    J * d_omega/dt = T_motor - T_load - T_friction(omega)

Energy stored:
    E_stored = 0.5 * J * omega^2

Friction model:
    T_friction = c_windage * omega^2 + T_bearing

Motor/generator with efficiency map:
    Charge:    T_motor = P_command * eta_motor * eta_pe / omega
    Discharge: T_load  = P_command / (eta_motor * eta_pe * omega)

SOC tracking:
    SOC = (E - E_min) / (E_max - E_min)
    E_min = 0.5 * J * omega_min^2

Reference:
    Amiryar & Pullen (2017), Applied Sciences 7(3):286
    Beacon Power 20 MW Stephentown Plant Data
"""

import numpy as np
from scipy.integrate import solve_ivp


class FlywheelStorage_F2a:
    """Flywheel Energy Storage -- dynamic rotational ODE model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.J = u["J"]["value"]
        self.omega_max = u["omega_max"]["value"]
        self.omega_min = u["omega_min"]["value"]
        self.P_rated = u["P_rated"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.eta_pe = u["eta_pe"]["value"]
        self.c_windage = u["c_windage"]["value"]
        self.T_bearing = u["T_bearing"]["value"]
        self.E_max = 0.5 * self.J * self.omega_max**2
        self.E_min = 0.5 * self.J * self.omega_min**2

    # ------------------------------------------------------------------
    # Friction torque
    # ------------------------------------------------------------------
    def friction_torque(self, omega):
        """Total friction torque [N.m] = windage + bearing."""
        return self.c_windage * omega**2 + self.T_bearing

    # ------------------------------------------------------------------
    # Motor/generator torque from power command
    # ------------------------------------------------------------------
    def motor_torque(self, P_command, omega):
        """
        Motor torque from electrical power command.
        P_command > 0: charge (motor mode), torque accelerates rotor
        P_command < 0: discharge (generator mode), torque decelerates rotor
        """
        if omega < 1.0:
            # Below minimum speed, apply startup torque if charging
            if P_command > 0:
                return min(P_command / max(omega, 0.1), self.P_rated / 10.0)
            return 0.0

        if P_command > 0:
            # Charge: electrical -> mechanical
            eta = self.eta_motor * self.eta_pe
            T = P_command * eta / omega
        elif P_command < 0:
            # Discharge: mechanical -> electrical
            eta = self.eta_motor * self.eta_pe
            T = P_command / (eta * omega)  # Negative torque
        else:
            T = 0.0

        return T

    # ------------------------------------------------------------------
    # Stored energy and SOC
    # ------------------------------------------------------------------
    def stored_energy(self, omega):
        """Kinetic energy stored [J]."""
        return 0.5 * self.J * omega**2

    def soc(self, omega):
        """State of charge [-]."""
        E = self.stored_energy(omega)
        if self.E_max <= self.E_min:
            return 0.0
        return np.clip((E - self.E_min) / (self.E_max - self.E_min), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Round-trip efficiency
    # ------------------------------------------------------------------
    def instantaneous_efficiency(self, P_command, omega):
        """Instantaneous efficiency including friction losses."""
        T_f = self.friction_torque(omega)
        P_friction = T_f * omega
        if P_command > 0:
            # Charging: how much of input actually stored
            P_stored = P_command * self.eta_motor * self.eta_pe - P_friction
            return max(P_stored / max(P_command, 1.0), 0.0)
        elif P_command < 0:
            # Discharging: output vs mechanical extraction
            P_mech = abs(P_command) / (self.eta_motor * self.eta_pe)
            return abs(P_command) / max(P_mech + P_friction, 1.0)
        return 0.0

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, P_func):
        omega = y[0]
        omega = max(omega, 0.0)

        P_cmd = P_func(t)

        # Enforce limits
        E = self.stored_energy(omega)
        if E >= self.E_max and P_cmd > 0:
            P_cmd = 0.0  # Fully charged
        if omega <= self.omega_min and P_cmd < 0:
            P_cmd = 0.0  # Fully discharged (at min speed)

        # Clamp power to rated
        P_cmd = np.clip(P_cmd, -self.P_rated, self.P_rated)

        T_motor = self.motor_torque(P_cmd, omega)
        T_friction = self.friction_torque(omega)

        domega_dt = (T_motor - T_friction) / self.J

        # Prevent omega from going negative
        if omega <= 0.01 and domega_dt < 0:
            domega_dt = 0.0

        return [domega_dt]

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, P_command, omega0=None, dt=0.1, duration_s=3600.0):
        """
        Simulate flywheel dynamics.

        Parameters
        ----------
        P_command : float or callable(t)
            Power command [W]. Positive=charge, negative=discharge.
        omega0 : float
            Initial angular speed [rad/s]. Default: omega_max (fully charged).
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation time [s]

        Returns
        -------
        dict with time series
        """
        if omega0 is None:
            omega0 = self.omega_max

        _P = P_command if callable(P_command) else lambda t: P_command

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, _P),
            (0.0, duration_s), [omega0],
            t_eval=t_eval, method="RK45",
            rtol=1e-8, atol=1e-10, max_step=dt
        )

        t_out = sol.t
        omega_out = sol.y[0]
        N = len(t_out)

        E_stored = np.zeros(N)
        soc_out = np.zeros(N)
        P_actual = np.zeros(N)
        P_loss = np.zeros(N)
        efficiency = np.zeros(N)
        T_friction_out = np.zeros(N)

        for i in range(N):
            w = omega_out[i]
            E_stored[i] = self.stored_energy(w)
            soc_out[i] = self.soc(w)
            P_cmd = _P(t_out[i])
            P_actual[i] = P_cmd
            T_f = self.friction_torque(w)
            T_friction_out[i] = T_f
            P_loss[i] = T_f * w
            efficiency[i] = self.instantaneous_efficiency(P_cmd, w)

        return {
            "t": t_out,
            "omega": omega_out,
            "E_stored": E_stored,
            "SOC": soc_out,
            "P_command": P_actual,
            "P_loss": P_loss,
            "efficiency": efficiency,
            "T_friction": T_friction_out,
        }
