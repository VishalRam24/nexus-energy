"""
EC019 -- NMC Battery -- F2a ECM 1-RC Model

1-RC Equivalent Circuit Model with SOC-dependent parameters:
    V_terminal = OCV(SOC) - I*R0(SOC) - V_rc
    dV_rc/dt   = I/C1(SOC) - V_rc / (R1(SOC) * C1(SOC))
    dSOC/dt    = -I / (Q_nom * 3600)

Where R0, R1, C1 are SOC-dependent via quadratic multipliers.

Reference:
    Chen et al. (2020). J. Electrochem. Soc., 167, 080534.
"""

import numpy as np


class NMCBatteryECM1RC:
    """1-RC equivalent circuit model for NMC811 battery."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]
        soc_dep = params["soc_dependent_params"]

        self.Q_nom = cell["capacity"]["value"]
        self.v_max = cell["voltage_max"]["value"]
        self.v_min = cell["voltage_min"]["value"]
        self.R0_nom = cell["R0"]["value"]
        self.R1_nom = cell["R1"]["value"]
        self.C1_nom = cell["C1"]["value"]

        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

        self.R0_soc_coeffs = np.array(soc_dep["R0_coeffs"])
        self.R1_soc_coeffs = np.array(soc_dep["R1_coeffs"])
        self.C1_soc_coeffs = np.array(soc_dep["C1_coeffs"])

        self.soc = 1.0
        self.v_rc = 0.0

    def reset(self, soc_init=1.0):
        """Reset state to initial conditions."""
        self.soc = np.clip(soc_init, 0.0, 1.0)
        self.v_rc = 0.0

    def _soc_factor(self, soc, coeffs):
        """Evaluate quadratic SOC multiplier: b0 + b1*SOC + b2*SOC^2."""
        soc = np.clip(soc, 0.0, 1.0)
        return coeffs[0] + coeffs[1] * soc + coeffs[2] * soc**2

    def R0(self, soc):
        """SOC-dependent series resistance (Ohm)."""
        return self.R0_nom * np.clip(self._soc_factor(soc, self.R0_soc_coeffs), 0.1, 10.0)

    def R1(self, soc):
        """SOC-dependent polarization resistance (Ohm)."""
        return self.R1_nom * np.clip(self._soc_factor(soc, self.R1_soc_coeffs), 0.1, 10.0)

    def C1(self, soc):
        """SOC-dependent polarization capacitance (F)."""
        return self.C1_nom * np.clip(self._soc_factor(soc, self.C1_soc_coeffs), 0.1, 10.0)

    def tau1(self, soc):
        """RC time constant (seconds)."""
        return self.R1(soc) * self.C1(soc)

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def step(self, current, dt):
        """
        Advance model state by dt seconds.

        Args:
            current: Current in A (positive = discharge, negative = charge)
            dt: Time step in seconds

        Returns:
            Terminal voltage (V)
        """
        current = float(current)
        dt = float(dt)

        r0 = self.R0(self.soc)
        r1 = self.R1(self.soc)
        c1 = self.C1(self.soc)
        tau = r1 * c1

        # Exact solution of RC ODE for constant current over dt
        exp_factor = np.exp(-dt / tau)
        self.v_rc = current * r1 * (1.0 - exp_factor) + self.v_rc * exp_factor

        # Coulomb counting
        self.soc -= current * dt / (self.Q_nom * 3600.0)
        self.soc = np.clip(self.soc, 0.0, 1.0)

        v_term = self.ocv(self.soc) - current * r0 - self.v_rc
        v_term = np.clip(v_term, self.v_min, self.v_max)

        return float(v_term)

    def simulate(self, current_profile, dt):
        """
        Run full time-series simulation.

        Args:
            current_profile: Array of current values at each timestep (A)
            dt: Time step in seconds

        Returns:
            dict with keys: voltage, soc, v_rc, power, time
        """
        current_profile = np.asarray(current_profile, dtype=float)
        n = len(current_profile)

        voltages = np.zeros(n)
        socs = np.zeros(n)
        v_rcs = np.zeros(n)
        powers = np.zeros(n)
        times = np.arange(n) * dt

        for i in range(n):
            v = self.step(current_profile[i], dt)
            voltages[i] = v
            socs[i] = self.soc
            v_rcs[i] = self.v_rc
            powers[i] = v * current_profile[i]

        return {
            "voltage": voltages,
            "soc": socs,
            "v_rc": v_rcs,
            "power": powers,
            "time": times,
        }

    def static_voltage(self, soc, current):
        """F1a-style static voltage (no RC dynamics) for comparison."""
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        v = self.ocv(soc) - current * self.R0(soc)
        return np.clip(v, self.v_min, self.v_max)
