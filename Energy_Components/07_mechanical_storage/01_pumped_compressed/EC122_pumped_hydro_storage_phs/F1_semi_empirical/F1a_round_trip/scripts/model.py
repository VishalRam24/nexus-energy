"""
EC122 — Pumped Hydro Storage (PHS) — F1a Round-Trip Model

Generation: P_gen = eta_turbine * eta_generator * rho * g * Q * H / 1000  [kW]
Pumping:    P_pump = rho * g * Q * H / (eta_pump * eta_motor * 1000)       [kW]
Round-trip eta = eta_turbine * eta_generator * eta_pump * eta_motor
Energy:     E = rho * g * V * H / 3.6e9                                    [GWh]

Reference:
    Rehman et al. (2015). Renewable and Sustainable Energy Reviews, 44, 586-598.
"""

import numpy as np


class PHSF1a:
    """Pumped Hydro Storage — round-trip semi-empirical model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_turbine = u["eta_turbine"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_generator = u["eta_generator"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.rho = u["rho_water"]["value"]       # kg/m3
        self.g = u["g"]["value"]                 # m/s2
        self.V_reservoir = u["reservoir_volume"]["value"]  # m3

    def generation_power(self, flow_rate, head):
        """Turbine-generator output power [kW]."""
        Q = np.asarray(flow_rate, dtype=float)
        H = np.asarray(head, dtype=float)
        return self.eta_turbine * self.eta_generator * self.rho * self.g * Q * H / 1000.0

    def pumping_power(self, flow_rate, head):
        """Motor-pump input power required [kW]."""
        Q = np.asarray(flow_rate, dtype=float)
        H = np.asarray(head, dtype=float)
        return self.rho * self.g * Q * H / (self.eta_pump * self.eta_motor * 1000.0)

    def round_trip_efficiency(self):
        """Overall round-trip efficiency (dimensionless)."""
        return self.eta_turbine * self.eta_generator * self.eta_pump * self.eta_motor

    def energy_capacity(self, head, reservoir_volume=None):
        """Stored energy capacity [GWh] for given reservoir volume."""
        V = reservoir_volume if reservoir_volume is not None else self.V_reservoir
        H = np.asarray(head, dtype=float)
        V = np.asarray(V, dtype=float)
        return self.rho * self.g * V * H / 3.6e9

    def generation_efficiency(self):
        """One-way generation efficiency (turbine × generator)."""
        return self.eta_turbine * self.eta_generator

    def pump_efficiency(self):
        """One-way pumping efficiency (pump × motor)."""
        return self.eta_pump * self.eta_motor
