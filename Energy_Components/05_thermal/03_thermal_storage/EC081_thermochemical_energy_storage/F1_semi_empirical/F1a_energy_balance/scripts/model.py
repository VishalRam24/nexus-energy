"""
EC081 — Thermochemical Energy Storage — F1a Energy Balance Model

E_stored = m * dH_rxn * x          (stored energy at reaction extent x)
E_dischargeable = E_stored * eta_rt (usable discharge energy)
Q_loss ~ 0 (thermochemical storage has near-zero standby losses)

Charge: AB + heat → A + B  (endothermic, x increases)
Discharge: A + B → AB + heat  (exothermic, x decreases)

References:
    Pardo et al. (2014). A review on high temperature thermochemical heat
        energy storage. Renewable and Sustainable Energy Reviews.
    N'Tsoukpoe et al. (2009). A review on long-term sorption solar energy storage.
"""

import numpy as np


class ThermochemicalStorageF1a:
    """Energy balance model for thermochemical storage."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.m = u["m"]["value"]
        self.dH = u["dH_rxn"]["value"]
        self.eta_rt = u["eta_rt"]["value"]
        self.E_max = self.m * self.dH  # J

    def predict(self, x, mode="discharge"):
        """
        Parameters
        ----------
        x : float or array, reaction extent [0,1] (1=fully charged)
        mode : 'charge' or 'discharge'

        Returns
        -------
        dict with E_stored_kWh, E_usable_kWh, SOC, P_discharge_potential_kW
        """
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0.0, 1.0)

        E_stored_J = self.m * self.dH * x
        E_usable_J = E_stored_J * self.eta_rt

        # Indicative discharge power (assuming 1-hour discharge)
        P_kW = E_usable_J / 3600.0 / 1000.0  # kW for 1-hour

        return {
            "E_stored_kWh": E_stored_J / 3.6e6,
            "E_usable_kWh": E_usable_J / 3.6e6,
            "SOC": x,
            "E_max_kWh": self.E_max / 3.6e6,
            "eta_rt": self.eta_rt,
            "P_1h_kW": P_kW,
        }
