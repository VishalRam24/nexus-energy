"""
EC090 -- Solar Water Heater Combi System -- F2a Physics-Lumped Stratified Tank
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarCombiF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Default diurnal profiles (a clear summer day) -- used when caller passes
# scalars / nothing. All are callables of time t [s] since midnight.
# ---------------------------------------------------------------------------
def default_irradiance(G_peak=900.0):
    """Half-sine plane-of-array irradiance, 0 before 06:00 and after 18:00."""
    def G(t):
        hour = (t / 3600.0) % 24.0
        if 6.0 <= hour <= 18.0:
            return G_peak * np.sin(np.pi * (hour - 6.0) / 12.0)
        return 0.0
    return G


def default_ambient(T_min=283.15, T_max=298.15):
    """Sinusoidal ambient temperature, min ~05:00, max ~15:00."""
    def T(t):
        hour = (t / 3600.0) % 24.0
        return 0.5 * (T_max + T_min) - 0.5 * (T_max - T_min) * np.cos(
            np.pi * (hour - 5.0) / 12.0
        )
    return T


def default_dhw_load(morning_lpm=8.0, evening_lpm=10.0):
    """DHW draw [kg/s]: morning (07:00-08:00) + evening (19:00-21:00) peaks."""
    def L(t):
        hour = (t / 3600.0) % 24.0
        lpm = 0.0
        if 7.0 <= hour <= 8.0:
            lpm = morning_lpm
        elif 19.0 <= hour <= 21.0:
            lpm = evening_lpm
        return lpm / 60.0  # L/min -> kg/s (rho=1)
    return L


def default_space_load(Q_peak=4000.0):
    """Space-heating load [W]: morning + evening, none midday."""
    def S(t):
        hour = (t / 3600.0) % 24.0
        if 6.0 <= hour <= 9.0 or 18.0 <= hour <= 23.0:
            return Q_peak
        return 0.0
    return S


class ComponentModel:
    """Standardised wrapper for EC090 F2a stratified-tank solar combi model."""

    component_id = "EC090"
    component_name = "Solar Water Heater Combi System"
    fidelity = "F2a -- Physics-Lumped Stratified Tank Combi System"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SolarCombiF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Simulate the solar combi system over a horizon.

        inputs (all optional, sensible defaults for a clear summer day):
            G_peak : float        peak plane-of-array irradiance [W/m2] (900)
            G_profile : callable  override full irradiance profile G(t)
            T_amb_profile : callable  override ambient T(t) [K]
            dhw_morning_lpm, dhw_evening_lpm : float DHW peak draws [L/min]
            load_profile : callable   override combined draw flow [kg/s]
            space_peak_W : float  space-heating peak [W] (4000)
            space_profile : callable  override space load [W]
            T_init : list(N)      initial node temps [K] (top..bottom)
            dt : float            output step [s] (300)
            duration_s : float    horizon [s] (86400 = 1 day)
        """
        Gf = inputs.get("G_profile") or default_irradiance(inputs.get("G_peak", 900.0))
        Tf = inputs.get("T_amb_profile") or default_ambient()
        Lf = inputs.get("load_profile") or default_dhw_load(
            inputs.get("dhw_morning_lpm", 8.0), inputs.get("dhw_evening_lpm", 10.0)
        )
        Sf = inputs.get("space_profile") or default_space_load(
            inputs.get("space_peak_W", 4000.0)
        )
        T_init = inputs.get("T_init", None)
        dt = inputs.get("dt", 300.0)
        dur = inputs.get("duration_s", 86400.0)

        return self._model.simulate(Gf, Tf, Lf, Sf, T_init, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "G_peak": {"unit": "W/m2", "range": [0, 1100]},
                "T_amb_profile": {"unit": "K", "range": [253.15, 318.15]},
                "dhw_morning_lpm": {"unit": "L/min", "range": [0, 20]},
                "dhw_evening_lpm": {"unit": "L/min", "range": [0, 20]},
                "space_peak_W": {"unit": "W", "range": [0, 8000]},
                "dt": {"unit": "s", "range": [10, 600]},
                "duration_s": {"unit": "s", "range": [60, 172800]},
            },
            "outputs": {
                "t": "s",
                "T_nodes": "K (N x M, top..bottom)",
                "T_top": "K",
                "T_bottom": "K",
                "Q_solar": "W",
                "Q_aux_fuel": "W",
                "Q_load": "W",
                "pump_on": "0/1",
                "E_solar_J": "J",
                "E_aux_delivered_J": "J",
                "solar_fraction": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"dt": 300.0, "duration_s": 86400.0})
    print(
        f"\nDay simulation: solar_fraction={r['solar_fraction']:.3f}, "
        f"E_solar={r['E_solar_J']/3.6e6:.2f} kWh, "
        f"E_aux(fuel)={r['E_aux_fuel_J']/3.6e6:.2f} kWh, "
        f"E_load={r['E_load_J']/3.6e6:.2f} kWh"
    )
    print(
        f"Tank top: {r['T_top'].min()-273.15:.1f}..{r['T_top'].max()-273.15:.1f} C, "
        f"bottom: {r['T_bottom'].min()-273.15:.1f}..{r['T_bottom'].max()-273.15:.1f} C, "
        f"pump on {r['pump_on'].mean()*100:.0f}% of day"
    )
