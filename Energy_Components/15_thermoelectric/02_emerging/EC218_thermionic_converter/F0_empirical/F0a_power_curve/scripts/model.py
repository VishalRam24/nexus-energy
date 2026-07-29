"""EC218 Thermionic Converter F0a - empirical power-density vs emitter-temp lookup.

Output power density of a Cs-vapor thermionic diode tabulated against emitter
temperature. Breakpoints from the ideal Richardson-Dushman emission relation
    J = A * Te^2 * exp(-phi_e / (k*Te)),  P = J * (phi_emitter - phi_collector).
The table also exposes total power for a given emitter area.

Data source: Hatsopoulos & Gyftopoulos (1979) Thermionic Energy Conversion;
Angrist (1982) Direct Energy Conversion.
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class ThermionicPowerCurve:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.area = p["emitter_area"]["value"]
        self.V_out = p["output_voltage"]["value"]
        self._te = np.asarray(p["lookup"]["T_emitter_K"]["value"], dtype=float)
        self._pd = np.asarray(p["lookup"]["power_density_W_m2"]["value"], dtype=float)

    def power_density(self, T_emitter_K):
        """Interpolated output power density (W/m^2) vs emitter temperature."""
        return float(np.interp(T_emitter_K, self._te, self._pd))

    def power(self, T_emitter_K):
        """Total electrical output power (W) for the rated emitter area."""
        return self.power_density(T_emitter_K) * self.area

    def breakpoints(self):
        return self._te.copy(), self._pd.copy()
