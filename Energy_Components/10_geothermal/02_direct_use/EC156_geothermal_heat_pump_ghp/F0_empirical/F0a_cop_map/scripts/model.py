"""F0a empirical COP-map model for EC156 Geothermal Heat Pump (GHP).

Simplest fidelity: a 2-D lookup of heating COP vs ground-loop source
temperature and sink (supply) temperature, tabulated from the F1a relation
    COP = carnot_fraction * T_sink / (T_sink - T_source)   [Kelvin]
on a breakpoint grid. Bilinear interpolation is done with two passes of
np.interp (interp along T_sink for each bracketing T_source row, then along
T_source). Electric power input is Q_thermal/COP plus auxiliary power.

Data source: Staffell et al. (2012); ASHRAE (2011). Breakpoints reuse the
EC156 F1a cop-map design numbers.

Pure NumPy. No scipy / ODEs / AI.
"""
import numpy as np


class CopMap:
    def __init__(self, T_source_degC, T_sink_degC, COP, carnot_fraction, aux_power):
        self.T_src = np.asarray(T_source_degC, dtype=float)
        self.T_sink = np.asarray(T_sink_degC, dtype=float)
        self.COP = np.asarray(COP, dtype=float)  # shape (n_src, n_sink)
        self.carnot_fraction = float(carnot_fraction)
        self.aux_power = float(aux_power)

    def cop(self, T_source, T_sink):
        """Bilinear COP lookup, clamped to grid edges."""
        ts = float(np.clip(T_source, self.T_src[0], self.T_src[-1]))
        tk = float(np.clip(T_sink, self.T_sink[0], self.T_sink[-1]))
        # interpolate along T_sink within every source row
        per_row = np.array([np.interp(tk, self.T_sink, row) for row in self.COP])
        # then interpolate along T_source
        return float(np.interp(ts, self.T_src, per_row))

    def electric_power_kW(self, Q_thermal_kW, T_source, T_sink, part_load_ratio=1.0):
        cop = self.cop(T_source, T_sink)
        q = Q_thermal_kW * part_load_ratio
        return q / cop + self.aux_power
