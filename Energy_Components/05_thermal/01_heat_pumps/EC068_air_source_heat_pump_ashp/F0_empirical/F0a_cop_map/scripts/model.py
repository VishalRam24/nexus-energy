"""F0a empirical COP lookup map for Air-Source Heat Pump (ASHP).

Data source: Staffell et al. (2012), Energy Environ. Sci., 5, 9291-9306
The COP table was generated from the Carnot-fraction relation
    COP = eta_c * T_sink / (T_sink - T_source)   (absolute temperatures)
and is reused here as a pure tabulated lookup (NumPy only, no scipy/AI).
"""
import numpy as np


class CopMap:
    def __init__(self, src_bp, sink_bp, table):
        self.src = np.asarray(src_bp, dtype=float)
        self.sink = np.asarray(sink_bp, dtype=float)
        self.table = np.asarray(table, dtype=float)  # [n_sink, n_src]

    def cop(self, t_source, t_sink):
        """Bilinear lookup of COP at (t_source, t_sink). Clipped to table bounds."""
        ts = float(np.clip(t_source, self.src[0], self.src[-1]))
        tk = float(np.clip(t_sink, self.sink[0], self.sink[-1]))
        per_sink = np.array([np.interp(ts, self.src, self.table[i]) for i in range(len(self.sink))])
        return float(np.interp(tk, self.sink, per_sink))
