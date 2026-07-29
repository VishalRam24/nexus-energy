"""
EC070 -- Water-Source Heat Pump -- F2a Vapor-Compression Cycle
Optional Plotly report. Plotly import wrapped so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()

    # --- COP vs sink temperature for several source temps ---
    sinks = np.linspace(25, 60, 36)
    sources = [5, 12, 20, 30]
    cop_curves = {}
    carnot_curves = {}
    for Ts in sources:
        cop_curves[Ts] = [cm.predict({"mode": "cycle", "T_source_c": Ts,
                                      "T_sink_c": Tk})["cop_heat"] for Tk in sinks]
        carnot_curves[Ts] = [cm.predict({"mode": "cycle", "T_source_c": Ts,
                                         "T_sink_c": Tk})["cop_carnot"] for Tk in sinks]

    # --- transient warm-up ---
    tr = cm.predict({"mode": "transient", "T_source_c": 12.0, "T_load0_c": 25.0,
                     "Q_demand_W": 18000.0, "duration_s": 3600.0, "dt": 30.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"[simulate] Sample COP @ src=12,sink=45: "
              f"{cm.predict({'mode':'cycle','T_source_c':12,'T_sink_c':45})['cop_heat']:.3f}")
        return

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("COP vs sink temperature",
                                        "Load-loop transient warm-up"))
    for Ts in sources:
        fig.add_trace(go.Scatter(x=sinks, y=cop_curves[Ts], mode="lines",
                                 name=f"COP src={Ts}C"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sinks, y=carnot_curves[Ts], mode="lines",
                                 line=dict(dash="dot"),
                                 name=f"Carnot src={Ts}C"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tr["t"] / 60.0, y=tr["T_load"], mode="lines",
                             name="T_load [C]"), row=1, col=2)

    fig.update_xaxes(title_text="Sink temperature [degC]", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_xaxes(title_text="Time [min]", row=1, col=2)
    fig.update_yaxes(title_text="Load temperature [degC]", row=1, col=2)
    fig.update_layout(title="EC070 WSHP F2a -- Vapor-Compression Cycle")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    main()
