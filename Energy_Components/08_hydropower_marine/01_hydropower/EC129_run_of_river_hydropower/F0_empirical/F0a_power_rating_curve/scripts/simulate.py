"""EC129 F0a — optional Plotly report (power & efficiency vs flow)."""
import numpy as np
from predict import ComponentModel


def main():
    m = ComponentModel()
    Q = np.linspace(12.5, 57.5, 60)
    r = m.predict({"flow_rate_m3s": Q, "head_m": 8.0})
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=Q, y=r["power_kw"], name="Power (kW)"))
        fig.add_trace(go.Scatter(x=Q, y=r["overall_efficiency"], name="Overall eta"), secondary_y=True)
        fig.update_layout(title="EC129 F0a — Run-of-river power-rating curve", xaxis_title="Flow (m3/s)")
        out = "simulation_report.html"
        fig.write_html(out)
        print(f"wrote {out}")
    except ImportError:
        print("plotly not installed; numeric summary:")
        for q, p, e in zip(Q[::10], r["power_kw"][::10], r["overall_efficiency"][::10]):
            print(f"  Q={q:5.1f}  P={p:8.0f} kW  eta={e:.3f}")


if __name__ == "__main__":
    main()
