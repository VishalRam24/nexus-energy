"""EC138 F0a — optional Plotly report (net efficiency & power vs dT)."""
import numpy as np
from predict import ComponentModel


def main():
    m = ComponentModel()
    dT = np.linspace(10, 28, 60)
    r = m.predict({"dT_c": dT})
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=dT, y=np.asarray(r["net_efficiency"]) * 100, name="Net eff (%)"))
        fig.add_trace(go.Scatter(x=dT, y=r["net_power_kw"], name="Net power (kW)"), secondary_y=True)
        fig.update_layout(title="EC138 F0a — OTEC net efficiency vs dT", xaxis_title="dT (degC)")
        fig.write_html("simulation_report.html")
        print("wrote simulation_report.html")
    except ImportError:
        print("plotly not installed; numeric summary:")
        for d, e, p in zip(dT[::10], r["net_efficiency"][::10], r["net_power_kw"][::10]):
            print(f"  dT={d:5.1f} C  eta={e*100:5.2f}%  P={p:6.1f} kW")


if __name__ == "__main__":
    main()
