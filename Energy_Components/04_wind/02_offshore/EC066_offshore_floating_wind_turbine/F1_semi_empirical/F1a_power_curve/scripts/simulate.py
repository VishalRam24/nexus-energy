"""EC066 — Offshore Floating Wind — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_report():
    model = ComponentModel()
    info = model.get_info()
    v = np.linspace(0, 30, 200)

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Power Curve", "Capacity Factor", "Power Coefficient (Cp)", "Density Effect"],
        vertical_spacing=0.12)

    r = model.predict({"wind_speed": v})
    fig.add_trace(go.Scatter(x=v, y=r["power_kw"], name="Power"), row=1, col=1)
    fig.add_trace(go.Scatter(x=v, y=r["capacity_factor"], name="CF"), row=1, col=2)
    fig.add_trace(go.Scatter(x=v, y=r["power_coefficient"], name="Cp"), row=2, col=1)
    fig.add_trace(go.Scatter(x=v, y=np.full_like(v, 0.593), name="Betz", line=dict(dash="dash")), row=2, col=1)

    for rho in [1.0, 1.1, 1.225, 1.35]:
        r = model.predict({"wind_speed": v, "air_density": rho})
        fig.add_trace(go.Scatter(x=v, y=r["power_kw"], name=f"rho={rho}"), row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Wind Speed (m/s)", row=i, col=j)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="CF", row=1, col=2)
    fig.update_yaxes(title_text="Cp", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}", height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

if __name__ == "__main__":
    generate_report()
