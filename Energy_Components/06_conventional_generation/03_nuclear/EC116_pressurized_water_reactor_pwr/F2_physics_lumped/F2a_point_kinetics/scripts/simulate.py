"""EC116 -- PWR Nuclear Reactor -- F2a Point Kinetics -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Step +100 pcm: Neutron Population",
            "Step +100 pcm: Temperatures",
            "Step +300 pcm: Neutron Population",
            "Step +300 pcm: Temperatures",
            "Ramp Insertion: Neutron Population",
            "Load-Following Scenario",
        ],
        vertical_spacing=0.10,
    )

    # -- Panel 1-2: Step +100 pcm --
    r1 = model.predict_step({"rho_step": 0.001, "dt": 0.1, "duration_s": 100.0})
    fig.add_trace(go.Scatter(
        x=r1["t"], y=r1["n"], name="n (+100pcm)",
        line=dict(color="#636EFA", width=2),
    ), row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=1, col=1)
    fig.update_yaxes(title_text="n / n0", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=r1["t"], y=r1["T_f"], name="T_fuel",
        line=dict(color="#EF553B", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=r1["t"], y=r1["T_m"], name="T_moderator",
        line=dict(color="#00CC96", width=2),
    ), row=1, col=2)
    fig.update_xaxes(title_text="Time [s]", row=1, col=2)
    fig.update_yaxes(title_text="Temperature [K]", row=1, col=2)

    # -- Panel 3-4: Step +300 pcm --
    r2 = model.predict_step({"rho_step": 0.003, "dt": 0.1, "duration_s": 100.0})
    fig.add_trace(go.Scatter(
        x=r2["t"], y=r2["n"], name="n (+300pcm)",
        line=dict(color="#AB63FA", width=2),
    ), row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="n / n0", row=2, col=1)

    fig.add_trace(go.Scatter(
        x=r2["t"], y=r2["T_f"], name="T_fuel (+300pcm)",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=r2["t"], y=r2["T_m"], name="T_mod (+300pcm)",
        line=dict(color="#00CC96", width=2, dash="dash"),
    ), row=2, col=2)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Temperature [K]", row=2, col=2)

    # -- Panel 5: Ramp insertion --
    r3 = model.predict_ramp({
        "rho_rate": 5e-5, "rho_max": 0.002,
        "dt": 0.1, "duration_s": 200.0,
    })
    fig.add_trace(go.Scatter(
        x=r3["t"], y=r3["n"], name="n (ramp)",
        line=dict(color="#FFA15A", width=2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=r3["t"], y=r3["rho_ext"] * 1e5, name="rho_ext [pcm]",
        line=dict(color="gray", width=1, dash="dot"),
    ), row=3, col=1)
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_yaxes(title_text="n / n0 (and rho_ext in pcm)", row=3, col=1)

    # -- Panel 6: Load-following --
    def rho_lf(t):
        if t < 50:
            return 0.0
        elif t < 100:
            return -0.002
        elif t < 200:
            return 0.0
        elif t < 250:
            return -0.001
        else:
            return 0.0

    r4 = model.predict({"rho_ext": rho_lf, "dt": 0.5, "duration_s": 400.0})
    fig.add_trace(go.Scatter(
        x=r4["t"], y=r4["P_thermal_W"] / 1e6, name="P_th [MW]",
        line=dict(color="#FF6692", width=2),
    ), row=3, col=2)
    fig.update_xaxes(title_text="Time [s]", row=3, col=2)
    fig.update_yaxes(title_text="Thermal Power [MW]", row=3, col=2)

    fig.update_layout(
        title_text=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} {info['sub_fidelity']}",
        height=1200, width=1100,
        showlegend=True,
    )

    out_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    generate_report()
