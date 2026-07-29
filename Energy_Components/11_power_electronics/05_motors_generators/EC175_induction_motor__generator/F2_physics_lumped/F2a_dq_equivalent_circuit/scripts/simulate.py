"""EC175 -- Induction Motor -- F2a dq-Frame -- Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Free Acceleration (T_load=0 Nm)",
            "Loaded Start (T_load=30 Nm)",
            "Load Step (T: 10 -> 40 Nm at t=1s)",
            "Speed & Slip vs Time (T=30 Nm)",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: Free acceleration
    r1 = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 0.0,
        "dt": 5e-4, "duration_s": 2.0,
    })
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["speed_rpm"], name="Speed (no load)", line=dict(color="#636EFA", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["torque"], name="Torque (no load)", line=dict(color="#EF553B", width=1.5)), row=1, col=1)

    # Panel 2: Loaded start
    r2 = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 30.0,
        "dt": 5e-4, "duration_s": 2.0,
    })
    fig.add_trace(go.Scatter(x=r2["t"], y=r2["speed_rpm"], name="Speed (30 Nm)", line=dict(color="#00CC96", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=r2["t"], y=r2["torque"], name="Torque (30 Nm)", line=dict(color="#FFA15A", width=1.5)), row=1, col=2)

    # Panel 3: Load step
    def t_step(t):
        return 10.0 if t < 1.0 else 40.0
    r3 = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": t_step,
        "dt": 5e-4, "duration_s": 3.0,
    })
    fig.add_trace(go.Scatter(x=r3["t"], y=r3["speed_rpm"], name="Speed (load step)", line=dict(color="#AB63FA", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=r3["t"], y=r3["torque"], name="Torque (load step)", line=dict(color="#FF6692", width=1.5)), row=2, col=1)

    # Panel 4: Speed and slip
    fig.add_trace(go.Scatter(x=r2["t"], y=r2["speed_rpm"], name="Speed", line=dict(color="#19D3F3", width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=r2["t"], y=r2["slip"] * 100, name="Slip (%)", line=dict(color="#B6E880", width=2)), row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Time (s)", row=r, col=c)
    fig.update_yaxes(title_text="Speed (rpm) / Torque (Nm)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (rpm) / Torque (Nm)", row=1, col=2)
    fig.update_yaxes(title_text="Speed (rpm) / Torque (Nm)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (rpm) / Slip (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} dq-Frame Model",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print(f"\n--- IM F2a Final Values (400V, 50Hz, T_load=30Nm) ---")
    print(f"Speed = {r2['speed_rpm'][-1]:.1f} rpm")
    print(f"Torque = {r2['torque'][-1]:.2f} Nm")
    print(f"Slip = {r2['slip'][-1]:.4f}")
    print(f"Current = {r2['current'][-1]:.2f} A")
    print(f"Power = {r2['power'][-1]:.1f} W")


if __name__ == "__main__":
    generate_report()
