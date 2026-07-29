"""EC158 -- Boost Converter -- F2a Averaged SSM -- Simulation & HTML Report"""
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
        rows=2, cols=2,
        subplot_titles=[
            "Startup Transient (V_in=12V, D=0.75, R=48 Ohm)",
            "Load Step (R: 48 -> 24 Ohm at t=25ms)",
            "Duty Cycle Step (D: 0.75 -> 0.80 at t=25ms)",
            "Input Voltage Step (V_in: 12 -> 10V at t=25ms)",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: Startup
    r1 = model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.05,
    })
    fig.add_trace(go.Scatter(
        x=r1["t"] * 1e3, y=r1["v_out"],
        name="V_out (startup)", line=dict(color="#636EFA", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=r1["t"] * 1e3, y=r1["i_L"],
        name="I_L (startup)", line=dict(color="#EF553B", width=2),
    ), row=1, col=1)

    # Panel 2: Load step
    def r_step(t):
        return 48.0 if t < 0.025 else 24.0
    r2 = model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": r_step,
        "dt": 5e-6, "duration_s": 0.05,
    })
    fig.add_trace(go.Scatter(
        x=r2["t"] * 1e3, y=r2["v_out"],
        name="V_out (load step)", line=dict(color="#00CC96", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=r2["t"] * 1e3, y=r2["i_L"],
        name="I_L (load step)", line=dict(color="#FFA15A", width=2),
    ), row=1, col=2)

    # Panel 3: Duty step
    def d_step(t):
        return 0.75 if t < 0.025 else 0.80
    r3 = model.predict({
        "v_in": 12.0, "duty_cycle": d_step, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.05,
    })
    fig.add_trace(go.Scatter(
        x=r3["t"] * 1e3, y=r3["v_out"],
        name="V_out (duty step)", line=dict(color="#AB63FA", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=r3["t"] * 1e3, y=r3["i_L"],
        name="I_L (duty step)", line=dict(color="#FF6692", width=2),
    ), row=2, col=1)

    # Panel 4: Vin step
    def vin_step(t):
        return 12.0 if t < 0.025 else 10.0
    r4 = model.predict({
        "v_in": vin_step, "duty_cycle": 0.75, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.05,
    })
    fig.add_trace(go.Scatter(
        x=r4["t"] * 1e3, y=r4["v_out"],
        name="V_out (Vin step)", line=dict(color="#19D3F3", width=2),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=r4["t"] * 1e3, y=r4["i_L"],
        name="I_L (Vin step)", line=dict(color="#B6E880", width=2),
    ), row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Time (ms)", row=r, col=c)
            fig.update_yaxes(title_text="V_out (V) / I_L (A)", row=r, col=c)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Averaged State-Space Model",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0})
    print(f"\n--- Boost F2a Steady-State (V_in=12V, D=0.75, R=48 Ohm) ---")
    print(f"V_out_ss = {ss['v_out_ss']:.4f} V")
    print(f"I_L_ss   = {ss['i_L_ss']:.4f} A")
    print(f"P_out_ss = {ss['power_ss']:.4f} W")
    print(f"\nSimulated final: V_out={r1['v_out'][-1]:.4f}V  I_L={r1['i_L'][-1]:.4f}A")


if __name__ == "__main__":
    generate_report()
