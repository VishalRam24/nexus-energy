"""EC158 — Boost Converter — F1a — Simulation & HTML Report"""
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
            "Efficiency vs Load Current (V_in=12V, V_out=48V)",
            "Input vs Output Current (current amplification)",
            "Efficiency vs V_out Target (V_in=12V, I_out=5A)",
            "Duty Cycle vs Conversion Ratio (V_in=12V)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Efficiency vs load current
    i_range = np.linspace(0.1, 10.0, 200)
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": i_range})
    fig.add_trace(go.Scatter(
        x=i_range, y=r["efficiency"] * 100,
        name="eta (%)", line=dict(color="#636EFA", width=2.5),
    ), row=1, col=1)

    # Plot 2: I_in vs I_out (current amplification)
    fig.add_trace(go.Scatter(
        x=i_range, y=r["i_input"],
        name="I_in (A)", line=dict(color="#EF553B", width=2),
    ), row=1, col=2)
    # Reference ideal line
    ideal_iin = i_range * 48.0 / 12.0
    fig.add_trace(go.Scatter(
        x=i_range, y=ideal_iin,
        name="Ideal I_in = 4*I_out", line=dict(color="gray", dash="dot"),
    ), row=1, col=2)

    # Plot 3: Efficiency vs V_out target
    v_out_range = np.linspace(14.0, 120.0, 100)
    r_vout = model.predict({"v_in": 12.0, "v_out_target": v_out_range, "i_load": 5.0})
    fig.add_trace(go.Scatter(
        x=v_out_range, y=r_vout["efficiency"] * 100,
        name="eta vs V_out", line=dict(color="#00CC96", width=2),
    ), row=2, col=1)
    fig.add_vline(x=48.0, line_dash="dot", line_color="red",
                  annotation_text="V_out=48V", row=2, col=1)

    # Plot 4: Duty cycle vs conversion ratio M = V_out/V_in
    M_range = np.linspace(1.1, 10.0, 100)
    v_out_M = 12.0 * M_range
    r_D = model.predict({"v_in": 12.0, "v_out_target": v_out_M, "i_load": 2.0})
    fig.add_trace(go.Scatter(
        x=M_range, y=r_D["duty_cycle"],
        name="D = 1 - 1/M", line=dict(color="#AB63FA", width=2),
    ), row=2, col=2)
    # Theoretical: D = 1 - 1/M
    D_theory = 1.0 - 1.0 / M_range
    fig.add_trace(go.Scatter(
        x=M_range, y=np.clip(D_theory, 0, 0.95),
        name="Theoretical D", line=dict(color="black", dash="dash", width=1),
    ), row=2, col=2)

    fig.update_xaxes(title_text="Output Current I_out (A)", row=1, col=1)
    fig.update_xaxes(title_text="Output Current I_out (A)", row=1, col=2)
    fig.update_xaxes(title_text="V_out Target (V)", row=2, col=1)
    fig.update_xaxes(title_text="Conversion Ratio M = V_out/V_in", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Current (A)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Duty Cycle D", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} (12V→48V, 100kHz)",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary table
    print("\n--- Boost Converter Summary (V_in=12V, V_out=48V) ---")
    print(f"{'I_out(A)':>9} {'D':>6} {'V_out(V)':>9} {'eta(%)':>8} {'I_in(A)':>8} {'P_loss(W)':>9}")
    for i in [0.5, 1.0, 2.0, 5.0, 8.0, 10.0]:
        rv = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": i})
        print(
            f"{i:>9.1f} {float(rv['duty_cycle']):>6.4f} {float(rv['v_out']):>9.3f} "
            f"{float(rv['efficiency'])*100:>8.3f} {float(rv['i_input']):>8.3f} "
            f"{float(rv['p_loss_w']):>9.4f}"
        )


if __name__ == "__main__":
    generate_report()
