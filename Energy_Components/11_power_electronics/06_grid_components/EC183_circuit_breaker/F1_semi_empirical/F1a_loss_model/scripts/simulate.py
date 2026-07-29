"""EC183 — Circuit Breaker — F1a Loss Model — Simulation & HTML Report"""
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
            "Conduction Loss vs Current (Closed State)",
            "Thermal Energy during Fault Clearing vs I_fault",
            "Loss vs Current (log scale)",
            "Interrupting Capability Map",
        ],
        vertical_spacing=0.14,
    )

    I_range = np.linspace(0, 630, 200)
    r1 = model.predict({"I_A": I_range, "state": "closed"})
    fig.add_trace(go.Scatter(x=I_range, y=r1["P_loss_W"] * 1000,
                             name="P_loss (mW)", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)
    fig.add_vline(x=630, line_dash="dash", line_color="red",
                  annotation_text="I_rated", row=1, col=1)

    I_fault_range = np.linspace(0, 30, 100)
    r2 = model.predict({"I_A": 0.0, "state": "closed", "I_fault_kA": I_fault_range})
    fig.add_trace(go.Scatter(x=I_fault_range, y=r2["E_fault_J"],
                             name="E_fault (J)", line=dict(color="#EF553B", width=2.5)),
                  row=1, col=2)
    fig.add_vline(x=25.0, line_dash="dash", line_color="red",
                  annotation_text="I_interrupt limit", row=1, col=2)

    fig.add_trace(go.Scatter(x=I_range[1:], y=r1["P_loss_W"][1:] * 1000,
                             name="P_loss log", line=dict(color="#00CC96", width=2)),
                  row=2, col=1)

    # Open vs closed state comparison
    fig.add_trace(go.Bar(x=["Open", "Closed (rated)"],
                         y=[0, model.predict({"I_A": 630.0, "state": "closed"})["P_loss_W"] * 1000],
                         name="State comparison",
                         marker_color=["#AB63FA", "#FFA15A"]),
                  row=2, col=2)

    fig.update_xaxes(title_text="Current (A)", row=1, col=1)
    fig.update_xaxes(title_text="Fault Current (kA)", row=1, col=2)
    fig.update_xaxes(title_text="Current (A)", row=2, col=1)
    fig.update_yaxes(title_text="P_loss (mW)", row=1, col=1)
    fig.update_yaxes(title_text="Fault Energy (J)", row=1, col=2)
    fig.update_yaxes(title_text="P_loss (mW)", row=2, col=1, type="log")
    fig.update_yaxes(title_text="P_loss (mW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Circuit Breaker Summary (11kV VCB, closed state) ---")
    print(f"{'I_A':>8} {'P_loss(mW)':>12} {'overload':>10} {'thermal_ok':>12}")
    for I in [100, 200, 400, 630, 700]:
        rv = model.predict({"I_A": float(I), "state": "closed"})
        print(f"{I:>8} {float(rv['P_loss_W'])*1000:>12.3f} {str(bool(rv['is_overloaded'])):>10} {str(bool(rv['thermal_rating_ok'])):>12}")


if __name__ == "__main__":
    generate_report()
