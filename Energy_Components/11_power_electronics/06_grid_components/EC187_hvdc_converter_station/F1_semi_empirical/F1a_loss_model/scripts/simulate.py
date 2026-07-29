"""EC187 — HVDC Converter Station — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    P_range = np.linspace(0, 1000, 200)
    r_rect = model.predict({"P_transfer_MW": P_range, "direction": "rectifier"})
    r_inv = model.predict({"P_transfer_MW": P_range, "direction": "inverter"})

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs Power Transfer",
            "Station Losses vs Power Transfer",
            "DC Current vs Power Transfer",
            "Rectifier vs Inverter P_out",
        ],
        vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(x=P_range, y=r_rect["efficiency"] * 100,
                             name="Rectifier eta (%)", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=P_range, y=r_inv["efficiency"] * 100,
                             name="Inverter eta (%)", line=dict(color="#EF553B", width=2.5, dash="dash")),
                  row=1, col=1)
    fig.add_hline(y=99.0, line_dash="dot", line_color="gray",
                  annotation_text="99%", row=1, col=1)

    fig.add_trace(go.Scatter(x=P_range, y=r_rect["P_loss_MW"],
                             name="P_loss (MW)", line=dict(color="#00CC96", width=2.5)),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=P_range, y=r_rect["I_dc_kA"],
                             name="I_dc (kA)", line=dict(color="#AB63FA", width=2.5)),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=P_range, y=r_rect["P_out_MW"],
                             name="P_out Rectifier", line=dict(color="#636EFA")),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=P_range, y=r_inv["P_out_MW"],
                             name="P_out Inverter", line=dict(color="#EF553B", dash="dash")),
                  row=2, col=2)

    for (ar, ac, xt, yt) in [
        (1, 1, "P_transfer (MW)", "Efficiency (%)"),
        (1, 2, "P_transfer (MW)", "P_loss (MW)"),
        (2, 1, "P_transfer (MW)", "I_dc (kA)"),
        (2, 2, "P_transfer (MW)", "P_out (MW)"),
    ]:
        fig.update_xaxes(title_text=xt, row=ar, col=ac)
        fig.update_yaxes(title_text=yt, row=ar, col=ac)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- HVDC Converter Station Summary (Rectifier mode) ---")
    print(f"{'P_transfer(MW)':>16} {'P_loss(MW)':>11} {'eta(%)':>8} {'I_dc(kA)':>10}")
    for P in [100, 250, 500, 750, 1000]:
        rv = model.predict({"P_transfer_MW": float(P), "direction": "rectifier"})
        print(f"{P:>16} {float(rv['P_loss_MW']):>11.2f} {float(rv['efficiency'])*100:>8.3f} {float(rv['I_dc_kA']):>10.4f}")


if __name__ == "__main__":
    generate_report()
