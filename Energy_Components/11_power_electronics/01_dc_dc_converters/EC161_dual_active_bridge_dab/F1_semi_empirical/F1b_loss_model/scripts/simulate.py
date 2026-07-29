"""EC161 -- Dual Active Bridge (DAB) -- F1b -- Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Efficiency vs Load Power", "Loss Breakdown vs Power",
                        "Phase Shift vs Power", "Junction Temperature vs Power"],
        vertical_spacing=0.14)

    p_range = np.linspace(100, 12000, 200)
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": p_range})

    fig.add_trace(go.Scatter(x=p_range/1000, y=r["efficiency"]*100, name="eta (%)",
        line=dict(color="#636EFA", width=2.5)), row=1, col=1)
    fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="95%", row=1, col=1)

    for key, col, name in [("p_mosfet_cond_w","#EF553B","MOSFET cond"),
                            ("p_switching_w","#AB63FA","Switching"),
                            ("p_transformer_w","#00CC96","Transformer")]:
        fig.add_trace(go.Scatter(x=p_range/1000, y=r[key], name=name,
            line=dict(color=col), stackgroup="losses"), row=1, col=2)

    fig.add_trace(go.Scatter(x=p_range/1000, y=np.degrees(r["phi_rad"]),
        name="phi (deg)", line=dict(color="#19D3F3", width=2)), row=2, col=1)
    fig.add_hline(y=90, line_dash="dot", annotation_text="phi_max=90deg", row=2, col=1)

    fig.add_trace(go.Scatter(x=p_range/1000, y=r["T_j_degC"],
        name="T_j (C)", line=dict(color="#FF6692", width=2)), row=2, col=2)

    fig.update_xaxes(title_text="Load Power (kW)", row=1, col=1)
    fig.update_xaxes(title_text="Load Power (kW)", row=1, col=2)
    fig.update_xaxes(title_text="Load Power (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Load Power (kW)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="Phase Shift (deg)", row=2, col=1)
    fig.update_yaxes(title_text="T_j (degC)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
