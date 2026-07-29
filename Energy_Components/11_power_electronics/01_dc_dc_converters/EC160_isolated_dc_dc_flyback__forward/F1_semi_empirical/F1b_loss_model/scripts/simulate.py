"""EC160 -- Isolated DC-DC (Flyback/Forward) -- F1b -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Efficiency vs Load Current", "Loss Breakdown vs Load",
                        "Junction Temperature vs Load", "Efficiency vs Vin"],
        vertical_spacing=0.14)

    i_range = np.linspace(0.1, 12.0, 200)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})

    fig.add_trace(go.Scatter(x=i_range, y=r["efficiency"]*100, name="eta (%)",
        line=dict(color="#636EFA", width=2.5)), row=1, col=1)
    fig.add_hline(y=90, line_dash="dash", line_color="gray", annotation_text="90%", row=1, col=1)

    for key, col, name in [("p_mosfet_cond_w","#EF553B","MOSFET"),
                            ("p_diode_cond_w","#FFA15A","Diode"),
                            ("p_transformer_pri_w","#00CC96","Xfmr Pri"),
                            ("p_transformer_sec_w","#19D3F3","Xfmr Sec"),
                            ("p_switching_w","#AB63FA","Switching")]:
        fig.add_trace(go.Scatter(x=i_range, y=r[key], name=name,
            line=dict(color=col), stackgroup="losses"), row=1, col=2)

    fig.add_trace(go.Scatter(x=i_range, y=r["T_j_degC"], name="T_j (C)",
        line=dict(color="#FF6692", width=2)), row=2, col=1)

    v_in_range = np.linspace(20, 100, 100)
    r2 = model.predict({"v_in": v_in_range, "v_out_target": 12.0, "i_load": 5.0})
    fig.add_trace(go.Scatter(x=v_in_range, y=r2["efficiency"]*100, name="eta vs Vin",
        line=dict(color="#FECB52", width=2)), row=2, col=2)

    fig.update_xaxes(title_text="Load Current (A)", row=1, col=1)
    fig.update_xaxes(title_text="Load Current (A)", row=1, col=2)
    fig.update_xaxes(title_text="Load Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="V_in (V)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="T_j (degC)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
