"""EC140 -- Anaerobic Digester -- F2a Monod Kinetics -- Simulation & HTML Report"""
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
            "Startup: Substrate and Biomass",
            "Startup: Methane Production",
            "Effect of HRT on Steady-State",
            "Effect of Temperature on Methane Yield",
            "Shock Load Response",
            "pH Inhibition Effect",
        ],
        vertical_spacing=0.10,
    )

    # -- Panel 1-2: Startup transient --
    r1 = model.predict({"dt": 0.5, "duration_d": 80.0})
    fig.add_trace(go.Scatter(
        x=r1["t"], y=r1["S"], name="S [gCOD/L]",
        line=dict(color="#636EFA", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=r1["t"], y=r1["X"], name="X [gVSS/L]",
        line=dict(color="#EF553B", width=2),
    ), row=1, col=1)
    fig.update_xaxes(title_text="Time [d]", row=1, col=1)
    fig.update_yaxes(title_text="Concentration [g/L]", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=r1["t"], y=r1["V_ch4_rate_L_d"] / 1000, name="CH4 rate [m^3/d]",
        line=dict(color="#00CC96", width=2),
    ), row=1, col=2)
    fig.update_xaxes(title_text="Time [d]", row=1, col=2)
    fig.update_yaxes(title_text="CH4 Production [m^3/d]", row=1, col=2)

    # -- Panel 3: HRT effect --
    hrts = np.arange(5, 45, 1)
    removal = []
    ch4_rate = []
    for hrt in hrts:
        ss = model.predict_steady_state({"HRT": float(hrt)})
        removal.append(ss["COD_removal_pct"])
        ch4_rate.append(ss["V_ch4_d"] / 1000 if not ss["washout"] else 0.0)
    fig.add_trace(go.Scatter(
        x=hrts, y=removal, name="COD Removal [%]",
        line=dict(color="#AB63FA", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=hrts, y=ch4_rate, name="CH4 [m^3/d]",
        line=dict(color="#FFA15A", width=2, dash="dash"),
    ), row=2, col=1)
    fig.update_xaxes(title_text="HRT [d]", row=2, col=1)
    fig.update_yaxes(title_text="Removal [%] / CH4 [m^3/d]", row=2, col=1)

    # -- Panel 4: Temperature effect --
    temps_C = np.arange(20, 50, 1)
    ch4_temp = []
    for tc in temps_C:
        ss = model.predict_steady_state({"T": float(tc + 273.15)})
        ch4_temp.append(ss["V_ch4_d"] / 1000 if not ss["washout"] else 0.0)
    fig.add_trace(go.Scatter(
        x=temps_C, y=ch4_temp, name="CH4 vs Temp",
        line=dict(color="#FF6692", width=2),
    ), row=2, col=2)
    fig.update_xaxes(title_text="Temperature [C]", row=2, col=2)
    fig.update_yaxes(title_text="CH4 [m^3/d]", row=2, col=2)

    # -- Panel 5: Shock load --
    def shock(t):
        return 80.0 if 20 < t < 25 else 40.0
    r3 = model.predict({
        "S_in": shock, "dt": 0.5, "duration_d": 80.0,
        "x0": [5.0, 10.0],
    })
    fig.add_trace(go.Scatter(
        x=r3["t"], y=r3["S"], name="S (shock)",
        line=dict(color="#636EFA", width=2, dash="dot"),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=r3["t"], y=r3["X"], name="X (shock)",
        line=dict(color="#EF553B", width=2, dash="dot"),
    ), row=3, col=1)
    fig.update_xaxes(title_text="Time [d]", row=3, col=1)
    fig.update_yaxes(title_text="Concentration [g/L]", row=3, col=1)

    # -- Panel 6: pH effect --
    phs = np.linspace(5.5, 9.0, 30)
    ch4_ph = []
    for ph in phs:
        ss = model.predict_steady_state({"pH": float(ph)})
        ch4_ph.append(ss["V_ch4_d"] / 1000 if not ss["washout"] else 0.0)
    fig.add_trace(go.Scatter(
        x=phs, y=ch4_ph, name="CH4 vs pH",
        line=dict(color="#19D3F3", width=2),
    ), row=3, col=2)
    fig.update_xaxes(title_text="pH", row=3, col=2)
    fig.update_yaxes(title_text="CH4 [m^3/d]", row=3, col=2)

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
