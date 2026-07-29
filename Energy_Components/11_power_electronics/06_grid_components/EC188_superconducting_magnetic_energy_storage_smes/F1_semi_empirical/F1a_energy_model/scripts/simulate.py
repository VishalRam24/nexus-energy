"""EC188 — SMES — F1a Energy Model — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    # E vs I curve
    I_range = np.linspace(0, model._model.I_max, 200)
    ef = model.energy_from_current({"I_A": I_range})

    # Efficiency vs power (discharge)
    P_range = np.linspace(0.5, 10, 100)
    eta_list = [float(model.predict({"SOC": 1.0, "P_request_MW": P,
                                     "mode": "discharge"})["eta_instantaneous"])
                for P in P_range]

    # SOC time series (charge-discharge cycle)
    dt = 1.0  # s
    N = 200
    SOC = 0.5
    soc_hist, P_hist = [], []
    for t in range(N):
        mode = "charge" if t < 80 else "discharge"
        P_req = 8.0
        r = model.predict({"SOC": SOC, "P_request_MW": P_req, "mode": mode, "dt_s": dt})
        SOC = float(r["SOC_new"])
        soc_hist.append(SOC)
        P_hist.append(float(r["P_grid_MW"]))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Stored Energy vs Coil Current",
            "Efficiency vs Discharge Power (SOC=1.0)",
            "SOC During Charge-Discharge Cycle",
            "Grid Power During Charge-Discharge Cycle",
        ],
        vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(x=I_range, y=ef["E_MJ"],
                             name="E (MJ)", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=P_range, y=np.array(eta_list) * 100,
                             name="eta (%)", line=dict(color="#EF553B", width=2.5)),
                  row=1, col=2)

    t_axis = np.arange(N)
    fig.add_trace(go.Scatter(x=t_axis, y=soc_hist,
                             name="SOC", line=dict(color="#00CC96", width=2)),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=t_axis, y=P_hist,
                             name="P_grid (MW)", line=dict(color="#AB63FA", width=2)),
                  row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    for (ar, ac, xt, yt) in [
        (1, 1, "Coil Current (A)", "Stored Energy (MJ)"),
        (1, 2, "Discharge Power (MW)", "Efficiency (%)"),
        (2, 1, "Time (s)", "SOC"),
        (2, 2, "Time (s)", "P_grid (MW)"),
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

    print(f"\n--- SMES Summary (E_max={model._model.E_max_MJ:.1f} MJ, P_rated={model._model.P_rated} MW) ---")
    print(f"{'SOC':>6} {'E_stored(MJ)':>14} {'P_grid(MW)':>12} {'eta(%)':>8}")
    for soc in [1.0, 0.8, 0.6, 0.4, 0.2]:
        rv = model.predict({"SOC": soc, "P_request_MW": 8.0, "mode": "discharge"})
        print(f"{soc:>6.2f} {float(rv['E_stored_MJ']):>14.3f} {float(rv['P_grid_MW']):>12.3f} "
              f"{float(rv['eta_instantaneous'])*100:>8.2f}")


if __name__ == "__main__":
    generate_report()
