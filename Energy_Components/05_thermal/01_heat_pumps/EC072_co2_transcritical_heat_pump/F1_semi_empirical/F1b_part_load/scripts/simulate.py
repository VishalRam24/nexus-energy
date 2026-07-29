"""EC072 — CO2 Transcritical HP — F1b Part-Load — Simulation Scenarios"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

BASE = Path(__file__).parent.parent


def run():
    model = ComponentModel()

    plr_arr  = np.linspace(0.1, 1.0, 50)
    Twin_arr = np.linspace(5.0, 50.0, 50)
    Tevap_arr = np.linspace(-20.0, 15.0, 50)

    r_pl  = model.predict({"T_evap": 0.0, "T_water_in": 15.0,
                           "T_water_out": 65.0, "part_load_ratio": plr_arr})
    r_tw  = model.predict({"T_evap": 0.0, "T_water_in": Twin_arr,
                           "T_water_out": 65.0, "part_load_ratio": 1.0})
    r_te  = model.predict({"T_evap": Tevap_arr, "T_water_in": 15.0,
                           "T_water_out": 65.0, "part_load_ratio": 0.75})

    print("=== EC072 CO2 Transcritical HP F1b — Simulation Summary ===")
    print("\n[PLR sweep] T_evap=0C, T_w_in=15C, T_w_out=65C:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  PLR={plr_arr[i]:.2f}: COP={float(r_pl['cop'][i]):.3f}, "
              f"degrad={float(r_pl['cop_degradation_factor'][i]):.3f}")

    print("\n[T_water_in sweep] PLR=1, T_evap=0C, T_w_out=65C:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  T_w_in={Twin_arr[i]:.0f}C: COP={float(r_tw['cop'][i]):.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["COP vs PLR", "COP vs T_w_in", "COP vs T_evap"])
        fig.add_trace(go.Scatter(x=plr_arr, y=r_pl["cop"], mode="lines",
                                 name="COP vs PLR"), row=1, col=1)
        fig.add_trace(go.Scatter(x=Twin_arr, y=r_tw["cop"], mode="lines",
                                 name="COP vs T_w_in"), row=1, col=2)
        fig.add_trace(go.Scatter(x=Tevap_arr, y=r_te["cop"], mode="lines",
                                 name="COP vs T_evap"), row=1, col=3)
        fig.update_layout(title="EC072 CO2 Transcritical HP — F1b Part-Load", height=450)
        fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
        fig.update_xaxes(title_text="T_w_in [°C]", row=1, col=2)
        fig.update_xaxes(title_text="T_evap [°C]", row=1, col=3)
        fig.update_yaxes(title_text="COP [-]", row=1, col=1)
        html_path = BASE / "simulation_report.html"
        fig.write_html(str(html_path))
        print(f"\nReport written to {html_path}")
    except ImportError:
        print("plotly not available — skipping HTML report")


if __name__ == "__main__":
    run()
