"""EC093 — Adsorption Chiller — F1b Part-Load — Simulation Scenarios"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

BASE = Path(__file__).parent.parent


def run():
    model = ComponentModel()

    plr_arr  = np.linspace(0.05, 1.0, 50)
    Thot_arr = np.linspace(60.0, 95.0, 50)
    Tchi_arr = np.linspace(7.0, 18.0, 50)

    r_pl  = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                           "part_load_ratio": plr_arr})
    r_th  = model.predict({"T_hot": Thot_arr, "T_cool": 30.0, "T_chilled": 14.0,
                           "part_load_ratio": 1.0})
    r_tc  = model.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": Tchi_arr,
                           "part_load_ratio": 0.75})

    print("=== EC093 Adsorption Chiller F1b — Simulation Summary ===")
    print("\n[PLR sweep] T_hot=85C, T_cool=30C, T_chilled=14C:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  PLR={plr_arr[i]:.2f}: COP={float(r_pl['cop'][i]):.3f}, "
              f"degrad={float(r_pl['cop_degradation_factor'][i]):.3f}")

    print("\n[T_hot sweep] PLR=1, T_cool=30C, T_chilled=14C:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  T_hot={Thot_arr[i]:.0f}C: COP={float(r_th['cop'][i]):.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["COP vs PLR", "COP vs T_hot", "COP vs T_chilled"])
        fig.add_trace(go.Scatter(x=plr_arr, y=r_pl["cop"], mode="lines",
                                 name="COP"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plr_arr, y=r_pl["cop_degradation_factor"],
                                 mode="lines", name="Degradation",
                                 line=dict(dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=Thot_arr, y=r_th["cop"], mode="lines",
                                 name="COP vs T_hot"), row=1, col=2)
        fig.add_trace(go.Scatter(x=Tchi_arr, y=r_tc["cop"], mode="lines",
                                 name="COP vs T_chilled"), row=1, col=3)
        fig.update_layout(title="EC093 Adsorption Chiller — F1b Part-Load", height=450)
        fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
        fig.update_xaxes(title_text="T_hot [°C]", row=1, col=2)
        fig.update_xaxes(title_text="T_chilled [°C]", row=1, col=3)
        fig.update_yaxes(title_text="COP [-]", row=1, col=1)
        html_path = BASE / "simulation_report.html"
        fig.write_html(str(html_path))
        print(f"\nReport written to {html_path}")
    except ImportError:
        print("plotly not available — skipping HTML report")


if __name__ == "__main__":
    run()
