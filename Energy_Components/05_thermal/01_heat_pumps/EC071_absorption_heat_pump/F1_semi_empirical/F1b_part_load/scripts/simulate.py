"""EC071 — Absorption HP — F1b Part-Load — Simulation Scenarios"""
import json, numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run():
    model = ComponentModel()

    # --- Scenario 1: Part-load sweep ---
    plr_arr = np.linspace(0.1, 1.0, 50)
    r_pl = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                          "part_load_ratio": plr_arr})

    # --- Scenario 2: Generator temperature sweep ---
    Tgen_arr = np.linspace(70.0, 110.0, 50)
    r_tg = model.predict({"T_gen": Tgen_arr, "T_evap": 10.0, "T_cond": 35.0,
                          "part_load_ratio": 1.0})

    # --- Scenario 3: Evaporator temperature sweep ---
    Tevap_arr = np.linspace(0.0, 25.0, 50)
    r_te = model.predict({"T_gen": 90.0, "T_evap": Tevap_arr, "T_cond": 35.0,
                          "part_load_ratio": 0.75})

    # Print summary table
    print("=== EC071 Absorption HP F1b — Simulation Summary ===")
    print("\n[Scenario 1] Part-load sweep (T_gen=90C, T_evap=10C, T_cond=35C):")
    for i in [0, 10, 25, 40, 49]:
        print(f"  PLR={plr_arr[i]:.2f}: COP={float(r_pl['cop'][i]):.3f}, "
              f"Q_h={float(r_pl['heating_capacity_kw'][i]):.1f}kW, "
              f"degradation={float(r_pl['cop_degradation_factor'][i]):.3f}")

    print("\n[Scenario 2] T_gen sweep (PLR=1, T_evap=10C, T_cond=35C):")
    for i in [0, 12, 25, 37, 49]:
        print(f"  T_gen={Tgen_arr[i]:.0f}C: COP={float(r_tg['cop'][i]):.3f}")

    print("\n[Scenario 3] T_evap sweep (PLR=0.75, T_gen=90C, T_cond=35C):")
    for i in [0, 12, 25, 37, 49]:
        print(f"  T_evap={Tevap_arr[i]:.0f}C: COP={float(r_te['cop'][i]):.3f}")

    # Generate HTML report
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["COP vs PLR", "COP vs T_gen", "COP vs T_evap"])

        fig.add_trace(go.Scatter(x=plr_arr, y=r_pl["cop"], mode="lines",
                                 name="COP vs PLR"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plr_arr, y=r_pl["cop_degradation_factor"],
                                 mode="lines", name="Degradation factor",
                                 line=dict(dash="dash")), row=1, col=1)

        fig.add_trace(go.Scatter(x=Tgen_arr, y=r_tg["cop"], mode="lines",
                                 name="COP vs T_gen"), row=1, col=2)

        fig.add_trace(go.Scatter(x=Tevap_arr, y=r_te["cop"], mode="lines",
                                 name="COP vs T_evap"), row=1, col=3)

        fig.update_layout(title="EC071 Absorption HP — F1b Part-Load", height=450)
        fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
        fig.update_xaxes(title_text="T_gen [°C]", row=1, col=2)
        fig.update_xaxes(title_text="T_evap [°C]", row=1, col=3)
        fig.update_yaxes(title_text="COP [-]", row=1, col=1)

        html_path = BASE / "simulation_report.html"
        fig.write_html(str(html_path))
        print(f"\nReport written to {html_path}")
    except ImportError:
        print("plotly not available — skipping HTML report")


if __name__ == "__main__":
    run()
