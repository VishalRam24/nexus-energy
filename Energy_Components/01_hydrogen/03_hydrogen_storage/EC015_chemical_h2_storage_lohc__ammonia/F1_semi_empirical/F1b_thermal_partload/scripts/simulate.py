"""EC015 -- Chemical H2 Storage -- F1b Thermal+Part-load -- Simulation Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly required"); sys.exit(1)

def generate_report():
    model = ComponentModel()
    T_lohc = np.linspace(470.0, 680.0, 100)
    T_nh3  = np.linspace(650.0, 850.0, 100)
    loads  = np.linspace(0.001, 0.01, 100)

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["LOHC: Efficiency vs T", "NH3: Efficiency vs T",
                        "LOHC: Part-load effect (T=573K)", "Round-trip efficiency vs T"],
        vertical_spacing=0.13, horizontal_spacing=0.10)

    eta_l = [float(model.predict({"carrier":"lohc","direction":"dehydrogenation","h2_mass_kg":1.0,"temperature_K":float(T)})["efficiency"]) for T in T_lohc]
    fig.add_trace(go.Scatter(x=T_lohc-273.15, y=eta_l, name="LOHC eta"), row=1, col=1)

    eta_n = [float(model.predict({"carrier":"ammonia","direction":"cracking","h2_mass_kg":1.0,"temperature_K":float(T)})["efficiency"]) for T in T_nh3]
    fig.add_trace(go.Scatter(x=T_nh3-273.15, y=eta_n, name="NH3 eta", line=dict(color="#ff7f0e")), row=1, col=2)

    eta_pl = [float(model.predict({"carrier":"lohc","direction":"dehydrogenation","h2_mass_kg":1.0,"temperature_K":573.15,"flow_rate_kg_s":float(F)})["efficiency"]) for F in loads]
    fig.add_trace(go.Scatter(x=loads*1000, y=eta_pl, name="LOHC part-load"), row=2, col=1)

    rt_l = [float(model.predict({"carrier":"lohc","direction":"dehydrogenation","h2_mass_kg":1.0,"temperature_K":float(T)})["roundtrip_efficiency"]) for T in T_lohc]
    rt_n = [float(model.predict({"carrier":"ammonia","direction":"cracking","h2_mass_kg":1.0,"temperature_K":float(T)})["roundtrip_efficiency"]) for T in T_nh3]
    fig.add_trace(go.Scatter(x=T_lohc-273.15, y=rt_l, name="LOHC RT-eta"), row=2, col=2)
    fig.add_trace(go.Scatter(x=T_nh3-273.15, y=rt_n, name="NH3 RT-eta", line=dict(color="#ff7f0e")), row=2, col=2)

    fig.update_layout(title_text="EC015 Chemical H2 Storage F1b Thermal+Part-load Simulation", height=700)
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report saved to {out}")

if __name__ == "__main__":
    generate_report()
