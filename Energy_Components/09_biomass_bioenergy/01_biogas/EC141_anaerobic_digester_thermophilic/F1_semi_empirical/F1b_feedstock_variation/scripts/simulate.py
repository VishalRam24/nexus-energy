"""EC141 -- Thermophilic AD -- F1b -- Simulation and HTML report"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    feedstocks = ["cattle_manure", "food_waste", "grass_silage", "sewage_sludge", "corn_silage"]

    T_arr = np.linspace(35.0, 65.0, 40)
    yields_T = {fs: [] for fs in feedstocks}
    for T in T_arr:
        for fs in feedstocks:
            r = model.predict({"feedstock_type": fs, "vs_loading_kg_m3_day": 3.0,
                                "temperature_degC": float(T)})
            yields_T[fs].append(r["methane_yield_m3_day"])

    M_arr = np.linspace(0.0, 0.6, 30)
    lhv_factors = [model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                                   "moisture_fraction": float(m)})["moisture_lhv_factor"]
                   for m in M_arr]

    hrt_arr = np.linspace(5.0, 30.0, 30)
    yields_hrt = [model.predict({"feedstock_type": "food_waste",
                                  "vs_loading_kg_m3_day": 3.0,
                                  "hrt_days": float(h)})["methane_yield_m3_day"]
                  for h in hrt_arr]

    traces = ""
    for fs in feedstocks:
        traces += f"{{x:{list(T_arr)}, y:{yields_T[fs]}, name:'{fs}'}},"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>EC141 Thermophilic AD F1b</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body>
<h1>EC141 -- Thermophilic Anaerobic Digester -- F1b</h1>
<div id="p1"></div><div id="p2"></div><div id="p3"></div>
<script>
Plotly.newPlot('p1', [{traces}],
  {{title:'Methane Yield vs Temperature (all feedstocks)',
   xaxis:{{title:'T [degC]'}}, yaxis:{{title:'m3_CH4/day'}}}});
Plotly.newPlot('p2', [{{x:{list(M_arr)}, y:{lhv_factors}, name:'LHV factor'}}],
  {{title:'Moisture-LHV Factor vs Moisture Content',
   xaxis:{{title:'Moisture fraction'}}, yaxis:{{title:'LHV_eff/LHV_dry'}}}});
Plotly.newPlot('p3', [{{x:{list(hrt_arr)}, y:{yields_hrt}, name:'Food waste'}}],
  {{title:'Methane Yield vs HRT (food waste, 55 degC)',
   xaxis:{{title:'HRT [days]'}}, yaxis:{{title:'m3_CH4/day'}}}});
</script></body></html>"""

    out = Path(__file__).parent.parent / "simulation_report.html"
    out.write_text(html)
    print(f"Report written: {out}")


if __name__ == "__main__":
    run_simulations()
