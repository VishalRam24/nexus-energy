"""Optional Plotly report for EC205 CO2 Electrolyzer (CO2 to CO/Fuels) F0a (safe if plotly missing)."""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    print("Sample:", m.predict({'co2_in_kg_s': 1.0, 'current_density_mA_cm2': 200.0}))
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        print("plotly not available; skipping HTML report.")
        return
    print("plotly available -- report stub.")


if __name__ == "__main__":
    main()
