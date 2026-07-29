"""Optional Plotly report for EC093 F0a. Safe to run without plotly."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from predict import ComponentModel  # noqa: E402

def main():
    m = ComponentModel()
    info = m.get_info()
    print("Model:", info["component_name"], info["fidelity"])
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        print("plotly not available - skipping HTML report")
        return
    print("plotly available - extend here to emit simulation_report.html")

if __name__ == "__main__":
    main()
