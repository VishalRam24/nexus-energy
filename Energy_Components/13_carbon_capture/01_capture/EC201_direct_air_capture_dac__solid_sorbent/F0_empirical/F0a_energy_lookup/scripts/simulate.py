"""Optional Plotly report for EC201 Direct Air Capture (DAC) -- Solid Sorbent F0a (safe if plotly missing)."""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    print("Sample:", m.predict({'air_flow_m3h': 1000000.0, 'ambient_temp': 25.0}))
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        print("plotly not available; skipping HTML report.")
        return
    print("plotly available -- report stub.")


if __name__ == "__main__":
    main()
