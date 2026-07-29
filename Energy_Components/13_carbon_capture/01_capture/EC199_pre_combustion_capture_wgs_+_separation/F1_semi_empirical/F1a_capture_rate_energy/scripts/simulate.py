"""EC199 F1a — Pre-Combustion Capture simulation."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    T = np.linspace(180, 400, 30)
    r = m.predict({"co_fraction": 0.40, "P_bar": 30.0, "T_WGS_C": T})
    print("EC199 F1a — Pre-Combustion Capture (WGS + Selexol)")
    print(f"  WGS conversion at 250C, 30 bar: {float(m.model.wgs_conversion(250, 30)):.4f}")
    print(f"  Overall capture rate at design: {float(m.model.capture_rate(250, 30)):.4f}")
    print(f"  E_sep at P_CO2=15 bar: {float(m.model.separation_energy_GJ_tCO2(15)):.4f} GJ/tCO2")
    print(f"  E_sep at P_CO2=5 bar:  {float(m.model.separation_energy_GJ_tCO2(5)):.4f} GJ/tCO2")
    print(f"  Total energy at design: {float(m.model.total_energy_GJ_tCO2(250, 30, 0.4)):.4f} GJ/tCO2")
    print("DONE")


if __name__ == "__main__":
    main()
