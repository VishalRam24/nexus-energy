"""EC182 F1b — Distribution Line Thermal Ampacity simulation scenarios."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    P = np.linspace(50, 2000, 30)
    r = m.predict({"V_s_kV": 11.0, "P_load_kW": P, "Q_load_kVAR": P * 0.3})
    print("EC182 F1b — Distribution Line Thermal Ampacity Simulation")
    print(f"  R_dc at 20C:  {m.model.r_ac_ohm_per_km(20.0):.4f} Ohm/km")
    print(f"  R_ac at 75C:  {m.model.r_ac_ohm_per_km(75.0):.4f} Ohm/km")
    print(f"  Ampacity 25C: {m.model.thermal_ampacity_A(25.0):.1f} A")
    print(f"  Ampacity 40C: {m.model.thermal_ampacity_A(40.0):.1f} A")
    print(f"  Derating 40C: {m.model.ampacity_derating_factor(40.0):.4f}")
    idx = 14
    print(f"  P=1025 kW: V_r={float(r['V_r_kV'][idx]):.3f} kV, I={float(r['I_line_A'][idx]):.1f} A")
    print(f"  Congestion: {float(r['congestion_factor'][idx]):.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
