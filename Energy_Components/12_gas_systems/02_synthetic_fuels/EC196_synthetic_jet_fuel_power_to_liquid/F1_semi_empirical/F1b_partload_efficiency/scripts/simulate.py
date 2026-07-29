"""EC196 F1b — FT Power-to-Liquid part-load simulation."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    plr = np.linspace(0.2, 1.0, 20)
    r = m.predict({"T_set": 220.0, "pressure_bar": 25.0, "plr": plr})
    print("EC196 F1b — FT Power-to-Liquid Part-Load + Thermal Integration")
    print(f"  CO conversion at PLR=1.0: {float(m.model.conversion(220, 25, plr=1.0)):.4f}")
    print(f"  CO conversion at PLR=0.5: {float(m.model.conversion(220, 25, plr=0.5)):.4f}")
    print(f"  Alpha at design T: {float(m.model._alpha_at_T(220)):.4f}")
    print(f"  Alpha at T_eff(PLR=0.5): {float(m.model._alpha_at_T(210)):.4f}")
    print(f"  Heat recovery at design (1 mol/s CO): {float(m.model.heat_recovery_kW(220, 25)):.1f} kW")
    print(f"  Deactivation after 10000h: {(1-m.model.deactivation_factor(10000))*100:.1f}%")
    print("DONE")


if __name__ == "__main__":
    main()
