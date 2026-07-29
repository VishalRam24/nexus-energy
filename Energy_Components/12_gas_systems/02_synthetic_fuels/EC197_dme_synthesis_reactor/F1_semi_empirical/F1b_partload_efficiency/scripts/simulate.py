"""EC197 F1b — DME Synthesis part-load simulation."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    plr = np.linspace(0.2, 1.0, 20)
    r = m.predict({"T_set": 260.0, "pressure_bar": 40.0, "plr": plr})
    print("EC197 F1b — DME Synthesis Reactor Part-Load + Thermal Integration")
    print(f"  CO conversion at PLR=1.0: {float(m.model.conversion(260, 40, plr=1.0)):.4f}")
    print(f"  CO conversion at PLR=0.5: {float(m.model.conversion(260, 40, plr=0.5)):.4f}")
    print(f"  DME selectivity at PLR=1.0: {float(m.model.selectivity_dme(260, plr=1.0)):.4f}")
    print(f"  DME selectivity at PLR=0.5: {float(m.model.selectivity_dme(260, plr=0.5)):.4f}")
    print(f"  MeOH slip at PLR=0.5: {float(m.model.meoh_slip_mol_s(260, 40, plr=0.5)):.4f} mol/s")
    print(f"  Deactivation after 5000h: {(1-m.model.deactivation_factor(5000))*100:.1f}%")
    print("DONE")


if __name__ == "__main__":
    main()
