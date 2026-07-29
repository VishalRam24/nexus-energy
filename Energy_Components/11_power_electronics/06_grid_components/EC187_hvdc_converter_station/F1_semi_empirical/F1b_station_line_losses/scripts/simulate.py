"""EC187 F1b — HVDC Full Link simulation scenarios."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    P = np.linspace(0, 1000, 50)

    r_cold = m.predict({"P_transfer_MW": P, "T_line_C": 10.0})
    r_hot  = m.predict({"P_transfer_MW": P, "T_line_C": 60.0})

    print("EC187 F1b — HVDC Full Link Simulation")
    print(f"  R_line at 20C: {m.model.dc_line_resistance(20.0):.3f} Ohm")
    print(f"  R_line at 60C: {m.model.dc_line_resistance(60.0):.3f} Ohm")
    idx = 40  # P~800 MW
    print(f"  P=800 MW: eta(10C)={float(r_cold['link_efficiency'][idx]):.4f}, "
          f"eta(60C)={float(r_hot['link_efficiency'][idx]):.4f}")
    print(f"  P=800 MW: P_loss_line(20C) = {float(m.predict({'P_transfer_MW':800.0})['P_loss_line_MW']):.2f} MW")
    print(f"  Q_reactive at 500 MW (LCC): {float(m.predict({'P_transfer_MW':500.0})['Q_reactive_demand_MVAR']):.1f} MVAR")
    print("DONE")


if __name__ == "__main__":
    main()
