"""EC070 — Water-Source Heat Pump — F1b Part-Load — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import WaterSourceHPF1b


class ComponentModel:
    """Standardized interface for EC070 WSHP — F1b part-load + flow rate."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = WaterSourceHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_source": degC (water source temperature),
                "T_sink": degC,
                "part_load_ratio": 0–1 (optional, default 1.0),
                "water_flow_rate_ls": L/s (optional, default rated)
            }
        Returns:
            {
                "cop": dimensionless (heating),
                "cooling_cop": dimensionless,
                "heating_capacity_kw": kW,
                "electrical_input_kw": kW,
                "cop_degradation_factor": dimensionless,
                "ua_effective": kW/K
            }
        """
        Ts = np.asarray(inputs["T_source"], dtype=float)
        Tk = np.asarray(inputs["T_sink"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        flow = inputs.get("water_flow_rate_ls", self._model.flow_rated)
        flow = np.asarray(flow, dtype=float)

        return {
            "cop": self._model.cop(Ts, Tk, plr, flow),
            "cooling_cop": self._model.cooling_cop(Ts, Tk, plr, flow),
            "heating_capacity_kw": self._model.heating_capacity(Ts, Tk, plr),
            "electrical_input_kw": self._model.electrical_input(Ts, Tk, plr, flow),
            "cop_degradation_factor": self._model.cop_degradation_factor(plr, Ts, Tk, flow),
            "ua_effective": self._model.ua_effective(flow),
        }

    def get_info(self) -> dict:
        return {
            "name": "Water-Source Heat Pump (WSHP)",
            "ec_id": "EC070",
            "fidelity": "F1b",
            "description": (
                "Part-load COP with PLF = 1 - C_d*(1-PLR) (EN 14825, C_d=0.15), "
                "cycling losses below PLR_min=0.30, and water-side flow rate effect "
                "on evaporator U_A (Dittus-Boelter). No defrost (water source)."
            ),
            "inputs": {
                "T_source": {"unit": "degC", "range": [5.0, 30.0]},
                "T_sink": {"unit": "degC", "range": [25.0, 65.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
                "water_flow_rate_ls": {"unit": "L/s", "range": [0.5, 6.0], "default": 3.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "cooling_cop": {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "electrical_input_kw": {"unit": "kW"},
                "cop_degradation_factor": {"unit": "dimensionless"},
                "ua_effective": {"unit": "kW/K"},
            },
            "source": "ASHRAE Handbook (2020); EN 14825:2016; Bagarella et al. (2016); Dittus-Boelter",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    print("\n--- PLR sweep at rated flow ---")
    for plr in [1.0, 0.75, 0.5, 0.30, 0.15, 0.05]:
        r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": plr})
        print(f"PLR={plr:.2f}: COP={float(r['cop']):.2f}, "
              f"Q={float(r['heating_capacity_kw']):.1f} kW, "
              f"W={float(r['electrical_input_kw']):.2f} kW, "
              f"degr={float(r['cop_degradation_factor']):.3f}")
    print("\n--- Flow rate sweep at PLR=1 ---")
    for flow in [0.5, 1.0, 2.0, 3.0, 5.0]:
        r = model.predict({"T_source": 15.0, "T_sink": 45.0,
                           "part_load_ratio": 1.0, "water_flow_rate_ls": flow})
        print(f"flow={flow:.1f} L/s: COP={float(r['cop']):.2f}, "
              f"UA_eff={float(r['ua_effective']):.2f} kW/K")
