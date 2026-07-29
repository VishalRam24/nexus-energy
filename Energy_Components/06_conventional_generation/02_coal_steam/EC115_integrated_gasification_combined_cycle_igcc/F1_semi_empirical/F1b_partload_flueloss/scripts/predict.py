"""EC115 — IGCC — F1b Part-Load / Flue-Loss — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import IGCCF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = IGCCF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio : float or array [-], range [0.40, 1.0]
            ambient_temp    : float or array [degC], range [-10, 45]
        returns:
            power_mw          : net electrical output [MW_e]
            efficiency        : net LHV efficiency [-]
            flue_loss_fraction: stack heat loss fraction [-]
            aux_power_fraction: auxiliary power fraction [-]
            coal_rate_kgs     : coal feed rate [kg/s]
            syngas_rate_nm3s  : syngas flow to CCGT [Nm3/s]
            co2_rate_kgs      : CO2 emission rate [kg/s]
            co2_intensity     : CO2 intensity [g_CO2/kWh_e] (no CCS)
        """
        plr   = np.asarray(inputs["part_load_ratio"], dtype=float)
        T_amb = np.asarray(inputs["ambient_temp"],     dtype=float)
        return {
            "power_mw":           self._model.power_mw(plr),
            "efficiency":         self._model.efficiency(plr, T_amb),
            "flue_loss_fraction": self._model.flue_loss_fraction(plr),
            "aux_power_fraction": self._model.aux_power_fraction(plr),
            "coal_rate_kgs":      self._model.coal_rate_kgs(plr, T_amb),
            "syngas_rate_nm3s":   self._model.syngas_rate_nm3s(plr, T_amb),
            "co2_rate_kgs":       self._model.co2_rate_kgs(plr, T_amb),
            "co2_intensity":      self._model.co2_intensity_g_per_kwh(plr, T_amb),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Integrated Gasification Combined Cycle (IGCC)",
            "ec_id":       "EC115",
            "fidelity":    "F1b",
            "model":       "Part-Load Flue / Auxiliary Loss",
            "description": (
                "Extends F1a with: (1) stack/flue heat loss rising at part load "
                "(gasifier part-load degrades HRSG recovery); "
                "(2) ASU auxiliary power fraction rising at part load; "
                "eta_net = eta_cycle * (1-flue_loss) * (1-aux_frac). "
                f"P_rated={m.P_rated:.0f} MW, eta_iso={m.eta_iso:.2f}, "
                f"min_PLR={m.min_plr:.2f}, aux_base={m.aux_base:.2f}"
            ),
            "inputs": {
                "part_load_ratio": {"unit": "dimensionless", "range": [0.40, 1.0]},
                "ambient_temp":    {"unit": "degC",          "range": [-10.0, 45.0]},
            },
            "outputs": {
                "power_mw":           {"unit": "MW_e"},
                "efficiency":         {"unit": "dimensionless (LHV net)"},
                "flue_loss_fraction": {"unit": "dimensionless"},
                "aux_power_fraction": {"unit": "dimensionless"},
                "coal_rate_kgs":      {"unit": "kg/s"},
                "syngas_rate_nm3s":   {"unit": "Nm3/s"},
                "co2_rate_kgs":       {"unit": "kg_CO2/s"},
                "co2_intensity":      {"unit": "g_CO2/kWh_e (no CCS)"},
            },
            "source": (
                "Cormos (2012) Int. J. Hydrogen Energy 37; "
                "IEA GHG R&D (2003); Higman & van der Burgt (2008) Gasification; "
                "Booras & Holt (2004) EPRI"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    print(f"Rated ISO: P={float(r['power_mw']):.1f} MW, "
          f"eta={float(r['efficiency'])*100:.2f}%, "
          f"flue={float(r['flue_loss_fraction'])*100:.2f}%, "
          f"aux={float(r['aux_power_fraction'])*100:.2f}%, "
          f"coal={float(r['coal_rate_kgs']):.2f} kg/s, "
          f"CO2={float(r['co2_intensity']):.0f} g/kWh")
    r2 = model.predict({"part_load_ratio": 0.50, "ambient_temp": 15.0})
    print(f"50% PLR:   P={float(r2['power_mw']):.1f} MW, "
          f"eta={float(r2['efficiency'])*100:.2f}%, "
          f"flue={float(r2['flue_loss_fraction'])*100:.2f}%, "
          f"aux={float(r2['aux_power_fraction'])*100:.2f}%, "
          f"CO2={float(r2['co2_intensity']):.0f} g/kWh")
