"""
EC058 -- Flat Plate Solar Collector -- F2a Dynamic Thermal -- predict interface.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import FlatPlateCollectorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC058"
    component_name = "Flat Plate Solar Collector"
    fidelity = "F2a -- Dynamic Lumped-Capacitance Thermal Model"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FlatPlateCollectorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Steady-state prediction (single operating point).

        inputs:
            irradiance_W_m2    : float, solar irradiance on collector plane [W/m2]
            T_inlet_C          : float, fluid inlet temperature [degC]
            T_ambient_C        : float, ambient temperature [degC]
            incidence_angle_deg: float, angle of incidence [deg] (default 0)
        """
        G = inputs.get("irradiance_W_m2", 800.0)
        T_in = inputs.get("T_inlet_C", 40.0)
        T_amb = inputs.get("T_ambient_C", 25.0)
        theta = inputs.get("incidence_angle_deg", 0.0)

        result = self._model.steady_state(G, T_in, T_amb, theta)
        result["iam_factor"] = float(self._model.iam(theta))
        return result

    def predict_dynamic(self, inputs: dict) -> dict:
        """
        Dynamic simulation over a time series.

        inputs:
            t_span     : (t_start, t_end) in seconds
            t_eval     : array of evaluation times [s]
            T_m0       : initial mean temperature [degC]
            G_series   : array of irradiance values [W/m2] (same length as t_eval)
            T_in_series: array of inlet temperatures [degC]
            T_amb_series: array of ambient temperatures [degC]
            theta_series: array of incidence angles [deg] (optional)
            m_dot      : mass flow rate [kg/s] (optional)
        """
        t_eval = np.asarray(inputs["t_eval"], dtype=float)
        t_span = inputs.get("t_span", (t_eval[0], t_eval[-1]))
        T_m0 = inputs.get("T_m0", inputs.get("T_inlet_C", 20.0))

        G_arr = np.asarray(inputs["G_series"], dtype=float)
        T_in_arr = np.asarray(inputs["T_in_series"], dtype=float)
        T_amb_arr = np.asarray(inputs["T_amb_series"], dtype=float)
        theta_arr = np.asarray(inputs.get("theta_series", np.zeros_like(t_eval)), dtype=float)

        # Build interpolation functions
        G_func = lambda t: float(np.interp(t, t_eval, G_arr))
        T_in_func = lambda t: float(np.interp(t, t_eval, T_in_arr))
        T_amb_func = lambda t: float(np.interp(t, t_eval, T_amb_arr))
        theta_func = lambda t: float(np.interp(t, t_eval, theta_arr))

        m_dot = inputs.get("m_dot", None)

        return self._model.simulate(
            t_span=t_span,
            t_eval=t_eval,
            T_m0=T_m0,
            G_func=G_func,
            T_in_func=T_in_func,
            T_amb_func=T_amb_func,
            theta_func=theta_func,
            m_dot=m_dot,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs_steady_state": {
                "irradiance_W_m2": {"unit": "W/m2", "range": [0, 1200]},
                "T_inlet_C": {"unit": "degC", "range": [-10, 120]},
                "T_ambient_C": {"unit": "degC", "range": [-20, 50]},
                "incidence_angle_deg": {"unit": "deg", "range": [0, 90]},
            },
            "inputs_dynamic": {
                "t_eval": {"unit": "s", "note": "time array"},
                "G_series": {"unit": "W/m2"},
                "T_in_series": {"unit": "degC"},
                "T_amb_series": {"unit": "degC"},
                "theta_series": {"unit": "deg", "optional": True},
                "T_m0": {"unit": "degC", "note": "initial mean temperature"},
                "m_dot": {"unit": "kg/s", "optional": True},
            },
            "outputs": {
                "Q_useful_W": "W",
                "T_outlet_C": "degC",
                "T_mean_C": "degC",
                "efficiency": "-",
                "Q_loss_W": "W (dynamic only)",
                "Q_solar_W": "W (dynamic only)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"irradiance_W_m2": 800.0, "T_inlet_C": 40.0, "T_ambient_C": 25.0})
    print(f"Q_useful={r['Q_useful_W']:.1f} W, T_out={r['T_outlet_C']:.1f} C, "
          f"eta={r['efficiency']:.3f}")
