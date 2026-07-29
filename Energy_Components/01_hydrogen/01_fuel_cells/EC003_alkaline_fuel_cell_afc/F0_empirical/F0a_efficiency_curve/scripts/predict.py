"""Standardized predict interface for EC003 F0a — Alkaline Fuel Cell (AFC)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import EfficiencyCurve  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC003"
    component_name = "Alkaline Fuel Cell (AFC)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        lk = self.params["lookup"]
        keys = list(lk.keys())
        self._x_key, self._y_key = keys[0], keys[1]
        self.model = EfficiencyCurve(lk[self._x_key]["value"], lk[self._y_key]["value"])

    def predict(self, inputs: dict) -> dict:
        """inputs: {'load_fraction': value-or-array} -> {'efficiency': ...}."""
        if self._x_key not in inputs:
            raise KeyError("expected input key '%s'" % self._x_key)
        xq = inputs[self._x_key]
        y = self.model.lookup(xq)
        out = {self._y_key: float(y) if np.ndim(y) == 0 else y.tolist()}
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "metric": self.params.get("metric"),
            "input": self._x_key,
            "output": self._y_key,
            "x_range": [self.model.x_min, self.model.x_max],
            "source": self.params.get("source"),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print("Model:", info["component_id"], info["component_name"], "|", info["fidelity"])
    print("Metric:", info["metric"], "| input:", info["input"], "-> output:", info["output"])
    xmid = 0.5 * (info["x_range"][0] + info["x_range"][1])
    print("Sample predict({%r: %.4g}) -> %s" % (info["input"], xmid,
          m.predict({info["input"]: xmid})))
