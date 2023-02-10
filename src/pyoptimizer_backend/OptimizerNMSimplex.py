from typing import Any, Dict

import json
import os

from pyoptimizer_backend.OptimizerABC import OptimizerABC
from pyoptimizer_backend.VenvManager import VenvManager


class OptimizerNMSimplex(OptimizerABC):
    def __init__(self):
        self.__name = "NMSimplex"
        self.__venv_name = "venv_{}".format(self.name)
        self.__packages = ["scipy", "benchmarking"]

    def check_install(self, install_prefix: str) -> bool:
        venv_m = VenvManager(os.path.join(install_prefix, self.venv_name))

        venv_m.start_venv()

        try:
            self._import_deps()
        except ModuleNotFoundError as e:
            print(e)
            return False

        return True

    def install(self, install_prefix: str, local_paths: Dict[str, str] = {}):
        self.__install_prefix = install_prefix

        venv_m = VenvManager(os.path.join(install_prefix, self.venv_name))

        venv_m.start_venv()

        # Install each package
        for package in self.__packages:
            # Install from local path if one is given
            if package in local_paths:
                venv_m.pip_install_e(local_paths[package])
            else:
                venv_m.pip_install(package)

        self._import_deps()

    def get_config(self):
        config = [
            {
                "name": "direction",
                "type": str,
                "value": ["min", "max"],
            },
            {
                "name": "feature_names",
                "type": list,
                "value": [],
            },
            {
                "name": "bounds",
                "type": list[list],
                "value": [[]],
            },
            {
                "name": "budget",
                "type": int,
                "value": 100,
            },
            {
                "name": "param_init",
                "type": list,
                "value": [],
            },
            {
                "name": "xatol",
                "type": float,
                "value": 1e-8,
            },
            {
                "name": "display",
                "type": bool,
                "value": False,
            },
            {
                "name": "server",
                "type": bool,
                "value": False,
            },
        ]
        return config

    def set_config(self, experiment_dir: str, config: Dict[str, Any]):
        # TODO: config validation?

        output_file = os.path.join(experiment_dir, "recent_config.json")

        with open(output_file, "w") as fout:
            json.dump(config, fout, indent=4)

    def train(self):
        """No training step for this algorithm."""

        pass

    def predict(self, prev_param, yield_value, experiment_dir, obj_func=None):
        with open(os.path.join(experiment_dir, "recent_config.json")) as fout:
            config = json.load(fout)

        self._imports["minimize"](
            obj_func,
            config["initial_parameters"],
            method="nelder-mead",
            options={
                "xatol": config["xatol"],
                "disp": config["display"],
                "bounds": config["bounds"],
            },
        )

    @property
    def name(self) -> str:
        return self.__name

    @property
    def venv_name(self) -> str:
        return self.__venv_name

    def _import_deps(self):
        """Import package needed to run the optimizer."""

        # import numpy as np
        from scipy.optimize import minimize

        self._imports = {
            "minimize": minimize,
            # "np": np,
        }
