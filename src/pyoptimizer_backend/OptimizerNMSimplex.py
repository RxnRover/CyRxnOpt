from typing import Any, Dict

import json
import os

from pyoptimizer_backend.optimizer_abc import OptimizerABC
from pyoptimizer_backend.venv_manager import Venv_manager


class OptimizerNMSimplex(OptimizerABC):
    def __init__(self):
        self.__name = "NMSimplex"
        self.__venv_name = "venv_{}".format(self.name)
        self.__packages = ["scipy", "benchmarking"]

    def check_install(self, install_prefix: str) -> bool:
        venv_m = Venv_manager(os.path.join(install_prefix, self.venv_name))

        venv_m.run()

        try:
            self._import_deps()
        except ModuleNotFoundError:
            return False

        return True

    def install(self, install_prefix: str, local_paths: Dict[str, str] = {}):
        self.__install_prefix = install_prefix

        venv_m = Venv_manager(os.path.join(install_prefix, self.venv_name))

        # TODO: Creates venv???
        venv_m.run()

        # Install each package
        for package in self.__packages:
            # Install from local path if one is given
            if package in local_paths:
                venv_m.pip_install_e(local_paths[package])
            else:
                venv_m.pip_install(package)

    def get_config(self):
        config = [
            {
                "name": "direction",
                "type": str,
                "value": ["min", "max"],
            },
            {
                "name": "feature_names",
                "type": list[list],
                "value": [[]],
            },
            {
                "name": "bounds",
                "type": list[list],
                "value": [[]],
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

        json.dump(config, os.path.join(experiment_dir, "recent_config.json"))

    def train(self):
        """No training step for this algorithm."""

        pass

    def predict(self, prev_param, yield_value, experiment_dir, obj_func = None):
        config = json.load(os.path.join(experiment_dir, "recent_config.json"))

        self.__imports["minimize"](
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
        """importing all the packages and libries needed for running amlro optimizer
        """

        # import numpy as np
        from scipy.optimize import minimize

        self.__imports = {
            "minimize": minimize,
            # "np": np,
        }
