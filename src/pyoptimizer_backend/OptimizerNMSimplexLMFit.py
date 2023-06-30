import json
import os
from typing import Any, Dict, List

from pyoptimizer_backend.OptimizerABC import OptimizerABC
from pyoptimizer_backend.VenvManager import VenvManager


class OptimizerNMSimplexLMFit(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    _packages = ["lmfit"]

    def __init__(self, venv: VenvManager = None) -> None:
        """Optimizer class for the Nelder-Mead Simplex algorithm from the
        ``scipy`` package.

        :param venv: Virtual environment manager to use, defaults to None
        :type venv: pyoptimizer_backend.VenvManager, optional
        """

        super().__init__(venv)

    def get_config(self) -> List[Dict[str, Any]]:
        """Get the configuration options available for this optimizer.

        :return: List of configuration options with option name, data type,
                 and information about which values are allowed/defaulted.
        :rtype: List[Dict[str, Any]]
        """

        self._import_deps()

        config = [
            {
                "name": "direction",
                "type": str,
                "value": ["min", "max"],
            },
            {
                "name": "continuous_feature_names",
                "type": list,
                "value": [],
            },
            {
                "name": "continuous_feature_bounds",
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

    def set_config(self, experiment_dir: str, config: Dict[str, Any]) -> None:
        """Set the configuration for this instance of the optimizer. Valid
        configuration options should be retrieved using `get_config()` before
        calling this function.

        :param experiment_dir: Output directory for the configuration file.
        :type experiment_dir: str
        :param config: Configuration options for this optimizer instance.
        :type config: Dict[str, Any]
        """

        self._import_deps()

        # TODO: config validation should be performed

        output_file = os.path.join(experiment_dir, "recent_config.json")

        # Write the configuration to a file for later use
        with open(output_file, "w") as fout:
            json.dump(config, fout, indent=4)

    def train(self) -> None:
        """No training step for this algorithm."""

        pass

    def predict(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func=None,
    ) -> None:
        """Find the desired optimum of the provided objective function.

        :param prev_param: Parameters provided from the previous prediction or
                           training step.
        :type prev_param: List[Any]
        :param yield_value: Result from the previous prediction or training
                            step.
        :type yield_value: float
        :param experiment_dir: Output directory for the optimizer algorithm.
        :type experiment_dir: str
        :param obj_func: Objective function to optimize, defaults to None
        :type obj_func: function, optional
        """

        self._import_deps()

        # Load the config file
        with open(os.path.join(experiment_dir, "recent_config.json")) as fout:
            config = json.load(fout)

        # Form the parameter list with initial values, names, and bounds
        parameters = self._imports["Parameters"]()
        for i in range(len(config["continuous_feature_bounds"])):
            # Get the feature name if one is provided
            if len(config["continuous_feature_names"]) > 0:
                feature_name = config["continuous_feature_names"][i]
            else:
                feature_name = "f{}".format(i)

            # Get the initial value if one is provided
            if len(config["param_init"]) > 0:
                feature_value = config["param_init"][i]
            else:
                feature_value = None

            bounds = config["continuous_feature_bounds"][i]

            # Add the feature to the parameter list
            parameters.add(
                feature_name, value=feature_value, min=bounds[0], max=bounds[1]
            )

        # Call the minimization function
        result = self._imports["minimize"](
            obj_func,
            parameters,
            method="nelder",
            max_nfev=config["budget"],
        )

        return result

    def _import_deps(self) -> None:
        """Import package needed to run the optimizer."""

        from lmfit import Parameters, minimize

        self._imports = {"minimize": minimize, "Parameters": Parameters}
