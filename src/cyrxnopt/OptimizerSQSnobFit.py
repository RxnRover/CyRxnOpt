import json
import os
from typing import Any, Callable, Dict, List, Optional

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerABC import OptimizerABC


class OptimizerSQSnobFit(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    _packages = ["SQSnobFit"]

    def __init__(self, venv: NestedVenv) -> None:
        """Optimizer class for the SQSnobFit algorithm from the
        ``SQSnobFit`` package.

        :param venv: Virtual environment manager to use
        :type venv: cyrxnopt.NestedVenv
        """

        super().__init__(venv)

    def get_config(self) -> List[Dict[str, Any]]:
        """Get the configuration options available for this optimizer.

        :return: List of configuration options with option name, data type,
                 and information about which values are allowed/defaulted.
        :rtype: List[Dict[str, Any]]
        """

        config: List[Dict[str, Any]] = [
            {
                "name": "direction",
                "type": "str",
                "value": ["min", "max"],
            },
            {
                "name": "continuous_feature_names",
                "type": "list",
                "value": [],
            },
            {
                "name": "continuous_feature_bounds",
                "type": "list[list]",
                "value": [[]],
            },
            {
                "name": "budget",
                "type": "int",
                "value": 100,
            },
            {
                "name": "param_init",
                "type": "list",
                "value": [],
            },
            {
                "name": "maxfail",
                "type": "int",
                "value": 5,
            },
            {
                "name": "verbose",
                "type": "bool",
                "value": False,
            },
        ]

        return config

    def set_config(self, experiment_dir: str, config: Dict[str, Any]) -> None:
        """Set the configuration for this instance of the optimizer.

        Valid configuration options should be retrieved using ``get_config()``
        before calling this function.

        :param experiment_dir: Output directory for the configuration file.
        :type experiment_dir: str
        :param config: CyRxnOpt-level config for the optimizer
        :type config: Dict[str, Any]
        """

        self._import_deps()

        # TODO: config validation should be performed

        output_file = os.path.join(experiment_dir, "recent_config.json")

        # Write the configuration to a file for later use
        with open(output_file, "w") as fout:
            json.dump(config, fout, indent=4)

    def train(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func: Optional[Callable] = None,
    ) -> List[Any]:
        """No training step for this algorithm.

        :returns: List will always be empty.
        :rtype: List[Any]
        """

        return []

    def predict(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func: Optional[Callable[..., float]] = None,
    ) -> List[Any]:
        """Find the desired optimum of the provided objective function.

        :param prev_param: Parameters provided from the previous prediction or
                           training step
        :type prev_param: List[Any]
        :param yield_value: Result from the previous prediction or training step
        :type yield_value: float
        :param experiment_dir: Output directory for the optimizer algorithm
        :type experiment_dir: str
        :param config: CyRxnOpt-level config for the optimizer
        :type config: Dict[str, Any]
        :param obj_func: Objective function to optimize, defaults to None
        :type obj_func: Optional[Callable[..., float]], optional

        :returns: The next suggested reaction to perform
        :rtype: List[Any]
        """

        self._import_deps()

        # Load the config file
        # with open(os.path.join(experiment_dir, "recent_config.json")) as fout:
        #     config = json.load(fout)

        # Convert initial parameters to tuple
        # param_init = tuple(config["param_init"])
        param_init = config["param_init"]

        # Convert bounds list to sequence of tuples
        # bounds = tuple([tuple(bound_list) for bound_list in config["bounds"]])
        bounds = config["continuous_feature_bounds"]

        options = {
            "minfcall": None,
            "maxmp": None,
            "maxfail": config["maxfail"],
            "verbose": config["verbose"],
        }
        options = self._imports["SQSnobFit"].optset(options)

        # Call the minimization function
        result, history = self._imports["SQSnobFit"].minimize(
            obj_func,
            param_init,
            bounds,
            config["budget"],
            options,
        )

        result.history = history

        # TODO: This is returning a result object, not the next suggested params
        return result

    def _import_deps(self) -> None:
        """Import package needed to run the optimizer."""

        import SQSnobFit  # type: ignore

        self._imports = {
            "SQSnobFit": SQSnobFit,
        }
