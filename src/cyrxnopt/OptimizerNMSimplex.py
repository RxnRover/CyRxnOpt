import json
import logging
import os
from collections.abc import Callable
from typing import Any, Optional

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerABC import OptimizerABC

logger = logging.getLogger(__name__)


class OptimizerNMSimplex(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    _packages = ["scipy"]

    def __init__(self, venv: NestedVenv) -> None:
        """Optimizer class for the Nelder-Mead Simplex algorithm from the
        ``scipy`` package.

        :param venv: Virtual environment manager to use
        :type venv: NestedVenv
        """

        super().__init__(venv)

    def get_config(self) -> list[dict[str, Any]]:
        """Gets the configuration options available for this optimizer.

        See :py:meth:`OptimizerABC.get_config` for more information about the
        config descriptions returned by this method and for general usage
        information.

        :return: List of configuration options with option name, data type,
                 and information about which values are allowed/defaulted.
        :rtype: list[dict[str, Any]]
        """

        config: list[dict[str, Any]] = [
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
                "name": "xatol",
                "type": "float",
                "value": 1e-8,
            },
            {
                "name": "display",
                "type": "bool",
                "value": False,
            },
            {
                "name": "server",
                "type": "bool",
                "value": False,
            },
        ]

        return config

    def set_config(self, experiment_dir: str, config: dict[str, Any]) -> None:
        """Sets the configuration for this instance of the optimizer.

        See :py:meth:`OptimizerABC.set_config` for more information about how
        to form the config dictionary and for general usage information.

        :param experiment_dir: Output directory for the configuration file
        :type experiment_dir: str
        :param config: Configuration options for this optimizer instance
        :type config: dict[str, Any]
        """

        self._import_deps()

        # TODO: config validation should be performed

        output_file = os.path.join(experiment_dir, "config.json")

        # Write the configuration to a file for later use
        with open(output_file, "w") as fout:
            json.dump(config, fout, indent=4)

    def train(
        self,
        prev_param: list[Any],
        yield_value: float,
        experiment_dir: str,
        config: dict[str, Any],
        obj_func: Optional[Callable] = None,
    ) -> list[Any]:
        """No training step for this algorithm.

        :returns: List will always be empty.
        :rtype: list[Any]
        """

        return []

    def predict(
        self,
        prev_param: list[Any],
        yield_value: float,
        experiment_dir: str,
        config: dict[str, Any],
        obj_func: Optional[Callable[..., float]] = None,
    ) -> list[Any]:
        """Find the desired optimum of the provided objective function.

        :param prev_param: Parameters provided from the previous prediction,
                           provide an empty list for the first call
        :type prev_param: list[Any]
        :param yield_value: Result from the previous prediction
        :type yield_value: float
        :param experiment_dir: Output directory for the optimizer algorithm
        :type experiment_dir: str
        :param config: CyRxnOpt-level config for the optimizer
        :type config: dict[str, Any]
        :param obj_func: Objective function to optimize, defaults to None
        :type obj_func: Optional[Callable[..., float]], optional

        :returns: The next suggested reaction to perform
        :rtype: list[Any]
        """

        self._import_deps()

        # Load the config file
        # with open(os.path.join(experiment_dir, "config.json")) as fout:
        #     config = json.load(fout)

        # Convert initial parameters to tuple
        param_init = tuple(config["param_init"])

        # Convert bounds list to sequence of tuples
        bounds = tuple(
            [
                tuple(bound_list)
                for bound_list in config["continuous_feature_bounds"]
            ]
        )

        # Call the minimization function
        results = self._imports["minimize"](
            obj_func,
            param_init,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": config["budget"],
                "xatol": config["xatol"],
                "disp": config["display"],
            },
            callback=self._create_writer(experiment_dir),
        )

        raw_results: list = []
        with open(os.path.join(experiment_dir, "results.csv")) as fin:
            for row in fin.readlines():
                row_list = row.split(",")
                row_list_float = [float(x) for x in row_list]
                raw_results.append(row_list_float)

        results.raw_results = raw_results

        # TODO: This is returning a result object, not the next suggested params
        return results

    def _create_writer(self, experiment_dir: str) -> Callable[..., None]:
        """Creates a callback function to write results for the optimizer.

        This function uses the "closure" technique to create and return
        a callback function that will write results to the correct location
        for the optimizer. These are typically considered an anti-pattern
        in Python, but I do not have a great way around it.

        :param experiment_dir: Experiment directory where files will be output.
        :type experiment_dir: str

        :return: Callback function to write results.
        :rtype: Callable[..., None]
        """

        def writer(intermediate_result) -> None:  # type: ignore
            """Callback function to write the results of each iteration to a
            results file in the experiment directory.

            This function will be called after each iteration of a
            ``scipy.optimize.minimize`` optimizer.

            :param intermediate_result: Intermediate result in the optimization
            :type intermediate_result: scipy.optimize.OptimizeResult.OptimizeResult
            """

            # TODO: Make this file name a constant for the package
            results_path = os.path.join(str(experiment_dir), "results.csv")

            # Create results list with parameters before results.
            # This will be the next row in the results file
            results = intermediate_result.x.tolist()
            results.append(intermediate_result.fun)

            # Convert results list to strings since ','.join() can't
            # be called on a list of numbers
            results = [str(x) for x in results]

            with open(results_path, "a") as fout:
                results_line = ",".join(results) + "\n"
                fout.write(results_line)

        return writer

    def _import_deps(self) -> None:
        """Import package needed to run the optimizer."""

        from scipy.optimize import minimize  # type: ignore

        self._imports = {
            "minimize": minimize,
        }
