import json
import os
from typing import Any, Dict, List

from pyoptimizer_backend.OptimizerABC import OptimizerABC
from pyoptimizer_backend.VenvManager import VenvManager


class OptimizerNMSimplex(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    __packages = ["scipy"]

    def __init__(self, venv: VenvManager = None) -> None:
        """Optimizer class for the Nelder-Mead Simplex algorithm from the
        ``scipy`` package.

        :param venv: Virtual environment manager to use, defaults to None
        :type venv: pyoptimizer_backend.VenvManager, optional
        """

        self._imports = {}  # Populated in self._import_deps()
        self.__venv = venv

    def check_install(self) -> bool:
        """Check if the installation for this optimizer exists or not.

        :return: Whether the optimizer is installed (True) or not (False).
        :rtype: bool
        """

        # Attempt to import all of the packages this optimizer depends
        # on. If this import fails, we consider the optimizer to not
        # be installed or to have a broken install.
        try:
            self._import_deps()
        except ModuleNotFoundError as e:
            # Printing the exception so the user knows what went wrong
            print(e)
            return False

        return True

    def install(self, local_paths: Dict[str, str] = {}) -> None:
        """Install the dependencies required for this optimizer class.

        The list of packages to be installed can be checked with

        .. code-block:: python

           print(OptimizerNMSimplex.dependencies)

        :param local_paths: Local paths to the packages to be installed,
                            defaults to {}
        :type local_paths: Dict[str, str], optional
        """

        # Install each package
        for package in self.__packages:
            # Install from local path if one is given
            if package in local_paths:
                self.__venv.pip_install_e(local_paths[package])
            else:
                self.__venv.pip_install(package)

        # Import the packages after they were installed
        self._import_deps()

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

        # Convert initial parameters to tuple
        param_init = tuple(config["param_init"])

        # Convert bounds list to sequence of tuples
        bounds = tuple([tuple(bound_list) for bound_list in config["bounds"]])

        # Call the minimization function
        result = self._imports["minimize"](
            obj_func,
            param_init,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": config["budget"],
                "xatol": config["xatol"],
                "disp": config["display"],
            },
        )

        return result

    @property
    def dependencies() -> List[str]:
        """This is a static property to print the dependencies required
        by this class.

        :return: List of dependency package names
        :rtype: List[str]
        """
        return OptimizerNMSimplex.__packages

    def _import_deps(self) -> None:
        """Import package needed to run the optimizer."""

        from scipy.optimize import minimize

        self._imports = {
            "minimize": minimize,
        }
