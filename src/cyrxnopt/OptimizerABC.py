import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cyrxnopt.NestedVenv import NestedVenv

logger = logging.getLogger(__name__)


class OptimizerABC(ABC):
    """This is the abstract class for general optimizer algorithms
    which include all the abstract functions need.
    """

    # Private static data member to list dependency packages required
    # by this class. This should be overwritten in children.
    _packages: List[str] = []

    def __init__(self, venv: NestedVenv) -> None:
        """Instantiates general Optimizer properties.

        :param venv: Virtual environment manager to use
        :type venv: cyrxnopt.NestedVenv
        """

        self._imports: Dict[str, Any] = {}  # Populated in self._import_deps()
        self.__venv = venv

    def check_install(self) -> bool:
        """Check if an installation for this optimizer exists or not.

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
            logger.error(e)
            return False

        return True

    def install(self, local_paths: Dict[str, str] = {}) -> None:
        """Install the optimizer and its dependencies.

        The list of packages to be installed can be checked with the
        ``OptimizerABC.dependencies`` property.

        :param local_paths: Mapping of package names to local paths to the
            packages to be installed. The package names in the mapping must
            match a name returned by ``OptimizerABC.dependencies``.
            Defaults to {}
        :type local_paths: Dict[str, str], optional
        """

        logger.info("Installing {}...".format(self.__class__.__name__))

        if not self.__venv.is_active():
            self.__venv.activate()

        # Install each package
        for package in self._packages:
            # Install from local path if one is given
            if package in local_paths.keys():
                self.__venv.pip_install_e(Path(local_paths[package]))
            else:
                self.__venv.pip_install(package)

        # Import the packages after they were installed
        self._import_deps()

    @abstractmethod
    def get_config(self) -> List[Dict[str, Any]]:
        """This abstract method should be included function calls for returning
        all the initial configuration requires for optimizer.

        The configuration descriptions returned by this function are
        dictionaries with three keys, "name", "type", and "value":

        - "name" will contain the name for the config option in snake_case,
          with "continuous" or "categorical" prepended to the name if two
          versions of the option are needed for continuous and categorical
          variables.
        - "type" will contain the Python type annotation describing the type
          to use for this option (for translating to strictly typed languages)
        - "value" will provide the default value of the option.
        - "range" is optional and used to specify the allowed range for
          numbers or to constrain what options a string input can accept.
          For example, a description of an option for the optimization
          direction could be::

             {
                 "name": "optimization_direction",
                 "type": str,
                 "value": "min",
                 "range": ["min", "max"]
             }
        - "description" is an optional description of the purpose of a config
          option, along with any caveats that may come with it.

        .. TODO::

           Think about how to define number bounds in only one direction,
           like saying "this integer must be >0 or >=0.

        .. TODO::

           Write a page in the documentation describing this, as well as
           expected mappings to traditional user interface widgets, like text
           inputs, combo boxes, and number sliders.
        """

        pass

    @abstractmethod
    def set_config(self, experiment_dir: str, config: Dict[str, Any]) -> None:
        """This abstract method should be included function calls required for
        genereting initial configurations and files for optimizer.

        The key for each option must match the corresponding "name" field
        and the value type must match the one assigned in the "type" field
        from ``get_config()``. Options prefixed with "continuous" or
        "categorical" should be stored as the name with the prefix removed
        under either the "continuous" or "categorical" key. For example,
        the "continuous_feature_names" config option should be stored in a
        ``config`` dict as ``config["continuous"]["feature_names"]``.

        :param experiment_dir: Experimental directory for saving data files.
        :type experiment_dir: str
        :param config: Configuration settings defined from `get_config()`.
        :type config: Dict[str, Any]
        """

        pass

    @abstractmethod
    def train(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func: Optional[Callable] = None,
    ) -> List[Any]:
        """Abstract optimizer training function.

        :param prev_param: experimental parameter combination for previous
                           experiment
        :type prev_param: List[Any]
        :param yield_value: experimental yield
        :type yield_value: float
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: Optimizer config
        :type config: Dict[str, Any]
        :param obj_func: Objective function to optimize.
        :type obj_func: Optional[Callable]

        :returns: The next suggested reaction to perform
        :rtype: List[Any]
        """

        pass

    @abstractmethod
    def predict(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func: Optional[Callable] = None,
    ) -> List[Any]:
        """Abstract optimizer prediction function.

        :param prev_param: Previous suggested reaction conditions
        :type prev_param: List[Any]
        :param yield_value: Yield value from previous reaction conditions
        :type yield_value: float
        :param experiment_dir: Output directory for the current experiment
        :type experiment_dir: str
        :param config: Optimizer configuration
        :type config: Dict[str, Any]
        :param obj_func: Objective function to optimize, defaults to None
        :type obj_func: Optional[Callable], optional

        :returns: The next suggested conditions to perform
        :rtype: List[Any]
        """

        pass

    @abstractmethod
    def _import_deps(self) -> None:
        """Imports required dependencies for the optimizer"""

        pass

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Verifies that an optimizer configuration is valid.

        :param config: Optimizer configuration to check
        :type config: Dict[str, Any]
        :raises RuntimeError: Invalid optimizer configuration
        """

        # Make sure that feature names are provided
        if (
            "continuous_feature_names" not in config
            and "categorical_feature_names" not in config
        ):
            raise RuntimeError(
                (
                    "Either 'continuous_feature_names' or "
                    "'categorical_feature_names' must be provided in the "
                    "configuration."
                )
            )

        # Make sure all continuous feature descriptors were provided
        if "continuous_feature_names" in config:
            no_bounds = False
            no_resolutions = False
            msg = "'continuous_feature_names' was provided, but "

            if "continuous_feature_bounds" not in config:
                no_bounds = True
                msg += "'continuous_feature_bounds' "
            if "continuous_feature_resolutions" not in config:
                no_resolutions = True
                if no_bounds:
                    msg += "and "
                msg += "'continuous_feature_resolutions' "

            if no_bounds and no_resolutions:
                msg += "were "
            else:
                msg += "was "

            msg += "not."

            if no_bounds or no_resolutions:
                raise RuntimeError(msg)

        # Make sure all categorical feature descriptors were provided
        if "categorical_feature_names" in config:
            msg = "'categorical_feature_names' was provided, but "

            if "categorical_feature_values" not in config:
                msg += "'categorical_feature_values' was not."
                raise RuntimeError(msg)

        if "budget" not in config:
            raise RuntimeError("'budget' must be provided in the config.")

    @property
    def dependencies(self) -> List[str]:
        """Dependencies required by this optimizer.

        :return: Dependency package names
        :rtype: List[str]
        """

        return self._packages
