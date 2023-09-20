from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pyoptimizer_backend.NestedVenv import NestedVenv


class OptimizerABC(ABC):
    """This is the abstract class for general optimizer algorithms
    which include all the abstract functions need.
    """

    # Private static data member to list dependency packages required
    # by this class. This should be overwritten in children.
    _packages = []

    def __init__(self, venv: NestedVenv = None) -> None:
        """Instantiates general Optimizer properties.

        :param venv: Virtual environment manager to use, defaults to None
        :type venv: pyoptimizer_backend.NestedVenv, optional
        """

        self._imports = {}  # Populated in self._import_deps()
        self.__venv = venv

    def check_install(self):
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
        for package in self._packages:
            # Install from local path if one is given
            if package in local_paths.keys():
                self.__venv.pip_install_e(local_paths[package])
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
          with "continuous_" or "categorical_" prepended to the name if two
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
    def set_config(self, experiment_dir: str, config: Dict[str, Any]):
        """This abstract method should be included function calls required for
        genereting initial configurations and files for optimizer.

        The key for each option must match the corresponding "name" field
        and the value type must match the one assigned in the "type" field
        from `get_config()`. Options prefixed with "continuous_" or
        "categorical_" should be stored as the name with the prefix removed
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
        itr: int,
        experiment_dir: str,
    ):
        """This abstract method should be overide with actual training function
        calls.

        :param prev_param: experimental parameter combination for previous
                           experiment
        :type prev_param: list[any]
        :param yield_value: experimental yield
        :type yield_value: float
        :param itr: experimental cycle number for training
        :type itr: int
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        """

        pass

    @abstractmethod
    def predict(
        self, prev_param: List[Any], yield_value: float, experiment_dir: str
    ):
        """This abstract method should be overide with actual predict function
        calls.
        :param prev_param: experimental parameter combination for previous
                           experiment
        :type prev_param: list[any]
        :param yield_value: experimental yield
        :type yield_value: float
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        """

        pass

    @property
    def dependencies(self) -> List[str]:
        """This is a static property to print the dependencies required
        by this class.

        :return: List of dependency package names
        :rtype: List[str]
        """

        return self._packages
