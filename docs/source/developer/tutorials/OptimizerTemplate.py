# TODO: Update this template for your optimizer algorithm. To help guide the
# process, TODO items like this one have been added to the template. These can
# be easily found by searching for "TODO".


from typing import Any, Dict, List

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerABC import OptimizerABC


# TODO: Rename ``OptimizerTemplate`` to ``OptimizerName``, where ``Name`` is
# the name of your optimizer or an abbreviation of it.
class OptimizerTemplate(OptimizerABC):

    # List of dependency packages required by this class. This is used to
    # to install the required packages in the ``install()`` method.
    # TODO: Update this list for the packages for your optimizer
    _packages = ["package1", "package2", ...]

    def __init__(self, venv: NestedVenv = None) -> None:
        """This code will initialize your optimizer class.

        :param venv: Virtual environment manager to use, defaults to None
        :type venv: cyrxnopt.NestedVenv, optional
        """

        self._imports = {}  # Populated in self._import_deps()
        self.__venv = venv

    def get_config(self) -> List[Dict[str, Any]]:
        """Get the configuration options available for this optimizer.

        TODO: Add the required configuration variables needed to run your
        optimizer algorithm. Follow the common dictionary keys and add new
        features end of the config list.
        TODO: See the documentation for ``OptimizerABC.get_config()`` for how
        to write configuration descriptions.

        :return: List of configuration options with option name, data type,
                 and information about which values are allowed/defaulted.
        :rtype: List[Dict[str, Any]]
        """

        self._import_deps()

        # TODO: Update this config description list for your optimizer
        config = [
            {
                "name": "continuous_feature_names",
                "type": List[str],
                "value": [],
            },
            {
                "name": "continuous_feature_bounds",
                "type": List[List[float]],
                "value": [],
            },
            {
                "name": "continuous_feature_resolutions",
                "type": List[float],
                "value": [],
            },
            {
                "name": "categorical_feature_names",
                "type": List[str],
                "value": [],
            },
            {
                "name": "categorical_feature_values",
                "type": List[List[str]],
                "value": [],
            },
            {
                "name": "budget",
                "type": int,
                "value": 100,
            },
            {
                "name": "objectives",
                "type": List[str],
                "value": ["yield"],
            },
            {
                "name": "direction",
                "type": List[str],
                "value": ["min"],
                "range": ["min", "max"],
            },
        ]

        return config

    def set_config(self, experiment_dir: str, config: Dict[str, Any]) -> None:
        """Set the configuration for this instance of the optimizer. Valid
        configuration options should be retrieved using `get_config()` before
        calling this function.

        TODO: Handle all the configuration options and include all the code
        required to initialize your algorithm. For example, reaction space
        generation or generation of initial files. Depending on your workflow
        you can break down your code into more functions.
        TODO: See the documentation for ``OptimizerABC.set_config()`` for more
        information about the incoming configuration format to expect.

        :param experiment_dir: Output directory for the configuration file.
        :type experiment_dir: str
        :param config: Configuration options for this optimizer instance.
        :type config: Dict[str, Any]
        """

        self._import_deps()

        # TODO: Add config validation code
        # TODO: Add initial file generation as needed
        # TODO: Add other initial configuration code as needed

    def train(
        self,
        prev_param: List[Any],
        yield_value: float,
        itr: int,
        experiment_dir: str,
        config: Dict,
        obj_func=None,
    ) -> List[Any]:
        """Generates initial training dataset needed for training.

        TODO: If your algorithm requires generating a training dataset or
        loading training files, add that code here. Otherwise, just set the
        function body to call ``pass`` so it is a no-op.

        :param prev_param: Parameters provided from the previous training step
        :type prev_param: List[Any]
        :param yield_value: Experimental result from the previous training step
        :type yield_value: float
        :param itr: experimental cycle number for training
        :type itr: int
        :param experiment_dir: Output directory for the optimizer algorithm
        :type experiment_dir: str
        :param config: Initial reaction feature configurations
        :type config: Dict
        :param obj_func: Objective function to optimize, defaults to None
        :type obj_func: function, optional

        :return: Next parameter combination for next experimental cycle.
        :rtype: LIst[Any]
        """

        # TODO: If your algorithm doesn't need training, uncomment the following
        #       "pass" line and delete all lines in the function after it.
        # pass

        self._import_deps()

        # TODO: Add necessary training code here

    def predict(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func=None,
    ) -> List[Any]:
        """Finds the desired optimum of the provided objective function.

        TODO: Call your optimizer to find the optimum reaction conditions.

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

        :return: Next suggested parameter combination
        :rtype: List[Any]
        """

        self._import_deps()

        # TODO: Add the code necessary to call your algorithm to optimize the
        #       reaction process.
        # TODO: If your algorithm does not map 1 call -> 1 reaction (e.g. uses
        #       an internal optimization loop), then an objective function must
        #       be provided to communicate individual reaction parameter
        #       suggestions and receive the result of each reaction.

    def _import_deps(self) -> None:
        """Imports packages needed to run the optimizer.

        TODO: Import all the required packages here, then add those packages to
        the self._imports dictionary to be used later to access packages as
        necessary.
        """

        # TODO: Update required package imports
        from YourLibrary import YourPackage

        # TODO: Update imported package dictionary
        self._imports = {"package": YourPackage}
