import os
import random
from typing import Any, Dict, List

from pyoptimizer_backend.NestedVenv import NestedVenv
from pyoptimizer_backend.OptimizerABC import OptimizerABC


class OptimizerEDBOp(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    _packages = ["benchmarking", "edboplus", "pandas"]

    # overidding methods
    def __init__(self, venv: NestedVenv = None) -> None:
        """initializing optimizer EDBO+ object

        :param venv: Virtual envirement class object, defaults to None
        :type venv: NestedVenv, optional
        """

        super(OptimizerEDBOp, self).__init__(venv)

        self._edbop_filename = "my_optimization.csv"
        self._reaction_order_filename = "reaction_order.csv"

    def get_config(self):
        """This function will return the configurations which are needed
        to initialize an optimizer through `set_config()`.

        :return: Configuration option descriptions.
        :rtype: List[Dict[str, Any]]
        """

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

    def set_config(self, experiment_dir: str, config: Dict) -> None:
        """Generate all the nessasry data files

        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: configuration dict which required for initializing edbo+
        :type config: dict
        """
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        config = self.config_translate(
            config
        )  # get reaction scope configurations
        # from general config file

        # generate reaction scope for EDBOp
        self._imports["EDBOplus"]().generate_reaction_scope(
            components=config["reaction_components"],
            directory=experiment_dir,
            filename=self._edbop_filename,
            check_overwrite=False,
        )

        # initialize the edbop optimization file will be use for prediction
        self._imports["EDBOplus"]().run(
            directory=experiment_dir,
            filename=self._edbop_filename,  # Previously generated scope.
            objectives=config["objectives"],  # ['yield', 'ee', 'side_product'],
            # Objectives to be optimized.
            objective_mode=config["direction"],  # ['max', 'max', 'min'],
            # Maximize yield and ee but minimize side_product.
            batch=1,  # Number of experiments in parallel that
            # we want to perform in this round.
            columns_features="all",  # features to be included in the model.
            init_sampling_method="seed",  # initialization method.
            seed=random.randint(0, 2**32 - 1),
        )

        # Create file for preserving reaction order
        # TODO: Rework this when we switch to multi-objective!
        with open(self._reaction_order_filename, "w") as fout:
            feature_names = config["continuous"]["feature_names"]
            # If categorical feature names is an empty list, list.extend leaves
            # the list unchanged
            feature_names.extend(config["categorical"]["feature_names"])

            objectives = config["objectives"][0]

            # Collect the feature names and objective names as headers
            headers = feature_names
            headers.extend(objectives)

            fout.write(",".join(headers), "\n")

    def train(
        self,
        prev_param: List[Any],
        yield_value: List[float],
        itr: int,
        experiment_dir: str,
        config: Dict,
        obj_func=None,
    ) -> List[Any]:
        """generate initial training dataset needed for optmizer. EDBOP doesent need
        initial training. traning function should be pass or empty.

        :param prev_param: experimental parameter combination for previous experiment
        :type prev_param: list
        :param yield_value: experimental yield
        :type yield_value: float
        :param itr: experimental cycle number for training
        :type itr: int
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: Initial reaction feature configurations
        :type config: Dict
        :return: next parameter combination for next experimental cycle.
        :rtype: list
        """
        pass

    def predict(
        self,
        prev_param: List[Any],
        yield_value: List[float],
        experiment_dir: str,
        config: Dict,
        obj_func=None,
    ) -> List[Any]:
        """prediction of next best combination of parameters and
         traning machine learning model from last experimental data for active learning.

        :param prev_param: experimental parameter combination for previous experiment
        :type prev_param: list
        :param yield_value: experimental yield
        :type yield_value: float
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: Initial reaction feature configurations
        :type config: Dict
        :return: best predicted parameter combination
        :rtype: list
        """
        filename = "my_optimization.csv"

        config = self.config_translate(
            config
        )  # get reaction scope configurations
        # from general config file

        # reading optimization file with reaction conditions
        df_edbo = self._imports["pd"].read_csv(
            os.path.join(experiment_dir, filename)
        )

        # TODO: Writing the entire dataframe of shape (2085136, 6),
        #       12,510,816 elements: 8.674756252000407 sec. This can probably
        #       be optimized quite a bit
        if len(prev_param) != 0:
            # [df_edbo.loc[0,config['objectives'][i]] =
            # yield_value[i] for i in range(len(yield_value))]
            df_edbo.loc[0, config["objectives"][0]] = yield_value
            df_edbo.to_csv(os.path.join(experiment_dir, filename), index=False)

            # Write the reaction parameters and results to the file preserving
            # reaction order
            # TODO: Rework this when we switch to multi-objective!
            with open(self._reaction_order_filename, "a") as fout:
                line = prev_param
                line.extend(yield_value)
                fout.write(",".join(line))
                fout.write("\n")

        # running the edbop prediction
        self._imports["EDBOplus"]().run(
            directory=experiment_dir,
            filename=filename,  # Previously generated scope.
            objectives=config[
                "objectives"
            ],  # ['yield', 'ee', 'side_product'],  # Objectives to be optimized.
            objective_mode=config["direction"],  # ['max', 'max', 'min'],
            # Maximize yield and ee but minimize side_product.
            batch=1,  # Number of experiments in parallel that
            # we want to perform in this round.
            columns_features="all",  # features to be included in the model.
            init_sampling_method="seed",  # initialization method.
            seed=random.randint(0, 2**32 - 1),
            write_extra_data=False,
        )

        # after one cycle of prediction again read the reaction condition file to
        #  getting the next reaction condition.
        df_edbo = self._imports["pd"].read_csv(
            os.path.join(experiment_dir, filename)
        )

        next_combo = df_edbo.iloc[:1].values.tolist()
        next_combo = next_combo[0][:-2]
        print("Next combo:", next_combo)

        return next_combo

    def config_translate(self, config: Dict) -> Dict:
        """This function convert general config dictionary into
        EDBOp reaction scope config dictionary format.

        :param config: general configuration dict
        :type config: Dict
        :return: translated configuration dict
        :rtype: Dict
        """
        self._import_deps()
        reaction_components = {}

        # If the keys are returned as they were given in `get_config` then
        # translate them to the format that works here
        config["continuous"] = {}
        if (
            "continuous_feature_names" in config
            and len(config["continuous_feature_names"]) > 0
        ):
            config["continuous"] = {}

            config["continuous"]["feature_names"] = config[
                "continuous_feature_names"
            ]
            config["continuous"]["bounds"] = config["continuous_feature_bounds"]
            config["continuous"]["resolutions"] = config[
                "continuous_feature_resolutions"
            ]
        else:
            config["continuous"]["feature_names"] = []
            config["continuous"]["bounds"] = []
            config["continuous"]["resolutions"] = []

        config["categorical"] = {}
        if (
            "categorical_feature_names" in config
            and len(config["categorical_feature_names"]) > 0
        ):
            config["categorical"] = {}

            config["categorical"]["feature_names"] = config[
                "categorical_feature_names"
            ]
            config["categorical"]["values"] = config[
                "categorical_feature_values"
            ]
        else:
            config["categorical"]["feature_names"] = []
            config["categorical"]["values"] = []

        for i in range(len(config["continuous"]["feature_names"])):
            low_bound = config["continuous"]["bounds"][i][0]
            upper_bound = config["continuous"]["bounds"][i][1]
            increment = config["continuous"]["resolutions"][i]

            values = self._imports["np"].arange(
                low_bound, upper_bound, increment
            )

            reaction_components[
                config["continuous"]["feature_names"][i]
            ] = values

        if bool(config["categorical"]["feature_names"]):
            for i in range(len(config["categorical"]["feature_names"])):
                reaction_components[
                    config["categorical"]["feature_names"][i]
                ] = config["categorical"]["values"][i]

        # EDBO+ supports multi-objective optimization, of which single-
        # objective optimization is a subset. When providing arguments
        # for single-objective optimization, only one objective and one
        # corresponding direction must be given. This catches when the user
        # does not provide single-element lists for the objectives and
        # their directions, which could be an easy mistake.
        if type(config["objectives"]) is str:
            config["objectives"] = [config["objectives"]]
        if type(config["direction"]) is str:
            config["direction"] = [config["direction"]]

        edbo_config = {
            "reaction_components": reaction_components,
            "objectives": config["objectives"],
            "direction": config["direction"],
        }

        return edbo_config

    def _import_deps(self) -> None:
        """importing all the packages and libries needed for running amlro
        optimizer
        """

        import numpy as np
        import pandas as pd
        from edbo.plus.optimizer_botorch import EDBOplus

        self._imports = {"EDBOplus": EDBOplus, "np": np, "pd": pd}
