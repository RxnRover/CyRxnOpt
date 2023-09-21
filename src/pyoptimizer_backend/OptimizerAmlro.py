import os
from typing import Any, Dict, List

from pyoptimizer_backend.OptimizerABC import OptimizerABC


class OptimizerAmlro(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    _packages = [
        "git+ssh://git@github.com/RxnRover/amlo",
        "numpy",
        "pandas",
    ]

    # overidding methods
    def __init__(self, venv=None) -> None:
        """initializing optimizer AMLRO object

        :param venv: Virtual envirement class object, defaults to None
        :type venv: NestedVenv, optional
        """

        super(OptimizerAmlro, self).__init__(venv)

    def train(
        self,
        prev_param: List[Any],
        yield_value: float,
        itr: int,
        experiment_dir: str,
        config: Dict,
        obj_func=None,
    ) -> List[Any]:
        """generate initial training dataset needed for AMLRO model training.

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
        self._import_deps()

        training_set_path = os.path.join(
            experiment_dir, "training_set_file.txt"
        )
        training_set_decoded_path = os.path.join(
            experiment_dir, "training_set_decoded_file.txt"
        )
        training_combo_path = os.path.join(
            experiment_dir, "training_combo_file.txt"
        )

        # training step
        next_parameters = self._imports[
            "training_set_generator"
        ].generate_training_data(
            training_set_path,
            training_set_decoded_path,
            training_combo_path,
            config,
            prev_param,
            yield_value,
            itr,
        )

        return next_parameters

    def predict(
        self,
        prev_param: List[Any],
        yield_value: float,
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
        self._import_deps()

        training_set_path = os.path.join(
            experiment_dir, "training_set_file.txt"
        )
        training_set_decoded_path = os.path.join(
            experiment_dir, "training_set_decoded_file.txt"
        )
        full_combo_path = os.path.join(experiment_dir, "full_combo_file.txt")

        # prediction step
        best_combo = self._imports["optimizer_main"].get_optimized_parameters(
            training_set_path,
            training_set_decoded_path,
            full_combo_path,
            config,
            prev_param,
            yield_value,
        )

        return best_combo

    def get_config(self) -> List[Dict[str, Any]]:
        """This function will return the configurations which are needed
        to initialize an optimizer through `set_config()`.

        :return: Configuration option descriptions.
        :rtype: List[Dict[str, Any]]
        """

        config = [
            {
                "name": "continuous_feature_names",
                "type": List[str],
                "value": [""],
            },
            {
                "name": "continuous_feature_bounds",
                "type": List[List[float]],
                "value": [[]],
            },
            {
                "name": "continuous_feature_resoultions",
                "type": List[float],
                "value": [],
            },
            {
                "name": "categorical_feature_names",
                "type": List[str],
                "value": [""],
            },
            {
                "name": "categorical_feature_values",
                "type": List[List[str]],
                "value": [[]],
            },
            {
                "name": "budget",
                "type": int,
                "value": 100,
            },
            {
                "name": "objectives",
                "type": List[str],
                "value": [""],
            },
            {
                "name": "objective_mode",
                "type": str,
                "value": "min",
                "range": ["min", "max"],
            },
        ]
        # TODO: Budget should be constrained to numbers greater than
        #       zero once that format is solidified.
        # TODO: Should the value of this "config" variable be moved into
        #       a JSON file to make it easier to modify without changing
        #       the code?

        return config

    def set_config(self, experiment_dir: str, config: Dict[str, Any]):
        """Generate all the necessary data files based on the given config.

        :param experiment_dir: Experimental directory for saving data files.
        :type experiment_dir: str
        :param config: Configuration settings defined from `get_config()`.
        :type config: Dict[str, Any]
        """

        self._import_deps()

        self._validate_config(config)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        # Add extra entries that AMLRO will understand
        config["continuous"] = {}
        config["categorical"] = {}
        config["continuous"]["feature_names"] = config[
            "continuous_feature_names"
        ]
        config["continuous"]["bounds"] = config["continuous_feature_bounds"]
        config["continuous"]["resolutions"] = config[
            "continuous_feature_resolutions"
        ]
        config["categorical"]["feature_names"] = config[
            "categorical_feature_names"
        ]
        config["categorical"]["values"] = config["categorical_feature_values"]

        full_combo_list = self._imports[
            "generate_combos"
        ].generate_uniform_grid(config)

        full_combo_list = self._imports["np"].around(
            full_combo_list, decimals=4
        )
        full_combo_df = self._imports["pd"].DataFrame(full_combo_list)
        training_combo_df = full_combo_df.sample(20)

        if bool(config["categorical_feature_names"]):
            feature_names_list = self._imports["np"].concatenate(
                (
                    config["continuous_feature_names"],
                    config["categorical_feature_names"],
                )
            )
        else:
            feature_names_list = config["continuous_feature_names"]

        full_combo_df.columns = feature_names_list

        full_combo_path = os.path.join(experiment_dir, "full_combo_file.txt")
        training_combo_path = os.path.join(
            experiment_dir, "training_combo_file.txt"
        )

        full_combo_df.to_csv(full_combo_path, index=False)
        training_combo_df.to_csv(training_combo_path, index=False)

        training_set_path = os.path.join(
            experiment_dir, "training_set_file.txt"
        )
        training_set_decoded_path = os.path.join(
            experiment_dir, "training_set_decoded_file.txt"
        )

        # Write the reaction conditions for training dataset into files
        # (encoded and decoded versions)
        with open(training_set_path, "w") as file_object:
            feature_names = ",".join([str(elem) for elem in feature_names_list])
            file_object.write(feature_names + ",Yield" + "\n")

        with open(training_set_decoded_path, "w") as file_object:
            feature_names = ",".join([str(elem) for elem in feature_names_list])
            file_object.write(feature_names + ",Yield" + "\n")

    def _import_deps(self) -> None:
        """importing all the packages and libries needed for running amlro optimizer"""
        import numpy as np
        import pandas as pd
        from amlro import (
            generate_combos,
            optimizer,
            optimizer_main,
            training_set_generator,
        )

        self._imports = {
            "generate_combos": generate_combos,
            "training_set_generator": training_set_generator,
            "optimizer": optimizer,
            "optimizer_main": optimizer_main,
            "np": np,
            "pd": pd,
        }

    def _validate_config(self, config):
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
