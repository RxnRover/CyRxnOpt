import os
from typing import Any, Dict, List

from pyoptimizer_backend.OptimizerABC import OptimizerABC


class OptimizerAmlro(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    _packages = ["amlro", "benchmarking", "numpy"]

    # overidding methods
    def __init__(self, venv=None) -> None:
        """initializing optimizer AMLRO object

        :param venv: Virtual envirement class object, defaults to None
        :type venv: VenvManager, optional
        """

        super(OptimizerAmlro, self).__init__(venv)

    def train(
        self,
        prev_param: List[Any],
        yield_value: float,
        itr: int,
        experiment_dir: str,
        config: Dict,
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

    def get_config(self) -> Dict:
        """This function will return the configurations which need to initialize a
        optimizer

        :return: configuration dictionary
        :rtype: Dict
        """
        config = {
            {
                "Name": "continuous_feature_names",
                "Type": List[str],
                "value": [""],
            },
            {
                "Name": "continuous_feature_bounds",
                "Type": List[List[float]],
                "value": [[]],
            },
            {
                "Name": "continuous_feature_resoultions",
                "Type": List[float],
                "value": [],
            },
            {
                "Name": "categorical_feature_names",
                "Type": List[str],
                "value": [""],
            },
            {
                "Name": "categorical_feature_values",
                "Type": List[List[str]],
                "value": [[]],
            },
            {
                "Name": "budget",
                "Type": int,
                "value": 100,
            },
            {
                "Name": "objectives",
                "Type": List[str],
                "value": [""],
            },
            {
                "Name": "objective_mode",
                "Type": List[str],
                "value": [""],
            },
        }
        return config

    def set_config(self, experiment_dir: str, config: Dict) -> None:
        """Generate all the nessasry data files

        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: configuration dict which required for initializing AMLRO
        :type config: dict
        """
        self._import_deps()

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        full_combo_list = self._imports[
            "generate_combos"
        ].generate_uniform_grid(config)

        full_combo_list = self._imports["np"].around(
            full_combo_list, decimals=4
        )
        full_combo_df = self._imports["pd"].DataFrame(full_combo_list)
        training_combo_df = full_combo_df.sample(20)

        if bool(config["categorical"]):
            feature_names_list = self._imports["np"].concatenate(
                (
                    config["continuous"]["feature_names"],
                    config["categorical"]["feature_names"],
                )
            )
        else:
            feature_names_list = config["continuous"]["feature_names"]

        full_combo_df.columns = feature_names_list

        print(len(full_combo_list))
        print(len(training_combo_df))

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

        # writting the reaction conditions for
        # training dataset into files(encoded and decoded versions)
        with open(training_set_path, "w") as file_object:
            # file_object.write("####" + "\n")
            feature_names = ",".join([str(elem) for elem in feature_names_list])
            file_object.write(feature_names + ",Yield" + "\n")

        with open(training_set_decoded_path, "w") as file_object:
            # file_object.write("####" + "\n")
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
