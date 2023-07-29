import csv
import os
import shutil
from typing import Any, Dict, List

from pyoptimizer_backend.OptimizerABC import OptimizerABC


class OptimizerEBDOp(OptimizerABC):
    # overidding methods
    def __init__(self, venv=None) -> None:
        """initializing optimizer EDBO+ object

        :param venv: Virtual envirement class object, defaults to None
        :type venv: VenvManager, optional
        """
        self._imports = {}
        self._venv = venv

    def install(self) -> None:
        """installing the virtual env for EDBO+ optimizer and install all
         the nessary packages for EDBO+.
        Also this function activate the EDBO+ virtual env.
        """

        self._venv.pip_install(
            "git+https://github.com/RxnRover/benchmarking.git"
        )  # this path should be get from labview
        self._venv.pip_install("pandas")
        self._venv.pip_install_e(
            "../../../edboplus-imp"
        )  # this path should be get from labview
        self._venv.pip_install_e("../../../pyoptimizer_backend")
        self._venv.pip_install_e("../../../datatools")
        # self._import_deps()
        self.check_install()

    def _import_deps(self) -> None:
        """importing all the packages and libries needed for running amlro optimizer"""

        import random

        import numpy as np
        import pandas as pd
        from edbo.plus.optimizer_botorch import EDBOplus

        self._imports = {
            "EDBOplus": EDBOplus,
            "np": np,
            "pd": pd,
            "random": random,
        }

    def check_install(self) -> bool:
        """checking whether edbo+ virtual env install or not

        :return: boolean veriable for virtual env installation check.
        :rtype: boolean
        """

        try:
            self._import_deps()
            print("check")
        except ModuleNotFoundError:
            return False

        return True

    def set_config(self, experiment_dir: str, config: Dict) -> None:
        """Generate all the nessasry data files

        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: configuration dict which required for initializing edbo+
        :type config: dict
        """
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        filename = "my_optimization.csv"
        config = self.config_translate(
            config
        )  # get reaction scope configurations
        # from general config file

        # generate reaction scope for EDBOp
        self._imports["EDBOplus"]().generate_reaction_scope(
            components=config["reaction_components"],
            directory=experiment_dir,
            filename=filename,
            check_overwrite=False,
        )

        # initialize the edbop optimization file will be use for prediction
        self._imports["EDBOplus"]().run(
            directory=experiment_dir,
            filename=filename,  # Previously generated scope.
            objectives=config["objectives"],  # ['yield', 'ee', 'side_product'],
            # Objectives to be optimized.
            objective_mode=config["objective_mode"],  # ['max', 'max', 'min'],
            # Maximize yield and ee but minimize side_product.
            batch=1,  # Number of experiments in parallel that
            # we want to perform in this round.
            columns_features="all",  # features to be included in the model.
            # init_sampling_method="cvtsampling",  # initialization method.
            init_sampling_method="seed",  # initialization method.
            seed=self._imports["random"].randint(0, 1000000),
        )

    def get_config(self):
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

    def train(
        self,
        prev_param: List[Any],
        yield_value: List[float],
        itr: int,
        experiment_dir: str,
        config: Dict,
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
        result_filename = "training_set_file.txt"

        config = self.config_translate(
            config
        )  # get reaction scope configurations
        # from general config file

        #  reading optimization file with reaction conditions
        df_edbo = self._imports["pd"].read_csv(
            os.path.join(experiment_dir, filename)
        )

        if len(prev_param) != 0:
            # [df_edbo.loc[0,config['objectives'][i]] =
            # #yield_value[i] for i in range(len(yield_value))]
            df_edbo.loc[0, config["objectives"][0]] = yield_value
            df_edbo.to_csv(os.path.join(experiment_dir, filename), index=False)
            parameters = (
                ",".join([str(elem) for elem in prev_param])
                + ","
                + str(yield_value)
            )
            # print(parameters)
            self.write_results(
                os.path.join(experiment_dir, result_filename), parameters
            )
            # self.write_results_edbop(experiment_dir, filename, config,yield_value)

        # running the edbop prediction
        self._imports["EDBOplus"]().run(
            directory=experiment_dir,
            filename=filename,  # Previously generated scope.
            objectives=config[
                "objectives"
            ],  # ['yield', 'ee', 'side_product'],  # Objectives to be optimized.
            objective_mode=config["objective_mode"],  # ['max', 'max', 'min'],
            # Maximize yield and ee but minimize side_product.
            batch=1,  # Number of experiments in parallel that
            # we want to perform in this round.
            columns_features="all",  # features to be included in the model.
            init_sampling_method="cvtsampling",  # initialization method.
        )

        # after one cycle of prediction again read the reaction condition file to
        #  getting the next reaction condition.
        df_edbo = self._imports["pd"].read_csv(
            os.path.join(experiment_dir, filename)
        )

        next_combo = df_edbo.iloc[:1].values.tolist()
        print(next_combo)

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

        for i in range(len(config["continuous"]["feature_names"])):
            low_bound = config["continuous"]["bounds"][i][0]
            upper_bound = config["continuous"]["bounds"][i][1]
            increment = config["continuous"]["resolutions"][i]

            values = self._imports["np"].arange(
                low_bound, upper_bound + increment, increment
            )

            reaction_components[
                config["continuous"]["feature_names"][i]
            ] = values

        if bool(config["categorical"]):
            for i in range(len(config["categorical"]["feature_names"])):
                reaction_components[
                    config["categorical"]["feature_names"][i]
                ] = config["categorical"]["values"][i]

        edbo_config = {
            "reaction_components": reaction_components,
            "objectives": config["objectives"],
            "objective_mode": config["objective_mode"],
        }

        return edbo_config

    def write_results_edbop(
        self, experiment_dir, filename, config, yield_value
    ):
        # Define the input and temporary output file paths
        input_file = os.path.join(experiment_dir, filename)
        temp_file = "temp_file.csv"

        # Define the new value to be updated
        new_value = yield_value

        # Open the input file and temporary output file
        with open(input_file, "r") as file, open(
            temp_file, "w", newline=""
        ) as temp:
            reader = csv.reader(file)
            writer = csv.writer(temp)

            # Iterate over each row in the input file
            for i, row in enumerate(reader):
                if i == 0:
                    value = config["objectives"][
                        0
                    ]  # column name of objective function
                    try:
                        index = row.index(value)
                        print(f"The index of '{value}' is: {index}")
                    except ValueError:
                        print(
                            f"The value '{value}' isnt found in the objective function"
                        )
                # Check if it's the second row (index 1)
                if i == 1:
                    # Update the value in the second column (index 1)
                    row[index] = new_value
                # Write the row to the temporary file
                writer.writerow(row)

        # Replace the original file with the updated file
        shutil.move(temp_file, input_file)

    def write_results(self, file_path: str, parameters: str) -> None:
        with open(file_path, "a+") as file_object:
            file_object.write(parameters + "\n")
