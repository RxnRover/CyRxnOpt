import os

from OptimizerABC import OptimizerABC
from pyoptimizer_backend.VenvManager import VenvManager


class OptimizerAMLRO(OptimizerABC):
    # overidding methods
    def install(self):
        """installing the virtual env for AMLRO optimizer and install all
         the nessary packages for AMLRO.
        Also this function activate the AMLRO virtual env.
        """
        pathToScriptDir = os.path.dirname(os.path.realpath(__file__))
        Venv_m = VenvManager(os.path.join(pathToScriptDir, "venv_AMLRO"))
        Venv_m.start_venv()
        Venv_m.pip_install_e("../benchmarking")  # this path should be get from labview
        Venv_m.pip_install_e("../amlo")  # this path should be get from labview
        self.check_install()

    def _import_deps(self):
        """importing all the packages and libries needed for running amlro optimizer"""
        import numpy as np
        import pandas as pd
        from amlro import generate_combos, main, optimizer, training_set_generator

        self._imports = {
            "generate_combos": generate_combos,
            "training_set_generator": training_set_generator,
            "optimizer": optimizer,
            "main": main,
            "np": np,
            "pd": pd,
        }

    def check_install(self):
        """checking whether amlro virtual env install or not

        :return: boolean veriable for virtual env installation check.
        :rtype: boolean
        """

        pathToScriptDir = os.path.dirname(os.path.realpath(__file__))

        Venv_m = VenvManager(os.path.join(pathToScriptDir, "venv_AMLRO"))
        Venv_m.start_venv()

        try:

            self._import_deps()
            print("check")
        except ModuleNotFoundError:
            return False

        return True

    def train(self, prev_param, yield_value, itr, experiment_dir):
        """generate initial training dataset needed for AMLRO model training.

        :param prev_param: experimental parameter combination for previous experiment
        :type prev_param: list
        :param yield_value: experimental yield
        :type yield_value: float
        :param itr: experimental cycle number for training
        :type itr: int
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :return: next parameter combination for next experimental cycle.
        :rtype: list
        """

        training_set_path = os.path.join(experiment_dir, "training_set_file.txt")
        training_combo_path = os.path.join(experiment_dir, "training_combo_file.txt")
        next_parameters = self._imports[
            "training_set_generator"
        ].generate_training_data(
            training_set_path, training_combo_path, prev_param, yield_value, itr
        )
        return next_parameters

    def predict(self, prev_param, yield_value, experiment_dir):
        """prediction of next best combination of parameters and
         traning machine learning model from last experimental data for active learning.

        :param prev_param: experimental parameter combination for previous experiment
        :type prev_param: list
        :param yield_value: experimental yield
        :type yield_value: float
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :return: best predicted parameter combination
        :rtype: list
        """

        # TODO: rename main
        training_set_path = os.path.join(experiment_dir, "training_set_file.txt")
        full_combo_path = os.path.join(experiment_dir, "full_combo_file.txt")
        best_combo = self._imports["main"].get_optimized_parameters(
            training_set_path, full_combo_path, prev_param, yield_value
        )
        return best_combo

    def get_config(self):
        config = {
            {
                "Name": "Min",  # or Max
                "Type": str,
                "value": ["Minimize", "Maximize"],
            },
            {
                "Name": "Feature_Count",
                "Type": int,
                "value": 0,
            },
            {
                "Name": "Feature_Names",
                "Type": list[list],
                "value": [[]],
            },
            {
                "Name": "Feature_Bounds",
                "Type": list[list],
                "value": [[]],
            },
            {
                "Name": "Feature_resoultions",
                "Type": list,
                "value": [],
            },
            {
                "name": "hyperparameters",
                "type": list[dict],
                "value": [
                    {"name": "unroll_length", "type": int, "value": [0, 100]},
                ],
            },
        }
        return config

    def set_config(self, experiment_dir, config):
        """Generate all the nessasry data files

        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: configuration dict which required for initializing AMLRO
        :type config: dict
        """

        full_combo_list = self._imports["generate_combos"].generate_uniform_grid(config)

        full_combo_list = self._imports["np"].around(full_combo_list, decimals=4)
        full_combo_df = self._imports["pd"].DataFrame(full_combo_list)
        training_combo_df = full_combo_df.sample(20)
        full_combo_df.columns = config["feature_names"]

        print(len(full_combo_list))
        print(len(training_combo_df))

        full_combo_path = os.path.join(experiment_dir, "full_combo_file.txt")
        training_combo_path = os.path.join(experiment_dir, "training_combo_file.txt")
        full_combo_df.to_csv(full_combo_path, index=False)
        training_combo_df.to_csv(training_combo_path, index=False)

        training_set_path = os.path.join(experiment_dir, "training_set_file.txt")
        with open(training_set_path, "w") as file_object:
            # file_object.write("####" + "\n")
            feature_names = ",".join([str(elem) for elem in config["feature_names"]])
            file_object.write(feature_names + ",Yield" + "\n")
