import logging
import os
from collections.abc import Callable
from typing import Any, Optional

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerABC import OptimizerABC
from cyrxnopt.utilities.config.transforms import use_subkeys

logger = logging.getLogger(__name__)


class OptimizerAmlro(OptimizerABC):
    # Private static data member to list dependency packages required
    # by this class
    _packages = [
        "git+ssh://git@github.com/RxnRover/amlo",
        "numpy",
        "pandas",
        "joblib",
    ]

    def __init__(self, venv: NestedVenv) -> None:
        """initializing optimizer AMLRO object

        :param venv: Virtual envirement class object
        :type venv: NestedVenv
        """

        super().__init__(venv)

    def get_config(self) -> list[dict[str, Any]]:
        """This function will return the configurations which are needed
        to initialize an optimizer through `set_config()`.

        :return: Configuration option descriptions.
        :rtype: list[dict[str, Any]]
        """

        config: list[dict[str, Any]] = [
            {
                "name": "continuous_feature_names",
                "type": "list[str]",
                "value": [],
            },
            {
                "name": "continuous_feature_bounds",
                "type": "list[list[float]]",
                "value": [],
            },
            {
                "name": "continuous_feature_resolutions",
                "type": "list[float]",
                "value": [],
            },
            {
                "name": "categorical_feature_names",
                "type": "list[str]",
                "value": [],
            },
            {
                "name": "categorical_feature_values",
                "type": "list[list[str]]",
                "value": [],
            },
            {
                "name": "budget",
                "type": "int",
                "value": 100,
            },
            {
                "name": "objectives",
                "type": "list[str]",
                "value": ["yield"],
            },
            {
                "name": "direction",
                "type": "str",
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

    def set_config(self, experiment_dir: str, config: dict[str, Any]) -> None:
        """Generate all the necessary data files based on the given config.

        :param experiment_dir: Experimental directory for saving data files.
        :type experiment_dir: str
        :param config: Configuration settings defined from `get_config()`.
        :type config: dict[str, Any]
        """

        self._import_deps()

        self._validate_config(config)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        full_combo_list = self._imports[
            "generate_combos"
        ].generate_uniform_grid(use_subkeys(config))

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

    def train(
        self,
        prev_param: list[Any],
        yield_value: float,
        experiment_dir: str,
        config: dict[str, Any],
        obj_func: Optional[Callable[..., float]] = None,
    ) -> list[Any]:
        """generate initial training dataset needed for AMLRO model training.

        :param prev_param: experimental parameter combination for previous experiment,
            provide an empty list for the first call
        :type prev_param: list
        :param yield_value: experimental yield
        :type yield_value: float
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: Initial reaction feature configurations
        :type config: dict

        :return: Next parameter combination to perform, or an empty list (``[]``)
                 if all training points have been performed
        :rtype: list[Any]
        """

        print("----- DEBUG train() called -----")

        self._import_deps()

        # TODO: Set these as properties?
        training_set_path = os.path.join(
            experiment_dir, "training_set_file.txt"
        )
        training_set_decoded_path = os.path.join(
            experiment_dir, "training_set_decoded_file.txt"
        )
        training_combo_path = os.path.join(
            experiment_dir, "training_combo_file.txt"
        )

        if config["direction"].lower() == "min":
            yield_value = -yield_value

        # Determine next training row to perform
        training_combos = self._imports["pd"].read_csv(training_combo_path)
        training_set = self._imports["pd"].read_csv(training_set_path)
        next_index = self._get_next_training_index_by_length(
            training_combos, training_set
        )
        # NOTE: Not used due to bug in amlo that causes training_set_file.txt
        # to have the decoded feature headers, so training_set_file.txt and
        # training_combo_file.txt will never have the prerequisite matching
        # column headers.
        # next_index = self._get_next_training_index_next_combo(
        #     training_combos, training_set
        # )

        # Exit early if all training points have already been performed
        if next_index == -1:
            return []

        # Workaround to avoid being stuck at the first parameter
        if len(prev_param) > 0:
            next_index += 1

        # training step
        next_parameters = self._imports[
            "training_set_generator"
        ].generate_training_data(
            training_set_path,
            training_set_decoded_path,
            training_combo_path,
            use_subkeys(config),
            prev_param,
            yield_value,
            next_index,
        )

        return next_parameters

    def predict(
        self,
        prev_param: list[Any],
        yield_value: float,
        experiment_dir: str,
        config: dict[str, Any],
        obj_func: Optional[Callable[..., float]] = None,
    ) -> list[Any]:
        """prediction of next best combination of parameters and
         traning machine learning model from last experimental data for active learning.

        :param prev_param: experimental parameter combination for previous experiment
        :type prev_param: list[Any]
        :param yield_value: experimental yield
        :type yield_value: float
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: Initial reaction feature configurations
        :type config: dict[str, Any]

        :return: best predicted parameter combination
        :rtype: list[Any]
        """

        self._import_deps()

        training_set_path = os.path.join(
            experiment_dir, "training_set_file.txt"
        )
        training_set_decoded_path = os.path.join(
            experiment_dir, "training_set_decoded_file.txt"
        )
        full_combo_path = os.path.join(experiment_dir, "full_combo_file.txt")

        if config["direction"].lower() == "min":
            yield_value = -yield_value

        # prediction step
        best_combo = self._imports["optimizer_main"].get_optimized_parameters(
            training_set_path,
            training_set_decoded_path,
            full_combo_path,
            use_subkeys(config),
            prev_param,
            yield_value,
        )

        return best_combo

    def _import_deps(self) -> None:
        """importing all the packages and libries needed for running amlro optimizer"""

        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from amlro import (  # type: ignore
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

    def _get_next_training_index_by_length(  # type: ignore
        self, training_combos, training_dataset
    ) -> int:
        """Gets the index for the next training condition to be performed.

        This is implemented by a simple check on the dataset length. It is
        assumed that the only rows present in the dataset are those from the
        training combo list. This means that if the dataset has 6 rows, then
        the index of the next training combo to perform is also 6.

        :param training_combos: Training conditions suggested by AMLRO
        :type training_combos: pd.DataFrame
        :param training_dataset: Current dataset of performed reactions used to
                                 train AMLRO
        :type training_dataset: pd.DataFrame

        :returns: Index in the training combo list of the next conditions
                  missing from the training dataset. An index of -1 is returned
                  if the dataset is longer than the training combo list.
        :rtype: int
        """

        # Get the row counts of the training combos and dataset
        combo_rows = len(training_combos.index)
        dataset_rows = len(training_dataset.index)
        print(f"DEBUG combo_rows, dataset_rows : {combo_rows}, {dataset_rows}")

        # Simple check assuming the dataset only contains the training combos
        # that have been run
        if dataset_rows >= combo_rows:
            # No more training points to run
            return -1

        # The row count of the dataset will be the next index in the combo
        # list due to zero indexing
        return dataset_rows

    def _get_next_training_index_next_combo(  # type: ignore
        self, training_combos, training_dataset
    ) -> int:
        """Gets the index for the next training condition to be performed.

        This is implemented by checking which training conditions are missing
        in the training dataset, then giving the index in the training condition
        list for the first missing condition. Importantly, this means that a
        training dataset with the correct number of entries, but none matching
        the training combo file, will still receive indices and not be
        considered "completed" yet.

        :param training_combos: Training conditions suggested by AMLRO
        :type training_combos: pd.DataFrame
        :param training_dataset: Current dataset of performed reactions used to
                                 train AMLRO
        :type training_dataset: pd.DataFrame

        :returns: Index in the training combo list of the next conditions
                  missing from the training dataset. An index of -1 is returned
                  if no more training conditions are missing from the dataset.
        :rtype: int
        """

        # Merge the current training dataset with the training combos suggested
        # by AMLRO. A left merge is used to only use keys from the training
        # combos and preserve the row index in the training combos.
        merged = training_combos.merge(
            training_dataset, how="left", indicator=True
        )

        # Determine training combos are missing
        is_missing = merged["_merge"].eq("left_only")

        # The next line to get the first missing row will return 0 for both
        # the first row and if no rows are missing, so exit early with -1
        # if no more training combos are missing
        if is_missing.eq(False).all():
            return -1

        # Get the index of the first training combo not found in the provided
        # training dataset. Apparently between True and False, True is the max
        # so idxmax() works.
        first_missing_index = merged["_merge"].eq("left_only").idxmax()

        return first_missing_index
