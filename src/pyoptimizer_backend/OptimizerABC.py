from abc import ABC, abstractmethod
from typing import Any, Dict, List


class OptimizerABC(ABC):
    """This is the abstract class for general optimizer algorithms
    which include all the abstract functions need.
    """

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
        :param prev_param: experimental parameter combination for previous experiment
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
    def check_install(self):
        """This abstract method should be included package installation checking
        codes."""
        pass

    @abstractmethod
    def predict(
        self, prev_param: List[Any], yield_value: float, experiment_dir: str
    ):
        """This abstract method should be overide with actual predict function calls.
        :param prev_param: experimental parameter combination for previous experiment
        :type prev_param: list[any]
        :param yield_value: experimental yield
        :type yield_value: float
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        """
        pass

    @abstractmethod
    def install(self):
        """This abstract method should be included function calls for installing
        required packages and activating virtual environment.
        """
        pass

    @abstractmethod
    def get_config(self):
        """This abstract method should be included function calls for returning
        all the initial configuration requires for optimizer.
        """
        pass

    @abstractmethod
    def set_config(self, experiment_dir: str, config: Dict):
        """This abstract method should be included function calls required for
        genereting initial  configurations and files for optimizer.
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: configuration dict which required for initializing AMLRO
        :type config: dict
        """
        pass
