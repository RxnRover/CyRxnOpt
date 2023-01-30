from abc import ABC, abstractmethod


class OptimizerABC(ABC):
    """This is the abstract class for general optimizer algorithms
    which include all the abstract functions need.
    """

    @abstractmethod
    def train(self):
        """This abstract method should be overide with actual training function
        calls."""
        pass

    @abstractmethod
    def check_install(self):
        """This abstract method should be included package installation checking
        codes."""
        pass

    @abstractmethod
    def predict(self):
        """This abstract method should be overide with actual predict function calls."""
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
    def set_config(self):
        """This abstract method should be included function calls required for
        genereting initial  configurations and files for optimizer.
        """
        pass
