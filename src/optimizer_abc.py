from abc import ABC, abstractmethod


class OptimizerABC(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def check_install(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def install(self):
        pass

    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def set_config(self):
        pass
