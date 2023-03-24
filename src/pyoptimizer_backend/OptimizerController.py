from typing import Any, Dict, List

from pyoptimizer_backend.OptimizerAmlro import OptimizerAmlro
from pyoptimizer_backend.OptimizerNMSimplex import OptimizerNMSimplex


def check_install(optimizer_name: str) -> bool:
    """This method will call to the actual check install function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :return: calling the check install function
    :rtype: bool
    """

    opt = get_optimizer(optimizer_name)

    return opt.check_install()


def install(optimizer_name: str, venv):
    """This method will call to the actual install function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param venv: Virtual envirement class object
    :type venv: VenvManager
    :return: calling the install function
    :rtype: none
    """

    opt = get_optimizer(optimizer_name, venv)

    return opt.install()


def train(
    optimizer_name: str,
    prev_param: List[Any],
    yield_value: float,
    itr: int,
    experiment_dir: str,
    config: Dict,
) -> List[Any]:
    """This method will call to the actual train function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
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
    :return: Calling the train function it will return next training parameters.
    :rtype: List[Any]
    """

    opt = get_optimizer(optimizer_name)

    opt.check_install()
    return opt.train(prev_param, yield_value, itr, experiment_dir, config)


def predict(
    optimizer_name: str,
    prev_param: List[Any],
    yield_value: float,
    experiment_dir: str,
    config: Dict,
) -> List[Any]:
    """This method will call to the actual predict function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param prev_param: experimental parameter combination for previous experiment
    :type prev_param: list
    :param yield_value: experimental yield
    :type yield_value: float
    :param experiment_dir: experimental directory for saving data files
    :type experiment_dir: str
    :param config: Initial reaction feature configurations
    :type config: Dict
    :return: Calling the train function it will return predicted parameter combination
    :rtype: List[Any]
    """

    opt = get_optimizer(optimizer_name)

    return opt.predict(prev_param, yield_value, experiment_dir)


def get_config(optimizer_name: str) -> Dict:
    """This method will call to the actual get config function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :return: calling the check install function
    :rtype: Dict
    """
    opt = get_optimizer(optimizer_name)

    return opt.get_config()


def set_config(optimizer_name: str, config: Dict):
    """This method will call to the actual set config function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param config: Initial reaction feature configurations
    :type config: Dict
    :return: Calling the set config function
    :rtype: none
    """
    opt = get_optimizer(optimizer_name)

    return opt.set_config()


def get_optimizer(optimizer_name):
    """This function generate the class object for requested optimkzer algorithm

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :return: optimizer object veriable
    :rtype: class object
    """

    if optimizer_name == "AMLRO":
        # calling the AMLRO optmizer class inherite from optimizer ABC
        Optimizer = OptimizerAmlro()
        print("AMLRO")
        # return Optimizer
    if optimizer_name == "NMSimplex":
        Optimizer = OptimizerNMSimplex()

    return Optimizer
