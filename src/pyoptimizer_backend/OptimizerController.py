from typing import Any, Dict, List

from pyoptimizer_backend.OptimizerAmlro import OptimizerAmlro
from pyoptimizer_backend.OptimizerEDBOp import OptimizerEBDOp
from pyoptimizer_backend.OptimizerNMSimplex import OptimizerNMSimplex
from pyoptimizer_backend.VenvManager import VenvManager


def check_install(optimizer_name: str, venv: VenvManager = "") -> bool:
    """This method will call to the actual check install function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param venv: VenvManager object of the environment with the optimizer
                 installation
    :type venv: VenvManager
    :return: calling the check install function
    :rtype: bool
    """

    opt = get_optimizer(optimizer_name, venv)

    return opt.check_install()


def install(
    optimizer_name: str, venv: VenvManager = "", local_paths: dict = {}
):
    """This method will call to the actual install function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param venv: VenvManager object of the environment with the optimizer
                 installation
    :type venv: VenvManager
    :return: calling the install function
    :rtype: none
    """

    opt = get_optimizer(optimizer_name, venv)

    return opt.install(local_paths=local_paths)


def train(
    optimizer_name: str,
    prev_param: List[Any],
    yield_value: float,
    itr: int,
    experiment_dir: str,
    config: Dict,
    venv: VenvManager = "",
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
    :param venv: VenvManager object of the environment with the optimizer
                 installation
    :type venv: VenvManager
    :return: Calling the train function it will return next training parameters.
    :rtype: List[Any]
    """

    opt = get_optimizer(optimizer_name, venv)

    opt.check_install()
    return opt.train(prev_param, yield_value, itr, experiment_dir, config)


def predict(
    optimizer_name: str,
    prev_param: List[Any],
    yield_value: float,
    experiment_dir: str,
    config: Dict,
    venv: VenvManager = "",
    obj_func=None,
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
    :param venv: VenvManager object of the environment with the optimizer
                 installation
    :type venv: VenvManager
    :param obj_func: Objective function needed to optimize
    :type config: function
    :return: Calling the train function it will return predicted parameter combination
    :rtype: List[Any]
    """

    opt = get_optimizer(optimizer_name, venv)

    try:
        result = opt.predict(
            prev_param, yield_value, experiment_dir, config, obj_func=obj_func
        )
    except TypeError:
        result = opt.predict(prev_param, yield_value, experiment_dir, config)

    return result


def get_config(optimizer_name: str, venv: VenvManager = "") -> Dict:
    """This method will call to the actual get config function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param venv: VenvManager object of the environment with the optimizer
                 installation
    :type venv: VenvManager
    :return: calling the check install function
    :rtype: Dict
    """
    opt = get_optimizer(optimizer_name, venv)

    return opt.get_config()


def set_config(
    optimizer_name: str,
    config: Dict,
    experiment_dir: str,
    venv: VenvManager = "",
):
    """This method will call to the actual set config function
    from given optimizer class.

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param config: Initial reaction feature configurations
    :type config: Dict
    :param venv: VenvManager object of the environment with the optimizer
                 installation
    :type venv: VenvManager
    :return: Calling the set config function
    :rtype: none
    """
    opt = get_optimizer(optimizer_name, venv)

    return opt.set_config(experiment_dir, config)


def get_optimizer(optimizer_name, venv: VenvManager = ""):
    """This function generate the class object for requested optimkzer algorithm

    :param optimizer_name: Name of the given optimizer algorithm
    :type optimizer_name: str
    :param venv: VenvManager object of the environment with the optimizer
                 installation
    :type venv: VenvManager
    :return: optimizer object veriable
    :rtype: class object
    """

    if optimizer_name == "amlro":
        optimizer = OptimizerAmlro(venv)
    elif optimizer_name == "edbop":
        optimizer = OptimizerEBDOp(venv)
    elif optimizer_name == "NMSimplex":
        optimizer = OptimizerNMSimplex(venv)

    else:
        raise RuntimeError(
            "Invalid optimizer name given: {}".format(optimizer_name)
        )

    return optimizer
