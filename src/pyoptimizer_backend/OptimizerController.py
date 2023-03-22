def check_install(optimizer_name: str) -> bool:

    opt = get_optimizer(optimizer_name)

    return opt.check_install()


def install(optimizer_name: str, venv):

    opt = get_optimizer(optimizer_name, venv)

    return opt.install()


def train(optimizer_name, prev_param, yield_value, itr, experiment_dir):

    opt = get_optimizer(optimizer_name)

    opt.check_install()
    return opt.train(prev_param, yield_value, itr, experiment_dir)


def predict(optimizer_name, prev_param, yield_value, itr, experiment_dir):

    opt = get_optimizer(optimizer_name)

    return opt.predict(prev_param, yield_value, itr, experiment_dir)


def get_config(optimizer_name: str):

    opt = get_optimizer(optimizer_name)

    return opt.get_config()


def set_config(optimizer_name: str):

    opt = get_optimizer(optimizer_name)

    return opt.set_config()


def get_optimizer(optimizer_name):
    if optimizer_name == "AMLRO":
        # calling the AMLRO optmizer class inherite from optimizer ABC
        # Optimizer = AMLRO()
        print("AMLRO")
        # return Optimizer
