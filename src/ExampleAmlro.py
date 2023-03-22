import os
import subprocess
import sys

from benchmarking.functions import branin

from pyoptimizer_backend.OptimizerAmlro import OptimizerAmlro

# from pyoptimizer_backend.OptimizerController import *
from pyoptimizer_backend.VenvManager import VenvManager

experiment_dir = "Amlro_example"  # should choose from lab view/user
# experiment_dir = os.path.dirname(os.path.abspath(experiment_dir))
venv_m = VenvManager(os.path.join(experiment_dir, "venv_amlro"))
if not venv_m.is_venv():
    print("in if statement")
    venv_m.install_virtual_env()
    subprocess.call([venv_m.virtual_python, __file__] + sys.argv[1:])
    exit(0)

opt = OptimizerAmlro(venv_m)

# checking whether nessasary packages install or not

check = opt.check_install()
print(check)

opt.install()

config = {
    "continuous": {
        "feature_names": ["f1", "f2", "f3"],
        "bounds": [[0, 0.3], [0, 0.3], [-10, 20]],
        "resolutions": [0.1, 0.1, 5],
    },
    "categorical": {
        "feature_names": ["animal", "color"],
        "values": [["cat", "dog"], ["brown", "black", "yellow"]],
    },
}

opt.set_config(experiment_dir, config)

# training loop


parameters = []
yield_val = 0

for i in range(20):
    next_parameters = opt.train(
        parameters, yield_val, i, experiment_dir, config
    )
    parameters = next_parameters
    yield_val = -branin.branin(x=next_parameters)

# active learning and prediction
for i in range(2):
    best_combo = opt.predict(parameters, yield_val, experiment_dir, config)
    parameters = best_combo
    yield_val = -branin.branin(x=best_combo)
