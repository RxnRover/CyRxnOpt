#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

from pyoptimizer_backend.OptimizerController import install
from pyoptimizer_backend.VenvManager import VenvManager


def main():
    args = parse_args()

    venv_path = os.path.join(args.output_dir, "venv_{}".format(args.optimizer))

    # Set up virtual environment
    venv_m = VenvManager(venv_path)
    if not venv_m.is_venv():
        venv_m.install_virtual_env()
        # venv_m.pip_install_r("./requirements.txt")
        # venv_m.pip_install_e(".")
        subprocess.call([venv_m.virtual_python, __file__] + sys.argv[1:])
        exit(0)

    return install(args.optimizer, venv_m)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument("optimizer", help="Optimizer to use.")
    parser.add_argument("output_dir", help="Location for output data.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
