import argparse
import json
import logging
import os
from pathlib import Path

from cyrxnopt.apps._utilities.gen_logfile import gen_logfile
from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerController import check_install, set_config


def main():
    args = parse_args()

    logfile = gen_logfile(__file__, args.location)
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)

    # Prepare virtual environment
    venv_path = os.path.join(args.location, "venv_{}".format(args.optimizer))
    venv = NestedVenv(venv_path)

    if not os.path.exists(venv_path) and not check_install(
        args.optimizer, venv
    ):
        print(
            (
                "No optimizer install found at the given location. Run ",
                "'install_optimizer --help' for more details on how to install ",
                "an optimizer.",
            )
        )
        return -1
    logging.debug("Activating virtual environment at: {}".format(venv_path))
    venv.activate()

    # TODO: Make args.config relative to args.location

    # Make sure the config file exists
    if not Path(args.config).exists():
        print("Config file not found at {}.".format(args.config))
        return -1

    with open(args.config, "r") as fin:
        logging.debug("Reading config to file: {}".format(args.config))
        config_contents = json.load(fin)

    print("Configuring optimizer: {}".format(args.optimizer))
    print("Potential output from the optimizer:")
    set_config(args.optimizer, config_contents, args.location, venv)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    # parser.add_argument("experiment_dir", help="Location for experiment data.")
    parser.add_argument("optimizer", help="Optimizer to use.")
    parser.add_argument(
        "-l",
        "--location",
        dest="location",
        default=".",
        type=str,
        help=("Location for experiment data."),
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.json",
        type=str,
        help=("Configuration file to use for the given optimizer."),
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
