import argparse
import json
import os
from pathlib import Path

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerController import check_install, get_config


def main():
    args = parse_args()

    # Prepare virtual environment
    venv_path = os.path.join(args.location, "venv_{}".format(args.optimizer))
    venv = NestedVenv(venv_path)

    if not os.path.exists(venv_path) and not check_install(
        args.optimizer, venv
    ):
        print(
            (
                "No optimizer install found at the given location. Run ",
                "'install_optimizer --help' for more details on how to ",
                "install an optimizer.",
            )
        )
        return -1
    print("Activating virtual environment at:", venv_path)
    venv.activate()

    config_descriptions = get_config(args.optimizer, venv)

    config_contents = {}
    for config in config_descriptions:
        value = config["value"]

        if config["type"] == "str":
            if type(config["value"]) is list:
                value = value[0]

        config_contents[config["name"]] = value

    config_file = os.path.join(args.location, "config.json")

    if Path(config_file).exists() and not args.force:
        print(
            (
                "Config file already exists at {}.".format(config_file),
                "Use the '-f' flag to overwrite this file with a new, ",
                "default config file.",
            )
        )
        return -1

    with open(config_file, "w") as fout:
        print("Writing config to file:", config_file)
        json.dump(config_contents, fout, indent=4)


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
        "-f",
        "--force",
        dest="force",
        action="store_true",
        help=("Forces a fresh configuration file to be created."),
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
