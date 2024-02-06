import argparse
import os

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerController import check_install, install


def main():
    args = parse_args()

    # Prepare virtual environment
    venv_path = os.path.join(args.location, "venv_{}".format(args.optimizer))
    venv = NestedVenv(venv_path)

    if not os.path.exists(venv_path) or args.force:
        print("Creating virtual environment at:", venv_path)
        venv.create()
    print("Activating virtual environment at:", venv_path)
    venv.activate()

    # Install the optimizer if it is not already installed
    if not check_install(args.optimizer, venv):
        install(
            args.optimizer,
            venv,
            local_paths={"amlro": "../amlo", "edboplus": "deps/edbop"},
        )
    else:
        print(
            (
                "Optimizer already installed. Use the '-f' flag to force ",
                "a fresh reinstall if needed.",
            )
        )


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
        help=(
            "Forces a fresh installation by recreating the virtual environment."
        ),
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
