import argparse
import logging
import os

from cyrxnopt.apps._utilities import arg_validation as validate
from cyrxnopt.apps._utilities import common_args as parsers
from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerController import check_install, install

logger = logging.getLogger(__name__)


def main() -> int:
    args = parse_args()

    optimizer = validate.optimizer(args.optimizer)
    location = validate.location(args.location)

    logging.basicConfig(level=args.log_level)

    # Prepare virtual environment
    venv_path = os.path.join(location, f"venv_{optimizer}")
    venv = NestedVenv(venv_path)

    if not os.path.exists(venv_path) or args.force:
        print(f"Creating virtual environment at: {venv_path}")
        venv.create()
    logger.info(f"Activating virtual environment at: {venv_path}")
    venv.activate()

    # Install the optimizer if it is not already installed
    if not check_install(optimizer, venv):
        install(
            optimizer,
            venv,
        )
        print(f'Optimizer "{optimizer}" installed in venv at {venv_path}')
    else:
        print(
            (
                "Optimizer already installed. Use the '-f' flag to force "
                "a fresh reinstall if needed."
            )
        )

        return 2

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        parents=[parsers.optimizer(), parsers.location(), parsers.logging()]
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
    exit(main())
