import argparse


def parser_optimizer() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("optimizer", help="Optimizer to use.")

    return parser


def parser_location() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l",
        "--location",
        dest="location",
        default=".",
        type=str,
        help=(
            "Location for experiment data. This location must exist! "
            "Defaults to the current working directory."
        ),
    )

    return parser


def parser_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        type=str,
        help=(
            "Configuration file to use for the given optimizer. "
            "Defaults to <location>/config.json"
        ),
    )

    return parser
