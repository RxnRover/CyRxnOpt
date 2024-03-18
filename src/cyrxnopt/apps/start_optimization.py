import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path

import zmq

from cyrxnopt.apps._utilities.gen_logfile import gen_logfile
from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerController import check_install
from cyrxnopt.util.predict_server import predict_server
from cyrxnopt.util.zmq import zmq_helpers
from cyrxnopt.util.zmq.zmq_obj_function import zmq_obj_function


def main():
    args = parse_args()

    logfile = gen_logfile(__file__, args.location)
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)

    logging.debug("Argparse arguments: {}".format(args))

    # Prepare virtual environment
    venv_path = os.path.join(args.location, "venv_{}".format(args.optimizer))
    venv = NestedVenv(venv_path)

    if not os.path.exists(venv_path) and not check_install(
        args.optimizer, venv
    ):
        print(
            (
                "No optimizer install found at the given location. Run "
                "'install_optimizer --help' for more details on how to "
                "install an optimizer."
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

    # address = config["ip_address"] + ":" + config["port"]
    address = "tcp://localhost:5555"
    socket = zmq_helpers.init_socket(address)

    # TODO: Temporary fix until we support multi-objective
    if type(config_contents["direction"]) is list:
        config_contents["direction"] = config_contents["direction"][0]

    obj_func = zmq_obj_function(socket, config_contents["direction"])

    print("Beginning optimization...")
    # The optimization thread is never used here after being spun up
    _ = start_optimization_thread(
        args.optimizer, [], 0, args.location, config_contents, venv, obj_func
    )
    user_input_thread = start_user_input_thread(config_contents["budget"])
    while user_input_thread.is_alive():
        time.sleep(1)


def start_optimization_thread(*args, **kwargs):
    # This thread is a daemon because the program should exit when no alive,
    # non-daemonic threads are left. Once the user input thread is done,
    # this thread should also exit and the program should terminate
    thread = threading.Thread(
        target=predict_server,
        name="Optimization Thread",
        args=args,
        kwargs=kwargs,
        daemon=True,
    )
    thread.start()

    return thread


def start_user_input_thread(*args, **kwargs):
    thread = threading.Thread(
        target=input_server,
        name="User Input Thread",
        args=args,
        kwargs=kwargs,
        daemon=False,
    )
    thread.start()

    return thread


def input_server(training_steps: int):
    SERVER_ENDPOINT = "tcp://*:5555"

    # Create the context and socket
    context = zmq.Context(1)
    socket = context.socket(zmq.REP)

    logging.debug("Binding to {}".format(SERVER_ENDPOINT))
    socket.bind(SERVER_ENDPOINT)

    # Register the socket with a poller
    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)

    steps = 1
    while steps <= training_steps:
        #  Wait for ready from the optimizer
        logging.debug("Waiting...")
        request = socket.recv()
        logging.debug("Received request: %s" % request)

        reply = b"invalid_request"

        if type(json.loads(request)) == list:
            params = json.loads(request)
            print("Reaction to perform:", params)
            user_input = input(
                "Step {}: Enter reaction result ('q' to quit): ".format(steps)
            )

            if is_quit_request(user_input):
                logging.debug(
                    "Received quit input from user. Exitting..."
                )  # DEBUG
                reply = b"quit"

                socket.close()
                context.term()
                return

            else:
                reply = float(user_input)

            steps += 1
            reply = json.dumps(reply).encode("utf-8")

        logging.debug("Sending reply: {}".format(reply))
        socket.send(reply)

    print("Training complete!")


def is_quit_request(request: str) -> bool:
    """Checks if the request is one of the "quit" keywords.

    :param request: Request received through zmq socket.
    :type request: str
    :return: Whether the request is a quit request (True) or not (False).
    :rtype: bool
    """

    _request = request.lower()

    is_quit = False

    is_quit = _request == "quit" or _request == "exit" or _request == "q"

    return is_quit


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
