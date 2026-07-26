from typing import TYPE_CHECKING, Any, Callable, Tuple

from cyrxnopt.OptimizerController import train

if TYPE_CHECKING:
    from cyrxnopt.NestedVenv import NestedVenv

problematic_optimizers = [
    "amlro",
    "edbop",
]


def train_server(
    optimizer_name: str,
    prev_param: list[Any],
    yield_value: float,
    training_steps: int,
    output_dir: str,
    config: dict[str, Any],
    venv: "NestedVenv",
    obj_func: Callable[[list[float]], float],
) -> Tuple[list[Any], float]:
    if optimizer_name.lower() in problematic_optimizers:
        prev_param, yield_value = train_faux_server(
            optimizer_name,
            prev_param,
            yield_value,
            training_steps,
            output_dir,
            config,
            venv,
            obj_func,
        )
    else:
        prev_param = train(
            optimizer_name,
            venv,
            prev_param,
            yield_value,
            output_dir,
            config,
            obj_func=obj_func,
        )

        # Algorithm doesn't support training
        if len(prev_param) == 0:
            yield_value = 0
        else:
            # TODO: Support grabbing the last yield value when an algorithm
            # behaves like this
            raise RuntimeError(
                (
                    "Algorithms with internal training loops are not yet "
                    f"supported by {__name__}. Please submit an issue "
                    "requesting this feature if you have an optimizer that"
                    " requires this."
                )
            )

    return prev_param, yield_value


def train_faux_server(
    optimizer_name: str,
    prev_param: list[Any],
    yield_value: float,
    training_steps: int,
    output_dir: str,
    config: dict[str, Any],
    venv: "NestedVenv",
    obj_func: Callable[[list[float]], float],
) -> Tuple[list[Any], float]:
    for i in range(training_steps):
        prev_param = train(
            optimizer_name,
            venv,
            prev_param,
            yield_value,
            output_dir,
            config,
        )

        yield_value = obj_func(prev_param)

    return prev_param, yield_value
