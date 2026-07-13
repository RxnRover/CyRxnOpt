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
    # TODO: Once we find an algorithm that uses this, we'll have to fix
    #       whatever it is returning
    # else:
    #     results = train(
    #         optimizer_name,
    #         prev_param,
    #         yield_value,
    #         training_steps,
    #         output_dir,
    #         config,
    #         venv,
    #         obj_func,
    #     )

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
