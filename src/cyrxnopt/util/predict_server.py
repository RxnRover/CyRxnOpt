from cyrxnopt.OptimizerController import predict

# TODO: Rename this variable
problematic_optimizers = [
    "amlro",
    "edbop",
]


def predict_server(
    optimizer_name, prev_param, yield_value, output_dir, config, venv, obj_func
):
    if optimizer_name.lower() in problematic_optimizers:
        results = predict_faux_server(
            optimizer_name,
            prev_param,
            yield_value,
            output_dir,
            config,
            venv,
            obj_func,
        )
    else:
        results = predict(
            optimizer_name,
            prev_param,
            yield_value,
            output_dir,
            config,
            venv,
            obj_func,
        )

    return results


def predict_faux_server(
    optimizer_name, prev_param, yield_value, output_dir, config, venv, obj_func
):
    results = {
        "total_iter": config["budget"],
        "best_coords": None,
        "best_value": None,
        "best_iter": None,
        "raw_results": [],
    }

    # TODO: This is a temporary fix until multi-objective is supported
    if type(config["direction"]) is list:
        direction = config["direction"][0]
    else:
        direction = config["direction"]

    if direction == "min":
        results["best_value"] = float("inf")
    else:
        results["best_value"] = float("-inf")

    # Loop over cycle iterations
    for i in range(config["budget"]):
        prev_param = predict(
            optimizer_name,
            prev_param,
            yield_value,
            output_dir,
            config,
            venv,
        )

        yield_value = obj_func(prev_param)

        if (direction == "min" and yield_value < results["best_value"]) or (
            direction == "max" and yield_value > results["best_value"]
        ):
            results["best_value"] = yield_value
            results["best_coords"] = prev_param
            results["best_iter"] = i

        # Add another line to the raw results
        result_line = [x for x in prev_param]
        result_line.append(yield_value)
        results["raw_results"].append(result_line)

    return results
