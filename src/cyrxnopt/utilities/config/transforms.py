import copy
from typing import Any


def use_subkeys(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Converts a config dictionary to use categorical and continuous subkeys.

    :param config: Config dictionary to convert. This dictionary is not modified.
    :type config: dict[str, Any]
    :return: Converted config dictionary
    :rtype: dict[str, Any]
    """

    new_config = copy.deepcopy(config)

    new_config["continuous"] = {}
    new_config["categorical"] = {}

    # Populate the sub-keys, removing the converted keys from the new_config
    # By using pop(), this will not fail if a key does not exist
    new_config["continuous"]["feature_names"] = new_config.pop(
        "continuous_feature_names", []
    )
    new_config["continuous"]["bounds"] = new_config.pop(
        "continuous_feature_bounds", []
    )
    new_config["continuous"]["resolutions"] = new_config.pop(
        "continuous_feature_resolutions", []
    )
    new_config["categorical"]["feature_names"] = new_config.pop(
        "categorical_feature_names", []
    )
    new_config["categorical"]["values"] = new_config.pop(
        "categorical_feature_values", []
    )

    return new_config
