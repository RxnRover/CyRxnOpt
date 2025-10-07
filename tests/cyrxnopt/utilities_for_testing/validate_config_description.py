import unittest
from typing import Any, Dict, List


def validate_config_description(
    test_case: unittest.TestCase, config_desc: List[Dict[str, Any]]
) -> None:
    """Validates the general formatting of a config description returned from
    an ``Optimizer.get_config()`` call.

    :param test_case: Unittest test case that has the assertion checks.
    :type test_case: unittest.TestCase
    :param config_desc: Collection of config descriptors
    :type config_desc: List[Dict[str, Any]]
    """

    # A config description should be a list of descriptions for each config
    # option of the optimizer
    test_case.assertEqual(type(config_desc), list)

    for description in config_desc:
        # Each config description is a dictionary with relevant information
        test_case.assertEqual(type(description), dict)

        # A description is complete by providing fields for the config option
        # "name", "type", and "value".
        test_case.assertIn("name", description)
        test_case.assertIn("type", description)
        test_case.assertIn("value", description)


def validate_config_description_pytest(
    config_desc: List[Dict[str, Any]],
) -> None:
    """Validates the general formatting of a config description returned from
    an ``Optimizer.get_config()`` call.

    :param config_desc: Collection of config descriptors
    :type config_desc: List[Dict[str, Any]]
    """

    # A config description should be a list of descriptions for each config
    # option of the optimizer
    assert type(config_desc) is list

    for description in config_desc:
        # Each config description is a dictionary with relevant information
        assert type(description) is dict

        # A description is complete by providing fields for the config option
        # "name", "type", and "value".
        assert "name" in description
        assert "type" in description
        assert "value" in description
