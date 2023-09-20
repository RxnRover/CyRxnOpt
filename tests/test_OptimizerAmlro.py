import os
import unittest

from utilities_for_testing.validate_config_description import (
    validate_config_description,
)

from pyoptimizer_backend.NestedVenv import NestedVenv
from pyoptimizer_backend.OptimizerAmlro import OptimizerAmlro


class TestOptimizerAmlro(unittest.TestCase):
    def setUp(self) -> None:
        self.test_prefix = os.path.join("tmp", self.id())
        self.venv_path = os.path.join(self.test_prefix, "venv_amlro")

        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_config(self):
        """This test checks that a config of the correct format was provided,
        but does not try to validate the actual values of each config
        description.
        """

        venv = NestedVenv(self.venv_path)
        venv.create()
        venv.activate()

        opt = OptimizerAmlro(venv)

        result = opt.get_config()

        validate_config_description(self, result)

    def test_set_config(self):
        venv = NestedVenv(self.venv_path)
        venv.create()
        venv.activate()

        opt = OptimizerAmlro(venv)
        opt.install()

        config = {
            "continuous_feature_names": ["f1", "f2"],
            "continuous_feature_bounds": [[-1, 1], [-5, 5]],
            "continuous_feature_resolutions": [1, 5, 1],
            "categorical_feature_names": ["f3"],
            "categorical_feature_values": [["a", "b", "c"]],
            "budget": 10,
            "objectives": ["yield"],
            "objective_mode": "min",
        }

        opt.set_config(self.test_prefix, config)
