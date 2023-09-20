import os
import unittest

from utilities_for_testing.validate_config_description import (
    validate_config_description,
)

from pyoptimizer_backend.NestedVenv import NestedVenv
from pyoptimizer_backend.OptimizerAmlro import OptimizerAmlro


class TestOptimizerAmlro(unittest.TestCase):
    def setUp(self) -> None:
        self.parent_venv_path = "tmp"
        self.venv_path = os.path.join(self.parent_venv_path, "test_venv_")

        # Append the test ID so each venv is separate
        self.venv_path += self.id()

        # Used to collect venvs created in a test
        self.venvs = []

        return super().setUp()

    def tearDown(self) -> None:
        for venv in self.venvs:
            venv.delete()

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

        venv.delete()
