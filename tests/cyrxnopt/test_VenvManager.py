import os
import unittest

from cyrxnopt.VenvManager import VenvManager


class TestVenvManager(unittest.TestCase):
    def setUp(self) -> None:
        self.pathToScriptDir = os.path.dirname(os.path.realpath(__file__))
        self.NewVenvManager = VenvManager(
            os.path.join(self.pathToScriptDir, "venv_test")
        )
        return super().setUp()

    def tearDown(self) -> None:
        del self.pathToScriptDir
        del self.NewVenvManager
        return super().tearDown()

    def test_is_venv_no_venv(self):
        result = self.NewVenvManager.is_venv()
        self.assertFalse(result)

    def test_start_venv(self):
        # self.assertEqual(self.NewVenvManager.start_venv(),
        # "Running under virtual environment")
        pass

    def test_restart_under_venv(self):
        pass
