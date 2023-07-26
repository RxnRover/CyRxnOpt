import os
import shutil
import unittest

from pyoptimizer_backend.VenvManager import NestedVenv


class TestNestedVenv(unittest.TestCase):
    def setUp(self) -> None:
        self.parent_venv_path = "tmp"
        self.venv_path = os.path.join(self.parent_venv_path, "test_venv")

        # Append the test ID so each venv is separate
        self.venv_path += self.id()

        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(self.venv_path):
            shutil.rmtree(self.venv_path)

        if os.path.exists(self.venv_path + "_2"):
            shutil.rmtree(self.venv_path + "_2")

        return super().tearDown()

    def test_activate_with_no_venv_created(self):
        venv = NestedVenv(self.venv_path)

        self.assertRaises(RuntimeError, venv.activate)

    def test_activate_with_venv_created(self):
        venv = NestedVenv(self.venv_path)
        venv.create()

        # Shouldn't throw an exception
        venv.activate()

    def test_activate_two_venvs(self):
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(self.venv_path + "_2")

        # Created first, will not be primary
        venv1.create()
        venv1.activate()

        # Created second, will be primary
        venv2.create()
        venv2.activate()

        self.assertFalse(venv1.is_primary())
        self.assertTrue(venv2.is_primary())

        # Both venvs should be active
        self.assertTrue(venv1.is_active())
        self.assertTrue(venv2.is_active())

    def test_activate_two_venvs_then_deactivate_first(self):
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(self.venv_path + "_2")

        # Created first, will not be primary
        venv1.create()
        venv1.activate()

        # Created second, will be primary
        venv2.create()
        venv2.activate()

        venv1.deactivate()

        self.assertFalse(venv1.is_primary())
        self.assertTrue(venv2.is_primary())

        # Only venv2 should be active
        self.assertFalse(venv1.is_active())
        self.assertTrue(venv2.is_active())

    def test_create_with_no_prior_creation(self):
        venv = NestedVenv(self.venv_path)

        venv.create()

        self.assertTrue(os.path.exists(self.venv_path))
        self.assertTrue(os.path.exists(venv.virtual_python))

    def test_create_when_dir_exists(self):
        venv = NestedVenv(self.venv_path)

        # Create the directory beforehand
        os.makedirs(self.venv_path)

        # Creating venv but path already exists
        venv.create()

        self.assertTrue(os.path.exists(self.venv_path))
        self.assertTrue(os.path.exists(venv.virtual_python))

    def test_create_twice(self):
        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.create()

        self.assertTrue(os.path.exists(self.venv_path))
        self.assertTrue(os.path.exists(venv.virtual_python))

    def test_deactivate_not_active(self):
        venv = NestedVenv(self.venv_path)

        self.assertFalse(venv.is_active())

        # Nothing should happen here
        venv.deactivate()

        self.assertFalse(venv.is_active())

    def test_is_active_not_active(self):
        venv = NestedVenv(self.venv_path)

        # No venv has been created or activated

        self.assertFalse(venv.is_active())

    def test_is_active(self):
        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.activate()

        self.assertTrue(venv.is_active())

    def test_is_primary_not_active(self):
        venv = NestedVenv(self.venv_path)

        # No venv has been created or activated

        self.assertFalse(venv.is_primary())

    def test_is_primary(self):
        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.activate()

        self.assertTrue(venv.is_primary())

    def test_pip_install_numpy(self):
        """This test attempts to install the 'numpy' package from online
        using 'pip'.
        """

        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.activate()

        venv.pip_install("numpy")

    def test_pip_install_e(self):
        """This test attempts to self-install this package into a new
        virtual environment using 'pip install -e .'.
        """

        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.activate()

        venv.pip_install_e(".")

    def test_pip_install_r(self):
        """This test attempts to self-install this package's requirements.txt
        file into a new virtual environment using
        'pip install -r requirements'.
        """

        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.activate()

        venv.pip_install_r("requirements.txt")
