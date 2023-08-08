import os
import shutil
import unittest

from pyoptimizer_backend.NestedVenv import NestedVenv


class TestNestedVenv(unittest.TestCase):
    def setUp(self) -> None:
        self.parent_venv_path = "tmp"
        self.venv_path = os.path.join(self.parent_venv_path, "test_venv_")

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

    def test_activate_when_active(self):
        venv = NestedVenv(self.venv_path)
        venv.create()

        # Shouldn't throw an exception
        venv.activate()

        # Also shouldn't throw an exception
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
        self.assertTrue(os.path.exists(venv.python))

    def test_create_when_dir_exists(self):
        venv = NestedVenv(self.venv_path)

        # Create the directory beforehand
        os.makedirs(self.venv_path)

        # Creating venv but path already exists
        venv.create()

        self.assertTrue(os.path.exists(self.venv_path))
        self.assertTrue(os.path.exists(venv.python))

    def test_create_twice(self):
        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.create()

        self.assertTrue(os.path.exists(self.venv_path))
        self.assertTrue(os.path.exists(venv.python))

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

        self.assertTrue(venv.check_package("numpy"))

    def test_pip_install_e(self):
        """This test attempts to self-install this package into a new
        virtual environment using an editable install.
        """

        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.activate()

        venv.pip_install_e(os.path.abspath("tests/test_assets/test_project"))

        # self.assertTrue(venv.check_package("test_project"))
        self.assertTrue(venv.check_package("numpy", "1.24.0"))

    def test_pip_install_r(self):
        """This test attempts to self-install this package's requirements.txt
        file into a new virtual environment using
        'pip install -r requirements.txt'.
        """

        venv = NestedVenv(self.venv_path)

        venv.create()
        venv.activate()

        venv.pip_install_r(
            os.path.abspath("tests/test_assets/requirements.txt")
        )

        # self.assertTrue(venv.check_package("test_project"))
        self.assertTrue(venv.check_package("numpy", "1.24.0"))
        self.assertTrue(venv.check_package("requests", "2.31.0"))

    def test_pip_install_numpy_first_of_two_venvs(self):
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(self.venv_path + "_2")

        # Created first, will not be primary
        venv1.create()
        venv1.activate()

        # Created second, will be primary
        venv2.create()
        venv2.activate()

        # Install numpy only into the first, non-primary venv
        venv1.pip_install("numpy")

        # The first venv should have numpy
        self.assertTrue(venv1.check_package("numpy"))

        # The second, primary venv should not
        self.assertFalse(venv2.check_package("numpy"))

    def test_pip_install_numpy_two_versions(self):
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(self.venv_path + "_2")

        # Created first, will not be primary
        venv1.create()
        venv1.activate()

        # Created second, will be primary
        venv2.create()
        venv2.activate()

        venv1.pip_install("numpy==1.25")
        venv2.pip_install("numpy==1.24")

        self.assertTrue(venv1.check_package("numpy", "1.25.0"))
        self.assertTrue(venv2.check_package("numpy", "1.24.0"))

        import numpy

        self.assertEqual(numpy.__version__, "1.24.0")

        venv1.deactivate()

        import numpy

        print(numpy.__version__, "1.25.0")
