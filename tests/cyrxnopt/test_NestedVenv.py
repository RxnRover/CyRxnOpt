import unittest
from pathlib import Path
from typing import List

from cyrxnopt.NestedVenv import NestedVenv


class TestNestedVenv(unittest.TestCase):
    def setUp(self) -> None:
        self.parent_venv_path = Path("tmp")
        self.venv_path = self.parent_venv_path / "test_venv_"

        # Append the test ID so each venv is separate
        self.venv_path = Path(str(self.venv_path) + self.id())

        # List of venvs created in the test
        self.venvs: List[NestedVenv] = []

        self.test_asset_path = Path("tests/cyrxnopt/test_assets").resolve()

        return super().setUp()

    def tearDown(self) -> None:
        # This serves as a test for venv.delete()
        for venv in self.venvs:
            venv.delete()

            self.assertFalse(venv.prefix.exists())

        return super().tearDown()

    def test_activate_with_no_venv_created(self) -> None:
        venv = NestedVenv(self.venv_path)

        self.assertRaises(RuntimeError, venv.activate)

    def test_activate_with_venv_created(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)
        venv.create()

        # Shouldn't throw an exception
        venv.activate()

    def test_activate_when_active(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)
        venv.create()

        # Shouldn't throw an exception
        venv.activate()

        # Also shouldn't throw an exception
        venv.activate()

    def test_activate_two_venvs(self) -> None:
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(Path(str(self.venv_path) + "_2"))
        self.venvs.append(venv1)
        self.venvs.append(venv2)

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

    def test_activate_two_venvs_then_deactivate_first(self) -> None:
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(Path(str(self.venv_path) + "_2"))
        self.venvs.append(venv1)
        self.venvs.append(venv2)

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

    def test_create_with_no_prior_creation(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        venv.create()

        self.assertTrue(self.venv_path.exists())
        self.assertTrue(venv.python.exists())

        venv.delete()

    def test_create_when_dir_exists(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        # Create the directory beforehand
        self.venv_path.mkdir(parents=True, exist_ok=False)

        # Creating venv but path already exists
        venv.create()

        self.assertTrue(self.venv_path.exists())
        self.assertTrue(venv.python.exists)

    def test_create_twice(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        venv.create()
        venv.create()

        self.assertTrue(self.venv_path.exists())
        self.assertTrue(venv.python.exists())

    def test_deactivate_not_active(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        self.assertFalse(venv.is_active())

        # Nothing should happen here
        venv.deactivate()

        self.assertFalse(venv.is_active())

    def test_is_active_not_active(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        # No venv has been created or activated

        self.assertFalse(venv.is_active())

    def test_is_active(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        venv.create()
        venv.activate()

        self.assertTrue(venv.is_active())

    def test_is_primary_not_active(self) -> None:
        venv = NestedVenv(self.venv_path)

        # No venv has been created or activated

        self.assertFalse(venv.is_primary())

    def test_is_primary_only_venv(self) -> None:
        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        venv.create()
        venv.activate()

        self.assertTrue(venv.is_primary())

        venv.deactivate()

        self.assertFalse(venv.is_primary())

    def test_is_primary_two_venvs(self) -> None:
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(Path(str(self.venv_path) + "_2"))
        self.venvs.append(venv1)
        self.venvs.append(venv2)

        venv1.create()
        venv2.create()

        # neither venv is active
        self.assertFalse(venv1.is_primary())
        self.assertFalse(venv2.is_primary())

        # venv1 is the only active
        venv1.activate()
        self.assertTrue(venv1.is_primary())
        self.assertFalse(venv2.is_primary())

        # venv1 and venv2 are active, venv2 should be primary
        venv2.activate()
        self.assertFalse(venv1.is_primary())
        self.assertTrue(venv2.is_primary())

        # only venv2 is active
        venv1.deactivate()
        self.assertFalse(venv1.is_primary())
        self.assertTrue(venv2.is_primary())

        # venv1 and venv2 active again, but venv2 should be primary
        venv1.activate()
        self.assertTrue(venv1.is_primary())
        self.assertFalse(venv2.is_primary())

        # neither venv is active
        venv1.deactivate()
        venv2.deactivate()
        self.assertFalse(venv1.is_primary())
        self.assertFalse(venv2.is_primary())

    def test_pip_install_numpy(self) -> None:
        """This test attempts to install the 'numpy' package from online
        using 'pip'.
        """

        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        venv.create()
        venv.activate()

        venv.pip_install("numpy")

        self.assertTrue(venv.check_package("numpy"))

    def test_pip_install_e(self) -> None:
        """This test attempts to self-install this package into a new
        virtual environment using an editable install.
        """

        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        venv.create()
        venv.activate()

        venv.pip_install_e(self.test_asset_path / "test_project")

        # self.assertTrue(venv.check_package("test_project"))
        self.assertTrue(venv.check_package("numpy", "1.24.0"))

    def test_pip_install_r(self) -> None:
        """This test attempts to self-install this package's requirements.txt
        file into a new virtual environment using
        'pip install -r requirements.txt'.
        """

        venv = NestedVenv(self.venv_path)
        self.venvs.append(venv)

        venv.create()
        venv.activate()

        venv.pip_install_r(self.test_asset_path / "requirements.txt")

        # self.assertTrue(venv.check_package("test_project"))
        self.assertTrue(venv.check_package("numpy", "1.24.0"))
        self.assertTrue(venv.check_package("requests", "2.31.0"))

    def test_pip_install_numpy_first_of_two_venvs(self) -> None:
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(Path(str(self.venv_path) + "_2"))
        self.venvs.append(venv1)
        self.venvs.append(venv2)

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

    def test_pip_install_numpy_two_versions(self) -> None:
        venv1 = NestedVenv(self.venv_path)
        venv2 = NestedVenv(Path(str(self.venv_path) + "_2"))
        self.venvs.append(venv1)
        self.venvs.append(venv2)

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
