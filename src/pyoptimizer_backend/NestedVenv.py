import copy
import importlib
import importlib.util
import os
import shutil
import site
import subprocess
import sys
import venv
from typing import List

# from pyoptimizer_backend.util.reset_module import reset_module


class NestedVenv(venv.EnvBuilder):
    def __init__(self, virtual_dir: str):
        """initializing the virtual environment directory

        :param virtual_dir: path to the virtual env directory
        :type virtual_dir: str
        """

        self.prefix = os.path.abspath(virtual_dir)
        self.python = self._get_python_path()
        self.site_packages = self._get_site_package_path()

        # Call the EnvBuilder constructor
        super().__init__(
            system_site_packages=False,
            clear=True,
            symlinks=False,
            upgrade=False,
            with_pip=True,
            prompt=None,
            upgrade_deps=True,
        )

    def activate(self):
        """Activates the current virtual environment as the primary virtual
        environment. If the venv is active but not primary, it will be
        reactivated as the primary venv.

        :raises RuntimeError: The virtual environment does not exist.
        """

        # Return early if already active
        if self.is_active():
            return

        # # Deactivate the virtual environment if it was already active
        # self.deactivate()

        if os.path.exists(self.prefix):
            # NOTE: This os.environ["PATH"] stuff is from a Google Groups
            #       conversation between users "voltron" and "Ian Bicking"
            #       but it doesn't seem to actually bring the venv into
            #       scope
            #
            # Source: https://groups.google.com/g/python-virtualenv/c/FfipsFBqvq4?pli=1
            env_path = os.environ["PATH"].split(":")
            env_path.insert(0, self._bin_dir)

            # Determine available modules before activating this, then
            # available packages afterward to diff what packages were
            # added by this virtual environment
            self._prior_site_packages = site.getsitepackages()

            # TODO: This adds the site to the end of sys.path. It should
            #       go before any other venv site paths to be the primary venv.
            # Activates the virtual environment, adding it to sys.path
            site.addsitedir(self.site_packages)
            # NOTE: This sitedir stuff is from the SO answer here:
            #       https://stackoverflow.com/a/68173529, which points
            #       to this in dcreager/virtualenv on GitHub:
            #       https://github.com/dcreager/virtualenv/blob/master/virtualenv_support/activate_this.py
            #
            # Docs for site: https://docs.python.org/3/library/site.html

            # Move the site package to the front of the sys.path so it is
            # picked up first
            # print("sys.path:", sys.path)
            # tmp_site_packages = sys.path.pop(-1)
            # print("tmp_site_packages:", tmp_site_packages)
            # sys.path.insert(0, tmp_site_packages)
            # print("sys.path:", sys.path)
        else:
            raise RuntimeError("Virtual environment has not been created yet!")

        os.environ["PATH"] = ":".join(env_path)

    def create(self):
        """Creates the virtual environment at the given location."""

        super().create(self.prefix)

    def deactivate(self):
        """Deactivates the virtual environment regardless of if it is the
        primary virtual environment.
        """

        # Do nothing if the virtual environment is not active
        if not self.is_active():
            return

        env_path = os.environ["PATH"].split(":")

        # Remove all instances of the virtual environment from the path
        env_path = [path for path in env_path if path != self._bin_dir]

        os.environ["PATH"] = ":".join(env_path)

        # TODO: We need to remove the virtual environment from sys.path
        #       and unimport the packages from it without affecting
        #       other virtual environments. Troubles might arise from
        #       venv1 and venv2 both having the same package. How do
        #       we determine if both venvs have the package?
        #
        # Remove module: https://stackoverflow.com/a/57891909

        # Remove this venv from the sys.path
        sys.path.remove(self.site_packages)

        # A start is to invalidate the internal cache to guarantee that
        # import finders will notice new modules
        importlib.invalidate_caches()

        # Unimport packages that originate from this venv
        venv_modules = self._unimport_packages()

        # Attempt to reimport modules from other venvs
        for pkg in venv_modules:
            try:
                importlib.import_module(pkg)
                # __import__(pkg)
                # import pkg
                # importlib.reload(pkg)
                # reset_module(pkg)
                # print("Successfully reimported:", pkg)
            except ModuleNotFoundError:
                # print("Failed to reimport:", pkg)
                continue

    def delete(self):
        self.deactivate()

        if os.path.exists(self.prefix):
            shutil.rmtree(self.prefix)

    def is_active(self) -> bool:
        """Checks if the virtual environment is active or not.

        This checks if the virtual environment path is in the PATH
        environment variable or not.

        :return: Whether the venv is active (True) or not (False).
        :rtype: bool
        """

        env_path = os.environ["PATH"].split(":")

        return self._bin_dir in env_path

    def is_primary(self) -> bool:
        """Checks if the virtual environment is the primary active
        virtual environment.

        This checks if the virtual environment path is the first entry
        in the PATH environment variable. This menas its packages
        will be found first.

        TODO: Recognize other virtual environments to ensure we are
              the first virtual environment without needing to be the
              first element in the PATH environment variable.

        :return: Whether the venv is primary (True) or not (False).
        :rtype: bool
        """

        env_path = os.environ["PATH"].split(":")

        return env_path[0] == self._bin_dir

    def pip_freeze(self) -> List[str]:
        """Returns the list of modules in the virtual environment as
        they would be returned by 'pip freeze'.
        """

        # TODO: This doesn't return the list of dependencies
        subprocess.call(
            [
                self.python,
                "-m",
                "pip",
                "freeze",
            ]
        )

    def pip_install(self, package):
        """installing the packages to the running virtual env using
        pip install commands.

        :param package: package name
        :type package: Str
        """

        # NOTE: In the 'importlib' package, it is noted that `import_module()`
        #       should be used instead of `__import__()`. Maybe it is better
        #       to use that here, too. More research needed.
        #
        # Source: https://docs.python.org/3/library/importlib.html#importlib.__import__
        try:
            __import__(package)
        except ModuleNotFoundError:
            subprocess.call(
                [
                    self.python,
                    "-m",
                    "pip",
                    "install",
                    package,
                    "--upgrade",
                ]
            )

    def pip_install_e(self, package):
        """installing the local packages (from folder) to the running virtual env
        using pip install -e commands.

        :param package: folder path/name to the package
        :type package: Str
        """
        try:
            __import__(package)
        except ModuleNotFoundError:
            subprocess.call(
                [self.python, "-m", "pip", "install", "-e", package]
            )

    def pip_install_r(self, filename):
        """Installs package requirements from a "requirements.txt"-style file.

        :param filename: Requirements file to use
        :type filename: str
        """

        # Read each line of the requirements file and install the packages
        with open(filename, "r") as fin:
            lines = fin.readlines()

            for line in lines:
                if line.startswith("-e"):
                    package = line.replace("-e", "").strip()
                    self.pip_install_e(os.path.abspath(package))

                else:
                    package = line
                    self.pip_install(package)

    def check_package(self, package: str, version: str = "") -> bool:
        # TODO: Should this be allowed even if the venv is inactive at
        #       the time of calling? I think it can still be checked without
        #       affecting anything, so I am allowing it on inactive venvs
        #       for now.

        # Default to the package being found
        package_found = True

        # Get the original, full PATH variable
        og_env_path = os.environ["PATH"]
        og_sys_path = copy.deepcopy(sys.path)

        # Remove other virtual environment site-packages paths temporarily
        for path in reversed(sys.path):
            if "site-packages" in path:
                sys.path.remove(path)
            else:
                break

        # Replace the PATH variable with only the virtual environment
        os.environ["PATH"] = self._bin_dir
        sys.path.append(self.site_packages)

        importlib.invalidate_caches()

        og_sys_modules = sys.modules

        venv_modules = []

        loaded_package_modules = [key for key, value in sys.modules.items()]
        for pkg in loaded_package_modules:
            try:
                # If the spec is None, skip the entry
                if importlib.util.find_spec(pkg) is None:
                    continue
            # Sometimes a ValueError is raised if no .__spec__ member
            # is found
            except ValueError:
                continue

            # Check if valid packages associated with the package by
            # checking if the first part of the module path matches.
            # For example, "numpy.random.mtrand" would match when searching
            # for package "numpy".
            if package == pkg.split(".")[0]:
                sys.modules.pop(pkg)

                venv_modules.append(pkg)

        try:
            # print("Attempting import of", package)
            module = importlib.import_module(package)

            # TODO: This version checking could be much more complex
            #       to allow for the full versioning syntax that pip can use.
            #       For example, a user could specify version ">=1.25" instead
            #       of only matching a specific version.
            if version != "":
                package_found = True if module.__version__ == version else False
            # print("Import succeeded.")
        except ModuleNotFoundError:
            # print("Import failed.")
            package_found = False

        os.environ["PATH"] = og_env_path
        sys.path = og_sys_path

        importlib.invalidate_caches()

        sys.modules = og_sys_modules

        return package_found

    def _get_python_path(self) -> str:
        python_path = ""

        # Decide, based on the operating system, what path to the Python binary
        # to use. Windows uses <venv>/Scripts/python.exe, while Linux (and Mac,
        # I think) use <venv>/bin/python.
        self._python_bin = "python.exe" if sys.platform == "win32" else "python"
        self._venv_dir = "Scripts" if sys.platform == "win32" else "bin"
        self._bin_dir = os.path.join(self.prefix, self._venv_dir)

        python_path = os.path.join(self._bin_dir, self._python_bin)

        return python_path

    def _get_site_package_path(self) -> str:
        site_package_path = ""

        if sys.platform == "win32":
            site_package_path = os.path.join(
                self.prefix, "Lib", "site-packages"
            )
        else:
            site_package_path = os.path.join(
                self.prefix,
                "lib",
                "python{}".format(self._get_python_version()),
                "site-packages",
            )

        return site_package_path

    def _get_python_version(self) -> str:
        # This grabs the full semver, for example, "3.11.3"
        python_version = sys.version.split(" ")[0]

        # Remove the patch version
        python_version = ".".join(python_version.split(".")[:2])

        return python_version

    def _unimport_packages(self) -> List[str]:
        """Unimports all packages that originate from this virtual environment.

        This code is based on information provided by DeepSOIC and wjandrea
        on StackOverflow: https://stackoverflow.com/a/57891909.

        :return: Names of packages that were unimported by this function.
        :rtype: List[str]
        """

        venv_modules = []

        loaded_package_modules = [key for key, value in sys.modules.items()]
        for pkg in loaded_package_modules:
            try:
                # If the spec is None, skip the entry
                if importlib.util.find_spec(pkg) is None:
                    continue
            # Sometimes a ValueError is raised if no .__spec__ member
            # is found
            except ValueError:
                continue

            # Check if valid packages are from this virtual environment
            if (
                importlib.util.find_spec(pkg).origin
                and self.site_packages in importlib.util.find_spec(pkg).origin
            ):
                # print("Unimporting:", pkg)
                sys.modules.pop(pkg)

                venv_modules.append(pkg)

        return venv_modules
