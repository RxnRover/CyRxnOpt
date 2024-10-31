import copy
import importlib
import importlib.util
import logging
import os
import shutil
import site
import subprocess
import sys
import venv
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import List, Optional, Union, cast

# from cyrxnopt.util.reset_module import reset_module
logger = logging.getLogger(__name__)


class NestedVenv(venv.EnvBuilder):
    def __init__(self, virtual_dir: Union[str, Path]):
        """initializing the virtual environment directory

        :param virtual_dir: path to the virtual env directory
        :type virtual_dir: str | Path
        """

        self.prefix = Path(virtual_dir)

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

    def activate(self) -> None:
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

        if self.prefix.exists():
            # NOTE: This os.environ["PATH"] stuff is from a Google Groups
            #       conversation between users "voltron" and "Ian Bicking"
            #       but it doesn't seem to actually bring the venv into
            #       scope
            #
            # Source: https://groups.google.com/g/python-virtualenv/c/FfipsFBqvq4?pli=1
            env_path = [Path(p) for p in os.environ["PATH"].split(":")]
            env_path.insert(0, self.binary_directory)

            # Determine available modules before activating this, then
            # available packages afterward to diff what packages were
            # added by this virtual environment
            self._prior_site_packages = site.getsitepackages()

            # TODO: This adds the site to the end of sys.path. It should
            #       go before any other venv site paths to be the primary venv.
            # Activates the virtual environment, adding it to sys.path
            site.addsitedir(str(self.site_packages.resolve()))
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

        os.environ["PATH"] = ":".join([str(p.resolve()) for p in env_path])

    def create(self) -> None:
        """Creates the virtual environment at the given location."""

        super().create(self.prefix)

    def deactivate(self) -> None:
        """Deactivates the virtual environment regardless of if it is the
        primary virtual environment.
        """

        # Do nothing if the virtual environment is not active
        if not self.is_active():
            return

        env_path = [Path(p) for p in os.environ["PATH"].split(":")]

        # Remove all instances of the virtual environment from the path
        env_path = [path for path in env_path if path != self.binary_directory]

        os.environ["PATH"] = ":".join([str(p.resolve()) for p in env_path])

        # TODO: We need to remove the virtual environment from sys.path
        #       and unimport the packages from it without affecting
        #       other virtual environments. Troubles might arise from
        #       venv1 and venv2 both having the same package. How do
        #       we determine if both venvs have the package?
        #
        # Remove module: https://stackoverflow.com/a/57891909

        # Remove this venv from the sys.path
        sys.path.remove(str(self.site_packages.resolve()))

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

    def delete(self) -> None:
        self.deactivate()

        if self.prefix.exists():
            shutil.rmtree(self.prefix)

    def is_active(self) -> bool:
        """Checks if the virtual environment is active or not.

        This checks if the virtual environment path is in the PATH
        environment variable or not.

        :return: Whether the venv is active (True) or not (False).
        :rtype: bool
        """

        env_path = [Path(p) for p in os.environ["PATH"].split(":")]

        return self.binary_directory in env_path

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

        env_path = [Path(p) for p in os.environ["PATH"].split(":")]

        return env_path[0].resolve() == self.binary_directory

    def pip_freeze(self) -> List[str]:
        """Returns the list of modules in the virtual environment as
        they would be returned by 'pip freeze'.

        :raises CalledProcessError: An error occurred when running pip freeze
        """

        # Run ``pip freeze`` and capture the output
        completed_process = subprocess.run(
            [self.python, "-m", "pip", "freeze"],
            capture_output=True,  # Capture stdout and stderr
            encoding="utf-8",  # Dencode the stdout and stderr bytestrings
        )

        # Raises CalledProcessError if the return code is non-zero
        completed_process.check_returncode()

        # The response is split by newlines since one package is
        # printed on each line
        return completed_process.stdout.split()

    def pip_install(
        self,
        package_name: str,
        package_path: Optional[Path] = None,
        editable: bool = False,
    ) -> None:
        """Install a package to the active virtual environment using
        ``pip install`` for an editable install.

        :param package_name: Name of the package
        :type package_name: str
        :param package_path: Path to the package location
        :type package_path: Path
        :param editable: Whether to use an editable install
        :type editable: bool

        :raises CalledProcessError: An error occurred when running pip freeze
        """

        # NOTE: In the 'importlib' package, it is noted that `import_module()`
        #       should be used instead of `__import__()`. Maybe it is better
        #       to use that here, too. More research needed.
        #
        # Source: https://docs.python.org/3/library/importlib.html#importlib.__import__
        try:
            __import__(package_name)
        except ModuleNotFoundError:
            # Decide whether this is a local path or PyPI package
            if package_path is not None:
                package: str = str(package_path)
            else:
                package = package_name

            # Do we need to prepend ``-e`` for an editable install?
            pre_args = []
            if editable:
                pre_args.append("-e")

            # Create the command list
            cmd: List[str] = [str(self.python), "-m", "pip", "install"]
            cmd.extend(pre_args)
            cmd.append(package)
            cmd.append("--upgrade")

            completed_process = subprocess.run(
                cmd,
                capture_output=True,  # Capture stdout and stderr
                encoding="utf-8",  # Dencode the stdout and stderr bytestrings
            )

            # Raises CalledProcessError if the return code is non-zero
            completed_process.check_returncode()

    def pip_install_e(self, package_path: Path, package_name: str = "") -> None:
        """Install a package to the active virtual environment using
        ``pip install`` for an editable install.

        :param package_path: Path to the package location
        :type package_path: Path
        :param package_name: Name of the package, defaults to "". If not provided,
            the package name is assumed to the the last part of ``package_path``.
        :type package_name: str, optional

        :raises CalledProcessError: An error occurred when running ``pip install``
        """

        # Derive the package name from the package path if a name is not
        # explicitly provided
        if package_name == "":
            package_name = package_path.stem

        # Attempt to install the package
        self.pip_install(package_name, package_path, editable=True)

    def pip_install_r(self, req_file: Path) -> None:
        """Installs package requirements from a "requirements.txt"-style file.

        :param req_file: Requirements file to use
        :type req_file: str

        :raises CalledProcessError: An error occurred when running
            ``pip install`` for a package
        """

        # Read each line of the requirements file and install the packages
        with open(req_file, "r") as fin:
            lines = fin.readlines()

            for line in lines:
                if line.startswith("-e"):
                    package_path = Path(line.replace("-e", "").strip())
                    package_name = package_path.stem

                    self.pip_install(
                        package_name,
                        package_path.resolve(strict=True),
                        editable=True,
                    )

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
        os.environ["PATH"] = str(self.binary_directory)
        sys.path.append(str(self.site_packages.resolve()))

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

    def _get_site_package_path(self) -> Path:
        if sys.platform == "win32":
            site_package_path = self.prefix / "Lib" / "site-packages"
        else:
            site_package_path = (
                self.prefix
                / "lib"
                / "python{}".format(self._get_python_version())
                / "site-packages"
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
                modulespec = importlib.util.find_spec(pkg)

                # If the spec is None, skip the entry
                if modulespec is None:
                    continue
            # Sometimes a ValueError is raised if no .__spec__ member
            # is found. Skip the entry as well
            except ValueError:
                continue

            # We can guarantee that importlib.util.find_spec(pkg) is not
            # None from the checks above
            modulespec = cast(ModuleSpec, modulespec)

            # Check if valid packages are from this virtual environment
            if (
                modulespec.origin is not None
                and str(self.site_packages) in modulespec.origin
            ):
                # print("Unimporting:", pkg)
                sys.modules.pop(pkg)

                venv_modules.append(pkg)

        return venv_modules

    @property
    def binary_directory(self) -> Path:
        """The venv subdirectory containing binaries based on operating system.

        :return: Full path to the venv binary directory
        :rtype: Path
        """

        return self.prefix / self._binary_directory_name

    @property
    def prefix(self) -> Path:
        """The prefix directory for this venv.

        :return: Full path to the prefix directory for this venv
        :rtype: Path
        """

        return self._prefix

    @prefix.setter
    def prefix(self, value: Path) -> None:
        full_prefix = value.resolve()

        logger.debug("Setting venv prefix to {}".format(full_prefix))
        self._prefix = full_prefix

    @property
    def python(self) -> Path:
        """The python binary of the venv based on operatingsystem.

        :return: Full path to the Python binary of the venv
        :rtype: Path
        """

        return self.binary_directory / self._python_binary_file_name

    @property
    def site_packages(self) -> Path:
        return self._get_site_package_path()

    @property
    def _python_binary_file_name(self) -> str:
        """The python binary file name based on operating system.

        :return: Python binary file name
        :rtype: str
        """

        return "python.exe" if sys.platform == "win32" else "python"

    @property
    def _binary_directory_name(self) -> str:
        """The name of the venv subdirectory containing binaries based on
        operating system.

        :return: Virtual environment binary directory
        :rtype: str
        """

        return "Scripts" if sys.platform == "win32" else "bin"
