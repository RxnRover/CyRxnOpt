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
from subprocess import CalledProcessError
from typing import Any, List, Optional, Union, cast

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

        logger.info("Activating venv: {}".format(self.prefix))

        # Return early if already active
        if self.is_active():
            logger.info("Venv already active.")
            return

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
        else:
            raise RuntimeError("Virtual environment has not been created yet!")

        os.environ["PATH"] = ":".join([str(p.resolve()) for p in env_path])

        logger.info("Venv activated.")

    def create(self, env_dir: Any = "") -> None:
        """Creates the virtual environment at the given location.

        :param env_dir: Desired venv directory
        :type env_dir: AnyPath
        """

        prefix = env_dir

        # Default to the provided prefix provided on instantiation
        if str(env_dir) == "":
            logger.debug(
                "env_dir argument not given, defaulting to: {}".format(
                    self.prefix
                )
            )
            prefix = self.prefix

        logger.info("Creating venv: {}".format(self.prefix))

        return super().create(prefix)

    def deactivate(self) -> None:
        """Deactivates the virtual environment regardless of if it is the
        primary virtual environment.
        """

        logger.info("Deactivating venv: {}".format(self.prefix))

        # Do nothing if the virtual environment is not active
        if not self.is_active():
            logger.info("Venv not active.")
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
        logger.debug("Removing deactivated venv packages...")
        venv_modules = self._unimport_packages()

        # Attempt to reimport modules from other venvs
        logger.debug(
            "Attempting to reimport packages still active in other venvs..."
        )
        for pkg in venv_modules:
            try:
                importlib.import_module(pkg)
                # __import__(pkg)
                # import pkg
                # importlib.reload(pkg)
                # reset_module(pkg)
                logger.debug("Successfully reimported: {}".format(pkg))
            except (KeyError, ModuleNotFoundError) as e:
                logger.debug("Could not reimport: {}".format(pkg))
                logger.debug(
                    "Reimport exception: {}({})".format(e.__class__.__name__, e)
                )
                continue

        logger.info("Venv deactivated.")

    def delete(self) -> None:
        logger.info("Deleting venv: {}".format(self.prefix))

        self.deactivate()

        if self.prefix.exists():
            shutil.rmtree(self.prefix)

        logger.info("Venv deleted.")

    def is_active(self) -> bool:
        """Checks if the virtual environment is active or not.

        This checks if the virtual environment path is in the PATH
        environment variable or not.

        :return: Whether the venv is active (True) or not (False).
        :rtype: bool
        """

        env_path = [Path(p) for p in os.environ["PATH"].split(":")]

        is_active = self.binary_directory in env_path

        logger.debug(
            "Venv at {} is currently {}.".format(
                self.prefix, "active" if is_active else "not active"
            )
        )

        return is_active

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

        is_primary = env_path[0].resolve() == self.binary_directory

        logger.debug(
            "Venv at {} is currently {}.".format(
                self.prefix, "primary" if is_primary else "not primary"
            )
        )

        return is_primary

    def pip_freeze(self) -> List[str]:
        """Returns the list of modules in the virtual environment as
        they would be returned by 'pip freeze'.

        :raises CalledProcessError: An error occurred when running pip freeze
        """

        # TODO: Add logging

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

        # TODO: Add logging

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
            try:
                completed_process.check_returncode()
            except CalledProcessError as e:
                logger.info("Return code nonzero: {}".format(e))
                logger.info("stdout: {}".format(completed_process.stdout))
                logger.info("stderr: {}".format(completed_process.stderr))

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

        # TODO: Add logging

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

        # TODO: Add logging

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

        # TODO: Add logging and docstring!

        logger.debug(
            "Checking for '{}' in venv: {}".format(package, self.prefix)
        )

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

        logger.debug(
            "{} {}".format(package, "found" if package_found else "not found")
        )

        return package_found

    def _get_site_package_path(self) -> Path:
        # TODO: Add logging and docstring!

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
        # TODO: Add logging and docstring!

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

        # TODO: Add logging

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
                logger.debug("Unimporting: {}".format(pkg))
                sys.modules.pop(pkg)

                venv_modules.append(pkg)

        venv_modules

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
