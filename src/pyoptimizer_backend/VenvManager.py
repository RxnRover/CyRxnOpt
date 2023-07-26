import os
import subprocess
import sys
import venv


class NestedVenv(venv.EnvBuilder):
    def __init__(self, virtual_dir: str):
        """initializing the virtual environment directory

        :param virtual_dir: path to the virtual env directory
        :type virtual_dir: str
        """

        # Decide, based on the operating system, what path to the Python binary
        # to use. Windows uses <venv>/Scripts/python.exe, while Linux (and Mac,
        # I think) use <venv>/bin/python.
        self.__python_bin = (
            "python.exe" if sys.platform == "win32" else "python"
        )
        self.__venv_dir = "Scripts" if sys.platform == "win32" else "bin"

        self.virtual_dir = os.path.abspath(virtual_dir)
        self.virtual_python = os.path.join(
            self.virtual_dir, self.__venv_dir, self.__python_bin
        )

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

        # Deactivate the virtual environment if it was already active
        self.deactivate()

        if os.path.exists(self.virtual_dir):
            env_path = os.environ["PATH"].split(":")
            env_path.insert(0, self.virtual_dir)
        else:
            raise RuntimeError("Virtual environment has not been created yet!")

        os.environ["PATH"] = ":".join(env_path)

    def create(self):
        """Creates the virtual environment at the given location."""

        super().create(self.virtual_dir)

    def deactivate(self):
        """Deactivates the virtual environment regardless of if it is the
        primary virtual environment.
        """

        # Do nothing if the virtual environment is not active
        if not self.is_active:
            return  # pramga: no qa

        env_path = os.environ["PATH"].split(":")

        # Remove all instances of the virtual environment from the path
        env_path = [path for path in env_path if path != self.virtual_dir]

        os.environ["PATH"] = ":".join(env_path)

    def is_active(self) -> bool:
        """Checks if the virtual environment is active or not.

        This checks if the virtual environment path is in the PATH
        environment variable or not.

        :return: Whether the venv is active (True) or not (False).
        :rtype: bool
        """

        env_path = os.environ["PATH"].split(":")

        return self.virtual_dir in env_path

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

        return env_path[0] == self.virtual_dir

    def pip_install(self, package):
        """installing the packages to the running virtual env using
        pip install commands.

        :param package: package name
        :type package: Str
        """
        try:
            __import__(package)
        except ModuleNotFoundError:
            subprocess.call(
                [
                    self.virtual_python,
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
                [self.virtual_python, "-m", "pip", "install", "-e", package]
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
                    self.pip_install_e(package)

                else:
                    package = line
                    self.pip_install(package)
