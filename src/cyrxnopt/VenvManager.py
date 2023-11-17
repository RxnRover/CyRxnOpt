import os
import subprocess
import sys


class VenvManager:
    def __init__(self, virtual_dir):
        """initializing the virtual environment directory

        :param virtual_dir: path to the virtual env directory
        :type virtual_dir: Str
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

    def install_virtual_env(self):
        """Create virtual environment if doesnt exists."""

        if not os.path.exists(self.virtual_python):
            import subprocess

            subprocess.call(
                [sys.executable, "-m", "virtualenv", self.virtual_dir]
            )
        else:
            print("found virtual python: " + self.virtual_python)

    def is_venv(self) -> bool:
        """This function checking  whether virtual environment exists or not

        :return: boolean parameter for virtual env directory existence
        :rtype: bool
        """
        return sys.prefix == self.virtual_dir

    def restart_under_venv(self):
        """Restarting the installed virtual env's."""
        print("Restarting under virtual environment " + self.virtual_dir)
        subprocess.call([self.virtual_python, __file__] + sys.argv[1:])
        # exit(0)

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
            test = subprocess.call(
                [self.virtual_python, "-m", "pip", "install", "-e", package]
            )
            print(test)

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

    def start_venv(self):
        """Up and running the virtual env manager class or activate the virtual env."""
        if not self.is_venv():
            self.install_virtual_env()
            self.restart_under_venv()
        else:
            print("Running under virtual environment")
            # self.restart_under_venv()
