import os
import subprocess
import sys


class VenvManager:
    def __init__(self, virtual_dir):
        """initializing the virtual environment directory

        :param virtual_dir: path to the virtual env directory
        :type virtual_dir: Str
        """
        self.virtual_dir = virtual_dir
        self.virtual_python = os.path.join(self.virtual_dir, "Scripts", "python.exe")

    def install_virtual_env(self):
        """Create virtual environment if doesnt exists."""
        self.pip_install("virtualenv")
        if not os.path.exists(self.virtual_python):
            import subprocess

            subprocess.call([sys.executable, "-m", "virtualenv", self.virtual_dir])
        else:
            print("found virtual python: " + self.virtual_python)

    def is_venv(self):
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
                [self.virtual_python, "-m", "pip", "install", package, "--upgrade"]
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

    def start_venv(self):
        """Up and running the virtual env manager class or activate the virtual env."""
        if not self.is_venv():
            self.install_virtual_env()
            self.restart_under_venv()
        else:
            print("Running under virtual environment")
