class OptimizerTempelete(OptimizerABC):
    
    __packages = ["Packages"]

    def __init__(self, venv: VenvManager = None) -> None:
        
        self._imports = {}  # Populated in self._import_deps()
        self.__venv = venv

    def check_install(self) -> bool:
        
        try:
            self._import_deps()
        except ModuleNotFoundError as e:
            print(e)
            return False

        return True

    def install(self, local_paths: Dict[str, str] = {}) -> None:
        
        for package in self.__packages:
            
            if package in local_paths:
                self.__venv.pip_install_e(local_paths[package])
            else:
                self.__venv.pip_install(package)

        self._import_deps()

    def get_config(self) -> List[Dict[str, Any]]:
        """Get the configuration options available for this optimizer.

        :return: List of configuration options with option name, data type,
                 and information about which values are allowed/defaulted.
        :rtype: List[Dict[str, Any]]
        """

        self._import_deps()

        config = [
            {
                "name": "direction",
                "type": str,
                "value": ["min", "max"],
            },
            
        ]

        return config

    def set_config(self, experiment_dir: str, config: Dict[str, Any]) -> None:
        
        self._import_deps()

    def train(self) -> None:

        pass

    def predict(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func=None,
    ) -> None:
        

        self._import_deps()
        Pass


    def _import_deps(self) -> None:
        """Import package needed to run the optimizer."""

        from Your_Library import Your_Package

        self._imports = {
            "package": Your_package
        }
