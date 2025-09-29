class OptimizerTemplate(OptimizerABC):
    
    __packages = ["Packages"]

    def __init__(self, venv: VenvManager = None) -> None:
        """This code will initialize your optimizer class .

        :param venv: Virtual environment manager to use, defaults to None
        :type venv: cyrxnopt.NestedVenv, optional
        """
        
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

        In this function you need to add all the required configuration variable
        that are needed to run your optimizer algorithm. Follow the common dictionary
        keys and add new features end of the dictionary.

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
        """Set the configuration for this instance of the optimizer. Valid
        configuration options should be retrieved using `get_config()` before
        calling this function.

        In this code block/function you need to handle all the configuration options
        and all the code required to intialize yor algoithm will go here. For example,
        reaction space generation or generation of initial files. Depending on your 
        workflow you can breakdown your code in this function.

        :param experiment_dir: Output directory for the configuration file.
        :type experiment_dir: str
        :param config: Configuration options for this optimizer instance.
        :type config: Dict[str, Any]
        """

    def train(
        self,
        prev_param: List[Any],
        yield_value: float,
        itr: int,
        experiment_dir: str,
        config: Dict,
        obj_func=None,
    ) -> List[Any]:
        """generate initial training dataset needed for AMLRO model training.

        If your algorithm required generate training dataset or loading training files
        this function can be used for adding those code lines.

        :param prev_param: experimental parameter combination for previous experiment
        :type prev_param: list
        :param yield_value: experimental yield
        :type yield_value: float
        :param itr: experimental cycle number for training
        :type itr: int
        :param experiment_dir: experimental directory for saving data files
        :type experiment_dir: str
        :param config: Initial reaction feature configurations
        :type config: Dict
        :return: next parameter combination for next experimental cycle.
        :rtype: list
        """
        self._import_deps()

    def predict(
        self,
        prev_param: List[Any],
        yield_value: float,
        experiment_dir: str,
        config: Dict[str, Any],
        obj_func=None,
    ) -> None:
        """Find the desired optimum of the provided objective function using your algorithm.

        In this code block, you can add codes required for the finding optimum reaction conditions.

        :param prev_param: Parameters provided from the previous prediction or
                           training step.
        :type prev_param: List[Any]
        :param yield_value: Result from the previous prediction or training
                            step.
        :type yield_value: float
        :param experiment_dir: Output directory for the optimizer algorithm.
        :type experiment_dir: str
        :param obj_func: Objective function to optimize, defaults to None
        :type obj_func: function, optional
        """


    def _import_deps(self) -> None:
        """Import package needed to run the optimizer.
        
        You need to add code lines to import all the required packages here.
        Then create _imports dictionary to update all the packages and later use this dictionary 
        to access packages as necessary.
        
        """

        from Your_Library import Your_Package

        self._imports = {
            "package": Your_package
        }
