# import os
# import unittest

# from utilities_for_testing.validate_config_description import (
#     validate_config_description,
# )

# from pyoptimizer_backend.NestedVenv import NestedVenv
# from pyoptimizer_backend.OptimizerEDBOp import OptimizerEDBOp


# class TestOptimizerEDBOp(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.base_prefix = "tmp"
#         cls.venv_path = os.path.join(cls.base_prefix, "venv_edbop")

#         # Create common venv to save time
#         cls.venv = NestedVenv(cls.venv_path)

#         # Always recreate the venv to start with a clean slate for testing
#         try:
#             cls.venv.activate()
#         except RuntimeError as e:
#             cls.venv.create()
#             cls.venv.activate()

#         opt = OptimizerEDBOp(cls.venv)

#         # This test will fail if this throws an error
#         opt.install(local_paths={"edboplus": "deps/edbop"})

#     @classmethod
#     def tearDownClass(cls):
#         # cls.venv.delete()
#         pass

#     def setUp(self) -> None:
#         self.test_prefix = os.path.join(self.base_prefix, self.id())

#         # self.venv.activate()

#         return super().setUp()

#     def tearDown(self) -> None:
#         return super().tearDown()

#     def test_get_config(self):
#         """This test checks that a config of the correct format was provided,
#         but does not try to validate the actual values of each config
#         description.
#         """

#         opt = OptimizerEDBOp(self.venv)

#         result = opt.get_config()

#         validate_config_description(self, result)

#     def test_set_config(self):
#         opt = OptimizerEDBOp(self.venv)

#         config = {
#             "continuous_feature_names": ["f1", "f2"],
#             "continuous_feature_bounds": [[-1, 1], [-5, 5]],
#             "continuous_feature_resolutions": [1, 5, 1],
#             "categorical_feature_names": ["f3"],
#             "categorical_feature_values": [["a", "b", "c"]],
#             "budget": 10,
#             "objectives": ["yield"],
#             "objective_mode": "min",
#         }

#         opt.set_config(self.test_prefix, config)

#         # Make sure that all files were created
#         full_combo_file = os.path.join(self.test_prefix, "full_combo_file.txt")
#         training_combo_file = os.path.join(
#             self.test_prefix, "training_combo_file.txt"
#         )
#         training_set_decoded_file = os.path.join(
#             self.test_prefix, "training_set_decoded_file.txt"
#         )
#         training_set_file = os.path.join(
#             self.test_prefix, "training_set_file.txt"
#         )

#         self.assertTrue(os.path.exists(full_combo_file))
#         self.assertTrue(os.path.exists(training_combo_file))
#         self.assertTrue(os.path.exists(training_set_decoded_file))
#         self.assertTrue(os.path.exists(training_set_file))

#     def test__validate_config_complete_config(self):
#         opt = OptimizerEDBOp(self.venv)
#         opt.install()

#         config = {
#             "continuous_feature_names": ["f1", "f2"],
#             "continuous_feature_bounds": [[-1, 1], [-5, 5]],
#             "continuous_feature_resolutions": [1, 5, 1],
#             "categorical_feature_names": ["f3"],
#             "categorical_feature_values": [["a", "b", "c"]],
#             "budget": 10,
#             "objectives": ["yield"],
#             "objective_mode": "min",
#         }

#         opt._validate_config(config)

#     def test__validate_config_continuous_config(self):
#         opt = OptimizerEDBOp(self.venv)
#         opt.install()

#         config = {
#             "continuous_feature_names": ["f1", "f2"],
#             "continuous_feature_bounds": [[-1, 1], [-5, 5]],
#             "continuous_feature_resolutions": [1, 5, 1],
#             "budget": 10,
#         }

#         opt._validate_config(config)

#     def test__validate_config_categorical_config(self):
#         opt = OptimizerEDBOp(self.venv)
#         opt.install()

#         config = {
#             "categorical_feature_names": ["f3"],
#             "categorical_feature_values": [["a", "b", "c"]],
#             "budget": 10,
#         }

#         opt._validate_config(config)

#     def test__validate_config_missing_parts(self):
#         opt = OptimizerEDBOp(self.venv)
#         opt.install()

#         config_no_names = {
#             "continuous_feature_bounds": [[-1, 1], [-5, 5]],
#             "continuous_feature_resolutions": [1, 5, 1],
#             "categorical_feature_values": [["a", "b", "c"]],
#             "budget": 10,
#         }

#         self.assertRaises(RuntimeError, opt._validate_config, config_no_names)

#         config_no_continuous_feature_bounds_or_res = {
#             "continuous_feature_names": ["f1", "f2"],
#             "budget": 10,
#         }

#         self.assertRaises(
#             RuntimeError,
#             opt._validate_config,
#             config_no_continuous_feature_bounds_or_res,
#         )

#         config_no_categorical_feature_values = {
#             "categorical_feature_names": ["f3"],
#             "budget": 10,
#         }

#         self.assertRaises(
#             RuntimeError,
#             opt._validate_config,
#             config_no_categorical_feature_values,
#         )

#         config_no_budget = {
#             "continuous_feature_names": ["f1", "f2"],
#             "continuous_feature_bounds": [[-1, 1], [-5, 5]],
#             "continuous_feature_resolutions": [1, 5, 1],
#             "categorical_feature_names": ["f3"],
#             "categorical_feature_values": [["a", "b", "c"]],
#         }

#         self.assertRaises(RuntimeError, opt._validate_config, config_no_budget)

#     # TODO: check_install test

#     # TODO: call train and predict tests
