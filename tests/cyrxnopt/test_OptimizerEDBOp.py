import sys

import pytest
from git import Repo

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerEDBOp import OptimizerEDBOp
from tests.cyrxnopt.utilities_for_testing.validate_config_description import (
    validate_config_description_pytest,
)

skip_libtorch_error = pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason=(
        "Issue with libtorch_cpu.so on Linux prevents successful import "
        "of EDBO+ during testing."
    ),
)


@pytest.fixture(scope="session")
def edboplus_local_path(tmp_path_factory):
    repo_location = tmp_path_factory.mktemp("edboplus")

    # EDBO+ starts to run into installation issues noticed 2025-10-07.
    # This, along with never merging in PR #6 with 40x performance improvements,
    # requires manually downloading an older version/branch of EDBO+.
    Repo.clone_from(
        "https://github.com/zachcran/edboplus",
        repo_location,
        branch="performance_improvements",
    )

    return repo_location


@pytest.fixture(scope="session")
def venv_edbop(tmp_path_factory, edboplus_local_path):
    venv_path = tmp_path_factory.mktemp("venv_edbop")

    test_venv = NestedVenv(venv_path)

    test_venv.create()
    test_venv.activate()

    assert test_venv.is_active()
    assert test_venv.is_primary()

    # Preinstall dependencies
    opt = OptimizerEDBOp(test_venv)
    opt.install(local_paths={"edboplus": edboplus_local_path})
    assert opt.check_install()

    yield test_venv

    test_venv.deactivate()
    assert not test_venv.is_active()
    test_venv.delete()


@skip_libtorch_error
def test_get_config_returns_valid_description_list(venv_edbop) -> None:
    opt = OptimizerEDBOp(venv_edbop)

    result = opt.get_config()

    validate_config_description_pytest(result)


@skip_libtorch_error
def test_set_config_creates_correct_config(venv_edbop, tmp_path) -> None:
    import json

    opt = OptimizerEDBOp(venv_edbop)

    expected_config_path = tmp_path / "config.json"

    config = {
        "continuous_feature_names": ["f1", "f2"],
        "continuous_feature_bounds": [[-1, 1], [-1, 1]],
        "continuous_feature_resolutions": [0.1, 0.1],
        "categorical_feature_names": ["f3"],
        "categorical_feature_values": [["a", "b", "c"]],
        "direction": ["min"],
        "budget": 10,
        "objectives": ["yield"],
    }

    opt.set_config(str(tmp_path), config)

    # Check if config file was created
    assert expected_config_path.exists()

    # Check for the correct contents
    with open(expected_config_path) as fin:
        content = json.load(fin)
        assert content == config


@skip_libtorch_error
def test_train_does_nothing(venv_edbop, tmp_path) -> None:
    opt = OptimizerEDBOp(venv_edbop)
    expected_suggestion = []

    suggestion = opt.train([], 0, tmp_path, {})

    assert expected_suggestion == suggestion


@skip_libtorch_error
def test_predict_basic_run(venv_edbop, tmp_path, obj_func_3d) -> None:
    opt = OptimizerEDBOp(venv_edbop)
    config = config = {
        "continuous_feature_names": ["f1", "f2"],
        "continuous_feature_bounds": [[-1, 1], [-1, 1]],
        "continuous_feature_resolutions": [0.1, 0.1],
        "categorical_feature_names": ["f3"],
        "categorical_feature_values": [[0, 1, 2]],
        "direction": ["min"],
        "budget": 10,
        "objectives": ["yield"],
    }

    result = opt.predict([], 0, tmp_path, config, obj_func_3d)  # noqa: F841

    # We're not verifying the value, since randomness can affect this
    # assert result == [0, 0]
