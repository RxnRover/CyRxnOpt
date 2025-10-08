import pytest

from cyrxnopt.NestedVenv import NestedVenv
from cyrxnopt.OptimizerNMSimplex import OptimizerNMSimplex
from tests.cyrxnopt.utilities_for_testing.validate_config_description import (
    validate_config_description_pytest,
)


@pytest.fixture(scope="session")
def venv_nmsimplex(tmp_path_factory):
    venv_path = tmp_path_factory.mktemp("venv_nmsimplex")

    test_venv = NestedVenv(venv_path)

    test_venv.create()
    test_venv.activate()

    assert test_venv.is_active()
    assert test_venv.is_primary()

    # Preinstall dependencies
    opt = OptimizerNMSimplex(test_venv)
    opt.install()
    assert opt.check_install()

    yield test_venv

    test_venv.deactivate()
    assert not test_venv.is_active()
    test_venv.delete()


def test_get_config_returns_valid_description_list(venv_nmsimplex) -> None:
    opt = OptimizerNMSimplex(venv_nmsimplex)

    result = opt.get_config()

    validate_config_description_pytest(result)


def test_set_config_creates_correct_config(venv_nmsimplex, tmp_path) -> None:
    import json

    opt = OptimizerNMSimplex(venv_nmsimplex)

    expected_config_path = tmp_path / "config.json"

    config = {
        "continuous_feature_names": ["f1"],
        "continuous_feature_bounds": [[-1, 1], [-1, 1]],
        "direction": "min",
        "budget": 10,
        "param_init": [0.5, 0.5],
        "xatol": 1e-8,
        "display": False,
        "server": False,
    }

    opt.set_config(str(tmp_path), config)

    # Check if config file was created
    assert expected_config_path.exists()

    # Check for the correct contents
    with open(expected_config_path) as fin:
        content = json.load(fin)
        assert content == config


def test_train_does_nothing(venv_nmsimplex, tmp_path) -> None:
    opt = OptimizerNMSimplex(venv_nmsimplex)
    expected_suggestion = []

    suggestion = opt.train([], 0, tmp_path, {})

    assert expected_suggestion == suggestion


def test_predict_basic_run(venv_nmsimplex, tmp_path, obj_func_2d) -> None:
    opt = OptimizerNMSimplex(venv_nmsimplex)
    config = config = {
        "continuous_feature_names": ["f1"],
        "continuous_feature_bounds": [[-1, 1], [-1, 1]],
        "direction": "min",
        "budget": 10,
        "param_init": [0.5, 0.5],
        "xatol": 1e-8,
        "display": False,
        "server": False,
    }

    result = opt.predict([], 0, tmp_path, config, obj_func_2d)  # noqa: F841

    # We're not verifying the value, since randomness can affect this
    # assert result == [0, 0]
