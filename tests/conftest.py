"""
Dummy conftest.py for cyrxnopt.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def obj_func_2d() -> Callable[[list[float]], float]:
    def obj_func(xs: list[float]) -> float:
        return xs[0] ** 2 + xs[1] ** 2

    return obj_func


@pytest.fixture
def obj_func_3d() -> Callable[[list[float]], float]:
    def obj_func(xs: list[float]) -> float:
        return xs[0] ** 2 + xs[1] ** 2 + xs[2]

    return obj_func


@pytest.fixture(scope="session")
def test_assets_path():
    this_file = Path(__file__)

    return this_file.parent / "cyrxnopt" / "test_assets"
