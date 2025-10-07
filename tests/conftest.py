"""
Dummy conftest.py for cyrxnopt.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

from typing import List

import pytest


@pytest.fixture
def obj_func_2d():
    def obj_func(xs: List[float]) -> float:
        return xs[0] ** 2 + xs[1] ** 2

    return obj_func
