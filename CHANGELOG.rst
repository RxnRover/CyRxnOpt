Changelog
=========

Version 0.3.0
-------------

Breaking Changes
~~~~~~~~~~~~~~~~

- ``venv`` argument is now required in all ``OptimizerController.*`` functions.
- ``venv`` argument of ``set_config``, ``train``, and ``predict`` functions in
  ``OptimizerController`` was moved to the second position to better group
  arguments specific to the functions vs those passed to the optimizer class.
  This also makes all ``OptimizerController`` functions start with the same
  arguments.
- ``itr`` argument removed from ``OptimizerABC.train()``
- Added ``config`` and ``obj_func`` arguments to ``OptimizerABC.train()`` and
  ``OptimizerABC.predict()``

Features
~~~~~~~~

- Added mypy type checking
- Added basic software logging
- Unified optimizer method signatures (this required some reimplementation under
  the hood for AMLO)
- Added and updated testing for all optimizers

Bug Fixes
~~~~~~~~~

- Fixed incorrect or missing type hints
- Updated some docstring wording
- Fixed some incorrect or missing types in docstrings
