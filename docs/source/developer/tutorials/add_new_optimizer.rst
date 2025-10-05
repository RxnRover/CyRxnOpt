.. _add_new_optimizer:

Creating a New Optimizer Class
==============================

1. Make a copy of ``OptimizerTemplate.py`` and rename it as
   ``OptimizerName.py``, replacing ``Name`` with the name or an abbreviation of
   your optimizer. For example, if this template was being used to add support
   for the optimizer, Nelder-Mead Simplex, we may rename
   ``OptimizerTemplate.py`` to ``OptimizerNMSimplex.py``. The template file is
   located at: ``CyRxnOpt/src/CyRxnOpt/OptimizerTemplate.py``. You can download
   the file here: :download:`OptimizerTemplate.py
   </../../src/cyrxnopt/OptimizerTemplate.py>`
2. Change the class name to your optimizer class name, for example,
   ``OptimizerName``.

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 1
       :language: python

3. Add all the necessary libraries required for your optimizer to the
   ``__packages`` list in the template file.

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 3
       :language: python

4. Check that the ``install()`` and ``check_install()`` functions do not need to
   be modified:

   ``install()`` - Install function is designed to add all the necessary
   packages for given optimizer class. When we run optimizer class first time,
   Install function will first create new virtual environment, and then install
   all the packages from pip commands. When we use them install function will
   activate relevant virtual environment.

   ``check install()`` - This function is checking whether necessary packages
   are install or not and return Boolean output.

5. In the ``get_config()`` function, you need to update the configuration
   dictionary. This function is designed to maintain common configuration
   dictionary for all the optimization algorithms and It will helps to design
   common front-end for all the optimization algorithms. Here you can add more
   variables that are only used by your algorithm and it should dynamically
   change the front-end user interface.

   Add all the necessary configurations and variables required as user input to
   run your optimizer. Follow the same dictionary keys as other optimizer
   classes. Example get_config dictionary will be like this,

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 5-47
       :language: python

6. In the ``set_config()`` function, you need to add the necessary code to
   handle and initialize the optimizer. This function mainly handles the steps
   before the optimizer cycle begins. Code lines that you need to generate your
   reaction space, handle the format of configuration data into your algorithm
   format, and generate initial files will go inside ``set_config()`` function.
7. If your optimizer requires training steps, add the necessary code for
   training inside the ``train()`` function. For example, AMLRO is required to
   train the initial ML model before starting the active learning prediction.
   Therefore, the AMLRO optimizer class ``train()`` function includes code lines
   to generate training dataset.
8. In the ``predict()`` function, add the code lines for predicting the optimum
   reaction conditions. This function will return the optimal reaction
   conditions that should be run in next cycle. Actual optimization
   loop/prediction step, code should implement here.
9. In the ``_import_deps()`` function, write necessary package import lines.
   Each package should be added to the ``_imports`` dictionary, and for the
   dictionary key, use the package name. As a example, ``numpy`` and ``pandas``
   are imported here:

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 49-54
       :language: python

   Then, when you want to use the imported library, you can access it through
   the ``self._imports`` dictionary.

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 56-57
       :language: python

10. Depending on your optimizer workflow, add more class methods as necessary.
    Refer to how existing optimizer classes are defined, for guidance.

Adding new Optimizer to CyRxnOpt
--------------------------------

<<<<<<< HEAD
   Then, when you want to use the imported library, you can access it through the
   `self._imports` dictionary.
=======
After implementing your optimizer class, update the ``OptimizerController.py``
file to use your optimizer.
>>>>>>> 592d385 (Apply docstrfmt, fixing code syntax errors by moving to separate file)

1. At the top of this file, add the optimizer import line, replacing
   ``OptimizerName`` with the name of your optimizer class:

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 59
       :language: python

2. Update the ``get_optimizer()`` function to include your optimizer:

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 61-62
       :language: python

3. All the function parameters should match with the corresponding abstract
   function defined in ``OptimizerABC``. If you want to add new parameter for
   any function first add that into the optimizer controller function and give
   default value as None. For Eexample, if your algorithm predict function
   requires a new parameter call, learning rate.

   .. literalinclude:: add_new_optimizers.py.txt
       :lines: 64-73
       :language: python
