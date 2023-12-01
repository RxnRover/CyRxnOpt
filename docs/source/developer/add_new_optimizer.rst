################################
 Creating a New Optimizer Class
################################

#. Make a copy of `OptimizerTemplete.py` and rename it as
   `OptimizerName.py`. The template file is located at:
   `CyRxnOpt/src/CyRxnOpt/OptimizerTemplete.py`

.. literalinclude:: /../../src/cyrxnopt/OptimizerTemplete.py

2. Change the class name to your optimizer class name, for example,
   `OptimizerName`.

#. Add all the necessary libraries required for your optimizer to the
   `__packages` list in the template file.

#. Check that the `install` and `uninstall` functions do not need
   modification.

#. In the `get_config` function, update the configuration dictionary.
   Add all the necessary configurations and variables required as user
   input to run your optimizer. Follow the same dictionary keys as other
   optimizer classes.

#. In the `set_config` function, add the necessary code to handle and
   initialize the optimizer.

#. If your optimizer requires training steps, add the necessary code for
   training inside the `train()` function.

#. In the `predict` function, add the code lines for predicting the
   optimum reaction conditions.

#. In the `_import_deps()` function, write necessary package import
   lines. Each package should be added to the `_imports` dictionary, and
   for the dictionary key, use the package name.

#. Depending on your optimizer workflow, add more child functions as
   necessary. Refer to how the original 4 optimizer classes are defined
   for guidance.

#############################
 Update Optimizer Controller
#############################

After implementing your optimizer class, update the
`OptimizerController.py` file to use your optimizer.

#. At the top of this file, add the optimizer import line:
      .. code:: python

         from CyRnxOpt.OptimizerName import OptimizerName

#. update `get_optimizer()` function if statements,
      .. code:: python

         elif optimizer_name == "name":
         optimizer = OptimizerName(venv)
