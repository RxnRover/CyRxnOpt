.. _supported-algorithms:

Supported Algorithms
====================

Our platform supports a variety of state-of-the-art optimization algorithms,
ranging from classical methods to modern machine learning-driven approaches.
These algorithms enable efficient exploration of reaction condition landscapes,
balance local and global search strategies, and facilitate multi-objective
decision-making.

Nelder-Mead Simplex
-------------------

The Nelder-Mead Simplex method [nelder1965]_ is a **derivative-free, local
optimization algorithm** widely used for smooth, low-dimensional problems. It
iteratively refines a simplex (a polytope with n+1 vertices in n-dimensional
space) through reflection, expansion, contraction, and shrink steps to approach
a local minimum. While efficient for unimodal surfaces, it may converge to local
rather than global optima.

References
~~~~~~~~~~

.. [nelder1965] Nelder, J. A.; Mead, R. A Simplex Method for Function
    Minimization. *The Computer Journal* **1965**, *7* (4), 308--313. DOI:
    `10.1093/comjnl/7.4.308 <https://doi.org/10.1093/comjnl/7.4.308>`__.

SQSnobFit
---------

SQSnobFit [sqsnobfit]_ is a Python implementation by scikit-quant of Huyer and
Neumaier's original SNOBFIT (Stable Noisy Optimization by Branch and Fit)
algorithm. [huyer2008]_ SNOBFIT is a **global optimization algorithm**
particularly well-suited for expensive, noisy black-box functions. It combines
local quadratic model fitting with global space-partitioning to balance
**exploration and exploitation**.

References
~~~~~~~~~~

.. [huyer2008] Huyer, W.; Neumaier, A. SNOBFIT -- Stable Noisy Optimization by
    Branch and Fit. *ACM Transactions on Mathematical Software* **2008**, *35*
    (2), 1--25. DOI: `10.1145/1377612.1377613
    <https://doi.org/10.1145/1377612.1377613>`__.

EDBO+
-----

EDBO+ [garridotorres2022]_ is an **open-source, Bayesian optimization
framework** specifically designed for chemical reaction optimization based on
the original EDBO (Experimental Design via Bayesian Optimization) algorithm
[shields2021]_ from the Doyle group. It leverages Gaussian process models with
acquisition functions (e.g., Expected Improvement, qEI, and EHVI for
multi-objective settings) to identify new experiments that are most informative.
EDBO+ has been successfully applied to multi-objective optimization of reaction
yields, selectivity, and sustainability metrics.

References
~~~~~~~~~~

.. [shields2021] Shields, B. J.; Stevens, J.; Li, J.; Parasram, M.; Damani, F.;
    Alvarado, J. I. M.; Janey, J. M.; Adams, R. P.; Doyle, A. G. Bayesian
    Reaction Optimization as a Tool for Chemical Synthesis. *Nature* **2021**,
    *590* (7844), 89--96. DOI: `10.1038/s41586-021-03213-y
    <https://doi.org/10.1038/s41586-021-03213-y>`__.

.. [garridotorres2022] Garrido Torres, J. A.; Lau, S. H.; Anchuri, P.; Stevens,
    J. M.; Tabora, J. E.; Li, J.; Borovika, A.; Adams, R. P.; Doyle, A. G. A
    Multi-Objective Active Learning Platform and Web App for Reaction
    Optimization. *J. Am. Chem. Soc.* **2022**, *144* (43), 19999--20007. DOI:
    `10.1021/jacs.2c08592 <https://doi.org/10.1021/jacs.2c08592>`__.

AMLRO
-----

AMLRO (Active Machine Learning Reaction Optimizer) is our **in-house active
learning framework** designed for **reaction space optimization**. [amlro]_ It
integrates regression models with iterative candidate selection. AMLRO is
capable of handling **continuous and categorical features**, dynamic objectives
with directions (minimization or maximization), and user feedback loops.

- **Status:** Source code is currently under development. Public release is
  planned, and the project will be hosted on GitHub (expected link: *TBA*).

References
~~~~~~~~~~

.. [amlro] Manuscript in preparation.
