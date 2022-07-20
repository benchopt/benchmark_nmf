
NMF Benchopt Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of Nonnegative Matrix Factorization:


$$\\min_{W\\in \\mathbb{R}^{m\\times r}_+, H\\in \\mathbb{R}^{r\\times n}_+} f(X, WH)$$


where $m, n$ stand for respectively for the number of rows and columns of the data matrix $X$ which may have negative entries, 

$$X \\in \\mathbb{R}^{m \\times n}$$

In short, matrix $X$ is approximated by a low rank matrix $WH$ where each low-rank factor $W$ and $H$ have nonnegative entries, which makes NMF a part-based decomposition.

The rank for the NMF must be provided in the dataset. Several values may be specified, but the responsability of chosing candidate rank values by default does not fall on the solvers, nor the objective.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/cohenjer/benchmark_nmf
   $ benchopt run benchmark_nmf

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_nmf -s apg -d simulated --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/cohenjer/benchmark_nmf/workflows/Tests/badge.svg
   :target: https://github.com/cohenjer/benchmark_nmf/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
