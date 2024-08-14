=====
Usage
=====

To use Pantea in a project:

.. code:: python

    import pantea


Just-In-Time (JIT) compilation is a technique used by `Pantea` to improve performance by compiling code at runtime. 
It's important to note that the first time the function is called, it can be significantly slower than subsequent calls.
This is because `JAX` library under the hood needs to compile the function's 
code and optimize it for the specific inputs it receives. 

To mitigate this overhead cost, it's recommended to call 
the function once with representative input data before timing or benchmarking it. 
This is commonly referred to as a `warm-up` call.

Warm-up call

.. code:: python

    %time potential(s)
    CPU times: user 6.59 s, sys: 410 ms, total: 7 s
    Wall time: 5.12 s

    %time potential.compute_forces(s)
    CPU times: user 35.8 s, sys: 437 ms, total: 36.3 s
    Wall time: 24.4 s


After JIT compilation

.. code:: python

    %time potential(s)
    CPU times: user 52.9 ms, sys: 8.59 ms, total: 61.5 ms
    Wall time: 51.4 ms

    %time potential.compute_forces(s)
    CPU times: user 135 ms, sys: 32.4 ms, total: 167 ms
    Wall time: 156 ms


Examples
========
Below you will find a list of examples and tutorials to help you in using `Pantea`.

* `Getting started with Pantea <https://pantea.readthedocs.io/en/latest/notebooks/getting_started.html>`_
* `Potential traning <https://pantea.readthedocs.io/en/latest/notebooks/potential_training.html>`_