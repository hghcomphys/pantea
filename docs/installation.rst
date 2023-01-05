.. highlight:: shell

============
Installation
============

Requirements
------------

The following packages are required:

* `JAX`_: Composable transformations of Python+NumPy programs
* `FLAX`_: A neural network library and ecosystem for JAX designed for flexibility
* `ASE`_: Atomic Simulation Environment


.. _JAX: https://github.com/google/jax
.. _FLAX: https://github.com/google/flax
.. _ASE: https://wiki.fysik.dtu.dk/ase/



Stable release
--------------

To install JAXIP, run this command in your terminal:

.. code-block:: console

    $ pip install jaxip

This is the preferred method to install JAXIP, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for JAXIP can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/hghcomphys/jaxip

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/hghcomphys/jaxip/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/hghcomphys/jaxip
.. _tarball: https://github.com/hghcomphys/jaxip/tarball/master
