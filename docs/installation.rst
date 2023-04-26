.. highlight:: shell

============
Installation
============

Requirements
------------

.. This package has the following dependencies:
.. * `JAX`_: An `Autograd` and `XLA` framework for high-performance numerical computing

`JAX`_ is the core of this package and its installation is necessary to use it.
The CPU version of JAX is included, as dependency, in the default installation of Jaxip.

For machines with an NVIDIA **GPU**, it is recommended to install JAX separately 
via `Conda` using the following command (only Linux users):

.. code-block:: bash

    $ conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia

Please refer to the `JAX Install Guide`_ for full installation instructions.


.. _JAX: https://github.com/google/jax
.. _`JAX Install Guide`: https://github.com/google/jax#installation


Stable release
--------------

To install Jaxip, run this command in your terminal:

.. code-block:: console

    $ pip install jaxip

This is the preferred method to install Jaxip, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Jaxip can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/hghcomphys/jaxip

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/hghcomphys/jaxip/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/hghcomphys/jaxip
.. _tarball: https://github.com/hghcomphys/jaxip/tarball/main
