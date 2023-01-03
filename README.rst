
.. image:: docs/images/logo.png
        :alt: logo
        
=====
MLPOT
=====

**A machine-learning framework for development of interatomic potentials**

.. image:: https://img.shields.io/pypi/v/mlpot.svg
        :target: https://pypi.python.org/pypi/mlpot

.. image:: https://img.shields.io/travis/hghcomphys/mlpot.svg
        :target: https://travis-ci.com/hghcomphys/mlpot

.. image:: https://readthedocs.org/projects/mlpot/badge/?version=latest
        .. :target: https://mlpot.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. * Free software: GNU General Public License v3
.. * Documentation: https://mlpot.readthedocs.io.


What is it? 
-----------
MLPOT is a Python library to facilitate development of the emerging machine-learning (ML) 
interatomic potentials in computational physics and chemistry. 
Such potentials are essential for performing large-scale molecular dynamics (MD) simulations 
of complex materials at the atomic scale and with ab initio accuracy.

Why MLPOT?
----------
* Offers a generic and flexible design simplifies introducing atomic descriptors and potentials
* Utilizes `autograd` that makes definition of new descriptors quite easy
* Pythonic design with an optimized implementation using just-in-time (JIT) compilations
* Supports GPU-computing that can speeds up preprocessing steps and model trainings order(s) of magnitude

Important
---------

MLPOT is not a molecular dynamics (MD) simulation package but a framework to 
develop ML-based potentials used for the MD simulations.

.. note::
        This library is under heavy development and the current focus is on the implementation of high-dimensional 
        neural network potential (HDNNP) proposed by Behler et al. (`2007 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401>`_).



Examples
--------

-----------------------------
Defining an atomic descriptor
-----------------------------

The below example shows how to define a vector of Atomic-centered Symmetry Functions
(`ACSF`_) for an element.
The defined descriptor can be calculated on a given structure and the evaluated vector of descriptor values are eventually used for constructing ML potentials.

.. _ACSF: https://aip.scitation.org/doi/10.1063/1.3553717

.. code-block:: python

        from mlpot.datasets import RunnerStructureDataset
        from mlpot.descriptors import CutoffFunction, G2, G3
        from mlpot.descriptors import ACSF
        

        # Read atomic structure dataset
        structures = RunnerStructureDataset('input.data')
        structure = structures[0]

        # Define descriptor and adding radial and angular symmetry functions
        descriptor = ACSF(element='H')
        cfn = CutoffFunction(r_cutoff=12.0, cutoff_type='tanh')
        descriptor.add( G2(cfn, eta=0.5, r_shift=0.0), 'H')
        descriptor.add( G3(cfn, eta=0.001, zeta=2.0, lambda0=1.0, r_shift=12.0), 'H', 'O')

        # Calculate descriptor values
        values = descriptor(structure)


--------------------
Training a potential
--------------------

This example demonstrates how to quickly create a high-dimensional neural network 
potential `HDNNP`_ and training on the input structures. The energy and force components 
can be evaluated for (new) structures from the trained potential.

.. _HDNNP: https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00868


.. code-block:: python

        from mlpot.datasets import RunnerStructureDataset
        from mlpot.potentials import NeuralNetworkPotential

        # Atomic data
        structures = RunnerStructureDataset("input.data")

        # Potential
        nnp = NeuralNetworkPotential("input.nn")

        # Descriptor
        nnp.fit_scaler(structures)
        #nnp.load_scaler()

        # Train
        nnp.fit_model(structures)
        #nnp.load_model()

        # Predict energy and force components
        structure = structures[0]
        energy = nnp(structure)
        force = nnp.compute_force(structure)


.. Credits
.. -------

.. This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. .. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. .. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
