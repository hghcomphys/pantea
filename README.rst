
.. .. image:: docs/images/logo.png
..         :alt: logo
        
=====
JAXIP
=====


**JAX-based Interatomic Potential**

.. image:: https://img.shields.io/pypi/v/jaxip.svg
        :target: https://pypi.python.org/pypi/jaxip

.. image:: https://img.shields.io/travis/hghcomphys/jaxip.svg
        :target: https://travis-ci.com/hghcomphys/jaxip

.. image:: https://readthedocs.org/projects/jaxip/badge/?version=latest
        :target: https://jaxip.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Description
-----------
JAXIP is an optimized Python library on basis of `JAX`_ that helps 
in the development of emerging machine learning interatomic potentials 
for use in computational physics and chemistry. These potentials are necessary for conducting 
large-scale molecular dynamics simulations of complex materials at the atomic level with ab initio accuracy.

JAXIP is designed to *develop* potentials for use in molecular dynamics simulations, 
rather than a package for *performing* the simulations themselves.




.. _JAX: https://github.com/google/jax


Documentation: https://jaxip.readthedocs.io.

Main features
-------------
* The design of JAXIP is `simple` and `flexible`, which makes it easy to incorporate atomic descriptors and potentials 
* It uses `autograd` to make defining new descriptors straightforward
* JAXIP is written purely in Python and optimized with `just-in-time` (JIT) compilation.
* It also supports `GPU-accelerated` computing, which can significantly speed up preprocessing and model training.

.. Important
.. ---------

.. note::
        This package is under heavy development and the current focus is on the implementation of high-dimensional 
        neural network potential (HDNNP) proposed by Behler et al. 
        (`2007 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401>`_).


Examples
--------

-----------------------------------------
Defining an atomic environment descriptor
-----------------------------------------

The following example shows how to create an array of `atomic-centered symmetry functions`
(`ACSF`_) for a specific element. 
This descriptor can be applied to a given structure to produce the 
descriptor values that are required to build machine learning potentials.

.. _ACSF: https://aip.scitation.org/doi/10.1063/1.3553717

.. code-block:: python

        from jaxip.datasets import RunnerStructureDataset
        from jaxip.descriptors import ACSF
        from jaxip.descriptors.acsf import CutoffFunction, G2, G3
        

        # Read atomic structure dataset (e.g. water molecules)
        structures = RunnerStructureDataset('input.data')
        structure = structures[0]

        # Define ACSF descriptor for hydrogen element 
        descriptor = ACSF(element='H')
        
        # Add radial and angular symmetry functions
        cfn = CutoffFunction(r_cutoff=12.0, cutoff_type='tanh')
        descriptor.add( G2(cfn, eta=0.5, r_shift=0.0), 'H')
        descriptor.add( G3(cfn, eta=0.001, zeta=2.0, lambda0=1.0, r_shift=12.0), 'H', 'O')

        # Calculate descriptor values
        values = descriptor(structure)


.. Outputs:

.. .. code-block:: bash

..         >> values.shape
..         (128, 2)

..         >> values[:3]
..         DeviceArray([[1.9689142e-03, 3.3253882e+00],
..                 [1.9877939e-03, 3.5034561e+00],
..                 [1.5204106e-03, 3.5458331e+00]], dtype=float32)

 
.. The **gradient** of the defined descriptor can be obtained by using the `grad` method.

.. .. code-block:: bash

..         >> descriptor.grad(structure, acsf_index=0, atom_index=2)
..         DeviceArray([[ 0.00576886,  0.00219238, -0.00206053]], dtype=float32)


-------------------------------------
Training a machine learning potential
-------------------------------------

This example illustrates how to quickly create a `high-dimensional neural network 
potential` (`HDNNP`_) and train it on input structures. 
The trained potential can then be used to evaluate the energy and force components for new structures.

.. _HDNNP: https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00868


.. code-block:: python

        from jaxip.datasets import RunnerStructureDataset
        from jaxip.potentials import NeuralNetworkPotential

        # Atomic data
        structures = RunnerStructureDataset("input.data")
        structure = structures[0]

        # Potential
        nnp = NeuralNetworkPotential("input.nn")

        # Descriptor
        nnp.fit_scaler(structures)
        #nnp.load_scaler()

        # Train
        nnp.fit_model(structures)
        #nnp.load_model()

        # Predict the total energy and force components
        total_energy = energy = nnp(structure)
        force = nnp.compute_force(structure)


License
-------

.. _license-file: LICENSE


This project is licensed under the GNU General Public License (GPL) version 3 - 
see the LICENSE file for details.

.. The GPL v3 is a free software license that allows users to share and modify the software, 
.. as long as the original copyright notice and license are included and the modified versions 
.. are marked as such. The GPL v3 also requires that users receive the source code or have the 
.. ability to obtain it, and that they are made aware of their rights under the license.

.. For more information about the GPL v3 license, please see the full text of the license in the "LICENSE" file.



.. Credits
.. -------

.. This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. .. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. .. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
