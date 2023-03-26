
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
for use in computational physics, chemistry, material science. These potentials are necessary for conducting 
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

.. note::
        This package is under heavy development and the current focus is on the implementation of high-dimensional 
        neural network potential (HDNNP) proposed by Behler et al. 
        (`2007 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401>`_).


Machine learning interatomic potential
--------------------------------------
Ab initio methods (e.g., density functional theory) provide accurate predictions of 
the electronic structure and energy of molecules, 
but they are computationally expensive and limited to small systems. 
On the other hand, molecular dynamics simulations require an accurate interatomic potential to describe 
the interactions between atoms and enable simulation of larger systems over longer timescales. 
To overcome this limitation, data-intensive approach using Machine learning methods is 
beginning to emerge as a different paradigm. 
Machine learning-based interatomic potentials can provide an accurate and efficient alternative 
to ab initio calculations for molecular dynamics simulations.
To construct an ML-based interatomic potential, 
one can collect a dataset of atomic positions and corresponding energies or forces from ab initio calculations. 
This dataset can then be used to train an ML model, such as a neural network, to predict the energy or 
forces for a given set of atomic positions. The accuracy of the ML model can be validated by comparing its 
predictions to the ab initio reference data.

After the ML potential has been trained and validated, 
it becomes possible to use it in molecular dynamics simulations of larger systems 
far beyond what is possible with direct ab initio molecular dynamics. 
Training ML-based interatomic potentials with ab initio reference data offers an 
accurate and computationally efficient technique for performing large-scale simulations.


Atomic environment descriptor
-----------------------------
Direct atomic positions are not suitable for machine learning-based interatomic potentials 
because they are not invariant under translation, rotation, and permutation. 
Translation refers to moving the entire system in space, rotation refers to rotating the system around an axis, 
and permutation refers to exchanging the positions of two or more atoms in the system.
To overcome this issue, *atomic descriptors* are used to encode information about the relative positions 
of the atoms in the system, such as distances between atoms or angles between bonds. 

An atomic descriptor is an effective representation of chemical environment of each individual atom
which provides a way to encode atomic properties such as the atomic position and bonding environment 
into a numerical form that can be used as input to an ML model.
This enables ML model to learn the complex relationships between atomic 
properties and their interactions in a more efficient and accurate way 
than traditional interatomic potential models.


Training a potential
--------------------
Here's steps involved in using ab initio reference data to train a ML potential:

1. Collect a dataset of atomic positions and corresponding energies or forces, for example from DFT calculations.

2. Select and calculate descriptor values for all atoms in the dataset.

3. Split the dataset into training, validation, and testing sets.

4. Define the architecture of the potential and relevant parameters.

5. Train the neural network potential on the training set using the input descriptors and the target energy and force values.

6. Validate the accuracy of the ML potential on the validation set by comparing its predictions to the DFT reference data.

7. Use the trained potential to perform molecular dynamics simulations of larger systems.


Examples
--------

-----------------------------
Defining an atomic descriptor
-----------------------------
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

        # Compute descriptor values
        descriptor(structure)

        # Compute gradient
        descriptor.grad(structure, atom_index=0)


-------------------------------------
Training a machine learning potential
-------------------------------------

This example illustrates how to quickly create a `high-dimensional neural network 
potential` (`HDNNP`_) instance from an in input setting files and train it on input structures. 
The trained potential can then be used to evaluate the energy and force components for new structures.

.. _HDNNP: https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00868


.. code-block:: python

        from jaxip.datasets import RunnerStructureDataset
        from jaxip.potentials import NeuralNetworkPotential

        # Read atomic data
        structures = RunnerStructureDataset("input.data")
        structure = structures[0]

        # Instantiate potential from input settings file
        nnp = NeuralNetworkPotential.create_from("input.nn")

        # Fit descriptor scaler and model weights
        nnp.fit_scaler(structures)
        nnp.fit_model(structures)
        nnp.save()

        # Or loading from files
        #nnp.load()

        # Total energy
        nnp(structure)

        # Force components
        nnp.compute_force(structure)


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
