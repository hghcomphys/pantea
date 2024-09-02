
.. .. image:: docs/images/logo.png
.. :alt: logo
        
======
Pantea
======


.. image:: https://img.shields.io/pypi/v/pantea.svg
        :target: https://pypi.python.org/pypi/pantea

.. image:: https://github.com/hghcomphys/pantea/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/hghcomphys/pantea/blob/main/.github/workflows/tests.yml

.. image:: https://readthedocs.org/projects/pantea/badge/?version=latest
        :target: https://pantea.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Description
-----------
Pantea is an optimized Python library based on Google `JAX`_ that enables 
development of machine learning interatomic potentials   
for use in computational material science. 
These potentials are particularly necessary for conducting large-scale molecular 
dynamics simulations of complex materials with ab initio accuracy.

.. _JAX: https://github.com/google/jax


See `documentation <https://pantea.readthedocs.io/en/latest/theory.html>`_ for more information.


-------------
Main Features
-------------
* The design of Pantea is `simple` and `flexible`, which makes it easy to incorporate atomic descriptors and potentials. 
* It uses `automatic differentiation` to make defining new descriptors straightforward.
* Pantea is written purely in Python and optimized with `just-in-time` (JIT) compilation.
* It also supports `GPU` computing, which can significantly speed up preprocessing and model training.

.. warning::
        This package is under development and the current focus is on the implementation of high-dimensional 
        neural network potential (HDNNP) proposed by Behler et al. 
        (`2007 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401>`_).


Installation
------------
To install Pantea, run this command in your terminal:

.. code-block:: console

    $ pip install pantea

For machines with an NVIDIA **GPU** please follow the
`installation <https://pantea.readthedocs.io/en/latest/installation.html>`_ 
instruction on the documentation. 


Examples
--------

--------------------
I. Descriptor (ACSF)
--------------------
Atom-centered Symmetry Function (`ACSF`_) descriptor captures information about the distribution of neighboring atoms around a 
central atom by considering both radial (two-body) and angular (three-body) symmetry functions. 
The values obtained from these calculations represent a fingerprint of the local atomic environment and can be used in various machine learning potentials. 

Script below demonstrates the process of defining multiple symmetry functions
for an element, which can be utilized to evaluate the descriptor values for any structure. 

.. _ACSF: https://aip.scitation.org/doi/10.1063/1.3553717


.. code-block:: python

        from pantea.datasets import Dataset
        from pantea.descriptors import ACSF
        from pantea.descriptors.acsf import CutoffFunction, NeighborElements, G2, G3

        # Read atomic structure dataset (e.g. water molecules)
        structures = Dataset.from_runner("input.data")
        structure = structures[0]
        print(structure)
        # >> Structure(natoms=12, elements=('H', 'O'), dtype=float64)

        # Define an ACSF descriptor for hydrogen atoms
        # It includes two radial (G2) and angular (G3) symmetry functions
        cfn = CutoffFunction.from_type("tanh", r_cutoff=12.0)
        g2 = G2(cfn, eta=0.5, r_shift=0.0)
        g3 = G3(cfn, eta=0.001, zeta=2.0, lambda0=1.0, r_shift=12.0)

        descriptor = ACSF(
                central_element='H',
                radial_symmetry_functions=(
                        (g2, NeighborElements('H')),
                ),
                angular_symmetry_functions=(
                        (g3, NeighborElements('H', 'O')),
                ),
        )

        print(descriptor)
        # >> ACSF(central_element='H', num_symmetry_functions=2)

        values = descriptor(structure)
        print("Descriptor values:\n", values)
        # >> Descriptor values:
        # [[0.01952943 1.13103234]
        #  [0.01952756 1.04312263]
        # ...
        #  [0.00228752 0.41445455]]

        gradient = descriptor.grad(structure)
        print("Descriptor gradient:\n", gradient)
        # >> Descriptor gradient:
        # [[[ 4.64523585e-02 -5.03786078e-02 -6.14621389e-02]
        #   [-1.04818547e-01 -1.84170755e-02  4.76021411e-02]]
        #  [[-9.67003098e-03 -5.45498827e-02  6.32422634e-03]
        #   [-1.59613454e-01 -5.94085256e-02  1.72978932e-01]]
        # ...
        #  [[-1.36223042e-03 -8.02832759e-03 -6.08306094e-05]
        #   [ 1.29199076e-02 -9.58762344e-03 -9.12714216e-02]]] 


-------------------
II. Potential (NNP)
-------------------
This example illustrates how to quickly create a `high-dimensional neural network 
potential` (`HDNNP`_) instance from an input setting file.

.. _HDNNP: https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00868

.. code-block:: python

        from pantea.datasets import Dataset
        from pantea.potentials import NeuralNetworkPotential

        # Dataset: reading structures from RuNNer input data file
        structures = Dataset.from_runner("input.data")
        structure = structures[0]

        # Potential: creating a NNP from the RuNNer potential file
        nnp = NeuralNetworkPotential.from_runner("input.nn")
        nnp.load()  # this will require loading scaler and model parameter files.

        total_energy = nnp(structure)
        print(total_energy)

        forces = nnp.compute_forces(structure)
        print(forces)


-------------------
III. Training (NNP) 
-------------------
This example shows the process of training a NNP potential on input structures. 
The trained potential can then be used to evaluate the energy and force components for new structures.

.. code-block:: python

        from pantea.datasets import Dataset
        from pantea.potentials import NeuralNetworkPotential
        from pantea.potentials.nnp import NeuralNetworkPotentialTrainer        

        # Dataset: reading structures from RuNNer input data file
        structures = Dataset.from_runner("input.data", persist=True)
        structures.preload()

        # Potential: creating a NNP from the RuNNer configuration file
        nnp = NeuralNetworkPotential.from_runner("input.nn")

        # Trainer: initializing a trainer from the NNP potential 
        trainer = NeuralNetworkPotentialTrainer.from_runner(potential=nnp)
        trainer.fit_scaler()
        trainer.fit_model()

        trainer.save()  # this will save scaler and model parameters into files


.. warning::
        Please note that the above examples are just for demonstration. 
        For training a NNP model in real world we surely need larger samples of data.

Download example input files from `here <https://drive.google.com/drive/folders/1vABOndAia41Bn0v1jPaJZmVGnbjg8UPE?usp=sharing>`_.


License
-------
This project is licensed under the GNU General Public License (GPL) version 3 - 
see the `LICENSE <https://github.com/hghcomphys/pantea/blob/main/LICENSE>`_ file for details.
