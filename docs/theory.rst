Theory
------

----------------------
Basics of ML potential
----------------------
Ab initio methods provide accurate predictions of the electronic structure and energy of molecules, 
but they are computationally expensive and limited to small systems. 
On the other hand, molecular dynamics (MD) simulations require an accurate interatomic potential to describe 
the interactions between atoms and enable simulation of larger systems over longer timescales. 
To overcome this limitation, data-intensive approach using Machine learning (ML) methods is 
beginning to emerge as a different paradigm. 

Machine learning-based potentials are shown to provide an accurate and efficient alternative 
to ab initio calculations for MD simulations
Behler et al. (`2021 <https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00868>`_).
To construct such potentials, 
one can collect a dataset of atomic positions and corresponding energies or forces 
from ab initio calculations. 
This dataset can then be used to train an ML model, for example a neural network, to predict the energy or 
forces for a given set of atomic positions. The accuracy of the ML model can be validated by comparing its 
predictions to the ab initio reference data.
After an ML potential has been trained and validated, 
it becomes possible to use it in molecular dynamics simulations of larger systems 
far beyond what is possible with direct ab initio molecular dynamics. 
Training ML potentials with ab initio reference data offers an 
**accurate** and computationally **efficient** technique for performing large-scale simulations.

--------------------------
What is atomic descriptor?
--------------------------
Direct atomic positions are not suitable for machine learning potentials 
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

.. image:: https://github.com/hghcomphys/pantea/raw/main/docs/images/flowchart.drawio.png
    :alt: workflow
    :class: with-shadow
    :name: An illustration of ML-potential workflow in training and prediction modes.

------------------
Training procedure
------------------
Training a Machine Learning Potential involves using a dataset of atomic configurations with corresponding energies and forces, typically obtained from quantum mechanical calculations like density functional theory (DFT). The process begins by computing atomic descriptors that capture the local atomic environments. The potential model (e.g. neural network) is then trained on this data to learn the relationship between atomic positions and their associated energies or forces. After training, the ML potential is validated against a separate set of data to ensure its accuracy before being used for simulations, such as molecular dynamics, to study larger and more complex systems.