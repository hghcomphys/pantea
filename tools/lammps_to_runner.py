import sys

from adaptor import (ANGSTROM_TO_BOHR, EV_TO_HARTREE, RuNNerAdaptorForLAMMPS,
                     UnitConversion)

# convert compatible units for RuNNer package
uc = UnitConversion(energy_conversion=EV_TO_HARTREE, length_conversion=ANGSTROM_TO_BOHR)

# Convert lammps dataset into RuNNer input data format
args = sys.argv
assert len(args) == 3, "Expected input and output file names!"
lammps_filename = args[1]  # 'airebo.data'
runner_filename = args[2]  # 'airebo.input.data'

data = RuNNerAdaptorForLAMMPS().read_lammps(lammps_filename, {"1": "H", "2": "O"}, uc)
print("number of samples:", data.dataset.number_of_samples)
print("number of atoms in each sample:", data.dataset.samples[0].number_of_atoms)
data.write_runner(runner_filename)

# Read predicted force
# nnenergy = RunnerAdaptor().read_nnenergy("RuNNer/mode3.out")
# print(nnenergy)
