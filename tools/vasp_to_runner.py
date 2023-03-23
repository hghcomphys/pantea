import sys

from adaptor import UnitConversion, RuNNerAdaptorForVASP
from adaptor import EV_TO_HARTREE, ANGSTROM_TO_BOHR

# convert compatible units for RuNNer package
uc = UnitConversion(energy_conversion=EV_TO_HARTREE, length_conversion=ANGSTROM_TO_BOHR)

# Convert lammps dataset into RuNNer input data format
args = sys.argv
assert len(args) == 2, "Expected input and output file names!"
filename = args[1]  # 'airebo.data'

uc = UnitConversion(energy_conversion=EV_TO_HARTREE, length_conversion=ANGSTROM_TO_BOHR)
data = (
    RuNNerAdaptorForVASP()
    .read_vasp(symbol_list=["H", "B", "N", "O"], uc=uc)
    .write_runner(filename=filename)
)
# print("number of atoms in each sample:", data.dataset.samples[0].number_of_atoms)
