import sys

from adaptor import UnitConversion, RuNNerAdaptorForVASP
from adaptor import EV_TO_HARTREE, ANGSTROM_TO_BOHR

# convert compatible units for RuNNer package
uc = UnitConversion(energy_conversion=EV_TO_HARTREE, length_conversion=ANGSTROM_TO_BOHR)

# Convert lammps dataset into RuNNer input data format
args = sys.argv
assert len(args) == 3, "Expected input filename and number of samples!"
filename = args[1]  # 'airebo.data'
nsamples = int(args[2])

uc = UnitConversion(energy_conversion=EV_TO_HARTREE, length_conversion=ANGSTROM_TO_BOHR)
RuNNerAdaptorForVASP().read_runner(filename).write_poscar(
    symbol_list=["H", "B", "N", "O"],
    uc=uc.inverse,
    number_of_strucure=nsamples,
    seed=1234,
)
