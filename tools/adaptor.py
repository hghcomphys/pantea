"""
A simple tool to covert RuNNer structure file into different formats.
"""

# ----------------------------------------------------------------------------
# Define utility constants, functions, and classes
# ----------------------------------------------------------------------------

ANGSTROM_TO_BOHR = 1.8897261328
EV_TO_HARTREE = 0.0367493254
KCALMOL_TO_HARTREE = 0.001593602


def read_and_tokenize_line(in_file):
    return next(in_file).rstrip("/n").split()


class UnitConversion:
    """A class for unit conversion of RuNNer package."""

    def __init__(self, energy_conversion=1.0, length_conversion=1.0):
        self.energy = energy_conversion
        self.length = length_conversion
        self.charge = 1.0
        self.force = energy_conversion / length_conversion

    @property
    def inverse(self):
        """A method that applies inverse unit conversion."""
        return UnitConversion(1.0 / self.energy, 1.0 / self.length)


# ----------------------------------------------------------------------------
# Setup classes for data
# ----------------------------------------------------------------------------


class AtomicData:
    """A class that holds atomic data such as positions, forces, total_energy, charges, etc."""

    def __init__(
        self,
        atomid=0,
        position=(0.0, 0.0, 0.0),
        symbol="X",
        charge=0.0,
        energy=0.0,
        force=(0.0, 0.0, 0.0),
    ):
        self.atomid = atomid
        self.position = position
        self.symbol = symbol
        self.charge = charge
        self.energy = energy
        self.force = force


class CollectiveData:
    """A class that holds collective quantities of simulated system such as total energy or charge."""

    def __init__(
        self,
        cell=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        total_energy=0.0,
        total_charge=0.0,
    ):
        self.cell = cell
        self.total_energy = total_energy
        self.total_charge = total_charge


class Sample:
    """A class that holds a list of atomic data and also collective data for a single sample."""

    def __init__(self):
        self.atomic = []
        self.collective = None

    @property
    def number_of_atoms(self):
        return len(self.atomic)

    @property
    def total_energy(self):
        tot = 0.0
        for atom in self.atomic:
            tot += atom.energy
        return tot

    @property
    def total_charge(self):
        tot = 0.0
        for atom in self.atomic:
            tot += atom.charge
        return tot

    def get_atoms_for_symbol(self, symbol):
        sel_atoms = []
        for atom in self.atomic:
            if atom.symbol == symbol:
                sel_atoms.append(atom)
        return sel_atoms

    def get_number_of_atoms_for_symbol(self, symbol):
        return len(self.get_atoms_for_symbol(symbol))


class DataSet:
    """This class holds a collection of samples."""

    def __init__(self):
        self.samples = []

    def append(self, sample):
        """Append a sample to list of samples."""
        self.samples.append(sample)

    @property
    def number_of_samples(self):
        return len(self.samples)


# ----------------------------------------------------------------------------
# Setup class for RuNNer adaptor
# ----------------------------------------------------------------------------


class RunnerAdaptor:
    """A bas class for conversion file formats of RuNNer package."""

    def __init__(self):
        self.dataset = DataSet()

    def clean(self):
        self.dataset = DataSet()

    def write_runner(self, filename):
        """Write RuNNer input data."""

        with open(filename, "w") as out_file:
            # loop over samples
            for sample in self.dataset.samples:
                out_file.write("begin\n")
                cell = [c for c in sample.collective.cell]
                out_file.write(
                    "lattice %.10f %.10f %.10f\n" % (cell[0], cell[1], cell[2])
                )
                out_file.write(
                    "lattice %.10f %.10f %.10f\n" % (cell[3], cell[4], cell[5])
                )
                out_file.write(
                    "lattice %.10f %.10f %.10f\n" % (cell[6], cell[7], cell[8])
                )
                # loop over atoms in a sample
                for atom in sample.atomic:
                    out_file.write("atom ")
                    out_file.write(
                        "%15.10f %15.10f %15.10f "
                        % tuple([pos for pos in atom.position])
                    )
                    out_file.write(
                        "%s %15.10f %15.10f "
                        % (atom.symbol, atom.charge, atom.energy * 0.0)
                    )
                    out_file.write(
                        "%15.10f %15.10f %15.10f\n" % tuple([frc for frc in atom.force])
                    )
                out_file.write("energy %.10f\n" % (sample.collective.total_energy))
                out_file.write("charge %.10f\n" % (sample.collective.total_charge))
                out_file.write("end\n")
        return self

    def read_runner(self, filename):
        """read RuNNer atomic structure file."""

        read_file = True
        in_file = open(filename, "r")

        line = in_file.readline()
        while line:

            # read a frame
            if "begin" in line.rstrip("/n").split()[0]:

                # prepare sample
                sample = Sample()
                cell = []
                atomid = 0
                total_energy = 0.0
                total_charge = 0.0

                while True:

                    line = read_and_tokenize_line(in_file)

                    if "comment" in line[0]:
                        continue

                    if "lattice" in line[0]:
                        for c in line[1:4]:
                            cell.append(float(c))

                    if "atom" in line[0]:
                        atomid += 1
                        position = [float(pos) for pos in line[1:4]]
                        symbol = line[4]
                        charge = float(line[5])
                        energy = float(line[6])
                        force = [float(frc) for frc in line[7:10]]
                        sample.atomic.append(
                            AtomicData(atomid, position, symbol, charge, energy, force)
                        )

                    if "energy" in line[0]:
                        total_energy = float(line[1])

                    if "charge" in line[0]:
                        total_charge = float(line[1])

                    if "end" in line[0]:
                        break

                # set collective data
                assert len(cell) == 9, "Unexpected number of cell dimension (%d)" % len(
                    cell
                )
                sample.collective = CollectiveData(cell, total_energy, total_charge)
                # add sample to DataSet (list of samples)
                self.dataset.append(sample)

            # next line
            line = in_file.readline()

        return self

    # def read_nnforces(self, filename, uc=UnitConversion()):
    #     """A method that reads predicted force for a given structure"""
    #     nnforces = []
    #     with open(filename, 'r') as infile:
    #         for line in infile:
    #             if "NNforces" in line:
    #                 line = line.rstrip("/n").split()
    #                 nnforces.append([float(_)*uc.force for _ in line[2:5]])
    #     return nnforces

    # def read_nnenergy(self, filename, uc=UnitConversion()):
    #     """A method that reads predicted force for a given structure"""
    #     nnenergy = None
    #     with open(filename, 'r') as infile:
    #         for line in infile:
    #             if "NNenergy" in line:
    #                 line = line.rstrip("/n").split()
    #                 nnenergy = float(line[1])*uc.energy
    #                 break
    #     return nnenergy


# ----------------------------------------------------------------------------
# Setup class for RuNNer adaptor to LAMMPS
# ----------------------------------------------------------------------------


class RuNNerAdaptorForLAMMPS(RunnerAdaptor):
    """An inherited class for conversion file formats between RuNNer and LAMMPS packages."""

    def __init__(self):
        RunnerAdaptor.__init__(self)

    def read_lammps(
        self,
        filename,
        symbol_dict=None,
        uc=UnitConversion(),
    ):
        with open(filename, "r") as in_file:
            # loop over lines in file
            for line in in_file:

                # create a instance of sample data
                sample = Sample()

                # number of steps
                line = next(in_file)
                steps = int(line.split()[0])
                # number of atoms
                next(in_file)
                line = next(in_file)
                number_of_atoms = int(line.split()[0])
                # read cell sizes
                cell = []
                line = next(in_file)
                for n in range(9):
                    if n in [0, 4, 8]:
                        line = next(in_file)
                        line = line.rstrip("/n").split()
                        cell.append((float(line[1]) - float(line[0])) * uc.length)
                    else:
                        cell.append(0.0)

                # read atomic positions, symbol, charge, forces, energy, etc.
                line = next(in_file)
                for n in range(number_of_atoms):
                    line = next(in_file).rstrip("/n").split()
                    atomid = int(line[0])
                    position = [float(pos) * uc.length for pos in line[1:4]]
                    symbol = line[4]
                    charge = float(line[5]) * uc.charge
                    energy = float(line[6]) * uc.energy
                    force = [float(frc) * uc.force for frc in line[7:10]]
                    # convert number to an atomic symbol
                    if symbol_dict is not None:
                        symbol = symbol_dict[symbol]

                    # create atomic data and append it to sample
                    sample.atomic.append(
                        AtomicData(atomid, position, symbol, charge, energy, force)
                    )

                # set collective data
                sample.collective = CollectiveData(
                    cell, sample.total_energy, sample.total_charge
                )

                # add sample to DataSet (list of samples)
                self.dataset.append(sample)

        return self

    def write_lammps(self, filename, uc=UnitConversion()):
        """A method that writes lammps input data."""
        ...


# ----------------------------------------------------------------------------
# Setup class for RuNNer adaptor to VASP
# ----------------------------------------------------------------------------


class RuNNerAdaptorForVASP(RunnerAdaptor):
    """An inherited class for conversion file formats between RuNNer and VASP packages."""

    def __init__(self):
        RunnerAdaptor.__init__(self)

    def write_poscar(
        self,
        filename="POSCAR",
        symbol_list=None,
        uc=UnitConversion(),
        scaling_factor=1.0,
        number_of_strucure=None,
        seed=1234,
    ):

        if number_of_strucure is None:
            samples = self.dataset.samples
        else:
            assert (
                number_of_strucure <= self.dataset.number_of_samples
            ), "Number of structures exceeds number of samples"
            import random

            random.seed(seed)
            samples = random.sample(self.dataset.samples, int(number_of_strucure))

        index = 0
        for sample in samples:

            index += 1
            with open(filename + "_%d" % index, "w") as out_file:

                # comment
                out_file.write(", ATOM=")
                for symbol in symbol_list:
                    out_file.write("%s " % symbol)
                out_file.write("\n")

                # write scaling factor
                out_file.write("%15.10f\n" % scaling_factor)

                # cell
                for i in range(3):
                    out_file.write(
                        "%15.10f %15.10f %15.10f\n"
                        % tuple(
                            [
                                c * uc.length
                                for c in sample.collective.cell[3 * i : 3 * (i + 1)]
                            ]
                        )
                    )

                # number of atoms for each symbol
                for symbol in symbol_list:
                    out_file.write(
                        "%d " % sample.get_number_of_atoms_for_symbol(symbol)
                    )
                out_file.write("\n")

                # atom positions
                out_file.write("Cartesian \n")
                for symbol in symbol_list:
                    for atom in sample.get_atoms_for_symbol(symbol):
                        out_file.write(
                            "%15.10f %15.10f %15.10f\n"
                            % tuple([f * uc.length for f in atom.position])
                        )

    def read_poscar(
        self,
        filename="POSCAR",
        symbol_list=None,
        uc=UnitConversion(),
    ):
        # create a instance of sample data
        sample = Sample()

        with open(filename, "r") as in_file:

            # loop over lines in file
            for line in in_file:

                # create a instance of sample data
                sample = Sample()

                # read scaling factor
                line = read_and_tokenize_line(in_file)
                scaling_factor = float(line[0])
                # print ("Scaling factor (POSCAR): ", scaling_factor)

                # read cell info
                cell = []
                for n in range(3):
                    line = read_and_tokenize_line(in_file)
                    for m in range(3):
                        cell.append(float(line[m]) * scaling_factor * uc.length)
                # print(cell)

                line = read_and_tokenize_line(in_file)
                natoms_each_type = [int(l) for l in line]
                # print (natoms_each_type)

                # skip the line
                line = next(in_file)
                if "select" in line.lower():
                    line = next(in_file)

                # check cartesian coordinates
                if "car" not in line.lower():
                    # print (line)
                    raise AssertionError("Expected cartesian coordinates!")

                # read atomic positions
                atomid = 0
                for natoms, n in zip(natoms_each_type, range(len(natoms_each_type))):

                    for i in range(natoms):

                        atomid += 1
                        line = read_and_tokenize_line(in_file)

                        position = [
                            float(pos) * scaling_factor * uc.length for pos in line[0:3]
                        ]
                        symbol = symbol_list[n]

                        # create atomic data and append it to sample
                        sample.atomic.append(
                            AtomicData(
                                atomid, position, symbol, 0.0, 0.0, (0.0, 0.0, 0.0)
                            )
                        )
                        # (charge, energy, and force) * uc = 0
                        # print (symbol, position)
                # Assuming it is the end of the POSCAR
                break

            # set collective data
            sample.collective = CollectiveData(
                cell, sample.total_energy, sample.total_charge
            )

            # add sample to DataSet (list of samples)
            self.dataset.append(sample)

        return self

    def read_outcar(
        self,
        filename="OUTCAR",
        uc=UnitConversion(),
    ):
        with open(filename, "r") as in_file:
            # loop over lines in file

            for line in in_file:

                # read line
                # line = next(in_file)
                # print(line)

                # read the force section
                if "POSITION" in line:
                    next(in_file)
                    for atom in self.dataset.samples[0].atomic:
                        line = read_and_tokenize_line(in_file)
                        force = [float(frc) * uc.force for frc in line[3:6]]
                        atom.force = tuple(force)
                        # print(line, atom.force)

                if "TOTEN" in line:
                    total_energy = float(line.rstrip("/n").split()[-2])
                    # print (total_energy)
                    self.dataset.samples[0].collective.total_energy = (
                        total_energy * uc.energy
                    )

        return self

    def read_vasp(
        self,
        symbol_list=None,
        uc=UnitConversion(),
    ):
        self.read_poscar(symbol_list=symbol_list, uc=uc)
        self.read_outcar(uc=uc)
        return self


# if __name__ == "__main__":
# data = RunnerAdaptor().read_runner(filename="input.data").write_runner(filename="input.2.data")
# uc = UnitConversion(energy_conversion=EV_TO_HARTREE, length_conversion=ANGSTROM_TO_BOHR)
# RuNNerAdaptorForVASP().read_vasp(symbol_list=['H', 'O'], uc=uc).write_runner(filename='input.vasp.data')
# vasp.read_POSCAR(symbol_list=['O', 'H'], uc=uc)
# vasp.read_OUTCAR(uc=uc)
# uc = UnitConversion(energy_conversion=EV_TO_HARTREE, length_conversion=ANGSTROM_TO_BOHR)
# RuNNerAdaptorForVASP().read_runner("input.data").write_poscar(symbol_list=['H', 'O'], uc=uc.inverse, number_of_strucure=1)
