# !env > .env
import os


def set_env(filename):
    """
    An utility function to import conda environment variables into notebook when running on GPUs.
    
    On liux system:
        $ source activate <your_conda_environment>
        $ env > .env
        
    then on the top of the notebook use the following:
        from utils import set_env
        set_env(".env")
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split("=")
            key, value = line[0], "".join(line[1:])
            os.environ[key] = value