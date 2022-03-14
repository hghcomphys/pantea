from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='structure_cpp',
    ext_modules=[
        CppExtension('structure_cpp', ['structure.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })