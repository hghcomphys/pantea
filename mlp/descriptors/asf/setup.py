from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cutoff_cpp',
    ext_modules=[
        CppExtension('cutoff_cpp', ['cutoff.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })