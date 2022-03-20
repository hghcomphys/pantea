from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='extension_cpp',
    ext_modules=[
      CppExtension(name = 'cutoff_cpp', sources = ['cutoff.cpp']),
      CppExtension(name = 'radial_cpp', sources = ['radial.cpp']),
      CppExtension(name = 'angular_cpp', sources = ['angular.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })