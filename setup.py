#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "tqdm>=4.6",
    "ase>=3.2",
    "frozendict>=2.3",
    "flax>=0.6",
    "pydantic>=1.10",
    "pandas>=1.5.3",
]

test_requirements = [
    "pytest>=3",
]

_dict = {}
with open("jaxip/_version.py") as f:
    exec(f.read(), _dict)
__version__ = _dict["__version__"]

setup(
    author="Hossein Ghorbanfekr",
    author_email="hgh.comphys@gmail.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        # "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        # "Programming Language :: Python :: 3.9",
        # "Programming Language :: Python :: 3.10",
    ],
    description="JAX-based Interatomic Potential",
    entry_points={
        "console_scripts": [
            "jaxip=jaxip.cli:main",
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,  # + "\n\n" + history,
    include_package_data=True,
    keywords="jaxip",
    name="jaxip",
    packages=find_packages(include=["jaxip", "jaxip.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/hghcomphys/jaxip",
    version=__version__,
    zip_safe=False,
)
