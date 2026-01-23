#!/usr/bin/env python
import setuptools
from distutils.core import setup

VERSION = "v0.1.0"


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="alquimia-fair-forge",
    version=VERSION,
    description="Alquimia Fair Forge library",
    author="Alquimia AI",
    url="https://github.com/Alquimia-ai/fair-forge.git",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
