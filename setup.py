#!/usr/bin/env python
import os
import sys
from distutils.core import setup

import setuptools
from setuptools.command.install import install

VERSION = "v0.0.1"


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
    install_requires=requirements
)