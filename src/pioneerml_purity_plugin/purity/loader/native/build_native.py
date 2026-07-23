#!/usr/bin/env python3
"""Build the experimental NumPy-only PURITY pybind11 extension in place."""

import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

setup(
    name="pioneerml-purity-native-loader",
    version="0.0.0",
    ext_modules=[
        Pybind11Extension(
            "_purity_loader_native",
            [str(HERE / "purity_loader_native.cpp")],
            cxx_std=20,
            extra_compile_args=["-O3", "-DNDEBUG"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    script_args=["build_ext", "--inplace", "--force"],
)
