from setuptools import setup, find_packages, Extension
import numpy as np

core_module = Extension('repast4py.core', sources = ['src/repast4py/coremodule.cpp'], language='c++',
    extra_compile_args=["-std=c++11"], depends=["core.h"])
space_module = Extension('repast4py.space', sources = ['src/repast4py/spacemodule.cpp'], language='c++',
    extra_compile_args=["-std=c++11"], depends=['space.h'])

setup(
    name = 'repast4py',
    version = 0.1,
    description = "repast4py package",
    packages = find_packages('src'),
    package_dir = {'':'src'},
    include_dirs = [np.get_include()],
    ext_modules = [core_module, space_module]
)