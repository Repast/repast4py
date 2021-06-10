from setuptools import setup, find_packages, Extension
import numpy as np
import mpi4py

core_module = Extension('repast4py._core', sources = ['src/repast4py/coremodule.cpp'], language='c++',
    extra_compile_args=["-std=c++11"], depends=["core.h"])
space_module = Extension('repast4py._space', sources = ['src/repast4py/spacemodule.cpp', 
    'src/repast4py/geometry.cpp', 'src/repast4py/space.cpp', 'src/repast4py/distributed_space.cpp', 'src/repast4py/SpatialTree.cpp'], language='c++',
    extra_compile_args=["-std=c++11"], depends=['space.h', 'grid.h', 'cspace.h', 'space_types.h', 
        'geometry.h', 'distributed_space.h', 'borders.h'])

setup(
    name = 'repast4py',
    version = 0.1,
    description = "repast4py package",
    packages = find_packages('src'),
    package_dir = {'':'src'},
    include_dirs = [np.get_include(), mpi4py.get_include()],
    ext_modules = [core_module, space_module]
)