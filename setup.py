from setuptools import setup, Extension
import numpy as np
import mpi4py
import shutil
import os
import shlex


def run_command(exe, args):
    cmd = shutil.which(exe)
    if not cmd:
        return []
    if not isinstance(args, str):
        args = ' '.join(args)
    try:
        with os.popen(cmd + ' ' + args) as f:
            return shlex.split(f.read())
    except Exception:
        return []


def get_linker_args():
    compiler = os.getenv('CC')
    if compiler is None:
        print('Error: MPI compiler is not specified. Please specify the MPI compiler using the "CC" environment variable')
    args = run_command(compiler, '-show')
    linker_args = []
    for arg in args:
        if arg.startswith('-l') or arg.startswith('-L'):
            linker_args.append(arg)
    # linker_args_str = ' '.join(linker_args)
    # return linker_args_str
    return linker_args


core_module = Extension('repast4py._core', sources=['src/repast4py/coremodule.cpp'], language='c++',
                        extra_compile_args=["-std=c++11"], depends=["core.h"], extra_link_args=get_linker_args(),
                        include_dirs=[np.get_include(), mpi4py.get_include()])
space_module = Extension('repast4py._space', sources=['src/repast4py/spacemodule.cpp',
                         'src/repast4py/geometry.cpp', 'src/repast4py/space.cpp', 'src/repast4py/distributed_space.cpp',
                         'src/repast4py/SpatialTree.cpp'], language='c++',
                         extra_compile_args=["-std=c++11"], depends=['space.h', 'grid.h', 'cspace.h', 'space_types.h',
                         'geometry.h', 'distributed_space.h', 'borders.h'], extra_link_args=get_linker_args(),
                         include_dirs=[np.get_include(), mpi4py.get_include()])

setup(
    description="repast4py package",
    ext_modules=[core_module, space_module]
)
