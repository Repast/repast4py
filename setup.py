from setuptools import setup, Extension
import numpy as np
import mpi4py
import shutil
import os, sys
import shlex

import platform

IS_WINDOWS = platform.system() == "Windows"
IS_DARWIN = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

def is_64bit() -> bool:
    """ Returns True if the python interpreter is 64-bit, independent of the OS arch.
    """
    return sys.maxsize > 2**32

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
        
    linker_args = []

    # NOTE Windows setuptools apparently does not use the CC env variable.
    if IS_WINDOWS:
        pass
    
    else:
        compiler = os.getenv('CC')
        if compiler is None:
            print('Error: MPI compiler is not specified. Please specify the MPI compiler using the "CC" environment variable')
        
        args = run_command(compiler, '-show')
    
        for arg in args:
            if arg.startswith('-l') or arg.startswith('-L'):
                linker_args.append(arg)
        # linker_args_str = ' '.join(linker_args)
        # return linker_args_str
    return linker_args

def get_compiler_args():
    if IS_WINDOWS:
        compile_args=["/std:c++latest"]
    else:
        compile_args=["-std=c++11"]

    return compile_args

def get_extra_includes():
    if IS_WINDOWS:
        return [os.environ['MSMPI_INC']]
    else:
        return []

def get_lib_dirs():
    if IS_WINDOWS:
        if is_64bit():
            return [os.environ['MSMPI_LIB64']]
        else: 
            return [os.environ['MSMPI_LIB32']]
    else:
        return []

def get_libs():
    if IS_WINDOWS:
        return ['msmpi']
    else:
        return []


core_module = Extension('repast4py._core', sources=['src/repast4py/coremodule.cpp'], language='c++',
                        extra_compile_args=get_compiler_args(), depends=["core.h"], extra_link_args=get_linker_args(),
                        include_dirs=[np.get_include(), mpi4py.get_include()])
space_module = Extension('repast4py._space', sources=['src/repast4py/spacemodule.cpp',
                         'src/repast4py/geometry.cpp', 'src/repast4py/space.cpp', 'src/repast4py/distributed_space.cpp',
                         'src/repast4py/SpatialTree.cpp'], language='c++',
                         extra_compile_args=get_compiler_args(), depends=['space.h', 'grid.h', 'cspace.h', 'space_types.h',
                         'geometry.h', 'distributed_space.h', 'borders.h'], extra_link_args=get_linker_args(),
                         include_dirs=[np.get_include(), mpi4py.get_include(), *get_extra_includes()],
                         library_dirs=get_lib_dirs(),
                         libraries=get_libs())

setup(
    description="repast4py package",
    ext_modules=[core_module,space_module]
)
