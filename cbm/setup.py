# To build the extension, follow the instructions below:
# Set the MKLROOT environment variables to the path of your MKL installation.
# This can be done by running `source /path/to/setvars.sh` in your terminal.
# Run `python setup.py build_ext --inplace` in the `cbm` directory.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Assuming MKLROOT is set in your environment variables
mkl_root = os.getenv('MKLROOT')  # Replace '/path/to/your/mkl' with a default path if MKLROOT is not set

extra_compile_args = [
    '-m64',
    '-I"{}/include"'.format(mkl_root),
    '-fopenmp',
    '-march=native',
    '-O2',
    '-std=c++20',
    '-I/home/alves/arbok/source/arbok/include',
    f'-I{os.path.dirname(os.path.abspath(__file__))}/../arbok/source/arbok/include',
]

extra_link_args = [
    '-m64',  
    '-L{}/lib'.format(mkl_root),
    '-Wl,--no-as-needed', 
    '-lmkl_intel_ilp64', 
    '-lmkl_gnu_thread', 
    '-lmkl_core',
    '-lgomp', 
    '-lpthread',
    '-lm',
    '-ldl',
    '-L/home/alves/arbok/build/',
    '-larbok',
    f'-L{os.path.dirname(os.path.abspath(__file__))}/../arbok/build/',
    '-larbok',
]

install_requires = [
    "torch>=2.0.0",
]

setup(
    name='cbm_mkl_cpp',
    packages=find_packages(),
    install_requires=install_requires,
    version="0.1",
    ext_modules=[
        CppExtension(
            'cbm_mkl_cpp',
            ['cbm_extensions.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)