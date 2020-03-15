from distutils.core import setup, Extension
from math import pi
import numpy as np 
import os, sys, shutil, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--omega', default= 1 * pi, type=float, metavar='\omega',
                    help='the angular frequency of the non-inverted harmonic oscillator (in units of time^{-1})')
parser.add_argument('--n_max', default = 160, type=int, metavar='n_{max}',
                    help='the highest energy level of simulation. The last output (Failure) will be True if the amplitude on this level is too high.')
parser.add_argument('--gamma', default = 2 * pi, type=float, metavar='\gamma',
                    help='the measurement strength \gamma on the particle')
args = parser.parse_args()




# Please rewrite the following arguments based on your OS and your prescription of compilation if necessary
# Please refer to https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor . Usually Python uses GCC as the default compiler, and then GNU compiler should be selected. The arguments starting with "-I" mean to "include" those directories.

link_options = ['-Wl,--start-group', os.environ['MKLROOT']+'/lib/intel64/libmkl_intel_ilp64.a', os.environ['MKLROOT']+'/lib/intel64/libmkl_intel_thread.a', os.environ['MKLROOT']+'/lib/intel64/libmkl_core.a', '-Wl,--end-group', '-liomp5', '-lpthread', '-lm', '-ldl']

compiler_options = ['-DMKL_ILP64','-m64']


##############################################################################
# The following is the compilation program. 

def compile(n_max, omega, gamma):
    assert type(n_max)==int and type(omega)==float

    # It invokes the native "distutils.core" of Python by setting the commandline arguments stored in sys.argv to the desired one ("build")

    # set the "build" command
    original_args_exist = False
    if len(sys.argv)>=2:
        original_args=sys.argv[1:]
        sys.argv = [sys.argv[0], "build"]
        original_args_exist = True
    else: sys.argv.append("build")

    os.environ["MKL_NUM_THREADS"] = "1"

    package_name = 'simulation'

    module1 = Extension(package_name,language='c++',
                    define_macros = [('N_MAX', str(n_max)), ('OMEGA', repr(omega))],
                    include_dirs = [np.get_include(), os.path.join(os.environ['MKLROOT'],'include')],
                    sources = ['simulation_i.cpp'], 
                    extra_compile_args = compiler_options + ['-std=c++14','-Ofast','-funroll-loops', '-march=native', '-flto','-fuse-linker-plugin','--param', 'ipcp-unit-growth=2000','-fno-stack-protector','-fmerge-all-constants'], 
                    extra_link_args = link_options + ['-std=c++14','-Ofast','-fdelete-null-pointer-checks','-funroll-loops', '-march=native', '-fwhole-program','-flto','-fuse-linker-plugin','--param', 'ipcp-unit-growth=2000','-fno-stack-protector','-fmerge-all-constants']
                    )

    setup (name = package_name,
       version = '1.0',
       description = 'do simulation steps',
       author = 'Wang Zhikang',
       ext_modules = [module1])

    # copy the compiled C module to the root to import
    compiled_files = glob.glob('build/**/*')
    for compiled_file in compiled_files:
        if 'temp' not in compiled_file:
            shutil.move(compiled_file, os.path.basename(compiled_file), copy_function=shutil.copy2)

    # restore the original commandline arguments
    if original_args_exist: sys.argv = [sys.argv[0]]+original_args
    else: sys.argv.pop(1)

compile(n_max=args.n_max, omega=args.omega, gamma=args.gamma)
