from distutils.core import setup, Extension
from math import pi
import numpy as np 
import os, sys, shutil, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lambda_', default= -pi/100., type=float, metavar='\lambda',
                    help='the coefficent of the inverted quartic anharmonic oscillator (it should be negative)')
parser.add_argument('--x_max', default=13, type=float, metavar='x_{max}',
                    help='the distance from the center to the border of the simulation space')
parser.add_argument('--grid_size', default = 0.05, type=float, metavar='h',
                    help='the grid size of the discretized simulation space')
parser.add_argument('--mass', default = 1./pi, type=float, metavar='m',
                    help='the mass of the simulated particle')
parser.add_argument('--moment', default = 5, type=int,
                    help='the order of the distribution moments to compute in the compiled function "get_moments"')
args = parser.parse_args()




# Please rewrite the following arguments based on your OS and your prescription of compilation if necessary
# Please refer to https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor . Usually Python uses GCC as the default compiler, and then GNU compiler should be selected. The arguments starting with "-I" mean to "include" those directories.

link_options = ['-Wl,--start-group', os.environ['MKLROOT']+'/lib/intel64/libmkl_intel_ilp64.a', os.environ['MKLROOT']+'/lib/intel64/libmkl_intel_thread.a', os.environ['MKLROOT']+'/lib/intel64/libmkl_core.a', '-Wl,--end-group', '-liomp5', '-lpthread', '-lm', '-ldl']

compiler_options = ['-DMKL_ILP64','-m64']

##############################################################################
# The following is the compilation program. 

def compile(x_max, grid_size, mass, lambda_, moment):
    assert lambda_< 0., 'inverted quartic oscillator coefficient \lambda should be negative'
    assert mass> 0., 'the mass should be positive'
    assert x_max> 0., 'the size of the simulation space (2 * x_max) should be positive'
    assert grid_size> 0., 'the simulation grid size should be positive'

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
                    define_macros = [('X_MAX', str(x_max)), ('GRID_SIZE', repr(grid_size)), ('MASS',repr(mass)), ('LAMBDA', repr(lambda_)), ('MOMENT', str(moment))], # pass the defining parameters
                    include_dirs = [np.get_include(), os.path.join(os.environ['MKLROOT'],'include')], # set the includes
                    sources = ['simulation_quart.cpp'], 
                    extra_compile_args = compiler_options+['-Ofast','-funroll-loops', '-march=native', '-flto','-fuse-linker-plugin','--param', 'ipcp-unit-growth=2000', '-std=c++14','-fno-stack-protector','-fmerge-all-constants'], 
                    extra_link_args = link_options+['-Ofast','-fdelete-null-pointer-checks','-funroll-loops', '-march=native', '-fwhole-program','-flto','-fuse-linker-plugin','--param', 'ipcp-unit-growth=2000','-std=c++14','-fno-stack-protector','-fmerge-all-constants'])

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

compile(x_max=args.x_max, grid_size=args.grid_size, mass=args.mass, lambda_=args.lambda_, moment=args.moment)
