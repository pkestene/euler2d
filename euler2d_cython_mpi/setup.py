#
# Build project:
# python setup.py build_ext --inplace
#

import os, re
import subprocess
#import commands

from os.path import join as pjoin

from setuptools import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

# enable to dump annotate html for each pyx source file
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

# clean command
from distutils.command.clean import clean as _clean
from distutils.dir_util import remove_tree

# numpy
import numpy

#from distutils.core import setup
#from Cython.Build import cythonize

#
# check if mpi4py is available
#
Have_MPI = True
try:
    import mpi4py
except ImportError:
    Have_MPI = False
    print("Please install mpi4py !")

compiler = 'gcc'
#compiler = 'intel'
if compiler == 'intel':
    extra_compile_args = ['-O3']
else:
    extra_compile_args = []

# mpi flags
mpi_inc_dirs = []
mpi_compile_args = []
mpi_link_args = []

if Have_MPI:
    mpic = 'mpicc'
    if compiler == 'intel':
        print("WARNING: Intel compiler setup has not been tested !")
        link_args = subprocess.getoutput(mpic + ' -cc=icc -link_info')
        link_args = link_args[3:]
        compile_args = subprocess.getoutput(mpic + ' -cc=icc -compile_info')
        compile_args = compile_args[3:]
    else:
        link_args = subprocess.getoutput(mpic + ' --showme:link').split()
        compile_args = subprocess.getoutput(mpic + ' --showme:compile').split()
    mpi_link_args = link_args
    mpi_compile_args = compile_args
    mpi_inc_dirs.append(mpi4py.get_include())

print("##################")
print("mpi_link_args, mpi_compile_args, mpi_inc_dirs")
print(mpi_link_args, mpi_compile_args, mpi_inc_dirs)
print("##################")

include_dirs = [numpy.get_include()]

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything
# --------------------------------------------------------------------
relist=['^.*~$','^core\.*$','^#.*#$','^.*\.aux$','^.*\.pyc$','^.*\.o$','^.*\.so$','^.*\.vti$','^.*\.pvti$','^.*\.c$','^.*\.html$']
reclean=[]

for restring in relist:
  reclean.append(re.compile(restring))

def wselect(args,dirname,names):
  for n in names:
    for rev in reclean:
      if (rev.match(n)):
        os.remove("%s/%s"%(dirname,n))
        break

def remove_dir(some_dir):
  if (os.path.exists(some_dir)): remove_tree(some_dir)

def remove_files(my_pattern):
  import glob
  fileList = glob.glob(my_pattern)

  # Iterate over the list of filepaths & remove each file.
  for filePath in fileList:
    try:
      os.remove(filePath)
    except:
      print("Error while deleting file : ", filePath)

class clean(_clean):
  def walkAndClean(self):
    os.walk(".",wselect,[])
  def run(self):
    remove_dir("./build")
    remove_dir("./dist")
    remove_dir("./euler2d/__pycache__")
    remove_files("euler2d/*.so")
    remove_files("euler2d/*.c")
    remove_files("euler2d/*.html")
    remove_files("euler2d/*.pyc")
    remove_files("test/*.so")
    remove_files("test/*.c")
    remove_files("test/*.html")
    remove_files("test/*.pyc")
    remove_files("./*.*vti")
    self.walkAndClean()

# --------------------------------------------------------------------
# Build extensions
# --------------------------------------------------------------------
setup(
    name="euler2d_mpi",
    author='Pierre Kestener',
    version='0.1',
    ext_modules = [
        Extension(name='test.cython_test',
                  include_dirs = include_dirs,
                  sources=['test/cython_test.pyx']),
        Extension(name='euler2d.hydroMonitoring',
                  include_dirs = include_dirs,
                  sources=['euler2d/hydroMonitoring.pyx']),
        Extension(name='euler2d.hydroUtils',
                  include_dirs = include_dirs+mpi_inc_dirs,
                  libraries=['mpi'],
                  extra_link_args=mpi_link_args,
                  extra_compile_args=mpi_compile_args,
                  sources=['euler2d/hydroUtils.pyx']),
        Extension(name='euler2d.hydroRun',
                  include_dirs = include_dirs+mpi_inc_dirs,
                  libraries=['mpi'],
                  extra_link_args=mpi_link_args,
                  extra_compile_args=mpi_compile_args,
                  sources=['euler2d/hydroRun.pyx'])
    ],
    cmdclass={'build_ext': build_ext, 'clean': clean}
)
