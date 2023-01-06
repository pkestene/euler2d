#
# Build project:
# python setup.py build_ext --inplace
#

import os, re
from os.path import join as pjoin
#from setuptools import setup
from distutils.core import setup
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

include_dirs = [numpy.get_include()]

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything
# --------------------------------------------------------------------
relist=['^.*~$','^core\.*$','^#.*#$','^.*\.aux$','^.*\.pyc$','^.*\.o$','^.*\.so$','^.*\.vti$','^.*\.c$','^.*\.html$']
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
    self.walkAndClean()

# --------------------------------------------------------------------
# Build extensions
# --------------------------------------------------------------------
setup(
  name="euler2d",
  author='Pierre Kestener',
  version='0.1',
  packages = ['euler2d'],
  package_data={'euler2d' : ['*.pxd']},
  ext_modules = [
    Extension(name='test.cython_test',
              include_dirs = include_dirs,
              sources=['test/cython_test.pyx']),
    Extension(name='euler2d.hydroMonitoring',
              include_dirs = include_dirs,
              #extra_compile_args=["-pg"],
              #extra_link_args=["-pg"],
              sources=['euler2d/hydroMonitoring.pyx']),
    Extension(name='euler2d.hydroUtils',
              include_dirs = include_dirs,
              #extra_compile_args=["-pg"],
              #extra_link_args=["-pg"],
              sources=['euler2d/hydroUtils.pyx']),
    Extension(name='euler2d.hydroRun',
              include_dirs = include_dirs,
              #extra_compile_args=["-pg"],
              #extra_link_args=["-pg"],
              sources=['euler2d/hydroRun.pyx'])
  ],

  cmdclass={'build_ext': build_ext, 'clean': clean}
)
