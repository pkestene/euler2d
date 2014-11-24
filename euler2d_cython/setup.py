# python setup.py build_ext --inplace

import  os, re
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

# enable to dump annotate html for each pyx source file
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

from distutils.command.clean import clean as _clean
from distutils.dir_util import remove_tree
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

class clean(_clean):
  def walkAndClean(self):
    os.path.walk(".",wselect,[])
  def run(self):
    if (os.path.exists('./build')): remove_tree('./build')
    if (os.path.exists('./dist')):  remove_tree('./dist')
    self.walkAndClean()

# --------------------------------------------------------------------
# Build extensions
# --------------------------------------------------------------------
setup(
    name="euler2d",
    author='Pierre Kestener',
    version='0.1',
    packages = ['euler2d'],
    package_data={'' : ['*.pxd']},
    ext_modules = [
        Extension(name='test.cython_test',
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

