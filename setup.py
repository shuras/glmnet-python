from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
import os, sys

config = Configuration(
    'glmnet',
    parent_package=None,
    top_path=None
)

f_sources = ['glmnet/glmnet.pyf','glmnet/glmnet.f']
fflags= ['-fdefault-real-8', '-ffixed-form']

config.add_extension(name='_glmnet',
                     sources=f_sources,
                     extra_f77_compile_args=fflags,
                     extra_f90_compile_args=fflags
)

config_dict = config.todict()
config_dict['packages'].extend(['glmnet.glmnet', 'glmnet.util', 'glmnet.cv'])
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(version='0.9',
          description='Python wrappers for the GLMNET package',
          author='Matthew Drury',
          author_email='matthew.drury.83@gmail.com',
          url='github.com/madrury/glmnet-python',
          license='GPL2',
          requires=['NumPy (>= 1.3)'],
          **config_dict
)

