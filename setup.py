import setuptools
from numpy.distutils.misc_util import Configuration

config = Configuration()

f_sources = ['glmnet/glmnet.pyf','glmnet/glmnet.f']
fflags= ['-fdefault-real-8', '-ffixed-form']

config.add_extension(name='_glmnet',
                     sources=f_sources,
                     extra_f77_compile_args=fflags,
                     extra_f90_compile_args=fflags)

config_dict = config.todict()

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(name='glmnet',
          version='0.9',
          packages=setuptools.find_packages(),
          description='Python wrappers for the GLMNET package',
          author='Matthew Drury',
          author_email='matthew.drury.83@gmail.com',
          url='github.com/madrury/glmnet-python',
          license='GPL2',
          setup_requires=['numpy >= 1.3'],
          **config_dict)

