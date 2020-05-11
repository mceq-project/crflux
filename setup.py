# -*- coding: utf-8 -*-
from setuptools import setup

__version__ = "1.0.6"

def setup_package():
    from os.path import abspath, dirname, join
    import sys

    this_directory = abspath(dirname(__file__))
    if sys.version_info.major == 3:
        with open(join(this_directory, 'README.md'), encoding='utf-8') as f:
            long_description = f.read()
    else:
        with open(join(this_directory, 'README.md')) as f:
            long_description = f.read()

    skip_marker = "# `crflux"
    long_description = long_description[long_description.index(skip_marker) :].lstrip()

    arguments = dict(
        name='crflux',
        version=__version__,
        author='Anatoli Fedynitch',
        author_email='afedynitch@gmail.com',
        description='Parameterizations of the cosmic ray flux at Earth',
        long_description=long_description,
        license="MIT",
        url='https://github.com/afedynitch/crflux',
        package_dir={'crflux': 'crflux'},
        packages=['crflux'],
        package_data={'crflux': ['GSF_spline_20171007.pkl.bz2']},
        install_requires=['setuptools', 'numpy', 'scipy'],
        extras_require={"tests": ["pytest", "matplotlib"]},
        py_modules=['six'],
        classifiers=[
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering :: Physics',
            'Intended Audience :: Science/Research',
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: MIT License'
        ],
    )

    setup(**arguments)

if __name__ == '__main__':
    setup_package()
