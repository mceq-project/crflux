# -*- coding: utf-8 -*-
from setuptools import setup

__version__ = "1.0.0rc3"

def setup_package():
    long_description = ''.join(open('README.rst').readlines()[4:])

    arguments = dict(
        name='particletools',
        version=__version__,
        author='Anatoli Fedynitch',
        author_email='afedynitch@gmail.com',
        description='Translating particle codes from CR models from/to PDG codes',
        long_description=long_description,
        license="MIT",
        url='https://github.com/afedynitch/ParticleDataTool',
        package_dir={'particletools': 'particletools'},
        packages=['particletools'],
        install_requires=['setuptools'],
        py_modules=["six"],
        package_data={'particletools': ['ParticleData.xml']},
        classifiers=[
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Physics',
            'Intended Audience :: Science/Research',
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: MIT License'
        ],
    )

    setup(**arguments)

if __name__ == '__main__':
    setup_package()
