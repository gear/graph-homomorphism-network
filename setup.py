#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='ghc',
    version='0.1.1',
    license='MIT',
    description='A prototype for graph homomorphism convolution.',
    author='Hoang NT',
    author_email='me@gearons.org',
    url='https://github.com/gear/graph-homomorphism-network',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'graph homomorphism', 'graph neural networks'
    ],
    python_requires='>=3.7',
)
