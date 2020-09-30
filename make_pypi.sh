#!/bin/bash
export CC=/usr/bin/gcc-7
export CXX=/usr/bin/g++-7

python setup.py sdist bdist_egg
# twine upload dist/*