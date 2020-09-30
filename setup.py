from os import path
from setuptools import setup, Extension

cwd = path.abspath(path.dirname(__file__))
with open(path.join(cwd, 'pybarycuda/README.md'), encoding='utf-8') as f:
    full_desc = f.read()

barycuda_core = Extension('pybarycuda.core',
	sources=['pybarycuda/core.cpp'],
	include_dirs = ['include'],
	runtime_library_dirs=["$ORIGIN/./lib/"],
	library_dirs = ['pybarycuda/lib'],
	libraries=['barycuda'])

setup(name='pybarycuda',
	author='Srimukh Sripada', 
	version='1.0.0',
	license='MIT',
	author_email='hi@srimukh.com',
	url='https://github.com/postmalloc/barycuda',
	description="A tiny CUDA library for fast barycentric operations",
	long_description=full_desc,
  long_description_content_type='text/markdown',
	ext_modules=[barycuda_core],
	packages=['pybarycuda'],
	include_package_data=True
)
