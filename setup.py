from setuptools import setup, Extension

barycuda_core = Extension('barycuda.core',
	sources=['barycuda/core.cpp'],
	include_dirs = ['include'],
	library_dirs = ['lib'],
	libraries=['barycuda'])

setup(name = 'barycuda',
	author = 'Srimukh Sripada', 
	version = '1.0.0',
	license = 'MIT',
	author_email = 'hi@srimukh.com',
	url = 'https://github.com/postmalloc/barycuda',
	description="A tiny CUDA library for fast barycentric operations",
	ext_modules=[barycuda_core],
	packages=['barycuda'],
	include_package_data=True,
	package_data={'barycuda': ['lib/libbarycuda.so']}
)
