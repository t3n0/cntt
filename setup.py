from setuptools import setup, find_packages

setup(name='cntt',
      version='0.4',
      description='CNTT, also C-entity, computes, displays and manipulates carbon nanotubes properties.',
      author='Stefano Dal Forno',
      author_email='tenobaldi@gmail.com',
      url='https://github.com/t3n0/cntt',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Physics',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 3',
      ],
      keywords=['swcnt', 'cnt', 'band', 'bandstructure', 'carbon', 'nanotubes'],
      #packages=find_packages(),
      packages = ['cntt'],
      python_requires=">=3.5",
      install_requires=[],
      entry_points={ "console_scripts": [ "cntt=cntt.main:main" ],},
      )